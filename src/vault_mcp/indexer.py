"""Vault indexer — ChromaDB + sentence-transformers for semantic search."""

import asyncio
import logging
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .catalog import build_inventory
from .chunker import chunk_document
from .parser import infer_domain, parse_document

logger = logging.getLogger(__name__)


class VaultIndexer:
    """Manages chunking, embedding, and semantic search over the vault."""

    def __init__(
        self,
        vault_path: Path,
        chroma_path: Path,
        model_name: str,
        watch_files: bool = True,
    ):
        self.vault_path = vault_path
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.watch_files = watch_files
        self._model: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None
        self._ready = False
        self._watch_task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def start(self):
        """Initialize ChromaDB + embedding model, then run incremental index."""
        loop = asyncio.get_event_loop()

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = await loop.run_in_executor(
            None, lambda: SentenceTransformer(self.model_name)
        )

        logger.info("Initializing ChromaDB at: %s", self.chroma_path)
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = client.get_or_create_collection(
            name="vault_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Running initial index...")
        stats = await self.index_all(force=False)
        logger.info(
            "Index ready — %d files, %d chunks in %.1fs",
            stats["files"], stats["chunks"], stats["elapsed"],
        )
        self._ready = True

        if self.watch_files:
            self._watch_task = asyncio.create_task(self._watch_vault())

    async def _watch_vault(self):
        """Background task: watch VAULT_PATH for .md changes and re-index live.

        Uses watchfiles (already a transitive dep via uvicorn) to react to
        external file changes — git pull, manual edits in Obsidian, sync
        scripts, etc. Without this, vault-mcp's index only catches changes
        made through its own write tools and at server startup.
        """
        try:
            from watchfiles import awatch, Change
        except ImportError:
            logger.warning(
                "watchfiles not installed — auto-reindex disabled. "
                "Install with: pip install watchfiles"
            )
            return

        logger.info("Filesystem watcher started on %s", self.vault_path)
        loop = asyncio.get_event_loop()

        try:
            async for changes in awatch(self.vault_path, recursive=True):
                for change_type, abs_path_str in changes:
                    abs_path = Path(abs_path_str)
                    if abs_path.suffix.lower() != ".md":
                        continue
                    try:
                        rel = str(abs_path.relative_to(self.vault_path))
                    except ValueError:
                        continue

                    if change_type == Change.deleted:
                        logger.info("Watcher: removing %s", rel)
                        try:
                            await loop.run_in_executor(
                                None, self._delete_files_chunks, [rel]
                            )
                        except Exception:
                            logger.exception("Watcher: failed to delete %s", rel)
                    else:
                        logger.info(
                            "Watcher: re-indexing %s (%s)", rel, change_type.name
                        )
                        try:
                            await self.index_file(rel)
                        except Exception:
                            logger.exception("Watcher: failed to index %s", rel)
        except asyncio.CancelledError:
            logger.info("Filesystem watcher cancelled")
            raise
        except Exception:
            logger.exception("Filesystem watcher crashed (auto-reindex disabled)")

    async def index_all(self, force: bool = False) -> dict:
        """Full or incremental index of the vault.

        Args:
            force: If True, delete collection and rebuild from scratch.

        Returns:
            Stats dict with keys: files, chunks, elapsed, skipped.
        """
        start = time.monotonic()
        loop = asyncio.get_event_loop()

        if force and self._collection is not None:
            # Wipe and recreate
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            client.delete_collection("vault_chunks")
            self._collection = client.get_or_create_collection(
                name="vault_chunks",
                metadata={"hnsw:space": "cosine"},
            )

        # Get all vault .md files
        vault_files = self._list_vault_files()

        # Get stored mtimes from ChromaDB for incremental check
        stored_mtimes = await loop.run_in_executor(None, self._get_stored_mtimes)

        files_processed = 0
        files_skipped = 0
        total_chunks = 0

        # Find files that need indexing
        to_index: list[str] = []
        current_files: set[str] = set()

        for rel_path in vault_files:
            current_files.add(rel_path)
            full_path = self.vault_path / rel_path
            try:
                current_mtime = full_path.stat().st_mtime
            except OSError:
                continue

            stored_mtime = stored_mtimes.get(rel_path)
            if not force and stored_mtime is not None and abs(current_mtime - stored_mtime) < 0.5:
                files_skipped += 1
                continue

            to_index.append(rel_path)

        # Remove chunks for deleted files
        deleted_files = set(stored_mtimes.keys()) - current_files
        if deleted_files:
            await loop.run_in_executor(
                None, self._delete_files_chunks, list(deleted_files)
            )

        # Index changed files: one inventory pass, chunk everything, then a
        # single length-bucketed encode() across ALL pending chunks, then
        # per-file upserts. (Per-file inventory+encode was O(n^2) walks plus
        # unbounded torch allocator growth -> the 2026-07-02 OOM crash loop.)
        if to_index:
            inventory = build_inventory(self.vault_path)
            pending: list[tuple[str, list]] = []
            for rel_path in to_index:
                canonical = inventory.resolve(rel_path) or rel_path
                try:
                    chunks = await loop.run_in_executor(
                        None, chunk_document, self.vault_path, canonical
                    )
                except Exception as e:
                    logger.warning("Failed to chunk %s: %s", canonical, e)
                    files_processed += 1
                    continue
                if not chunks:
                    files_processed += 1
                    continue
                pending.append((rel_path, chunks))

            if pending:
                all_texts = [
                    c.embedding_text for _, chunks in pending for c in chunks
                ]
                all_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._model.encode(
                        all_texts, batch_size=32, show_progress_bar=False
                    ).tolist(),
                )
                offset = 0
                for rel_path, chunks in pending:
                    embs = all_embeddings[offset : offset + len(chunks)]
                    offset += len(chunks)
                    count = await self._index_single_file(
                        rel_path, inventory=inventory,
                        embeddings=embs, chunks=chunks,
                    )
                    files_processed += 1
                    total_chunks += count

        elapsed = time.monotonic() - start

        # Count total chunks in collection
        if self._collection is not None:
            total_in_db = self._collection.count()
        else:
            total_in_db = total_chunks

        return {
            "files": files_processed,
            "chunks": total_in_db,
            "skipped": files_skipped,
            "elapsed": elapsed,
        }

    async def index_file(self, rel_path: str):
        """Re-index a single file (called after vault_write/vault_edit)."""
        await self._index_single_file(rel_path)

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search across the vault.

        Returns list of dicts with: path, title, section, content, score, start_line.
        """
        if self._collection is None or self._model is None:
            return []

        loop = asyncio.get_event_loop()

        # Embed the query
        query_embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(query).tolist()
        )

        # Query ChromaDB
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            ),
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output: list[dict] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1.0 - (distance / 2.0)

            output.append({
                "path": meta.get("doc_path", ""),
                "title": meta.get("doc_title", ""),
                "section": meta.get("section_header", ""),
                "content": results["documents"][0][i],
                "score": score,
                "start_line": meta.get("start_line", 0),
                "sensitivity": meta.get("sensitivity", ""),
            })

        return output

    # --- Private helpers ---

    def _list_vault_files(self) -> list[str]:
        """List all .md files in the vault."""
        return build_inventory(self.vault_path).canonical_files

    def _get_stored_mtimes(self) -> dict[str, float]:
        """Get file mtimes stored in ChromaDB metadata."""
        if self._collection is None or self._collection.count() == 0:
            return {}

        all_data = self._collection.get(include=["metadatas"])
        mtimes: dict[str, float] = {}
        for meta in all_data["metadatas"]:
            doc_path = meta.get("doc_path", "")
            mtime = meta.get("file_mtime", 0.0)
            if doc_path and doc_path not in mtimes:
                mtimes[doc_path] = mtime
        return mtimes

    def _delete_files_chunks(self, rel_paths: list[str]):
        """Delete all chunks for given file paths."""
        if self._collection is None:
            return
        for rel_path in rel_paths:
            # Get IDs matching this file
            results = self._collection.get(
                where={"doc_path": rel_path},
                include=[],
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])

    async def _index_single_file(
        self, rel_path: str, inventory=None, embeddings: list | None = None,
        chunks: list | None = None,
    ) -> int:
        """Chunk, embed, and store a single file. Returns chunk count.

        inventory/chunks/embeddings may be precomputed by index_all(): building
        the inventory does a full vault walk + SHA-256 of every file, so doing
        it per-file made a cold index O(n^2); and per-file encode() calls (each
        a unique batch shape) grow PyTorch's CPU allocator arena without bound
        (the 2026-07-02 OOM crash loop).
        """
        loop = asyncio.get_event_loop()
        if inventory is None:
            inventory = build_inventory(self.vault_path)
        canonical_rel_path = inventory.resolve(rel_path) or rel_path

        # Delete existing chunks for this file
        await loop.run_in_executor(
            None, self._delete_files_chunks, [rel_path, canonical_rel_path]
        )

        # Chunk the document (unless precomputed)
        if chunks is None:
            try:
                chunks = await loop.run_in_executor(
                    None, chunk_document, self.vault_path, canonical_rel_path
                )
            except Exception as e:
                logger.warning("Failed to chunk %s: %s", canonical_rel_path, e)
                return 0

        if not chunks:
            return 0

        # Get file mtime
        try:
            file_mtime = (self.vault_path / canonical_rel_path).stat().st_mtime
        except OSError:
            file_mtime = 0.0

        document = parse_document(self.vault_path, canonical_rel_path)
        domain = infer_domain(canonical_rel_path)

        # Embed all chunks for this file at once (unless precomputed)
        if embeddings is None:
            embedding_texts = [c.embedding_text for c in chunks]
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    embedding_texts, batch_size=32, show_progress_bar=False
                ).tolist(),
            )

        # Prepare batch for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{canonical_rel_path}::{chunk.section_header}::{i}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "doc_path": chunk.doc_path,
                "doc_title": chunk.doc_title,
                "section_header": chunk.section_header,
                "file_mtime": file_mtime,
                "start_line": chunk.start_line,
                "domain": domain,
                "sensitivity": document.metadata.sensitivity or "",
                "scope": document.metadata.scope or "",
                "tour": document.metadata.tour or "",
                "doc_type": document.metadata.doc_type or "",
            })

        # Upsert to ChromaDB
        await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )

        return len(chunks)
