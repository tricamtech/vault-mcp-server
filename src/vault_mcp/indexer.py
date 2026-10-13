"""Vault indexer — ChromaDB + sentence-transformers for semantic search."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .catalog import build_inventory, is_indexable
from .chunker import chunk_document
from .parser import infer_domain, parse_document

logger = logging.getLogger(__name__)


import os
# Minimum cosine-similarity score a chunk must reach to be returned at all.
_MIN_SCORE = float(os.environ.get("VAULT_MIN_SCORE", "0.75"))

#: The collection semantic search reads. Never rebuilt in place — see
#: :meth:`VaultIndexer._rebuild_into_new_collection`.
LIVE_COLLECTION = "vault_chunks"
#: A rebuild under construction. Worthless if interrupted, so any leftover is
#: dropped at startup.
REBUILD_PREFIX = "vault_chunks_rebuild"
#: The previous live collection, between the two renames of a swap. Only ever
#: seen by a process that crashed mid-swap; startup renames it back.
RETIRED_PREFIX = "vault_chunks_retired"

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


class RebuildFailed(RuntimeError):
    """A forced rebuild produced nothing usable, so the old index was kept."""


def _collection_names(client) -> list[str]:
    """Collection names, across chroma's two shapes for ``list_collections``.

    0.6 returns ``CollectionName`` — a ``str`` subclass whose ``.name`` raises
    ``NotImplementedError`` on purpose, to push callers off the pre-0.6 API. So
    the str check has to come FIRST; ``getattr(item, "name", item)`` looks
    version-agnostic and blows up on exactly the version we run.
    """
    return [
        str(item) if isinstance(item, str) else item.name
        for item in client.list_collections()
    ]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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

        # --- forced-rebuild state (FIX 3: the caller must be able to tell
        # "running" from "finished" from "failed" without waiting) ---
        self._rebuild_task: asyncio.Task | None = None
        self._rebuild_status: dict = {
            "state": "idle",
            "handle": None,
            "phase": None,
            "started_at": None,
            "finished_at": None,
            "elapsed": None,
            "files": None,
            "chunks": None,
            "error": None,
        }
        #: Non-None only while a rebuild is in flight: paths written (or deleted)
        #: through the live collection meanwhile, replayed into the new one after
        #: the swap so a concurrent write is not silently dropped.
        self._rebuild_dirty: set[str] | None = None

    @property
    async def stop(self):
        """Cancel the filesystem watcher and wait for it to actually finish.

        Without this ``_watch_task`` outlives shutdown: the task keeps a
        reference to the indexer and to an open ``awatch`` generator, so the
        process will not exit cleanly and tests leak a pending task.
        """
        task = getattr(self, "_watch_task", None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._watch_task = None

    def is_ready(self) -> bool:
        return self._ready

    @property
    def rebuild_running(self) -> bool:
        return self._rebuild_status["state"] == "running"

    def rebuild_status(self) -> dict:
        """Snapshot of the forced-rebuild state machine. Safe from any thread."""
        return dict(self._rebuild_status)

    def chunk_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:  # pragma: no cover - chroma handle went away
            return 0

    async def start(self):
        """Initialize ChromaDB + embedding model, then run incremental index."""
        loop = asyncio.get_running_loop()

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = await loop.run_in_executor(
            None, lambda: SentenceTransformer(self.model_name)
        )

        logger.info("Initializing ChromaDB at: %s", self.chroma_path)
        await loop.run_in_executor(None, self.open_collection)

        logger.info("Running initial index...")
        stats = await self.index_all(force=False)
        logger.info(
            "Index ready — %d files, %d chunks in %.1fs",
            stats["files"], stats["chunks"], stats["elapsed"],
        )
        self._ready = True

        if self.watch_files:
            self._watch_task = asyncio.create_task(self._watch_vault())

    def _client(self):
        return chromadb.PersistentClient(path=str(self.chroma_path))

    def open_collection(self):
        """Attach to the live collection, repairing a crashed swap first.

        Blocking; call from an executor. Split out of :meth:`start` so the
        recovery path is testable and so a rebuild can reuse the same client.
        """
        client = self._client()
        self._recover_collections(client)
        self._collection = client.get_or_create_collection(
            name=LIVE_COLLECTION,
            metadata=dict(_COLLECTION_METADATA),
        )
        return self._collection

    def _recover_collections(self, client):
        """Clean up whatever a crash or an abort left behind.

        Two shapes exist:

        * a ``vault_chunks_rebuild_*`` collection — a rebuild that never
          finished. It is partial by definition, so it is dropped.
        * a ``vault_chunks_retired_*`` with no ``vault_chunks`` — the process
          died in the microsecond between the two renames of a swap. The retired
          collection IS the last good index, so it is renamed back rather than
          left for a cold rebuild to replace.
        """
        names = _collection_names(client)

        for name in names:
            if name.startswith(REBUILD_PREFIX):
                logger.warning("Dropping abandoned rebuild collection %s", name)
                self._drop_collection(client, name)

        retired = sorted(n for n in names if n.startswith(RETIRED_PREFIX))
        if LIVE_COLLECTION not in names and retired:
            newest = retired[-1]
            logger.warning(
                "No %s but %s exists — a swap was interrupted; restoring it",
                LIVE_COLLECTION, newest,
            )
            try:
                client.get_collection(newest).modify(name=LIVE_COLLECTION)
            except Exception:
                logger.exception("Failed to restore %s", newest)
            retired = retired[:-1]

        for name in retired:
            self._drop_collection(client, name)

    @staticmethod
    def _drop_collection(client, name: str):
        try:
            client.delete_collection(name)
        except Exception:
            logger.debug("delete_collection(%s) failed (already gone?)", name)

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
        loop = asyncio.get_running_loop()

        try:
            async for changes in awatch(self.vault_path, recursive=True):
                for change_type, abs_path_str in changes:
                    abs_path = Path(abs_path_str)
                    try:
                        rel = str(abs_path.relative_to(self.vault_path))
                    except ValueError:
                        continue
                    # Same predicate as the full walk -- see catalog.is_indexable.
                    if not is_indexable(rel):
                        continue

                    # A delete is only honoured if the path is ACTUALLY gone.
                    # `awatch` yields an unordered SET, so a batch containing both a
                    # delete and an add for one path (exactly what `git reset --hard`
                    # and any bulk edit produce) could apply the delete last and wipe
                    # the chunks just written. Measured cost before this guard: 374 of
                    # 834 files unsearchable, and climbing. The file stayed readable on
                    # disk, so only search went blind -- which is why it hid for so long.
                    if change_type == Change.deleted and not abs_path.exists():
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
        """Index the vault.

        Args:
            force: If True, rebuild every file from scratch — into a SEPARATE
                collection that is swapped in only once it is complete and
                verified non-empty. The live collection is never emptied first;
                an interrupted rebuild leaves search exactly as it was.
                If False (default), skip files whose mtime is unchanged and
                prune chunks for files that have disappeared.

        Returns:
            Stats dict with keys: mode, files, chunks, skipped, deleted, elapsed.
        """
        if force:
            return await self._rebuild_into_new_collection()
        return await self._index_incremental()

    async def _index_incremental(self) -> dict:
        """mtime-skipping pass over the LIVE collection (cheap, in place)."""
        start = time.monotonic()
        loop = asyncio.get_running_loop()

        # Get all vault .md files. build_inventory sha256s every file in the
        # vault, so this must not run on the event loop (it is the same stall
        # that made every other tool wait behind an index pass).
        vault_files = await loop.run_in_executor(None, self._list_vault_files)

        # Get stored mtimes from ChromaDB for incremental check
        stored_mtimes = await loop.run_in_executor(None, self._get_stored_mtimes)

        files_skipped = 0
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
            if stored_mtime is not None and abs(current_mtime - stored_mtime) < 0.5:
                files_skipped += 1
                continue

            to_index.append(rel_path)

        # Remove chunks for deleted files
        deleted_files = set(stored_mtimes.keys()) - current_files
        if deleted_files:
            await loop.run_in_executor(
                None, self._delete_files_chunks, list(deleted_files)
            )

        files_processed = await self._embed_and_store(to_index, self._collection)

        return {
            "mode": "incremental",
            "files": files_processed,
            "chunks": self.chunk_count(),
            "skipped": files_skipped,
            "deleted": len(deleted_files),
            "elapsed": time.monotonic() - start,
        }

    async def _rebuild_into_new_collection(self) -> dict:
        """Full rebuild that CANNOT blind the vault.

        The 2026-07-30 outage: ``index_all(force=True)`` deleted ``vault_chunks``
        and then re-embedded 970 files into the hole it had just made. The
        rebuild takes ~505 s and the MCP client gives up at 300 s, so the vault
        answered "No results found" for everything until the service was
        restarted and had re-indexed — 8.5 minutes blind, from a call whose whole
        purpose was to drop 12 files.

        So: build into ``vault_chunks_rebuild_<ts>``, and only when it is
        complete AND non-empty rename it over the live one. Nothing about the
        live collection changes until that last step, which means an exception, a
        cancellation, an OOM or a power cut all leave the previous index serving.
        """
        start = time.monotonic()
        loop = asyncio.get_running_loop()
        client = self._client()

        temp_name = f"{REBUILD_PREFIX}_{int(time.time() * 1000)}"
        previous = self._collection
        self._rebuild_dirty = set()

        def _make_temp():
            self._drop_collection(client, temp_name)
            return client.create_collection(
                name=temp_name, metadata=dict(_COLLECTION_METADATA)
            )

        temp = await loop.run_in_executor(None, _make_temp)

        try:
            vault_files = await loop.run_in_executor(None, self._list_vault_files)
            self._set_phase("chunking", files=len(vault_files))
            files_processed = await self._embed_and_store(
                list(vault_files), temp, report_phase=True, prune=False
            )

            chunk_total = await loop.run_in_executor(None, temp.count)

            # The degenerate case: a vault full of files that produced no chunks
            # means the chunker, the model or the disk failed, NOT that the vault
            # is empty. Swapping that in would be the outage all over again.
            if vault_files and chunk_total == 0:
                raise RebuildFailed(
                    f"rebuild produced 0 chunks from {len(vault_files)} files — "
                    "keeping the previous index"
                )

            self._set_phase("swapping")
            await loop.run_in_executor(None, self._swap_in, client, temp_name)
            await self._replay_dirty()

            return {
                "mode": "full-rebuild",
                "files": files_processed,
                "chunks": self.chunk_count(),
                "skipped": 0,
                "deleted": 0,
                "elapsed": time.monotonic() - start,
            }
        except BaseException:
            # BaseException, not Exception: CancelledError is the case that
            # actually happened, and it must not skip the cleanup either.
            logger.exception("Rebuild failed — previous index left in place")
            self._collection = previous
            await loop.run_in_executor(
                None, self._drop_collection, client, temp_name
            )
            raise
        finally:
            self._rebuild_dirty = None

    def _swap_in(self, client, temp_name: str):
        """Rename the finished rebuild over the live collection. Blocking.

        chromadb 0.6.3 supports ``Collection.modify(name=...)``, a metadata-only
        rename of an existing collection — no re-embedding and no copy — so the
        swap is two renames, not a data move. ``self._collection`` is a handle on
        a collection *id*, so a reader holding it keeps reading the old data
        right up until the attribute is rebound, which is a single atomic
        assignment.
        """
        names = _collection_names(client)
        retired_name = f"{RETIRED_PREFIX}_{int(time.time() * 1000)}"
        had_live = LIVE_COLLECTION in names

        if had_live:
            client.get_collection(LIVE_COLLECTION).modify(name=retired_name)
        client.get_collection(temp_name).modify(name=LIVE_COLLECTION)

        self._collection = client.get_collection(LIVE_COLLECTION)

        if had_live:
            self._drop_collection(client, retired_name)

    async def _replay_dirty(self):
        """Re-apply writes that landed on the old collection during the rebuild.

        A rebuild takes minutes; vault_write / the filesystem watcher keep
        running. Those writes went to the collection that is now retired, so
        without this they would vanish at the swap.
        """
        dirty = self._rebuild_dirty or set()
        self._rebuild_dirty = None  # replayed writes go straight to the new one
        if not dirty:
            return
        loop = asyncio.get_running_loop()
        gone = [p for p in dirty if not (self.vault_path / p).exists()]
        if gone:
            await loop.run_in_executor(None, self._delete_files_chunks, gone)
        for rel_path in sorted(dirty):
            if rel_path in gone:
                continue
            try:
                await self._index_single_file(rel_path)
            except Exception:
                logger.exception("Replay after swap failed for %s", rel_path)

    async def _embed_and_store(
        self, to_index: list[str], collection, report_phase: bool = False,
        prune: bool = True,
    ) -> int:
        """Chunk, batch-encode and upsert a list of files into ``collection``.

        One inventory pass, chunk everything, then a single length-bucketed
        encode() across ALL pending chunks, then per-file upserts. (Per-file
        inventory+encode was O(n^2) walks plus unbounded torch allocator growth
        -> the 2026-07-02 OOM crash loop.)

        ``prune=False`` skips the per-file "delete what was there before" step.
    Only the rebuild path sets it, and only because it writes into a collection
    that was created empty moments earlier: there is nothing to delete. That
    delete is a metadata scan per file, and chroma keeps every collection's
    metadata in ONE sqlite table, so during a rebuild each scan also walks the
    8000 rows of the live collection sitting alongside — 1788 scans of a table
    that is temporarily twice its normal size. Measured: it dominated the pass.

    ``report_phase`` publishes chunking/encoding/storing transitions into the
        rebuild status. Only the rebuild path asks for it: the single batched
        encode() is ~95% of a full pass, so a per-file counter alone reads as
        "0/894" for seven minutes, which is the sort of ambiguous progress signal
        this whole change exists to remove.
        """
        if not to_index:
            return 0

        loop = asyncio.get_running_loop()
        files_processed = 0

        inventory = await loop.run_in_executor(None, build_inventory, self.vault_path)
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
            all_texts = [c.embedding_text for _, chunks in pending for c in chunks]
            if report_phase:
                self._set_phase("encoding", chunks=len(all_texts))
            all_embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    all_texts, batch_size=32, show_progress_bar=False
                ).tolist(),
            )
            if report_phase:
                self._set_phase("storing")
            offset = 0
            skipped_gone = 0
            for rel_path, chunks in pending:
                embs = all_embeddings[offset : offset + len(chunks)]
                offset += len(chunks)
                # A file can VANISH between chunking and storing. On a synced vault
                # that is routine, not exceptional: this corpus re-syncs from git
                # every ~20 min while a full pass takes hours, so a 2026-10-07
                # bge-base rebuild died on FileNotFoundError for a file deleted
                # mid-run and wrote only 466 of ~13,000 chunks. One missing file
                # must not abort the whole index. Mirrors the guard that already
                # exists in _replay_dirty_after_swap.
                try:
                    await self._index_single_file(
                        rel_path, inventory=inventory,
                        embeddings=embs, chunks=chunks,
                        collection=collection, prune=prune,
                    )
                except FileNotFoundError:
                    skipped_gone += 1
                    logger.warning("Skipping %s: removed from the vault mid-index", rel_path)
                except Exception:
                    skipped_gone += 1
                    logger.exception("Failed to store %s; continuing", rel_path)
                files_processed += 1
                if report_phase:
                    self._bump_phase_progress(files_processed)
            if skipped_gone:
                logger.warning("Index pass completed with %d file(s) skipped "
                               "(vanished or failed to store)", skipped_gone)

        return files_processed

    # --- Forced-rebuild task control (FIX 3) ---

    async def start_rebuild(self) -> dict:
        """Kick off a forced rebuild in the background and return immediately.

        A full rebuild is ~500 s and the MCP client idle timeout is 300 s, so the
        synchronous version could never report success over MCP even when it
        worked — which is exactly what made the outage look like a no-op and
        invited a retry. Callers get a handle now and poll for the outcome.
        """
        if self.rebuild_running:
            status = self.rebuild_status()
            status["already_running"] = True
            return status

        handle = f"rebuild-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        self._rebuild_status = {
            "state": "running",
            "handle": handle,
            "phase": "starting",
            "started_at": _utcnow(),
            "finished_at": None,
            "elapsed": None,
            "files": None,
            "chunks": None,
            "error": None,
        }
        self._rebuild_task = asyncio.create_task(self._run_rebuild())
        status = self.rebuild_status()
        status["already_running"] = False
        return status

    async def _run_rebuild(self):
        try:
            stats = await self.index_all(force=True)
        except asyncio.CancelledError:
            self._finish_rebuild("failed", error="cancelled")
            raise
        except BaseException as exc:
            self._finish_rebuild("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        self._finish_rebuild(
            "succeeded",
            files=stats["files"],
            chunks=stats["chunks"],
            elapsed=stats["elapsed"],
        )
        return stats

    def _finish_rebuild(self, state: str, **fields):
        self._rebuild_status.update(
            state=state,
            phase=None,
            finished_at=_utcnow(),
            **fields,
        )

    def _set_phase(self, phase: str, **fields):
        if self._rebuild_status["state"] == "running":
            self._rebuild_status.update(phase=phase, **fields)

    def _bump_phase_progress(self, done: int):
        if self._rebuild_status["state"] == "running":
            self._rebuild_status["indexed"] = done

    async def index_file(self, rel_path: str):
        """Re-index a single file (called after vault_write/vault_edit)."""
        await self._index_single_file(rel_path)

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search across the vault.

        Returns list of dicts with: path, title, section, content, score, start_line.
        """
        if self._collection is None or self._model is None:
            return []

        loop = asyncio.get_running_loop()

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

            # RELEVANCE FLOOR (2026-07-28). Chroma always returns the top_k nearest
            # chunks, however far away they are, so this search could NEVER say "I
            # don't know": an audit fired a nonsense control query and got 5 confident
            # -looking hits at 0.687-0.723, against a documented real success at 0.79.
            # A retrieval layer that always answers is worse than one that abstains,
            # because a caller cannot tell a real hit from noise. Tunable via
            # VAULT_MIN_SCORE; set 0 to restore the old always-answer behaviour.
            if score < _MIN_SCORE:
                continue

            output.append({
                "path": meta.get("doc_path", ""),
                "title": meta.get("doc_title", ""),
                "section": meta.get("section_header", ""),
                "content": results["documents"][0][i],
                "score": score,
                "start_line": meta.get("start_line", 0),
                "sensitivity": meta.get("sensitivity", ""),
                "status": meta.get("status", ""),
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

    def _delete_files_chunks(self, rel_paths: list[str], collection=None):
        """Delete all chunks for given file paths."""
        target = collection if collection is not None else self._collection
        if target is None:
            return
        if collection is None and self._rebuild_dirty is not None:
            self._rebuild_dirty.update(rel_paths)
        for rel_path in rel_paths:
            # Get IDs matching this file
            results = target.get(
                where={"doc_path": rel_path},
                include=[],
            )
            if results["ids"]:
                target.delete(ids=results["ids"])

    async def _index_single_file(
        self, rel_path: str, inventory=None, embeddings: list | None = None,
        chunks: list | None = None, collection=None, prune: bool = True,
    ) -> int:
        """Chunk, embed, and store a single file. Returns chunk count.

        inventory/chunks/embeddings may be precomputed by index_all(): building
        the inventory does a full vault walk + SHA-256 of every file, so doing
        it per-file made a cold index O(n^2); and per-file encode() calls (each
        a unique batch shape) grow PyTorch's CPU allocator arena without bound
        (the 2026-07-02 OOM crash loop).
        """
        loop = asyncio.get_running_loop()
        target = collection if collection is not None else self._collection
        if target is None:
            return 0
        if collection is None and self._rebuild_dirty is not None:
            # Live write during a rebuild — replayed into the new collection
            # after the swap (see :meth:`_replay_dirty`).
            self._rebuild_dirty.add(rel_path)
        if inventory is None:
            # Off-loop: this is the per-write reindex path, so an inline full-vault
            # hash here blocked every concurrent read for its duration.
            inventory = await loop.run_in_executor(
                None, build_inventory, self.vault_path
            )
        canonical_rel_path = inventory.resolve(rel_path) or rel_path

        # Delete existing chunks for this file (skipped when the target
        # collection was created empty — see _embed_and_store's `prune`).
        if prune:
            await loop.run_in_executor(
                None,
                self._delete_files_chunks,
                [rel_path, canonical_rel_path],
                target,
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
                # Freshness ranking (server._freshness_weight) demotes superseded docs.
                "status": str(document.metadata.status or ""),
                "updated": str(document.metadata.updated_date or ""),
            })

        # Upsert to ChromaDB
        await loop.run_in_executor(
            None,
            lambda: target.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )

        return len(chunks)
