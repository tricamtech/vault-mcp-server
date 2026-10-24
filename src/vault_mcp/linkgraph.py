"""Traversable link graph over the vault: forward links, backlinks, dangling links.

Vaults written Obsidian-style point notes at each other with ``[[wiki-links]]``,
usually without a ``.md`` suffix and often by bare basename. Nothing indexed
those, so "what points at this file?" was unanswerable, and a link to a note that
was never written (or has since been renamed) was invisible.

This module does **not** parse markdown. Extraction is
:func:`vault_mcp.parser.extract_wiki_links` / ``extract_cross_references``, the
same parser ``vault_read`` already uses. A second parser here is how the read
tools and the graph would drift apart.

Dangling links are a first-class output, not an error: a link whose target does
not exist means either a stale rename or a note somebody promised and never
wrote. Both are worth surfacing.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .parser import parse_document

logger = logging.getLogger(__name__)

WIKI = "wiki"
MARKDOWN = "markdown"


@dataclass(frozen=True)
class Link:
    """One reference from one vault file to a target (resolved or not)."""

    source: str  # vault-relative path of the linking file
    target_raw: str  # exactly what was written between the brackets
    kind: str  # WIKI | MARKDOWN
    target: str | None = None  # vault path of the target, or None when dangling

    @property
    def dangling(self) -> bool:
        return self.target is None


@dataclass
class LinkGraph:
    forward: dict[str, list[Link]] = field(default_factory=dict)
    backlinks: dict[str, list[Link]] = field(default_factory=dict)
    dangling: list[Link] = field(default_factory=list)
    files: int = 0
    signature: tuple = ()

    @property
    def link_count(self) -> int:
        return sum(len(links) for links in self.forward.values())

    @property
    def resolved_count(self) -> int:
        return self.link_count - len(self.dangling)

    def forward_links(self, path: str) -> list[Link]:
        return self.forward.get(path, [])

    def backlinks_for(self, path: str) -> list[Link]:
        return self.backlinks.get(path, [])


# --------------------------------------------------------------------------
# corpus walk
# --------------------------------------------------------------------------


def _skipped(name: str) -> bool:
    # Same rule as the server's file listing: hidden and underscore-prefixed
    # entries (.obsidian/, _review/, ...) are not part of the vault.
    return name.startswith((".", "_"))


def list_markdown_files(vault_path: Path) -> list[str]:
    """Vault-relative POSIX paths of every markdown note, sorted."""
    files: list[str] = []
    for root, dirs, filenames in os.walk(vault_path):
        dirs[:] = [d for d in dirs if not _skipped(d)]
        for name in filenames:
            if not name.endswith(".md") or _skipped(name):
                continue
            files.append((Path(root) / name).relative_to(vault_path).as_posix())
    return sorted(files)


def vault_signature(vault_path: Path) -> tuple:
    """Cheap fingerprint of the corpus: count, newest mtime, mtime sum, total bytes.

    Stats only -- no reads -- so it is cheap enough to check on every call, while
    a graph build parses every file. Any write moves the newest mtime (and
    usually the byte total), so it invalidates the cached graph. The mtime sum is
    there so an edit that leaves the byte count identical still moves the
    fingerprint, and a deletion moves the count.
    """
    count = 0
    newest = 0.0
    mtime_sum = 0.0
    total = 0
    for root, dirs, filenames in os.walk(vault_path):
        dirs[:] = [d for d in dirs if not _skipped(d)]
        for name in filenames:
            if not name.endswith(".md") or _skipped(name):
                continue
            try:
                stat = (Path(root) / name).stat()
            except OSError:
                continue
            count += 1
            total += stat.st_size
            mtime_sum += stat.st_mtime
            newest = max(newest, stat.st_mtime)
    return (str(vault_path), count, round(newest, 6), round(mtime_sum, 6), total)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def _priority(rel_path: str) -> tuple[int, str]:
    """Deterministic owner when several files share a name: shallowest, then A-Z."""
    return (rel_path.count("/"), rel_path)


class NoteResolver:
    """Maps what a link says to the file it means.

    Supported target forms, each with or without the ``.md`` suffix:

    - full vault path: ``[[projects/alpha]]``
    - trailing path segment: ``[[alpha/notes]]`` matches ``work/alpha/notes.md``
    - bare basename: ``[[notes]]``

    When a bare basename (or trailing segment) is shared by several files the
    shallowest path wins, then alphabetical order, so the answer is stable.
    Matching falls back to case-insensitive only after an exact match fails. No
    fuzzy matching beyond that: ``my_note`` and ``my-note`` are different names,
    and inventing a match would manufacture a connection that does not exist,
    which is worse than reporting a dangling link.
    """

    def __init__(self, files: list[str]):
        self.files = files
        self._exact: dict[str, str] = {}
        self._suffix: dict[str, str] = {}
        for rel_path in sorted(files, key=_priority):
            self._exact[rel_path] = rel_path
            parts = PurePosixPath(rel_path).parts
            # Every trailing run of segments: "a/b/c.md" -> "b/c.md", "c.md".
            for start in range(1, len(parts)):
                self._suffix.setdefault("/".join(parts[start:]), rel_path)
        self._exact_lower: dict[str, str] = {}
        self._suffix_lower: dict[str, str] = {}
        for key in sorted(self._exact):
            self._exact_lower.setdefault(key.lower(), self._exact[key])
        for key in sorted(self._suffix):
            self._suffix_lower.setdefault(key.lower(), self._suffix[key])

    def _lookup(self, candidate: str) -> str | None:
        for table, key in (
            (self._exact, candidate),
            (self._suffix, candidate),
            (self._exact_lower, candidate.lower()),
            (self._suffix_lower, candidate.lower()),
        ):
            hit = table.get(key)
            if hit:
                return hit
        return None

    def resolve(self, raw: str, source: str = "", kind: str = WIKI) -> str | None:
        """Resolve a link target to a vault path, or None if no such file exists."""
        text = raw.strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        text = text.lstrip("/")
        if not text:
            return None

        candidates: list[str] = []
        if kind == MARKDOWN and source:
            # Markdown links are written relative to the linking file.
            joined = os.path.normpath(os.path.join(os.path.dirname(source), raw.strip()))
            joined = joined.replace("\\", "/")
            if not joined.startswith(".."):
                candidates.append(joined)
        candidates.append(text if text.endswith(".md") else f"{text}.md")

        for candidate in candidates:
            hit = self._lookup(candidate)
            if hit:
                return hit
        return None


# --------------------------------------------------------------------------
# build + cache
# --------------------------------------------------------------------------

_cache: LinkGraph | None = None
_cache_resolver: NoteResolver | None = None
_cache_lock = threading.Lock()


def cached_link_graph(vault_path: Path) -> LinkGraph | None:
    """The memoized graph IF it still matches the corpus, else ``None``.

    For callers on a hot path that must never pay a full rebuild. Building the
    graph parses every file in the vault, and the cache is keyed on a corpus
    signature that any write invalidates -- so a tool that called
    ``build_link_graph`` inline would be one write away from a multi-second
    read on a large vault. ``vault_read`` uses this and simply omits backlinks
    on a miss.
    """
    signature = vault_signature(vault_path)
    with _cache_lock:
        if _cache is not None and _cache.signature == signature:
            return _cache
    return None


def build_link_graph(vault_path: Path, use_cache: bool = True) -> tuple[LinkGraph, NoteResolver]:
    """Build (or return the cached) link graph and the resolver it was built with."""
    global _cache, _cache_resolver

    signature = vault_signature(vault_path)
    if use_cache:
        with _cache_lock:
            if _cache is not None and _cache_resolver is not None and _cache.signature == signature:
                return _cache, _cache_resolver

    files = list_markdown_files(vault_path)
    resolver = NoteResolver(files)

    graph = LinkGraph(files=len(files), signature=signature)
    for rel_path in files:
        try:
            doc = parse_document(vault_path, rel_path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("link graph: skipping %s (%s)", rel_path, exc)
            continue

        links: list[Link] = []
        for raw in doc.wiki_links:
            target = resolver.resolve(raw, rel_path, WIKI)
            links.append(Link(source=rel_path, target_raw=raw, kind=WIKI, target=target))
        for raw in doc.cross_references:
            target = resolver.resolve(raw, rel_path, MARKDOWN)
            links.append(Link(source=rel_path, target_raw=raw, kind=MARKDOWN, target=target))

        if links:
            graph.forward[rel_path] = links
        for link in links:
            if link.target is None:
                graph.dangling.append(link)
            elif link.target != rel_path:  # ignore self-links in backlinks
                graph.backlinks.setdefault(link.target, []).append(link)

    with _cache_lock:
        _cache = graph
        _cache_resolver = resolver
    return graph, resolver


def clear_cache():
    """Drop the memoized graph (tests, or after a bulk corpus change)."""
    global _cache, _cache_resolver
    with _cache_lock:
        _cache = None
        _cache_resolver = None


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def format_links(
    graph: LinkGraph,
    file_path: str,
    resolved: str,
    direction: str = "both",
    limit: int = 50,
) -> str:
    """Render forward links and/or backlinks for one file as markdown."""
    lines = [f"# Links for {resolved}"]
    if resolved != file_path.strip():
        lines.append(f"_(resolved from `{file_path}`)_")
    lines.append("")

    if direction in ("both", "forward"):
        forward = graph.forward_links(resolved)
        found = [link for link in forward if not link.dangling]
        missing = [link for link in forward if link.dangling]
        lines.append(f"## Forward links ({len(forward)})")
        if not forward:
            lines.append("- none")
        for link in found[:limit]:
            lines.append(f"- {link.target}  ({link.kind}: `{link.target_raw}`)")
        for link in missing[:limit]:
            lines.append(f"- DANGLING `{link.target_raw}` ({link.kind}) — no such file")
        lines.append("")

    if direction in ("both", "backlinks"):
        back = graph.backlinks_for(resolved)
        lines.append(f"## Backlinks ({len(back)})")
        if not back:
            lines.append("- none")
        for link in back[:limit]:
            lines.append(f"- {link.source}  ({link.kind}: `{link.target_raw}`)")
        if len(back) > limit:
            lines.append(f"- … {len(back) - limit} more")
    return "\n".join(lines).rstrip()


def format_dangling(graph: LinkGraph, kind: str = WIKI, limit: int = 50, source_prefix: str = "") -> str:
    """Render the dangling-link report, grouped by missing target."""
    links = graph.dangling
    if kind and kind != "all":
        links = [link for link in links if link.kind == kind]
    if source_prefix:
        prefix = source_prefix.strip().lstrip("./")
        links = [link for link in links if link.source.startswith(prefix)]

    grouped: dict[str, list[Link]] = {}
    for link in links:
        grouped.setdefault(link.target_raw, []).append(link)

    kind_label = "all kinds" if kind == "all" else kind
    header = (
        f"# Dangling links ({len(links)} reference(s) to {len(grouped)} missing target(s), "
        f"{kind_label})"
    )
    if not links:
        return header + "\n\nNo dangling links — every link target resolves to a file."

    lines = [
        header,
        "",
        f"_Graph: {graph.files} files, {graph.link_count} links, "
        f"{graph.resolved_count} resolved._",
        "",
    ]
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    for target, refs in ordered[:limit]:
        sources = ", ".join(sorted({link.source for link in refs})[:6])
        more = "" if len(refs) <= 6 else f" (+{len(refs) - 6} more)"
        lines.append(f"- `{target}` — {len(refs)} ref(s) from: {sources}{more}")
    if len(ordered) > limit:
        lines.append(f"- … {len(ordered) - limit} more missing targets")
    return "\n".join(lines)
