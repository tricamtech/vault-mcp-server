"""Traversable link graph over the vault: forward links, backlinks, dangling links.

The corpus is written Obsidian-style: notes point at each other with
``[[wiki-links]]``, usually without a ``.md`` suffix and often by bare basename.
Nothing indexed those, so "what points at this file?" was unanswerable and a link
to a note that was never written (or has since been renamed) was invisible.

ONE CHOKE POINT: this module does **not** parse markdown. Extraction is
:func:`vault_mcp.parser.extract_wiki_links` / ``extract_cross_references`` (the same
parser the indexer and every read tool already use), and path resolution is
:func:`vault_mcp.catalog.build_inventory`'s alias map (the same resolver the write
tools use, including its tier priority for duplicate basenames). A second parser or
a second resolver here is how the two halves of this codebase would drift apart.

Dangling links are a first-class output, not an error: a link whose target does not
exist means either a stale rename or a note somebody promised and never wrote. Both
are worth surfacing.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import VaultInventory, build_inventory, vault_signature
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
    target: str | None = None  # canonical vault path, or None when dangling

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

    def dangling_by_target(self) -> dict[str, list[Link]]:
        grouped: dict[str, list[Link]] = {}
        for link in self.dangling:
            grouped.setdefault(link.target_raw, []).append(link)
        return grouped


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def _lower_alias_map(inventory: VaultInventory) -> dict[str, str]:
    """Case-insensitive view of the alias map, deterministic on collision."""
    lowered: dict[str, str] = {}
    for alias in sorted(inventory.alias_map):
        lowered.setdefault(alias.lower(), inventory.alias_map[alias])
    return lowered


def resolve_target(
    raw: str,
    source: str,
    kind: str,
    inventory: VaultInventory,
    lowered: dict[str, str],
) -> str | None:
    """Resolve a link target to a canonical vault path, or None if it does not exist.

    Candidate order matters: an exact alias beats a suffix guess, and a
    markdown-relative path is tried first for markdown links because those are
    written relative to the linking file.
    """
    target = raw.strip().lstrip("./")
    if not target:
        return None

    candidates: list[str] = []
    if kind == MARKDOWN:
        # ../foo/bar.md relative to the source file's directory.
        joined = os.path.normpath(os.path.join(os.path.dirname(source), raw.strip()))
        if not joined.startswith(".."):
            candidates.append(joined)
    candidates.append(target)
    if not target.endswith(".md"):
        candidates.append(f"{target}.md")
        candidates.append(f"{Path(target).name}.md")
    else:
        candidates.append(Path(target).name)

    for candidate in candidates:
        hit = inventory.resolve(candidate)
        if hit:
            return hit
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    return None


# --------------------------------------------------------------------------
# build + cache
# --------------------------------------------------------------------------

_cache: LinkGraph | None = None
_cache_key: tuple = ()
_cache_lock = threading.Lock()


def cached_link_graph(vault_path: Path) -> LinkGraph | None:
    """The memoized graph IF it still matches the corpus, else ``None``.

    For callers on a hot path that must never pay a full rebuild. Building the
    graph parses every file in the vault (~900 of them), and the cache is keyed
    on a corpus signature that ANY write invalidates — so a tool that calls
    ``build_link_graph`` inline is one write away from a multi-second read.
    ``vault_read`` uses this instead and simply omits link data on a miss.
    """
    signature = vault_signature(vault_path)
    with _cache_lock:
        if _cache is not None and _cache_key == signature:
            return _cache
    return None


def build_link_graph(vault_path: Path, use_cache: bool = True) -> LinkGraph:
    """Build (or return a cached) link graph for the vault."""
    global _cache, _cache_key

    signature = vault_signature(vault_path)
    if use_cache:
        with _cache_lock:
            if _cache is not None and _cache_key == signature:
                return _cache

    inventory = build_inventory(vault_path)
    lowered = _lower_alias_map(inventory)

    graph = LinkGraph(files=len(inventory.canonical_files), signature=signature)
    for rel_path in inventory.canonical_files:
        try:
            doc = parse_document(vault_path, rel_path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("link graph: skipping %s (%s)", rel_path, exc)
            continue

        links: list[Link] = []
        for raw in doc.wiki_links:
            target = resolve_target(raw, rel_path, WIKI, inventory, lowered)
            links.append(Link(source=rel_path, target_raw=raw, kind=WIKI, target=target))
        for raw in doc.cross_references:
            target = resolve_target(raw, rel_path, MARKDOWN, inventory, lowered)
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
        _cache_key = signature
    return graph


def clear_cache():
    """Drop the memoized graph (tests, or after a bulk corpus change)."""
    global _cache, _cache_key
    with _cache_lock:
        _cache = None
        _cache_key = ()


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def format_links(
    graph: LinkGraph,
    file_path: str,
    canonical: str,
    direction: str = "both",
    limit: int = 50,
) -> str:
    """Render forward links and/or backlinks for one file as markdown."""
    lines = [f"# Links for {canonical}"]
    if canonical != file_path.strip().lstrip("./"):
        lines.append(f"_(resolved from `{file_path}`)_")
    lines.append("")

    if direction in ("both", "forward", "out"):
        forward = graph.forward_links(canonical)
        resolved = [link for link in forward if not link.dangling]
        dangling = [link for link in forward if link.dangling]
        lines.append(f"## Forward links ({len(forward)})")
        if not forward:
            lines.append("- none")
        for link in resolved[:limit]:
            lines.append(f"- {link.target}  ({link.kind}: `{link.target_raw}`)")
        for link in dangling[:limit]:
            lines.append(f"- DANGLING `{link.target_raw}` ({link.kind}) — no such file")
        lines.append("")

    if direction in ("both", "back", "in", "backlinks"):
        back = graph.backlinks_for(canonical)
        lines.append(f"## Backlinks ({len(back)})")
        if not back:
            lines.append("- none")
        for link in back[:limit]:
            lines.append(f"- {link.source}  ({link.kind}: `{link.target_raw}`)")
        if len(back) > limit:
            lines.append(f"- … {len(back) - limit} more")
    return "\n".join(lines)


def format_dangling(graph: LinkGraph, kind: str = WIKI, limit: int = 50, source_prefix: str = "") -> str:
    """Render the dangling-link report, grouped by missing target."""
    links = graph.dangling
    if kind and kind != "all":
        links = [link for link in links if link.kind == kind]
    if source_prefix:
        normalized = source_prefix.strip().lstrip("./")
        links = [link for link in links if link.source.startswith(normalized)]

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
        lines.append(f"- `[[{target}]]` — {len(refs)} ref(s) from: {sources}{more}")
    if len(ordered) > limit:
        lines.append(f"- … {len(ordered) - limit} more missing targets")
    return "\n".join(lines)
