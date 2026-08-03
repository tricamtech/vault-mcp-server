"""Catalog helpers for canonical vault file inventory and alias resolution."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaultInventory:
    """Canonical vault view with compatibility aliases."""

    canonical_files: list[str]
    alias_map: dict[str, str]
    duplicate_map: dict[str, str]

    def resolve(self, requested_path: str) -> str | None:
        """Resolve a requested path or alias to a canonical file path."""
        return self.alias_map.get(requested_path)


def _list_markdown_files(vault_path: Path) -> list[str]:
    # Use os.walk with followlinks=True so symlinked subtrees (e.g. tiered
    # vault layouts pointing at curated subdirs) are included. Path.rglob
    # does not follow symlinks for directories.
    files: list[str] = []
    for root, dirs, filenames in os.walk(vault_path, followlinks=True):
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("_"))]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            if filename.startswith(".") or filename.startswith("_"):
                continue
            full = Path(root) / filename
            files.append(str(full.relative_to(vault_path)))
    return sorted(files)


def _content_hash(vault_path: Path, rel_path: str) -> str:
    return hashlib.sha256((vault_path / rel_path).read_bytes()).hexdigest()


def _priority(rel_path: str) -> tuple[int, int, str]:
    """Lower tuples win when picking canonical copies of duplicate content."""
    path = rel_path.replace("\\", "/")
    if path.startswith("shared/reference/"):
        return (0, len(path), path)
    if path.startswith("shared/"):
        return (1, len(path), path)
    if path.startswith("bob/memory/"):
        return (2, len(path), path)
    if path.startswith("bob/auto-memory/"):
        return (3, len(path), path)
    if path.startswith("bob/"):
        return (4, len(path), path)
    return (5, len(path), path)


def _alias_priority(rel_path: str) -> tuple[int, int, str]:
    """Priority for resolving a bare basename to one canonical owner.

    2026-07-29 (memory-system audit IA-06): removed a HANDOFF-* special case that
    ranked top-level ``shared/HANDOFF-*.md`` ABOVE ``shared/reference/HANDOFF-*.md``.
    That inversion was correct when the top-level copies were the live handoffs, but
    the tour sessions now maintain the reference/ tier and the top-level copies are
    frozen, superseded snapshots. Effect of the old rule: vault_read("HANDOFF-BB.md")
    returned the copy frozen 2026-06-16 (still naming Princeton NJ as the next show)
    instead of the 66 KB file updated the same day -- and because vault write tools
    share this resolver, writes landed on the fossil too.

    Basename resolution now uses the same tier order as everything else
    (shared/reference/ > shared/ > bob/...). Full explicit paths are unaffected:
    every canonical file still self-resolves, so ``shared/HANDOFF-BB.md`` remains
    readable by its full path. Verified blast radius at the time of the change was
    exactly one alias entry (HANDOFF-BB.md), zero aliases added or removed.
    """
    return _priority(rel_path.replace("\\", "/"))


def _candidate_aliases(rel_path: str) -> set[str]:
    aliases = {rel_path}
    if rel_path.startswith("shared/reference/"):
        remainder = rel_path[len("shared/reference/"):]
        aliases.add(f"reference/{remainder}")
        aliases.add(remainder)
    elif rel_path.startswith("shared/"):
        aliases.add(rel_path[len("shared/"):])
    return aliases


def vault_signature(vault_path: Path) -> tuple:
    """Cheap fingerprint of the corpus: count, newest mtime, mtime sum, total bytes.

    Only stats -- no reads and no hashing -- so it is safe to compute on every
    call. :func:`build_inventory`, by contrast, sha256s every file in the vault.
    ONE fingerprint for the whole codebase: ``linkgraph`` re-exports this one
    rather than keeping a second copy that could drift.

    Microsecond resolution on the newest mtime, plus the mtime sum, so that an
    edit which happens to leave the byte count identical still moves the
    fingerprint. At millisecond rounding a same-size rewrite inside the same
    millisecond would have been invisible -- tolerable when this only fed the link
    graph, not once it gates path resolution for reads and writes.
    """
    count = 0
    newest = 0.0
    mtime_sum = 0.0
    total = 0
    for root, dirs, files in os.walk(vault_path, followlinks=True):
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("_"))]
        for name in files:
            if not name.endswith(".md") or name.startswith((".", "_")):
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


_cache: dict[str, tuple[tuple, VaultInventory]] = {}
_cache_lock = threading.Lock()


def clear_cache():
    """Drop the memoized inventory (tests, or after a bulk corpus change)."""
    with _cache_lock:
        _cache.clear()


def cached_signature(vault_path: Path) -> tuple | None:
    """Signature recorded by the most recent :func:`build_inventory`, or None.

    For a caller that has ALREADY validated the inventory in this same operation
    and needs to check another signature-keyed cache. Computing a fresh
    :func:`vault_signature` costs ~83 ms on the 894-file corpus — most of a whole
    ``vault_read`` — so paying for a second walk within one tool call is not a
    rounding error, it is a 30% latency regression.

    Deliberately does NOT re-stat: the point is to reuse a fingerprint taken
    microseconds ago. Callers that need certainty call ``vault_signature``.
    """
    with _cache_lock:
        cached = _cache.get(str(vault_path))
        return cached[0] if cached is not None else None


def build_inventory(vault_path: Path, use_cache: bool = True) -> VaultInventory:
    """Scan the vault and produce canonical files plus compatibility aliases.

    MEMOIZED (2026-07-30). This is the hot path of every read and every write:
    ``_resolve_markdown_path`` calls it once per tool call, and on the live corpus
    (982 files) a rebuild costs ~165 ms — a walk plus a SHA-256 of every file. That
    single call *was* the whole of vault_read's 151 ms solo latency, and because it
    is GIL-bound Python it got worse, not better, once tool bodies started running
    in threads (read p50 638 ms -> 1847 ms under 6 clients).

    The key is :func:`vault_signature`: stat-only, ~28 ms, and already the caching
    strategy ``linkgraph`` uses for the same corpus. Any write bumps a file's mtime
    (and almost always the byte total), so the fingerprint moves and the next call
    rebuilds. Pass ``use_cache=False`` for a guaranteed-fresh scan.
    """
    signature = vault_signature(vault_path)
    key = str(vault_path)
    if use_cache:
        with _cache_lock:
            cached = _cache.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]

    files = _list_markdown_files(vault_path)

    groups: dict[str, list[str]] = {}
    for rel_path in files:
        digest = _content_hash(vault_path, rel_path)
        groups.setdefault(digest, []).append(rel_path)

    canonical_files: list[str] = []
    duplicate_map: dict[str, str] = {}
    for rel_paths in groups.values():
        canonical = sorted(rel_paths, key=_priority)[0]
        canonical_files.append(canonical)
        for rel_path in rel_paths:
            duplicate_map[rel_path] = canonical

    canonical_files = sorted(set(canonical_files))
    alias_candidates: dict[str, set[str]] = {}
    basename_owners: dict[str, set[str]] = {}

    for canonical in canonical_files:
        for alias in _candidate_aliases(canonical):
            alias_candidates.setdefault(alias, set()).add(canonical)
        basename_owners.setdefault(Path(canonical).name, set()).add(canonical)

    for rel_path, canonical in duplicate_map.items():
        for alias in _candidate_aliases(rel_path):
            alias_candidates.setdefault(alias, set()).add(canonical)

    basename_map = {
        basename: sorted(owners, key=_alias_priority)[0]
        for basename, owners in basename_owners.items()
    }

    alias_map = {
        alias: sorted(owners, key=_priority)[0]
        for alias, owners in alias_candidates.items()
        if len(owners) == 1
    }

    alias_map.update(basename_map)

    for canonical in canonical_files:
        alias_map[canonical] = canonical

    inventory = VaultInventory(
        canonical_files=canonical_files,
        alias_map=alias_map,
        duplicate_map=duplicate_map,
    )
    with _cache_lock:
        _cache[key] = (signature, inventory)
    return inventory
