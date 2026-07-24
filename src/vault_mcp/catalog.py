"""Catalog helpers for canonical vault file inventory and alias resolution."""

from __future__ import annotations

import hashlib
import os
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
    path = rel_path.replace("\\", "/")
    name = Path(path).name
    if name.startswith("HANDOFF-") and path.startswith("shared/") and not path.startswith("shared/reference/"):
        return (0, len(path), path)
    return _priority(path)


def _candidate_aliases(rel_path: str) -> set[str]:
    aliases = {rel_path}
    if rel_path.startswith("shared/reference/"):
        remainder = rel_path[len("shared/reference/"):]
        aliases.add(f"reference/{remainder}")
        aliases.add(remainder)
    elif rel_path.startswith("shared/"):
        aliases.add(rel_path[len("shared/"):])
    return aliases


def build_inventory(vault_path: Path) -> VaultInventory:
    """Scan the vault and produce canonical files plus compatibility aliases."""
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

    return VaultInventory(
        canonical_files=canonical_files,
        alias_map=alias_map,
        duplicate_map=duplicate_map,
    )
