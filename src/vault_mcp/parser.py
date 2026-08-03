"""Markdown parsing helpers for vault metadata, sections, and cross-references."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .models import Metadata, VaultDocument

# **Key:** Value pattern in early markdown body lines.
_META_RE = re.compile(r"\*\*(.+?):\*\*\s*(.+)")

# Markdown links to .md files.
_LINK_RE = re.compile(r"\[.*?\]\(([^)]+\.md)\)")

# Obsidian-style wiki links: [[target]], [[target|alias]], [[target#Section]],
# and embeds ![[target]]. The corpus uses these heavily and almost never with a
# .md suffix ([[feedback_standing_deploy_auth]]), so resolution to a real file is
# the link graph's job -- this only extracts the target text.
# Non-greedy and newline-free so an unclosed "[[" cannot swallow the document.
_WIKILINK_RE = re.compile(r"!?\[\[([^\[\]\n]+?)\]\]")

# Inline code spans, stripped before wiki-link extraction (see _strip_code).
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# Header pattern.
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)

# Status normalization mapping.
_STATUS_MAP: dict[str, str] = {
    "active": "Active",
    "activo": "Active",
    "en progreso": "Active",
    "draft": "Active",
    "🟡": "Active",
    "pending": "Pending",
    "pendiente": "Pending",
    "🔴": "Pending",
    "paused": "Paused",
    "pausado": "Paused",
    "⏸️": "Paused",
    "done": "Done",
    "completado": "Done",
    "completed": "Done",
    "🟢": "Done",
}

# Metadata key normalization across YAML and legacy inline markdown.
_KEY_MAP: dict[str, str] = {
    "created": "created_date",
    "created_date": "created_date",
    "creado": "created_date",
    "created date": "created_date",
    "status": "status",
    "estado": "status",
    "updated": "updated_date",
    "updated_date": "updated_date",
    "last updated": "updated_date",
    "ultima actualización": "updated_date",
    "última actualización": "updated_date",
    "type": "doc_type",
    "doc_type": "doc_type",
    "scope": "scope",
    "sensitivity": "sensitivity",
    "tour": "tour",
    "tags": "tags",
}

_SEMANTIC_INDEX_BLOCKLIST = {"restricted", "secret"}
_SENSITIVITY_ORDER = {"": 0, "restricted": 1, "secret": 2}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

_TOUR_HINTS: dict[str, tuple[str, ...]] = {
    "bb": ("bacon-brothers", "bb-salk"),
    "ct": ("celtic-thunder",),
    "jk": ("john-kim",),
    "ll": ("lyle-lovett", "touradmin-japan"),
    "mp": ("martin-pixie", "mp-canada", "touradmin-canada"),
    "oa": ("old-97s",),
    "yola": ("yola",),
}

_PATH_OVERRIDES: dict[str, dict[str, str]] = {
    "shared/people.cameron.md": {
        "scope": "personal",
        "doc_type": "person",
        "sensitivity": "restricted",
    },
    "shared/reference/people.cameron.full.md": {
        "scope": "personal",
        "doc_type": "person",
        "sensitivity": "restricted",
    },
    "shared/people.md": {
        "scope": "personal",
        "doc_type": "reference",
    },
    "shared/actual-budget.md": {
        "scope": "personal",
        "doc_type": "finance",
        "sensitivity": "restricted",
    },
    "shared/reference/services.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/homeassistant.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/homelab.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/quick-reference.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/gmail-management.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/memory-non-tour.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/bob-agent.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/reference/shesynergy.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "shared/nanostation-camera-bridge.md": {
        "scope": "ops",
        "doc_type": "reference",
        "sensitivity": "restricted",
    },
    "bob/MEMORY.md": {
        "scope": "ops",
        "doc_type": "mission-control",
        "sensitivity": "secret",
    },
    "air/MEMORY.md": {
        "scope": "ops",
        "doc_type": "host-memory",
        "sensitivity": "secret",
    },
    "mbp/MEMORY.md": {
        "scope": "ops",
        "doc_type": "host-memory",
        "sensitivity": "secret",
    },
    "argus/memory/MEMORY.md": {
        "scope": "ops",
        "doc_type": "host-memory",
        "sensitivity": "secret",
    },
}


def normalize_status(raw: Any) -> str:
    """Normalize status strings and emojis to standard vocabulary."""
    cleaned = str(raw).strip().lower()
    for token, normalized in _STATUS_MAP.items():
        if token in cleaned:
            return normalized
    return str(raw).strip()


def normalize_tags(value: Any) -> list[str]:
    """Normalize tag values from frontmatter or legacy metadata."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def normalize_sensitivity(value: Any) -> str | None:
    """Normalize sensitivity values to lowercase labels."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def is_semantic_index_blocked(sensitivity: str | None) -> bool:
    """Whether a document should be excluded from semantic indexing."""
    return bool(sensitivity and sensitivity.lower() in _SEMANTIC_INDEX_BLOCKLIST)


def _stronger_sensitivity(current: str | None, inferred: str | None) -> str | None:
    current_key = normalize_sensitivity(current) or ""
    inferred_key = normalize_sensitivity(inferred) or ""
    if _SENSITIVITY_ORDER.get(inferred_key, 0) > _SENSITIVITY_ORDER.get(current_key, 0):
        return inferred_key
    return current


def split_frontmatter(content: str) -> tuple[dict[str, Any], str, int]:
    """Split YAML frontmatter from a markdown document.

    Returns:
        frontmatter dict, markdown body without frontmatter, and the 1-based line
        number where the body starts in the original file.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content, 1

    for idx in range(1, len(lines)):
        if lines[idx].strip() != "---":
            continue
        frontmatter_text = "\n".join(lines[1:idx])
        body_lines = lines[idx + 1:]
        body = "\n".join(body_lines)
        try:
            parsed = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return parsed, body, idx + 2

    return {}, content, 1


def _apply_metadata_key(meta: Metadata, key_raw: str, value: Any):
    """Map heterogeneous metadata keys into the normalized Metadata shape."""
    key_norm = _KEY_MAP.get(key_raw.strip().lower())
    if key_norm == "created_date":
        meta.created_date = str(value).strip()
    elif key_norm == "updated_date":
        meta.updated_date = str(value).strip()
    elif key_norm == "status":
        meta.status = normalize_status(value)
    elif key_norm == "tags":
        meta.tags = normalize_tags(value)
    elif key_norm == "doc_type":
        meta.doc_type = str(value).strip()
    elif key_norm == "scope":
        meta.scope = str(value).strip()
    elif key_norm == "sensitivity":
        meta.sensitivity = normalize_sensitivity(value)
    elif key_norm == "tour":
        meta.tour = str(value).strip()


def _infer_tour(rel_path: str) -> str | None:
    name = Path(rel_path).name.lower().replace("_", "-")
    tokens = {token for token in re.split(r"[^a-z0-9]+", name) if token}
    for tour, hints in _TOUR_HINTS.items():
        if tour in tokens:
            return tour
        if any(hint in name for hint in hints):
            return tour
    return None


def _infer_path_metadata(rel_path: str) -> dict[str, str]:
    path = rel_path.replace("\\", "/")
    override = _PATH_OVERRIDES.get(path)
    if override:
        data = dict(override)
        if "tour" not in data:
            inferred_tour = _infer_tour(path)
            if inferred_tour:
                data["tour"] = inferred_tour
        return data

    parts = Path(path).parts
    top = parts[0] if parts else ""
    name = Path(path).name
    stem = Path(path).stem
    lowered = stem.lower().replace("_", "-")

    inferred: dict[str, str] = {}

    if top == "shared":
        inferred["scope"] = "shared"
        if len(parts) > 1 and parts[1] == "reference":
            inferred["doc_type"] = "reference"
        elif len(parts) > 1 and parts[1] == "plans":
            inferred["doc_type"] = "plan"
        elif name.startswith("HANDOFF-"):
            inferred["scope"] = "tour"
            inferred["doc_type"] = "handoff"
        elif name.startswith("feedback-"):
            inferred["doc_type"] = "feedback"
        elif lowered.startswith(("mp-", "bb-", "jk-", "ll-", "oa-", "yola-", "ct-")):
            inferred["doc_type"] = "reference"

    elif top == "bob":
        inferred["scope"] = "ops"
        if len(parts) > 1 and parts[1] == "memory":
            if _DATE_RE.match(name):
                inferred["scope"] = "journal"
                inferred["doc_type"] = "journal"
            elif name.startswith("archive"):
                inferred["doc_type"] = "history"
            elif name.startswith("project-"):
                inferred["doc_type"] = "project"
            else:
                inferred["doc_type"] = "reference"
        elif len(parts) > 1 and parts[1] == "auto-memory":
            if name == "MEMORY.md":
                inferred["doc_type"] = "mission-control"
                inferred["sensitivity"] = "secret"
            elif name.startswith("feedback_"):
                inferred["scope"] = "shared"
                inferred["doc_type"] = "feedback"
            elif name.startswith("project_"):
                inferred["doc_type"] = "project"
            else:
                inferred["doc_type"] = "reference"
        elif name == "CLAUDE.md":
            inferred["doc_type"] = "playbook"
        elif name == "SOUL.md":
            inferred["doc_type"] = "persona"

    elif top in {"air", "mbp", "argus"}:
        inferred["scope"] = "ops"
        if name == "CLAUDE.md":
            inferred["doc_type"] = "playbook"
        elif name == "MEMORY.md":
            inferred["doc_type"] = "host-memory"
        elif "plans" in parts:
            inferred["scope"] = "shared"
            inferred["doc_type"] = "plan"
        else:
            inferred["doc_type"] = "reference"

    else:
        inferred["scope"] = "shared"
        inferred["doc_type"] = "reference"

    inferred_tour = _infer_tour(path)
    if inferred_tour:
        inferred["tour"] = inferred_tour
        if inferred.get("doc_type") == "reference":
            inferred.setdefault("scope", "tour")

    return inferred


def apply_path_inference(meta: Metadata, rel_path: str):
    """Fill in metadata defaults based on the source repo path and filename."""
    inferred = _infer_path_metadata(rel_path)
    if not meta.scope and inferred.get("scope"):
        meta.scope = inferred["scope"]
    elif inferred.get("scope") in {"tour", "journal"}:
        meta.scope = inferred["scope"]
    if inferred.get("doc_type") and (
        not meta.doc_type
        or inferred["doc_type"] in {"handoff", "journal", "feedback", "mission-control", "host-memory"}
    ):
        meta.doc_type = inferred["doc_type"]
    if inferred.get("tour") and not meta.tour:
        meta.tour = inferred["tour"]
    meta.sensitivity = _stronger_sensitivity(meta.sensitivity, inferred.get("sensitivity"))


def parse_metadata(lines: list[str], frontmatter: dict[str, Any] | None = None) -> Metadata:
    """Extract normalized metadata from frontmatter and early markdown body lines."""
    meta = Metadata(raw={})
    frontmatter = frontmatter or {}

    for key, value in frontmatter.items():
        meta.raw[str(key)] = value
        _apply_metadata_key(meta, str(key), value)

    for line in lines[:20]:
        match = _META_RE.search(line)
        if not match:
            continue
        key_raw = match.group(1).strip()
        value = match.group(2).strip()
        meta.raw[key_raw] = value
        _apply_metadata_key(meta, key_raw, value)

    for line in lines[:80]:
        stripped = line.strip()
        if stripped.startswith("# "):
            meta.title = stripped[2:].strip()
            break

    if not meta.title:
        for key in ("title", "name"):
            value = frontmatter.get(key)
            if value:
                meta.title = str(value).strip()
                break

    if not meta.title:
        for line in lines[:20]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                meta.title = stripped
                break

    return meta


def extract_sections(content: str) -> list[str]:
    """Extract header names from markdown content."""
    return [match.group(2).strip() for match in _HEADER_RE.finditer(content)]


def extract_cross_references(content: str) -> list[str]:
    """Extract unique .md file references from markdown links."""
    refs = _LINK_RE.findall(content)
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        unique.append(ref)
    return unique


def _strip_code(content: str) -> str:
    """Blank out fenced code blocks and inline code spans.

    Only wiki-link extraction uses this. Reason it exists: the corpus documents
    CLI usage, and a JSON example like ``[["Header1","Header2"]]`` in a fenced
    block is not a link -- but it parses as one, and it then shows up in the
    dangling-link report as a missing note. Three of those were the loudest rows
    in the first real run. Lines are preserved (blanked, not removed) so nothing
    downstream shifts.
    """
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in content.splitlines():
        stripped = line.lstrip()
        if in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
            out.append("")
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_marker = stripped[:3]
            out.append("")
            continue
        out.append(_INLINE_CODE_RE.sub("", line))
    return "\n".join(out)


def extract_wiki_links(content: str) -> list[str]:
    """Extract unique ``[[wiki-link]]`` targets, alias and anchor stripped.

    This is the ONE wiki-link parser in the codebase -- the link graph reuses it
    rather than re-implementing markdown scanning (see linkgraph module docstring).
    ``[[note|shown text]]`` and ``[[note#Heading]]`` both yield ``note``; a bare
    ``[[#Heading]]`` (same-document anchor) yields nothing.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for raw in _WIKILINK_RE.findall(_strip_code(content)):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if not target or "://" in target:
            continue
        if target in seen:
            continue
        seen.add(target)
        unique.append(target)
    return unique


def parse_document(vault_root: Path, rel_path: str) -> VaultDocument:
    """Parse a markdown file into a VaultDocument with normalized metadata."""
    full_path = vault_root / rel_path
    content = full_path.read_text(encoding="utf-8")
    frontmatter, body, _body_start_line = split_frontmatter(content)
    lines = body.splitlines()

    metadata = parse_metadata(lines, frontmatter=frontmatter)
    apply_path_inference(metadata, rel_path)
    sections = extract_sections(body)
    cross_refs = extract_cross_references(body)
    wiki_links = extract_wiki_links(body)

    return VaultDocument(
        path=rel_path,
        content=content,
        metadata=metadata,
        sections=sections,
        cross_references=cross_refs,
        wiki_links=wiki_links,
    )


def infer_domain(rel_path: str) -> str:
    """Infer logical domain from path-aware scope metadata."""
    meta = Metadata()
    apply_path_inference(meta, rel_path)
    if meta.scope:
        return meta.scope
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0]
    return "root"
