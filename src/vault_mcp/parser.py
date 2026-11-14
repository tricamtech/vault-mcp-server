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

# Document lifecycle vocabulary (schema 2026-11-13). `status` means ONLY where a
# document is in its life; "what is the project up to" belongs in `progress`.
LIFECYCLE_STATUSES: frozenset[str] = frozenset(
    {"current", "draft", "superseded", "archived", "historical"}
)

# Status normalization: EXACT match on the whole (lower-cased, stripped) value.
# Until 2026-11-13 this was a substring scan, so `draft` became Active, `inactive`
# became Active, and any prose containing "done" became Done. Lifecycle words map
# to themselves; the legacy project words keep their legacy capitalised form so
# existing filters (`vault_list(status="Active")`, memory-api) keep working.
_STATUS_MAP: dict[str, str] = {
    "current": "current",
    "draft": "draft",
    "superseded": "superseded",
    "archived": "archived",
    "archive": "archived",
    "historical": "historical",
    "active": "Active",
    "activo": "Active",
    "en progreso": "Active",
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

# Everything normalize_status can return on purpose. Anything else is a
# non-standard value that passes through verbatim and is counted by vault_summary.
KNOWN_STATUSES: frozenset[str] = frozenset(_STATUS_MAP.values())

# Subject = what the document is about, defaulted from its directory. Distinct
# from `domain`, which is the path-derived storage partition and is unchanged.
SUBJECTS: frozenset[str] = frozenset(
    {"touring", "fleet", "homelab", "mtassist", "business", "personal", "comms"}
)
_SUBJECT_BY_DIR: dict[str, str] = {
    "shared/tours": "touring",
    "shared/tour-cache": "touring",
    "shared/fleet": "fleet",
    "shared/handoffs": "fleet",
    "shared/skills": "fleet",
    "shared/homelab": "homelab",
    "shared/mtassist": "mtassist",
    "shared/mastertour": "mtassist",
    "shared/business": "business",
    "shared/personal": "personal",
    "shared/comms": "comms",
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
    "progress": "progress",
    "subject": "subject",
    "as_of": "as_of",
    "as of": "as_of",
    "superseded_by": "superseded_by",
    "superseded by": "superseded_by",
}

_SEMANTIC_INDEX_BLOCKLIST = {"restricted", "secret"}
_SENSITIVITY_ORDER = {"": 0, "restricted": 1, "secret": 2}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

# Filename hints per tour code, taken from shared/tour-registry.md (Aliases +
# Memory Files) on 2026-11-13. The previous table named the wrong artists for
# jk / oa / ct and a tour (ct) that is not in the registry.
_TOUR_HINTS: dict[str, tuple[str, ...]] = {
    "bb": ("bacon-brothers", "bacon", "bb-salk"),
    "jk": ("jonah-kagen", "jonah", "kagen"),
    "ll": ("lisa-loeb", "loeb", "ll-shames", "shames", "touradmin-japan"),
    "mp": ("madeleine-peyroux", "peyroux", "mp-canada", "touradmin-canada"),
    "oa": ("oliver-anthony",),
    "yola": ("yola",),
}

# Path overrides. Sensitivity is meant to live in each file's own
# `sensitivity:` frontmatter (schema 2026-11-13); the entries here are a backstop
# for the four personal/ops files that carry it, and the only mechanism for the
# MEMORY.md mirrors, which are copies of files owned elsewhere. The eight
# `shared/reference/*` entries that used to exclude services.md, homeassistant.md,
# homelab.md, quick-reference.md, gmail-management.md, memory-non-tour.md,
# bob-agent.md and shesynergy.md keyed on paths that stopped existing when
# reference/ was dissolved; Cameron (2026-11-13) chose to keep those eight
# searchable after a raw-secret scan found only keychain entry NAMES in them.
_PATH_OVERRIDES: dict[str, dict[str, str]] = {
    "shared/people.cameron.md": {
        "scope": "personal",
        "doc_type": "person",
        "sensitivity": "restricted",
    },
    "shared/personal/people.cameron.full.md": {
        "scope": "personal",
        "doc_type": "person",
        "sensitivity": "restricted",
    },
    "shared/people.md": {
        "scope": "personal",
        "doc_type": "reference",
    },
    "shared/personal/actual-budget.md": {
        "scope": "personal",
        "doc_type": "finance",
        "sensitivity": "restricted",
    },
    "shared/homelab/nanostation-camera-bridge.md": {
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
    """Normalize a status value to the standard vocabulary.

    Exact match on the whole value (case-insensitive). Unknown values pass
    through verbatim: ``inactive`` stays ``inactive`` and prose stays prose,
    instead of a substring hit turning either into ``Active``.
    """
    cleaned = str(raw).strip()
    return _STATUS_MAP.get(cleaned.lower(), cleaned)


def is_known_status(value: Any) -> bool:
    """Whether a (normalized) status is one the vocabulary defines."""
    return str(value).strip() in KNOWN_STATUSES


def frontmatter_error(content: str) -> str | None:
    """Why a document that opens with ``---`` has unusable frontmatter, or None.

    ``split_frontmatter`` returns ``{}`` for a YAML error so the rest of the
    parser keeps working; this names the problem so vault_summary can count it.
    63 tour-cache files were silently metadata-less this way (2026-11-13).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() != "---":
            continue
        try:
            parsed = yaml.safe_load("\n".join(lines[1:idx]))
        except yaml.YAMLError as exc:
            first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            return f"yaml: {first}"
        if parsed is not None and not isinstance(parsed, dict):
            return f"yaml: frontmatter is a {type(parsed).__name__}, not a mapping"
        return None
    return "unterminated frontmatter (opening --- without a closing ---)"


def infer_subject(rel_path: str) -> str | None:
    """Directory default for `subject`, or None when the directory implies nothing."""
    parts = Path(rel_path.replace("\\", "/")).parts
    if len(parts) > 2:
        return _SUBJECT_BY_DIR.get("/".join(parts[:2]))
    return None


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
    elif key_norm == "progress":
        meta.progress = str(value).strip()
    elif key_norm == "subject":
        # Vocabulary only. Cached email-thread files already carry a `subject:`
        # that is the MAIL subject ("Monitor engineer for Sept 6th show"); those
        # stay in `raw` and the directory default applies (4 files, 2026-11-13).
        candidate = str(value).strip().lower()
        if candidate in SUBJECTS:
            meta.subject = candidate
    elif key_norm == "as_of":
        meta.as_of = str(value).strip()
    elif key_norm == "superseded_by":
        meta.superseded_by = str(value).strip()


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
    if not meta.subject:
        meta.subject = infer_subject(rel_path)
    meta.sensitivity = _stronger_sensitivity(meta.sensitivity, inferred.get("sensitivity"))


def parse_metadata(lines: list[str], frontmatter: dict[str, Any] | None = None) -> Metadata:
    """Extract normalized metadata from frontmatter and early markdown body lines.

    Frontmatter is authoritative. A ``**Status:** ...`` line in the first 20
    body lines used to OVERWRITE the YAML status (16 files showed prose or the
    wrong lifecycle because of it, 2026-11-13). Now: when YAML already states a
    status, the body line is read as ``progress``; when YAML has none, the body
    line is the status only if it normalizes to a known value, otherwise it is
    ``progress`` and the document has no lifecycle status.
    """
    meta = Metadata(raw={})
    frontmatter = frontmatter or {}

    for key, value in frontmatter.items():
        meta.raw[str(key)] = value
        _apply_metadata_key(meta, str(key), value)
    yaml_status = meta.status

    for line in lines[:20]:
        match = _META_RE.search(line)
        if not match:
            continue
        key_raw = match.group(1).strip()
        value = match.group(2).strip()
        if _KEY_MAP.get(key_raw.lower()) == "status":
            meta.raw.setdefault(key_raw, value)
            if yaml_status is not None:
                if not meta.progress:
                    meta.progress = value
                continue
            normalized = normalize_status(value)
            if is_known_status(normalized):
                meta.status = normalized
            elif not meta.progress:
                meta.progress = value
            continue
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


_NOT_A_NOTE_NAME = re.compile(r'["$={}<>^`\\]')

# Bash test operators, matched as whole space-delimited tokens (see _is_shell_test).
_SHELL_TEST_OPERATOR = re.compile(r"\s(?:-eq|-ne|-lt|-le|-gt|-ge|-nt|-ot|-ef|==|!=|=~|&&|\|\|)\s")


def _is_shell_test(raw: str) -> bool:
    """Whether the text between ``[[`` and ``]]`` is a bash test, not a link.

    Bash requires whitespace on both sides inside ``[[ ... ]]``, but padding
    alone is not enough to tell: ``[[ note ]]`` is written as a link and must
    still resolve. So padded text only counts as a test when it also looks like
    one -- a unary test or negation (``[[ -f x ]]``, ``[[ ! -d y ]]``) or a binary
    operator (``[[ a -lt b ]]``, ``[[ a || b ]]``). ``[[ -f x ]]`` was still a
    dangling "-f x" after _NOT_A_NOTE_NAME (2026-10-26).
    """
    if not (raw[:1].isspace() and raw[-1:].isspace()):
        return False
    inner = raw.strip()
    return inner.startswith(("-", "!")) or bool(_SHELL_TEST_OPERATOR.search(f" {inner} "))


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
        # Checked on the raw text, before the alias split: `[[ a || b ]]` would
        # otherwise become a link to "a".
        if _is_shell_test(raw):
            continue
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if not target or "://" in target:
            continue
        # Bash tests (`[[ "$x" -ge 13 ]]`, `[[ ${a,,} =~ ^(y ]]`) and chat-citation
        # markup use the same double brackets outside a recognisable code fence.
        # Note names never contain these characters, so such a "link" is not one
        # (4 false dangling links in imported chat transcripts, 2026-10-19).
        if _NOT_A_NOTE_NAME.search(target):
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
        frontmatter_error=frontmatter_error(content) if not frontmatter else None,
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
