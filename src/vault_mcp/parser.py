"""Markdown parsing — metadata extraction, sections, cross-references."""

import re
from pathlib import Path

from .models import Metadata, VaultDocument

# **Key:** Value pattern in first 15 lines
_META_RE = re.compile(r"\*\*(.+?):\*\*\s*(.+)")

# Markdown links to .md files
_LINK_RE = re.compile(r"\[.*?\]\(([^)]+\.md)\)")

# Obsidian-style wiki links: [[target]], [[target|alias]], [[target#Section]],
# and embeds ![[target]]. Targets are usually written without a .md suffix and
# often as a bare basename, so resolving them to a real file is the link graph's
# job -- this only extracts the target text.
# Non-greedy and newline-free so an unclosed "[[" cannot swallow the document.
_WIKILINK_RE = re.compile(r"!?\[\[([^\[\]\n]+?)\]\]")

# Inline code spans, stripped before wiki-link extraction (see _strip_code).
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# Characters that never appear in a note name but do appear inside other
# double-bracket syntax: shell tests (`[[ "$x" -ge 13 ]]`, `[[ ${a,,} =~ ^y ]]`),
# JSON arrays (`[["a","b"]]`) and footnote/citation markup (`[[^]]`).
_NOT_A_NOTE_NAME = re.compile(r'["$={}<>^`\\]')

# Binary operators of a bash ``[[ ... ]]`` test, as whitespace-separated tokens.
_SHELL_TEST_OPERATOR = re.compile(r"\s(?:-eq|-ne|-lt|-le|-gt|-ge|-nt|-ot|-ef|==|!=|=~|&&|\|\|)\s")

# Header pattern
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)", re.MULTILINE)

# Status normalization mapping
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

# Metadata key normalization
_KEY_MAP: dict[str, str] = {
    "creado": "created_date",
    "created": "created_date",
    "status": "status",
    "estado": "status",
    "última actualización": "updated_date",
    "ultima actualización": "updated_date",
    "last updated": "updated_date",
    "updated": "updated_date",
}


def normalize_status(raw: str) -> str:
    """Normalize status strings and emojis to standard vocabulary."""
    cleaned = raw.strip().lower()
    # Check emoji first (they may be prefix in table cells like "🔴 Pendiente")
    for token, normalized in _STATUS_MAP.items():
        if token in cleaned:
            return normalized
    return raw.strip()


def parse_metadata(lines: list[str]) -> Metadata:
    """Extract metadata from the first 15 lines of a markdown file."""
    meta = Metadata(raw={})
    scan_lines = lines[:15]

    for line in scan_lines:
        m = _META_RE.search(line)
        if m:
            key_raw = m.group(1).strip()
            value = m.group(2).strip()
            meta.raw[key_raw] = value

            key_norm = _KEY_MAP.get(key_raw.lower())
            if key_norm == "created_date":
                meta.created_date = value
            elif key_norm == "updated_date":
                meta.updated_date = value
            elif key_norm == "status":
                meta.status = normalize_status(value)

    # Title: first H1, or first non-empty line
    for line in lines[:5]:
        stripped = line.strip()
        if stripped.startswith("# "):
            meta.title = stripped[2:].strip()
            break
        elif stripped and not stripped.startswith("---") and meta.title is None:
            meta.title = stripped

    return meta


def extract_sections(content: str) -> list[str]:
    """Extract header names from markdown content."""
    return [m.group(2).strip() for m in _HEADER_RE.finditer(content)]


def extract_cross_references(content: str) -> list[str]:
    """Extract unique .md file references from markdown links."""
    refs = _LINK_RE.findall(content)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return unique


def _strip_code(content: str) -> str:
    """Blank out fenced code blocks and inline code spans.

    Only wiki-link extraction uses this. Notes that document CLI usage often
    contain examples like ``[["Header1","Header2"]]`` or ``[[ -f file ]]`` in a
    fenced block. Those are not links, but they parse as one and would then show
    up in the dangling-link report as missing notes. Lines are blanked rather
    than removed so line numbers do not shift.
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


def _is_shell_test(raw: str) -> bool:
    """Whether the text between ``[[`` and ``]]`` is a bash test, not a link.

    Bash requires whitespace on both sides inside ``[[ ... ]]``, but padding
    alone is not enough to tell: some vaults write links as ``[[ note ]]``, and
    those must still resolve. So padded text only counts as a test when it also
    looks like one -- a unary test or negation (``[[ -f x ]]``, ``[[ ! -d x ]]``)
    or a binary operator (``[[ a -lt b ]]``, ``[[ x == y ]]``).
    """
    if not (raw[:1].isspace() and raw[-1:].isspace()):
        return False
    inner = raw.strip()
    return inner.startswith(("-", "!")) or bool(_SHELL_TEST_OPERATOR.search(f" {inner} "))


def extract_wiki_links(content: str) -> list[str]:
    """Extract unique ``[[wiki-link]]`` targets, alias and heading stripped.

    ``[[note|shown text]]`` and ``[[note#Heading]]`` both yield ``note``; a bare
    ``[[#Heading]]`` (same-document anchor) yields nothing. This is the only
    wiki-link parser; the link graph reuses it rather than scanning markdown
    a second way.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for raw in _WIKILINK_RE.findall(_strip_code(content)):
        # Shell tests and citation markup use the same double brackets, and not
        # always inside a code fence. Without these checks each of them would be
        # reported as a dangling link.
        if _is_shell_test(raw):
            continue
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if not target or "://" in target:
            continue
        # Note names never contain these characters (see _NOT_A_NOTE_NAME).
        if _NOT_A_NOTE_NAME.search(target):
            continue
        if target in seen:
            continue
        seen.add(target)
        unique.append(target)
    return unique


def parse_document(vault_root: Path, rel_path: str) -> VaultDocument:
    """Parse a markdown file into a VaultDocument with metadata."""
    full_path = vault_root / rel_path
    content = full_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    metadata = parse_metadata(lines)
    sections = extract_sections(content)
    cross_refs = extract_cross_references(content)
    wiki_links = extract_wiki_links(content)

    return VaultDocument(
        path=rel_path,
        content=content,
        metadata=metadata,
        sections=sections,
        cross_references=cross_refs,
        wiki_links=wiki_links,
    )


def infer_domain(rel_path: str) -> str:
    """Infer domain from the file's top-level directory."""
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0]
    return "root"
