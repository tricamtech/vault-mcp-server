"""Markdown chunking for semantic indexing with frontmatter and heading awareness."""

from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk
from .parser import (
    apply_path_inference,
    is_semantic_index_blocked,
    parse_metadata,
    split_frontmatter,
)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_MAX_CHUNK_CHARS = 2200
_OVERLAP_CHARS = 280


def _hard_split_line(line: str) -> list[str]:
    """Split a single line longer than the chunk budget into word-boundary
    pieces. Without this, a mega-line (pocket transcripts run to 50K chars)
    becomes ONE oversized chunk and the embedder truncates at ~128 tokens,
    leaving ~95% of the content invisible to semantic search."""
    pieces: list[str] = []
    while len(line) > _MAX_CHUNK_CHARS:
        cut = line.rfind(" ", _MAX_CHUNK_CHARS - 200, _MAX_CHUNK_CHARS)
        if cut <= 0:
            cut = _MAX_CHUNK_CHARS
        pieces.append(line[:cut])
        line = line[cut:].lstrip()
    pieces.append(line)
    return pieces


def _split_section_lines(lines: list[str], start_line: int) -> list[tuple[int, str]]:
    """Split a section into overlapping chunks while keeping rough line anchors."""
    if not lines:
        return []

    # Pre-split any line longer than the chunk budget, keeping a map back to
    # the original line index so chunk anchors stay on real file lines.
    split_lines: list[str] = []
    orig_idx: list[int] = []
    for i, line in enumerate(lines):
        if len(line) > _MAX_CHUNK_CHARS:
            for piece in _hard_split_line(line):
                split_lines.append(piece)
                orig_idx.append(i)
        else:
            split_lines.append(line)
            orig_idx.append(i)
    lines = split_lines

    chunks: list[tuple[int, str]] = []
    chunk_lines: list[str] = []
    chunk_start_idx = 0
    line_idx = 0

    while line_idx < len(lines):
        if not chunk_lines:
            chunk_start_idx = line_idx

        candidate_lines = chunk_lines + [lines[line_idx]]
        candidate_text = "\n".join(candidate_lines).strip()

        if chunk_lines and len(candidate_text) > _MAX_CHUNK_CHARS:
            text = "\n".join(chunk_lines).strip()
            if len(text) >= 20:
                chunks.append((start_line + orig_idx[chunk_start_idx], text))

            overlap_lines: list[str] = []
            overlap_chars = 0
            for line in reversed(chunk_lines):
                overlap_lines.insert(0, line)
                overlap_chars += len(line) + 1
                if overlap_chars >= _OVERLAP_CHARS:
                    break

            # Progress guard: if the overlap plus the CURRENT line still busts
            # the cap (a single line longer than the chunk budget -- e.g.
            # pocket transcripts with 50K-char lines), keeping the overlap
            # means the next iteration emits the identical chunk and never
            # advances line_idx: infinite loop, unbounded chunk list, OOM
            # (the 2026-05-22..07-02 vault-mcp crash saga). Drop the overlap
            # so the long line starts a fresh chunk; it is emitted as a single
            # oversized chunk, which the embedder truncates -- lossy but finite.
            overlap_text_len = sum(len(l) + 1 for l in overlap_lines)
            if overlap_text_len + len(lines[line_idx]) > _MAX_CHUNK_CHARS:
                overlap_lines = []

            chunk_lines = overlap_lines
            chunk_start_idx = max(0, line_idx - len(overlap_lines))
            continue

        chunk_lines.append(lines[line_idx])
        line_idx += 1

    text = "\n".join(chunk_lines).strip()
    if len(text) >= 20:
        anchor = orig_idx[chunk_start_idx] if chunk_start_idx < len(orig_idx) else 0
        chunks.append((start_line + anchor, text))

    return chunks


def _iter_heading_sections(body_lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """Split body lines into heading-aware sections with nested heading paths."""
    sections: list[tuple[str, int, list[str]]] = []
    heading_stack: list[str] = []
    current_header = "preamble"
    current_lines: list[str] = []
    current_start_line = 1

    for line_no, line in enumerate(body_lines, start=1):
        match = _HEADING_RE.match(line)
        if not match:
            current_lines.append(line)
            continue

        if current_lines and any(part.strip() for part in current_lines):
            sections.append((current_header, current_start_line, current_lines))

        level = len(match.group(1))
        title = match.group(2).strip()
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)

        current_header = " > ".join(heading_stack)
        current_lines = [line]
        current_start_line = line_no

    if current_lines and any(part.strip() for part in current_lines):
        sections.append((current_header, current_start_line, current_lines))

    return sections


# Heading sections excluded from the semantic index.
#
# The pocket re-summarizer (on Bob: /root/scripts/maintenance/pocket-resummarize.py) replaces
# Pocket API summaries -- which misattribute speakers, and were caught inventing a person
# outright -- and preserves Pocket's original in-file under "## Summary (Pocket, unverified)"
# as a rollback and diff signal. That copy must stay in the FILE but must NOT be searchable:
# a search snippet does not always make the "unverified" heading obvious, so indexing it
# doubles the wrong-attribution surface. The corrected "## Summary" is indexed as normal.
_EXCLUDED_SECTION_MARKERS = ("(pocket, unverified)",)


def _section_excluded(section_header: str) -> bool:
    """True if any heading in the section path is marked unverified."""
    parts = [part.strip().lower() for part in section_header.split(" > ")]
    return any(marker in part for part in parts for marker in _EXCLUDED_SECTION_MARKERS)


def chunk_document(vault_root: Path, rel_path: str) -> list[Chunk]:
    """Split a markdown file into semantically useful chunks."""
    full_path = vault_root / rel_path
    content = full_path.read_text(encoding="utf-8")
    frontmatter, body, body_start_line = split_frontmatter(content)
    body_lines = body.splitlines()

    meta = parse_metadata(body_lines, frontmatter=frontmatter)
    apply_path_inference(meta, rel_path)
    if is_semantic_index_blocked(meta.sensitivity):
        return []

    doc_title = meta.title or Path(rel_path).stem
    sections = _iter_heading_sections(body_lines)
    if not sections:
        sections = [("preamble", 1, body_lines)]

    chunks: list[Chunk] = []
    for section_header, start_line, lines in sections:
        if _section_excluded(section_header):
            continue
        for chunk_start_line, chunk_text in _split_section_lines(lines, start_line):
            if section_header and section_header != "preamble":
                embedding_text = f"{doc_title} ({rel_path}) > {section_header}\n\n{chunk_text}"
            else:
                embedding_text = f"{doc_title} ({rel_path})\n\n{chunk_text}"

            chunks.append(
                Chunk(
                    doc_path=rel_path,
                    doc_title=doc_title,
                    section_header=section_header,
                    content=chunk_text,
                    embedding_text=embedding_text,
                    start_line=body_start_line + chunk_start_line - 1,
                )
            )

    return chunks
