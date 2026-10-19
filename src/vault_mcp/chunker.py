"""Markdown chunking — split documents by H2 headers for semantic indexing."""

import re
from pathlib import Path

from .models import Chunk
from .parser import parse_metadata

# H2 header line (## ) and its text
_H2_RE = re.compile(r"^## +(.+)")

# Any markdown heading line
_HEADING_RE = re.compile(r"^#{1,6}\s")

# Opening/closing line of a fenced code block
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _split_h2_sections(lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """Split lines into (header, start_line, lines) sections on H2 headers.

    The first section is the preamble (header ""). `##` lines inside a fenced
    code block are comments in the pasted code (shell, python, a markdown
    sample), not headings: treating them as headings chopped every such
    block into fake sections named after a code comment.
    """
    sections: list[tuple[str, int, list[str]]] = []
    header = ""
    start_line = 1
    current: list[str] = []
    in_fence = False

    for line_no, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _H2_RE.match(line)
        if not match:
            current.append(line)
            continue
        sections.append((header, start_line, current))
        header = match.group(1).strip()
        start_line = line_no
        current = [line]

    sections.append((header, start_line, current))
    return sections


def _has_body(lines: list[str]) -> bool:
    """True if a section holds anything besides heading lines.

    A heading with nothing under it would otherwise become its own chunk;
    being short, it scores high in search while carrying no answer.
    """
    return any(line.strip() and not _HEADING_RE.match(line) for line in lines)


def chunk_document(vault_root: Path, rel_path: str) -> list[Chunk]:
    """Split a markdown file into chunks by H2 headers.

    Returns one chunk per H2 section, plus a preamble chunk for content
    before the first H2. Files without H2 headers produce a single chunk.
    Sections with no content under their heading are skipped.
    """
    full_path = vault_root / rel_path
    content = full_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Get title from metadata
    meta = parse_metadata(lines)
    doc_title = meta.title or Path(rel_path).stem

    # Split on H2 boundaries
    sections = _split_h2_sections(lines)

    if len(sections) <= 1:
        # No H2 headers — single chunk for entire file
        text = content.strip()
        if len(text) < 20:
            return []
        return [
            Chunk(
                doc_path=rel_path,
                doc_title=doc_title,
                section_header="",
                content=text,
                embedding_text=f"{doc_title} ({rel_path})\n\n{text}",
                start_line=1,
            )
        ]

    chunks: list[Chunk] = []

    for header, start_line, section_lines in sections:
        text = "\n".join(section_lines).strip()
        if len(text) < 20 or not _has_body(section_lines):
            continue

        # Build contextualized text for embedding
        if header:
            embedding_text = f"{doc_title} ({rel_path}) > {header}\n\n{text}"
        else:
            embedding_text = f"{doc_title} ({rel_path})\n\n{text}"

        chunks.append(
            Chunk(
                doc_path=rel_path,
                doc_title=doc_title,
                section_header=header or "preamble",
                content=text,
                embedding_text=embedding_text,
                start_line=start_line,
            )
        )

    return chunks
