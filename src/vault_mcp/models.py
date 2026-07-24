"""Domain models for vault documents and metadata."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Metadata:
    title: Optional[str] = None
    status: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    doc_type: Optional[str] = None
    scope: Optional[str] = None
    sensitivity: Optional[str] = None
    tour: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VaultDocument:
    path: str  # relative to vault root
    content: str
    metadata: Metadata
    sections: list[str] = field(default_factory=list)  # header names
    cross_references: list[str] = field(default_factory=list)  # linked .md files


@dataclass
class Chunk:
    doc_path: str  # relative path to vault root
    doc_title: str  # document H1 title
    section_header: str  # H2 header (or "" for preamble/whole-file)
    content: str  # raw chunk text
    embedding_text: str  # contextualized text for embedding
    start_line: int  # line number in source file
