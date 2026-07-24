"""Vault MCP server for markdown memory with semantic search and structured writes."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .auth import CfAccessBypassMiddleware, SimpleOAuthProvider, create_approve_routes
from .catalog import build_inventory
from .indexer import VaultIndexer
from .parser import (
    infer_domain,
    parse_document,
    split_frontmatter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
_config: dict[str, Any] = {}
if _config_path.exists():
    with open(_config_path, encoding="utf-8") as handle:
        _config = yaml.safe_load(handle) or {}

VAULT_PATH = Path(os.environ.get("VAULT_PATH", _config.get("vault", {}).get("path", "/vault")))
SERVER_HOST = os.environ.get("SERVER_HOST", _config.get("server", {}).get("host", "0.0.0.0"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", _config.get("server", {}).get("port", 8080)))

_search_config = _config.get("search", {})
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", _search_config.get("chroma_path", "/app/chroma_data")))
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    _search_config.get("model", "paraphrase-multilingual-MiniLM-L12-v2"),
)

OAUTH_ISSUER_URL = os.environ.get("OAUTH_ISSUER_URL", "")
OAUTH_PIN = os.environ.get("OAUTH_PIN", "")
# Optional path to a JSON file used to persist registered OAuth clients +
# issued tokens across server restarts. When unset (default), all OAuth
# state lives in memory only — every restart forces clients to re-auth.
OAUTH_STATE_PATH = os.environ.get("OAUTH_STATE_PATH", "")

# DNS rebinding protection allowed hosts (comma-separated). When binding to
# 127.0.0.1/localhost FastMCP auto-enables protection with only localhost in
# the allowlist; if you front the server with a reverse proxy or tunnel that
# rewrites the Host header (e.g. Cloudflare Tunnel for vault.example.com),
# add those hostnames here. Wildcards like "vault.example.com:*" are honored.
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
VAULT_WATCH = os.environ.get("VAULT_WATCH", "true").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

_mcp_kwargs: dict[str, Any] = dict(
    name="vault-mcp-server",
    instructions=(
        "MCP server for a personal Markdown knowledge vault. "
        "Provides tools to list, read, write, append, update sections, set metadata, "
        "search, and summarize vault documents. "
        "Use vault_search for semantic queries across the vault and vault_read when you "
        "already know the file path."
    ),
    host=SERVER_HOST,
    port=SERVER_PORT,
)

if ALLOWED_HOSTS:
    from mcp.server.transport_security import TransportSecuritySettings

    _allowed = ["127.0.0.1:*", "localhost:*", "[::1]:*", *ALLOWED_HOSTS]
    _mcp_kwargs["transport_security"] = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed,
    )
    logger.info("DNS rebinding allowed_hosts: %s", _allowed)

oauth_provider: SimpleOAuthProvider | None = None
if OAUTH_ISSUER_URL:
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions

    oauth_provider = SimpleOAuthProvider(
        OAUTH_ISSUER_URL,
        pin=OAUTH_PIN,
        state_path=Path(OAUTH_STATE_PATH) if OAUTH_STATE_PATH else None,
    )
    _mcp_kwargs["auth"] = AuthSettings(
        issuer_url=OAUTH_ISSUER_URL,
        resource_server_url=OAUTH_ISSUER_URL,
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )
    _mcp_kwargs["auth_server_provider"] = oauth_provider
    logger.info("OAuth enabled — issuer: %s", OAUTH_ISSUER_URL)

mcp = FastMCP(**_mcp_kwargs)
indexer = VaultIndexer(VAULT_PATH, CHROMA_PATH, EMBEDDING_MODEL, watch_files=VAULT_WATCH)

_FRONTMATTER_SHOWN_KEYS = {
    "created",
    "created date",
    "created_date",
    "creado",
    "doc_type",
    "last updated",
    "name",
    "scope",
    "sensitivity",
    "status",
    "tags",
    "title",
    "tour",
    "type",
    "updated",
    "updated_date",
    "ultima actualización",
    "última actualización",
}


def _list_md_files() -> list[str]:
    """List all .md files in the vault, relative paths."""
    return build_inventory(VAULT_PATH).canonical_files


def _default_create_path(file_path: str) -> str:
    """Map legacy shorthand paths into the repo-shaped write surface."""
    normalized = file_path.strip().lstrip("./")
    if normalized.startswith("shared/") or normalized.startswith("bob/"):
        return normalized
    if normalized.startswith(("air/", "mbp/", "argus/")):
        return normalized
    if normalized.startswith("reference/"):
        return f"shared/{normalized}"
    if "/" not in normalized:
        return f"shared/{normalized}"
    return normalized


def _resolve_markdown_path(file_path: str, allow_create: bool = False) -> tuple[Path | None, str | None]:
    """Validate a relative markdown path or compatibility alias against the vault root."""
    if not file_path.endswith(".md"):
        return None, "Error: only .md files supported."

    normalized = file_path.strip().lstrip("./")
    inventory = build_inventory(VAULT_PATH)
    rel_path = inventory.resolve(normalized)
    if rel_path is None:
        if allow_create:
            rel_path = _default_create_path(normalized)
        else:
            return None, f"Error: file not found: {file_path}"

    # Logical path: do NOT collapse intended symlinked subtrees (e.g.
    # shared -> /root/claude-memory/shared). Downstream callers rely on
    # full_path.relative_to(VAULT_PATH) and parse_document(VAULT_PATH, rel_path),
    # so full_path must stay under VAULT_PATH. The OS transparently follows the
    # symlink when the file is actually read.
    full_path = VAULT_PATH / rel_path
    # Traversal guard: validate the *resolved* path does not escape the vault
    # root nor the real targets of symlinked subtrees directly under it.
    # rel_path already came from the trusted inventory (os.walk followlinks=True
    # from VAULT_PATH), so membership is the security boundary; this only blocks
    # genuine ../ escape while permitting intended symlinked corpora.
    resolved = full_path.resolve()
    allowed_roots = [VAULT_PATH.resolve()]
    try:
        for entry in VAULT_PATH.iterdir():
            if entry.is_symlink() and entry.is_dir():
                allowed_roots.append(entry.resolve())
    except OSError:
        pass
    if not any(
        resolved == root or str(resolved).startswith(str(root) + "/")
        for root in allowed_roots
    ):
        return None, "Error: path traversal not allowed."
    return full_path, None


def _queue_reindex(file_path: str):
    """Schedule a single-file reindex if the index is ready."""
    if indexer.is_ready:
        asyncio.create_task(indexer.index_file(file_path))


def _load_frontmatter_and_body(full_path: Path) -> tuple[dict[str, Any], str]:
    """Read a markdown file and split out frontmatter."""
    content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    frontmatter, body, _body_start_line = split_frontmatter(content)
    return frontmatter, body


def _render_markdown(frontmatter: dict[str, Any], body: str) -> str:
    """Render a markdown document with optional YAML frontmatter."""
    body = body.strip("\n")
    parts: list[str] = []
    if frontmatter:
        dumped = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        parts.extend(["---", dumped, "---"])
    if body:
        if parts:
            parts.append("")
        parts.append(body)
    rendered = "\n".join(parts).rstrip()
    return rendered + "\n"


def _set_frontmatter_value(frontmatter: dict[str, Any], key: str, value: Any):
    """Set or remove a frontmatter value with light normalization."""
    normalized_key = key.strip()
    if isinstance(value, str):
        value = value.strip()
    if value in ("", None, []):
        frontmatter.pop(normalized_key, None)
        return
    if normalized_key == "tags" and isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    frontmatter[normalized_key] = value


def _upsert_updated(frontmatter: dict[str, Any]):
    """Stamp a simple ISO date into the document frontmatter."""
    frontmatter["updated"] = datetime.now(timezone.utc).date().isoformat()


def _section_boundary_pattern(level: int) -> re.Pattern[str]:
    """Pattern for headings that terminate a section at the given depth."""
    return re.compile(rf"^#{{1,{level}}}\s+")


def _upsert_markdown_section(body: str, section_header: str, content: str, level: int) -> str:
    """Insert or replace a markdown section by heading text."""
    heading = f"{'#' * level} {section_header.strip()}"
    new_section_lines = [heading]
    if content.strip():
        new_section_lines.append(content.strip("\n"))
    new_section = "\n".join(new_section_lines).rstrip()

    body_lines = body.splitlines()
    section_start = None
    for idx, line in enumerate(body_lines):
        if line.strip() == heading:
            section_start = idx
            break

    if section_start is None:
        if not body.strip():
            return new_section + "\n"
        return body.rstrip() + "\n\n" + new_section + "\n"

    boundary = _section_boundary_pattern(level)
    section_end = len(body_lines)
    for idx in range(section_start + 1, len(body_lines)):
        if boundary.match(body_lines[idx]):
            section_end = idx
            break

    replacement_lines = new_section.splitlines()
    updated_lines = body_lines[:section_start] + replacement_lines + body_lines[section_end:]
    return "\n".join(updated_lines).rstrip() + "\n"


def _document_header(doc) -> list[str]:
    """Build the metadata summary shown above vault_read output."""
    header_parts = [f"# {doc.metadata.title or doc.path}"]
    header_parts.append(f"**Path:** {doc.path}")
    header_parts.append(f"**Domain:** {infer_domain(doc.path)}")
    if doc.metadata.status:
        header_parts.append(f"**Status:** {doc.metadata.status}")
    if doc.metadata.created_date:
        header_parts.append(f"**Created:** {doc.metadata.created_date}")
    if doc.metadata.updated_date:
        header_parts.append(f"**Updated:** {doc.metadata.updated_date}")
    if doc.metadata.doc_type:
        header_parts.append(f"**Type:** {doc.metadata.doc_type}")
    if doc.metadata.scope:
        header_parts.append(f"**Scope:** {doc.metadata.scope}")
    if doc.metadata.tour:
        header_parts.append(f"**Tour:** {doc.metadata.tour}")
    if doc.metadata.sensitivity:
        header_parts.append(f"**Sensitivity:** {doc.metadata.sensitivity}")
    if doc.metadata.tags:
        header_parts.append(f"**Tags:** {', '.join(doc.metadata.tags)}")
    if doc.sections:
        header_parts.append(f"**Sections:** {', '.join(doc.sections[:15])}")
    if doc.cross_references:
        header_parts.append(f"**Cross-references:** {', '.join(doc.cross_references[:10])}")
    extra = {
        key: value
        for key, value in doc.metadata.raw.items()
        if key.lower() not in _FRONTMATTER_SHOWN_KEYS
    }
    if extra:
        header_parts.append(f"**Other metadata:** {extra}")
    header_parts.append("\n---\n")
    return header_parts


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_list(
    domain: str = "",
    status: str = "",
    scope: str = "",
    sensitivity: str = "",
) -> str:
    """List vault files with optional metadata filters."""
    results: list[str] = []
    for rel_path in _list_md_files():
        doc = parse_document(VAULT_PATH, rel_path)
        file_domain = infer_domain(rel_path)
        if domain and file_domain.lower() != domain.lower():
            continue
        if status and (doc.metadata.status or "").lower() != status.lower():
            continue
        if scope and (doc.metadata.scope or "").lower() != scope.lower():
            continue
        if sensitivity and (doc.metadata.sensitivity or "").lower() != sensitivity.lower():
            continue

        labels: list[str] = []
        if doc.metadata.status:
            labels.append(doc.metadata.status)
        if doc.metadata.scope:
            labels.append(f"scope={doc.metadata.scope}")
        if doc.metadata.sensitivity:
            labels.append(f"sensitivity={doc.metadata.sensitivity}")
        label_text = f" [{' | '.join(labels)}]" if labels else ""
        results.append(f"- {rel_path}  —  {doc.metadata.title or Path(rel_path).stem}{label_text}")

    if not results:
        return "No files found matching the filters."

    filters = []
    if domain:
        filters.append(f"domain={domain}")
    if status:
        filters.append(f"status={status}")
    if scope:
        filters.append(f"scope={scope}")
    if sensitivity:
        filters.append(f"sensitivity={sensitivity}")
    header = f"Vault files ({len(results)})"
    if filters:
        header += " | " + ", ".join(filters)
    return f"{header}\n\n" + "\n".join(results)


@mcp.tool()
def vault_read(file_path: str) -> str:
    """Read a vault file with parsed metadata, sections, and cross-references."""
    full_path, error = _resolve_markdown_path(file_path)
    if error:
        return error
    if full_path is None or not full_path.exists():
        return f"Error: file not found: {file_path}"

    rel_path = str(full_path.relative_to(VAULT_PATH))
    doc = parse_document(VAULT_PATH, rel_path)
    return "\n".join(_document_header(doc)) + doc.content


@mcp.tool()
def vault_write(file_path: str, content: str) -> str:
    """Create or overwrite a vault file."""
    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    assert full_path is not None

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Written: {rel_path} ({len(content)} chars)"


@mcp.tool()
def vault_edit(file_path: str, old_text: str, new_text: str) -> str:
    """Edit a vault file via exact string replacement."""
    full_path, error = _resolve_markdown_path(file_path)
    if error:
        return error
    if full_path is None or not full_path.exists():
        return f"Error: file not found: {file_path}"

    content = full_path.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        return "Error: old_text not found in file."
    if count > 1:
        return f"Error: old_text found {count} times — must be unique. Provide more context."

    full_path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Edited: {rel_path} (replaced 1 occurrence)"


@mcp.tool()
def vault_append(file_path: str, content: str) -> str:
    """Append content to the end of a markdown file, creating it if needed."""
    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    assert full_path is not None

    full_path.parent.mkdir(parents=True, exist_ok=True)
    existing = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if existing and not existing.endswith("\n\n"):
        existing += "\n"
    full_path.write_text(existing + content.strip("\n") + "\n", encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Appended: {rel_path}"


@mcp.tool()
def vault_set_frontmatter(file_path: str, key: str, value: str) -> str:
    """Set or clear a YAML frontmatter field on a vault document."""
    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    assert full_path is not None

    frontmatter, body = _load_frontmatter_and_body(full_path)
    _set_frontmatter_value(frontmatter, key, value)
    _upsert_updated(frontmatter)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(_render_markdown(frontmatter, body), encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Updated frontmatter: {rel_path} ({key})"


@mcp.tool()
def vault_upsert_section(file_path: str, section_header: str, content: str, level: int = 2) -> str:
    """Insert or replace a markdown section by heading text."""
    if level < 1 or level > 4:
        return "Error: level must be between 1 and 4."

    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    assert full_path is not None

    frontmatter, body = _load_frontmatter_and_body(full_path)
    _upsert_updated(frontmatter)
    new_body = _upsert_markdown_section(body, section_header, content, level)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(_render_markdown(frontmatter, new_body), encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Upserted section: {rel_path} -> {section_header}"


@mcp.tool()
def vault_create_handoff(file_path: str, title: str, body: str, tour: str = "", status: str = "Active") -> str:
    """Create a standardized handoff document."""
    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    if full_path is None:
        return "Error: invalid file path."
    if full_path.exists():
        return f"Error: file already exists: {file_path}"

    today = datetime.now(timezone.utc).date().isoformat()
    frontmatter: dict[str, Any] = {
        "title": title,
        "type": "handoff",
        "status": status,
        "created": today,
        "updated": today,
    }
    if tour:
        frontmatter["tour"] = tour

    body_text = f"# {title}\n\n{body.strip()}\n"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(_render_markdown(frontmatter, body_text), encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Created handoff: {rel_path}"


@mcp.tool()
def vault_append_digest(file_path: str, heading: str, content: str) -> str:
    """Append a digest entry as a new H2 section."""
    full_path, error = _resolve_markdown_path(file_path, allow_create=True)
    if error:
        return error
    assert full_path is not None

    frontmatter, body = _load_frontmatter_and_body(full_path)
    if not frontmatter:
        today = datetime.now(timezone.utc).date().isoformat()
        frontmatter = {
            "type": "digest",
            "created": today,
        }
    _upsert_updated(frontmatter)
    digest_section = f"## {heading.strip()}\n\n{content.strip()}"
    new_body = body.rstrip()
    if new_body:
        new_body += "\n\n" + digest_section
    else:
        new_body = digest_section
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(_render_markdown(frontmatter, new_body), encoding="utf-8")
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    return f"Appended digest: {rel_path}"


@mcp.tool()
def vault_summary() -> str:
    """Dashboard summary of the vault — file counts by domain, status, and sensitivity."""
    files = _list_md_files()
    domain_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    sensitivity_counts: Counter[str] = Counter()
    active_projects: list[str] = []
    file_mtimes: list[tuple[float, str]] = []

    for rel_path in files:
        doc = parse_document(VAULT_PATH, rel_path)
        domain_counts[infer_domain(rel_path)] += 1
        if doc.metadata.status:
            status_counts[doc.metadata.status] += 1
            if doc.metadata.status == "Active":
                active_projects.append(f"  - {doc.metadata.title or Path(rel_path).stem} ({rel_path})")
        if doc.metadata.sensitivity:
            sensitivity_counts[doc.metadata.sensitivity] += 1

        try:
            mtime = (VAULT_PATH / rel_path).stat().st_mtime
            file_mtimes.append((mtime, rel_path))
        except OSError:
            pass

    file_mtimes.sort(reverse=True)
    lines_out = ["# Vault Summary\n", f"**Total files:** {len(files)}\n", "## By Domain\n"]
    for domain, count in domain_counts.most_common():
        lines_out.append(f"  - {domain}: {count}")

    lines_out.append("\n## By Status\n")
    for status, count in status_counts.most_common():
        lines_out.append(f"  - {status}: {count}")
    unstated = len(files) - sum(status_counts.values())
    if unstated:
        lines_out.append(f"  - (no status): {unstated}")

    lines_out.append("\n## By Sensitivity\n")
    if sensitivity_counts:
        for sensitivity, count in sensitivity_counts.most_common():
            lines_out.append(f"  - {sensitivity}: {count}")
    else:
        lines_out.append("  - none tagged")

    if active_projects:
        lines_out.append(f"\n## Active Projects ({len(active_projects)})\n")
        lines_out.extend(active_projects[:20])

    if file_mtimes:
        lines_out.append("\n## Recently Modified\n")
        for mtime, rel_path in file_mtimes[:10]:
            dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            lines_out.append(f"  - [{dt}] {rel_path}")

    return "\n".join(lines_out)


@mcp.tool()
async def vault_search(query: str, top_k: int = 10) -> str:
    """Semantic search across the vault."""
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    top_k = min(max(top_k, 1), 20)
    results = await indexer.search(query, top_k)
    if not results:
        return "No results found."

    parts = [f'Search results for: "{query}" ({len(results)} matches)\n']
    for idx, result in enumerate(results, start=1):
        section_label = (
            f" > {result['section']}"
            if result["section"] and result["section"] != "preamble"
            else ""
        )
        parts.append(f"### {idx}. {result['title']}{section_label}")
        meta_bits = [f"**File:** {result['path']}", f"**Score:** {result['score']:.3f}"]
        if result.get("sensitivity"):
            meta_bits.append(f"**Sensitivity:** {result['sensitivity']}")
        parts.append("  |  ".join(meta_bits))
        parts.append(f"```markdown\n{result['content']}\n```")
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def vault_reindex() -> str:
    """Force a full re-index of the vault for semantic search."""
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    stats = await indexer.index_all(force=True)
    return (
        "Re-index complete.\n"
        f"Files processed: {stats['files']}\n"
        f"Chunks indexed: {stats['chunks']}\n"
        f"Time: {stats['elapsed']:.1f}s"
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AcceptHeaderMiddleware(BaseHTTPMiddleware):
    """Ensure Accept header includes required media types for Streamable HTTP."""

    async def dispatch(self, request: Request, call_next):
        accept = request.headers.get("accept", "")
        if "text/event-stream" not in accept or "application/json" not in accept:
            new_accept = "application/json, text/event-stream"
            request._headers = request.headers.mutablecopy()
            request._headers["accept"] = new_accept
            request.scope["headers"] = [
                (key.encode(), value.encode()) for key, value in request._headers.items()
            ]
        return await call_next(request)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server with Streamable HTTP transport."""
    if oauth_provider:
        mcp._custom_starlette_routes.extend(create_approve_routes(oauth_provider))

    app = mcp.streamable_http_app()
    app.add_middleware(AcceptHeaderMiddleware)

    if oauth_provider:
        app.add_middleware(
            CfAccessBypassMiddleware,
            bypass_token=oauth_provider.cf_bypass_token,
        )

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_index(app_instance):
        async with original_lifespan(app_instance) as state:
            asyncio.create_task(indexer.start())
            yield state

    app.router.lifespan_context = lifespan_with_index
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    main()
