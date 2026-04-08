"""Vault MCP Server — intelligent layer over a Markdown knowledge vault."""

import asyncio
import logging
import os
from collections import Counter
from pathlib import Path

import uvicorn
import yaml
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .auth import CfAccessBypassMiddleware, SimpleOAuthProvider, create_approve_routes
from .indexer import VaultIndexer
from .parser import (
    extract_cross_references,
    extract_sections,
    infer_domain,
    normalize_status,
    parse_document,
    parse_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
_config: dict = {}
if _config_path.exists():
    with open(_config_path) as f:
        _config = yaml.safe_load(f) or {}

VAULT_PATH = Path(os.environ.get("VAULT_PATH", _config.get("vault", {}).get("path", "/vault")))
SERVER_HOST = os.environ.get("SERVER_HOST", _config.get("server", {}).get("host", "0.0.0.0"))
SERVER_PORT = int(os.environ.get("SERVER_PORT", _config.get("server", {}).get("port", 8080)))

_search_config = _config.get("search", {})
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", _search_config.get("chroma_path", "/app/chroma_data")))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", _search_config.get("model", "paraphrase-multilingual-MiniLM-L12-v2"))

# OAuth (empty ISSUER_URL = disabled, for backward compatibility)
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
    h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
]

# Filesystem watcher: re-index .md files in VAULT_PATH on external changes
# (git pull, manual edits, sync scripts). Set VAULT_WATCH=false to disable.
VAULT_WATCH = os.environ.get("VAULT_WATCH", "true").strip().lower() not in (
    "false", "0", "no", "off",
)

# ---------------------------------------------------------------------------
# MCP Server (with optional OAuth)
# ---------------------------------------------------------------------------

_mcp_kwargs: dict = dict(
    name="vault-mcp-server",
    instructions=(
        "MCP server for a personal Markdown knowledge vault. "
        "Provides tools to list, read, write, edit, search, and summarize vault documents. "
        "Use vault_search for semantic queries across the vault — it returns only the "
        "relevant chunks instead of full files, saving tokens. "
        "All file paths are relative to the vault root."
    ),
    host=SERVER_HOST,
    port=SERVER_PORT,
)

if ALLOWED_HOSTS:
    from mcp.server.transport_security import TransportSecuritySettings

    # Merge user-provided hosts with the localhost defaults FastMCP would
    # otherwise inject so we don't break local stdio/HTTP testing.
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

# Semantic search indexer
indexer = VaultIndexer(VAULT_PATH, CHROMA_PATH, EMBEDDING_MODEL, watch_files=VAULT_WATCH)


def _list_md_files() -> list[str]:
    """List all .md files in the vault, relative paths."""
    files: list[str] = []
    for p in sorted(VAULT_PATH.rglob("*.md")):
        rel = str(p.relative_to(VAULT_PATH))
        # Skip hidden dirs and _review/
        if any(part.startswith(".") or part.startswith("_") for part in Path(rel).parts):
            continue
        files.append(rel)
    return files


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_list(domain: str = "", status: str = "") -> str:
    """List vault files with optional filters.

    Args:
        domain: Filter by top-level directory (e.g. "homeserver", "career", "developments").
                 Empty string = all domains.
        status: Filter by normalized status (Active, Pending, Paused, Done).
                 Empty string = no status filter.

    Returns:
        Formatted list of files with metadata (title, status, domain).
    """
    files = _list_md_files()
    results: list[str] = []

    for rel_path in files:
        file_domain = infer_domain(rel_path)

        if domain and file_domain.lower() != domain.lower():
            continue

        # Quick metadata scan (only read first 15 lines for performance)
        full_path = VAULT_PATH / rel_path
        try:
            with open(full_path, encoding="utf-8") as f:
                lines = [next(f, "").rstrip("\n") for _ in range(15)]
        except (OSError, StopIteration):
            continue

        meta = parse_metadata(lines)

        if status and meta.status and meta.status.lower() != status.lower():
            continue
        if status and not meta.status:
            continue

        title = meta.title or Path(rel_path).stem
        status_str = f" [{meta.status}]" if meta.status else ""
        results.append(f"- {rel_path}  —  {title}{status_str}")

    if not results:
        return "No files found matching the filters."

    header = f"Vault files ({len(results)})"
    if domain:
        header += f" | domain={domain}"
    if status:
        header += f" | status={status}"
    return f"{header}\n\n" + "\n".join(results)


@mcp.tool()
def vault_read(file_path: str) -> str:
    """Read a vault file with parsed metadata, sections, and cross-references.

    Args:
        file_path: Path relative to vault root (e.g. "homeserver/homeserver-backlog.md").

    Returns:
        File content prefixed with parsed metadata summary.
    """
    full_path = VAULT_PATH / file_path
    if not full_path.exists():
        return f"Error: file not found: {file_path}"
    if not full_path.suffix == ".md":
        return f"Error: only .md files supported, got: {file_path}"
    if not str(full_path.resolve()).startswith(str(VAULT_PATH.resolve())):
        return "Error: path traversal not allowed."

    doc = parse_document(VAULT_PATH, file_path)

    # Build header
    header_parts = [f"# {doc.metadata.title or file_path}"]
    header_parts.append(f"**Path:** {file_path}")
    header_parts.append(f"**Domain:** {infer_domain(file_path)}")
    if doc.metadata.status:
        header_parts.append(f"**Status:** {doc.metadata.status}")
    if doc.metadata.created_date:
        header_parts.append(f"**Created:** {doc.metadata.created_date}")
    if doc.metadata.updated_date:
        header_parts.append(f"**Updated:** {doc.metadata.updated_date}")
    if doc.sections:
        header_parts.append(f"**Sections:** {', '.join(doc.sections[:15])}")
    if doc.cross_references:
        header_parts.append(f"**Cross-references:** {', '.join(doc.cross_references[:10])}")

    # Extra raw metadata not already shown
    shown_keys = {"status", "estado", "creado", "created", "última actualización",
                  "ultima actualización", "last updated", "updated"}
    extra = {k: v for k, v in doc.metadata.raw.items() if k.lower() not in shown_keys}
    if extra:
        header_parts.append(f"**Other metadata:** {extra}")

    header_parts.append("\n---\n")

    return "\n".join(header_parts) + doc.content


@mcp.tool()
def vault_write(file_path: str, content: str) -> str:
    """Create or overwrite a vault file.

    Args:
        file_path: Path relative to vault root (e.g. "developments/new-project.md").
        content: Full file content to write.

    Returns:
        Confirmation message.
    """
    full_path = VAULT_PATH / file_path
    if not str(full_path.resolve()).startswith(str(VAULT_PATH.resolve())):
        return "Error: path traversal not allowed."
    if not file_path.endswith(".md"):
        return "Error: only .md files supported."

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")

    if indexer.is_ready:
        asyncio.create_task(indexer.index_file(file_path))

    return f"Written: {file_path} ({len(content)} chars)"


@mcp.tool()
def vault_edit(file_path: str, old_text: str, new_text: str) -> str:
    """Edit a vault file via string replacement (like Claude's Edit tool).

    Args:
        file_path: Path relative to vault root.
        old_text: Exact text to find and replace. Must be unique in the file.
        new_text: Replacement text.

    Returns:
        Confirmation or error message.
    """
    full_path = VAULT_PATH / file_path
    if not full_path.exists():
        return f"Error: file not found: {file_path}"
    if not str(full_path.resolve()).startswith(str(VAULT_PATH.resolve())):
        return "Error: path traversal not allowed."

    content = full_path.read_text(encoding="utf-8")
    count = content.count(old_text)

    if count == 0:
        return "Error: old_text not found in file."
    if count > 1:
        return f"Error: old_text found {count} times — must be unique. Provide more context."

    new_content = content.replace(old_text, new_text, 1)
    full_path.write_text(new_content, encoding="utf-8")

    if indexer.is_ready:
        asyncio.create_task(indexer.index_file(file_path))

    return f"Edited: {file_path} (replaced 1 occurrence)"


@mcp.tool()
def vault_summary() -> str:
    """Dashboard summary of the vault — file counts by domain, active projects, recent files.

    Returns:
        Formatted summary with domain breakdown, status counts, and recently modified files.
    """
    files = _list_md_files()
    domain_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    active_projects: list[str] = []
    file_mtimes: list[tuple[float, str]] = []

    for rel_path in files:
        domain = infer_domain(rel_path)
        domain_counts[domain] += 1

        full_path = VAULT_PATH / rel_path
        try:
            mtime = full_path.stat().st_mtime
            file_mtimes.append((mtime, rel_path))
        except OSError:
            pass

        # Quick metadata scan
        try:
            with open(full_path, encoding="utf-8") as f:
                lines = [next(f, "").rstrip("\n") for _ in range(15)]
        except (OSError, StopIteration):
            continue

        meta = parse_metadata(lines)
        if meta.status:
            status_counts[meta.status] += 1
            if meta.status == "Active":
                title = meta.title or Path(rel_path).stem
                active_projects.append(f"  - {title} ({rel_path})")

    # Recent files (top 10 by mtime)
    file_mtimes.sort(reverse=True)
    recent = file_mtimes[:10]

    lines_out: list[str] = []
    lines_out.append(f"# Vault Summary\n")
    lines_out.append(f"**Total files:** {len(files)}\n")

    lines_out.append("## By Domain\n")
    for domain, count in domain_counts.most_common():
        lines_out.append(f"  - {domain}: {count}")

    lines_out.append("\n## By Status\n")
    for status, count in status_counts.most_common():
        lines_out.append(f"  - {status}: {count}")
    unstated = len(files) - sum(status_counts.values())
    if unstated:
        lines_out.append(f"  - (no status): {unstated}")

    if active_projects:
        lines_out.append(f"\n## Active Projects ({len(active_projects)})\n")
        lines_out.extend(active_projects[:20])

    if recent:
        lines_out.append(f"\n## Recently Modified\n")
        from datetime import datetime
        for mtime, path in recent:
            dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            lines_out.append(f"  - [{dt}] {path}")

    return "\n".join(lines_out)


@mcp.tool()
async def vault_search(query: str, top_k: int = 10) -> str:
    """Semantic search across the vault. Returns the most relevant chunks.

    Use this instead of vault_read when you don't know which file contains the
    information — it searches all documents and returns only the relevant sections,
    saving tokens compared to reading full files.

    Args:
        query: Natural language search query (works in English and Spanish).
        top_k: Number of results to return (default 10, max 20).

    Returns:
        Ranked list of relevant chunks with file path, section, and content.
    """
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    top_k = min(max(top_k, 1), 20)
    results = await indexer.search(query, top_k)

    if not results:
        return "No results found."

    parts = [f"Search results for: \"{query}\" ({len(results)} matches)\n"]
    for i, r in enumerate(results, 1):
        section_label = f" > {r['section']}" if r["section"] and r["section"] != "preamble" else ""
        parts.append(f"### {i}. {r['title']}{section_label}")
        parts.append(f"**File:** {r['path']}  |  **Score:** {r['score']:.3f}")
        parts.append(f"```markdown\n{r['content']}\n```")
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def vault_reindex() -> str:
    """Force a full re-index of the vault for semantic search.

    Use this after external changes to the vault (e.g. git pull, manual edits)
    that weren't made through vault_write/vault_edit.

    Returns:
        Index statistics (files processed, chunks created, time elapsed).
    """
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    stats = await indexer.index_all(force=True)
    return (
        f"Re-index complete.\n"
        f"Files processed: {stats['files']}\n"
        f"Chunks indexed: {stats['chunks']}\n"
        f"Time: {stats['elapsed']:.1f}s"
    )


# ---------------------------------------------------------------------------
# Middleware — inject Accept header if missing (Claude Code compatibility)
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
                (k.encode(), v.encode()) for k, v in request._headers.items()
            ]
        return await call_next(request)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server with Streamable HTTP transport."""
    # If OAuth is enabled, register approval page routes before building the app
    if oauth_provider:
        mcp._custom_starlette_routes.extend(create_approve_routes(oauth_provider))

    app = mcp.streamable_http_app()
    app.add_middleware(AcceptHeaderMiddleware)

    # If OAuth is enabled, add CF Access bypass (outermost middleware — runs first)
    if oauth_provider:
        app.add_middleware(CfAccessBypassMiddleware, bypass_token=oauth_provider.cf_bypass_token)

    # Wrap the existing lifespan to add background indexing
    original_lifespan = app.router.lifespan_context

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan_with_index(app_instance):
        async with original_lifespan(app_instance) as state:
            asyncio.create_task(indexer.start())
            yield state

    app.router.lifespan_context = lifespan_with_index

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    main()
