"""Vault MCP server for markdown memory with semantic search and structured writes."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import re
import tempfile
import threading
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NoReturn

import anyio
import uvicorn
import yaml
from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from . import gitstore
from .activity import format_activity, recent_activity
from .auth import CfAccessBypassMiddleware, SimpleOAuthProvider, create_approve_routes
from .catalog import build_inventory, cached_signature
from .indexer import VaultIndexer
from .linkgraph import (
    build_link_graph,
    cached_link_graph,
    format_dangling,
    format_links,
)
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

#: Run tool bodies in worker threads instead of on the event loop. See ``_tool``.
#:
#: DEFAULT OFF, and the default is a measurement, not an opinion. Every tool body
#: here is GIL-bound Python (the dominant cost of a read is a stat walk of ~1000
#: files), so threads contend rather than overlap. Measured on ct123, 6 clients,
#: reads only: inline 29.6 reads/s @ 201 ms p50, threaded 16.7 reads/s @ 354 ms
#: p50. On a mixed read/search/write load: inline 9.1 calls/s, threaded 4.2.
#: Threading does help ``vault_search`` (826 ms -> 335 ms p50, because torch and
#: chroma release the GIL and no longer queue behind read bodies) — so flip this
#: on for a search-dominated deployment, or on a free-threaded interpreter. The
#: mechanism is implemented and tested either way; only the default is off.
VAULT_THREADED = os.environ.get("VAULT_THREADED", "false").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
)
#: How many worker threads write tools may occupy at once. Writes serialize on
#: ``_write_lock`` regardless; this only stops a burst of blocked writers from
#: eating the shared thread pool that reads and searches also draw from.
VAULT_WRITE_SLOTS = max(1, int(os.environ.get("VAULT_WRITE_SLOTS", "4")))

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
    # The MCP endpoint is mounted at /mcp, so the PROTECTED RESOURCE is
    # <issuer>/mcp -- not the bare origin. This matters twice:
    #   1. the advertised "resource" must match the URL the client registered,
    #      or the connector rejects it as a mismatch; and
    #   2. the SDK derives the RFC 9728 path-aware metadata route from this
    #      value (routes.py: /.well-known/oauth-protected-resource + <path>),
    #      which is the location clients probe FIRST. With a pathless value it
    #      only ever served the root variant and /...-resource/mcp returned 404,
    #      so Claude/ChatGPT reported "Couldn't discover OAuth settings".
    _resource_url = OAUTH_ISSUER_URL.rstrip("/") + "/mcp"
    _mcp_kwargs["auth"] = AuthSettings(
        issuer_url=OAUTH_ISSUER_URL,
        resource_server_url=_resource_url,
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


# ---------------------------------------------------------------------------
# Failure choke point
# ---------------------------------------------------------------------------


class VaultToolError(ToolError):
    """A vault tool refused or failed. Raised only by :func:`_fail`."""


def _fail(message: str) -> NoReturn:
    """THE single failure exit for every tool.

    ``return "Error: ..."`` makes MCP report ``isError=false``: the protocol says
    the call succeeded and the failure is discoverable only by parsing prose. A
    6-client load run made exactly that mistake — 129 of 348 ``vault_edit`` calls
    came back "Error: old_text not found in file." and the harness counted every
    one of them as a win. A lost update that looks like a success is the worst
    shape a write API can have.

    ONE CHOKE POINT: every failure path in every tool raises through here, and
    ``tests/test_error_envelope.py`` fails the build if an eighth
    ``return "Error: ..."`` ever reappears anywhere under ``src/vault_mcp``.
    """
    text = message if message.startswith("Error:") else f"Error: {message}"
    logger.warning("vault tool failure: %s", text)
    raise VaultToolError(text)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

#: The server's event loop, captured so worker threads can hand coroutines back
#: to it (see :func:`_queue_reindex`). ``None`` outside a running server.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None

#: Every write tool serializes on this. Before threading, the event loop gave that
#: for free: one tool body ran to completion before the next one started, so a
#: read-modify-write (vault_edit, vault_upsert_section, vault_append_digest, …)
#: could never interleave with another writer. Running bodies in threads removes
#: that guarantee, and two threads upserting different sections of one file would
#: lose an update. Writes therefore keep exactly the serialization they already
#: had — reads and searches are what this change parallelizes.
_write_lock = threading.RLock()

_write_limiter: Any = None


def _get_write_limiter() -> Any:
    """Lazily built: anyio limiters must be constructed on a running event loop."""
    global _write_limiter
    if _write_limiter is None:
        _write_limiter = anyio.CapacityLimiter(VAULT_WRITE_SLOTS)
    return _write_limiter


def _serialized(fn: Callable) -> Callable:
    """Wrap a write tool so *every* caller — MCP or direct — holds the write lock."""

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        with _write_lock:
            return fn(*args, **kwargs)

    return guarded


def _tool(writes: bool = False) -> Callable[[Callable], Callable]:
    """Register a vault tool. THE single dispatch path — do not add a second one.

    ``mcp.server.fastmcp`` awaits an async tool and *calls a sync tool inline*
    (``FuncMetadata.call_fn_with_arg_validation`` ends in
    ``return await fn(...)`` / ``return fn(...)``; there is no ``to_thread``
    anywhere in ``tools/base.py`` or ``func_metadata.py``). Every tool here used
    to be a plain ``def``, so every call — a 4 ms search, a 300 ms read, a 200 ms
    git commit alike — blocked the whole event loop for its full duration and six
    clients serialized single-file: search 4 ms solo → 908 ms p50 under load.

    So the object handed to FastMCP is an async wrapper that runs the body in a
    worker thread, while the name this module exports stays the plain sync
    function (tests and any in-process caller keep calling it normally). Both go
    through the *same* wrapped callable, so a write cannot acquire the lock on
    one path and skip it on the other.

    ``functools.wraps`` is load-bearing: ``inspect.signature`` follows
    ``__wrapped__``, so FastMCP still derives the real parameter schema from the
    original signature. ``tests/test_tool_schema.py`` asserts that, because a
    silently-mangled schema is how a caller ends up passing ``section`` instead
    of ``section_header`` 314 times without noticing.
    """

    def decorator(fn: Callable) -> Callable:
        sync_fn = _serialized(fn) if writes else fn
        registered = sync_fn

        if VAULT_THREADED and not inspect.iscoroutinefunction(fn):

            @functools.wraps(sync_fn)
            async def dispatch(*args, **kwargs):
                global _MAIN_LOOP
                _MAIN_LOOP = asyncio.get_running_loop()
                call = functools.partial(sync_fn, *args, **kwargs)
                if writes:
                    return await to_thread.run_sync(call, limiter=_get_write_limiter())
                return await to_thread.run_sync(call)

            registered = dispatch

        mcp.tool()(registered)
        return sync_fn

    return decorator


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


def _resolve_markdown_path(file_path: str, allow_create: bool = False) -> Path:
    """Validate a relative markdown path or compatibility alias against the vault root.

    Raises through :func:`_fail` rather than returning an ``(path, error)`` pair:
    the old pair forced every tool to re-implement ``if error: return error``,
    which is precisely how seven "Error: …" strings ended up leaving this module
    inside success envelopes.
    """
    if not file_path.endswith(".md"):
        _fail("only .md files supported.")

    normalized = file_path.strip().lstrip("./")
    inventory = build_inventory(VAULT_PATH)
    rel_path = inventory.resolve(normalized)
    if rel_path is None:
        if allow_create:
            rel_path = _default_create_path(normalized)
        else:
            _fail(f"file not found: {file_path}")

    # Logical path: do NOT collapse intended symlinked subtrees (e.g.
    # shared -> /root/claude-memory/shared). Downstream callers rely on
    # full_path.relative_to(VAULT_PATH) and parse_document(VAULT_PATH, rel_path),
    # so full_path must stay under VAULT_PATH. The OS transparently follows the
    # symlink when the file is actually read.
    full_path = VAULT_PATH / rel_path
    # Traversal guard: validate the *resolved* path does not escape the vault
    # root nor the real targets of the symlinked subtrees on this path.
    # rel_path already came from the trusted inventory (os.walk followlinks=True
    # from VAULT_PATH), so membership is the security boundary; this only blocks
    # genuine ../ escape while permitting intended symlinked corpora.
    #
    # 2026-07-30: the roots used to come from a single ``VAULT_PATH.iterdir()``
    # pass, so only symlinks DIRECTLY under the vault root were trusted. The live
    # curated tree symlinks at depth two (bob/memory, bob/*-auto-memory, air/notes
    # …), and every one of those files was rejected as "path traversal" — 9 of the
    # vault's 10 subtrees were unreadable through vault_read. It went unnoticed for
    # months because the refusal was returned as an ordinary string and MCP
    # reported isError=false: the caller saw a successful call whose body happened
    # to say Error. Walking the actual path components instead is both correct for
    # nesting and *stricter*, since it trusts only the symlinks a request really
    # traverses rather than every symlink at the root.
    resolved = full_path.resolve()
    allowed_roots = [VAULT_PATH.resolve()]
    probe = VAULT_PATH
    for part in Path(rel_path).parts[:-1]:
        probe = probe / part
        try:
            if probe.is_symlink():
                allowed_roots.append(probe.resolve())
        except OSError:
            break
    if not any(
        resolved == root or str(resolved).startswith(str(root) + "/")
        for root in allowed_roots
    ):
        _fail("path traversal not allowed.")
    return full_path


def _reindex_finished(future):
    """Post-write reindex is fire-and-forget, but a failure must not be silent."""
    try:
        error = future.exception()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - cancelled future
        return
    if error is not None:
        logger.warning("post-write reindex failed: %s", error)


def _queue_reindex(file_path: str):
    """Schedule a single-file reindex if the index is ready.

    THREAD SAFETY. Tool bodies now run in worker threads, and
    ``asyncio.create_task`` from a foreign thread raises ``RuntimeError`` — the
    old code caught that and returned, which under threading would have quietly
    disabled post-write reindexing for *every* write and left semantic search
    stale until the filesystem watcher happened to notice. ``run_coroutine_
    threadsafe`` is the one asyncio entry point that is safe to call from another
    thread, so the coroutine is handed to the loop that owns the indexer.
    """
    if not indexer.is_ready:
        return

    coro = indexer.index_file(file_path)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:  # already on the server loop (VAULT_THREADED=false, or the watcher)
        task = asyncio.ensure_future(coro)
        task.add_done_callback(_reindex_finished)
        return

    loop = _MAIN_LOOP
    if loop is None or loop.is_closed():
        # No server loop at all (tests, one-shot CLI use). The filesystem watcher
        # is the backstop; never crash the write over an index refresh.
        coro.close()
        return
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        coro.close()
        return
    future.add_done_callback(_reindex_finished)


def _atomic_write(full_path: Path, content: str) -> Path:
    """Write ``content`` durably: temp file in the same directory, then rename.

    A half-written markdown file is worse than no write at all -- readers,
    the indexer and git would all see a truncated document. ``os.replace`` is
    atomic within a filesystem, so a reader sees either the old or the new file.
    Symlinked *files* are followed (write through to the target) so this keeps
    the previous ``write_text`` semantics for the curated vault's symlinks.
    """
    target = Path(os.path.realpath(full_path)) if full_path.is_symlink() else full_path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".vault-tmp-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            os.chmod(tmp_name, target.stat().st_mode & 0o7777)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _persist(full_path: Path, content: str, tool: str) -> tuple[str, gitstore.CommitResult]:
    """THE single write path. Every vault write tool goes through this function.

    ONE CHOKE POINT DOCTRINE: do not add ``write_text`` / ``open(..., "w")`` to a
    tool body. The worst bugs in this codebase have all been a fix applied to one
    write path and missed on a second one (see gitstore module docstring). If a
    write needs new behaviour, it goes here and every tool inherits it.

    Order is load-bearing:
      1. disk   -- atomically, so the user's content survives anything below;
      2. index  -- best effort, the fs watcher is the backstop;
      3. git    -- last, and its failure is reported, never raised.
    """
    _atomic_write(full_path, content)
    rel_path = str(full_path.relative_to(VAULT_PATH))
    _queue_reindex(rel_path)
    result = gitstore.commit_write(full_path, tool, vault_rel_path=rel_path)
    if not result.ok:
        logger.error(
            "WRITE SAVED BUT NOT COMMITTED: %s (%s) — %s", rel_path, tool, result.detail
        )
    return rel_path, result


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


#: Both link directions truncate at the same number, so a header stays a header.
_LINK_DISPLAY_LIMIT = 10

#: Backlinks in vault_read cost one cached link-graph lookup. Env kill-switch
#: because vault_read is the hottest tool here and a nicety must never be the
#: reason a read gets slow (measured: ~0.1 ms added to a ~3 ms read, warm).
VAULT_READ_BACKLINKS = os.environ.get("VAULT_READ_BACKLINKS", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


_graph_refresh_lock = threading.Lock()
_graph_refreshing = False


def _schedule_link_graph_refresh():
    """Rebuild the link graph off the read path, one refresh at a time.

    Without this the graph would only warm up when somebody happened to call
    ``vault_links``, so backlinks would silently vanish after every write and
    stay gone. With it they come back within a few seconds, and no reader ever
    waits for the parse.
    """
    global _graph_refreshing
    with _graph_refresh_lock:
        if _graph_refreshing:
            return
        _graph_refreshing = True

    def refresh():
        global _graph_refreshing
        try:
            build_link_graph(VAULT_PATH)
        except Exception:
            logger.warning("link graph refresh failed", exc_info=True)
        finally:
            with _graph_refresh_lock:
                _graph_refreshing = False

    threading.Thread(target=refresh, name="linkgraph-refresh", daemon=True).start()


def _backlinks_for(rel_path: str) -> list[str]:
    """Files that link AT ``rel_path`` — "what points at this?".

    A vault file has always advertised what it links TO (``Cross-references``);
    the other direction existed only in ``vault_links``, so an agent reading a
    document saw half of every connection. This surfaces the other half at read
    time, with no file mutation: the alternative on the table was rewriting 1011
    files to insert reciprocal links.

    Goes through the link graph — the SAME extractor and the SAME alias
    resolution ``vault_links``/``vault_dangling_links`` use. Emphatically not a
    second link parser, and no fuzzy matching: the corpus has a real
    underscore-vs-hyphen naming split (~8 of 26 dangling links), and inventing
    matches across it would manufacture connections that do not exist, which is
    worse than reporting none.

    ``cached_link_graph``, never ``build_link_graph``: vault_read is the hottest
    tool here and a full graph build parses every file in the vault (measured
    4.3 s on the 894-file corpus). The cache is invalidated by any write, so
    building inline would mean the first read after every write pays seconds. On
    a miss this returns nothing and warms the graph in the background instead —
    a backlink line that is occasionally absent is a fair price; a read that
    occasionally takes four seconds is not.

    It also reuses the corpus fingerprint ``_resolve_markdown_path`` just took
    via ``build_inventory``. Taking a fresh one is a stat walk of every file
    (~83 ms, i.e. most of a vault_read), and measuring that mistake is what
    turned a +26 ms regression into +0 ms.

    Never raises. A read that fails because an ornament broke is the same class
    of trade this codebase has already been burned by.
    """
    if not VAULT_READ_BACKLINKS:
        return []
    try:
        signature = cached_signature(VAULT_PATH)
        if signature is None:  # inventory not warm either — do no work here
            _schedule_link_graph_refresh()
            return []
        graph = cached_link_graph(VAULT_PATH, signature=signature)
        if graph is None:
            _schedule_link_graph_refresh()
            return []
        return sorted({link.source for link in graph.backlinks_for(rel_path)})
    except Exception:
        logger.warning("backlinks unavailable for %s", rel_path, exc_info=True)
        return []


def _document_header(doc, backlinks: list[str] | None = None) -> list[str]:
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
        header_parts.append(
            f"**Cross-references:** "
            f"{', '.join(doc.cross_references[:_LINK_DISPLAY_LIMIT])}"
        )
    if backlinks:
        shown = backlinks[:_LINK_DISPLAY_LIMIT]
        line = ", ".join(shown)
        if len(backlinks) > len(shown):
            # Say so rather than cutting silently — a truncated list that looks
            # complete is how "nothing else references this" gets believed.
            line += f" ({len(backlinks)} total, showing {len(shown)})"
        header_parts.append(f"**Referenced by:** {line}")
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


@_tool()
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


@_tool()
def vault_read(file_path: str) -> str:
    """Read a vault file with metadata, sections, cross-references and backlinks."""
    full_path = _resolve_markdown_path(file_path)
    if not full_path.exists():
        _fail(f"file not found: {file_path}")

    rel_path = str(full_path.relative_to(VAULT_PATH))
    doc = parse_document(VAULT_PATH, rel_path)
    header = _document_header(doc, backlinks=_backlinks_for(rel_path))
    return "\n".join(header) + doc.content


@_tool(writes=True)
def vault_write(file_path: str, content: str) -> str:
    """Create or overwrite a vault file."""
    full_path = _resolve_markdown_path(file_path, allow_create=True)

    rel_path, git = _persist(full_path, content, "vault_write")
    return f"Written: {rel_path} ({len(content)} chars){git.suffix()}"


@_tool(writes=True)
def vault_edit(file_path: str, old_text: str, new_text: str) -> str:
    """Edit a vault file via exact string replacement."""
    full_path = _resolve_markdown_path(file_path)
    if not full_path.exists():
        _fail(f"file not found: {file_path}")

    content = full_path.read_text(encoding="utf-8")
    count = content.count(old_text)
    if count == 0:
        _fail("old_text not found in file.")
    if count > 1:
        _fail(f"old_text found {count} times — must be unique. Provide more context.")

    rel_path, git = _persist(full_path, content.replace(old_text, new_text, 1), "vault_edit")
    return f"Edited: {rel_path} (replaced 1 occurrence){git.suffix()}"


@_tool(writes=True)
def vault_append(file_path: str, content: str) -> str:
    """Append content to the end of a markdown file, creating it if needed."""
    full_path = _resolve_markdown_path(file_path, allow_create=True)

    existing = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if existing and not existing.endswith("\n\n"):
        existing += "\n"
    rel_path, git = _persist(full_path, existing + content.strip("\n") + "\n", "vault_append")
    return f"Appended: {rel_path}{git.suffix()}"


@_tool(writes=True)
def vault_set_frontmatter(file_path: str, key: str, value: str) -> str:
    """Set or clear a YAML frontmatter field on a vault document."""
    full_path = _resolve_markdown_path(file_path, allow_create=True)

    frontmatter, body = _load_frontmatter_and_body(full_path)
    _set_frontmatter_value(frontmatter, key, value)
    _upsert_updated(frontmatter)
    rel_path, git = _persist(
        full_path, _render_markdown(frontmatter, body), "vault_set_frontmatter"
    )
    return f"Updated frontmatter: {rel_path} ({key}){git.suffix()}"


@_tool(writes=True)
def vault_upsert_section(file_path: str, section_header: str, content: str, level: int = 2) -> str:
    """Insert or replace a markdown section by heading text."""
    if level < 1 or level > 4:
        _fail("level must be between 1 and 4.")

    full_path = _resolve_markdown_path(file_path, allow_create=True)

    frontmatter, body = _load_frontmatter_and_body(full_path)
    _upsert_updated(frontmatter)
    new_body = _upsert_markdown_section(body, section_header, content, level)
    rel_path, git = _persist(
        full_path, _render_markdown(frontmatter, new_body), "vault_upsert_section"
    )
    return f"Upserted section: {rel_path} -> {section_header}{git.suffix()}"


@_tool(writes=True)
def vault_create_handoff(file_path: str, title: str, body: str, tour: str = "", status: str = "Active") -> str:
    """Create a standardized handoff document."""
    full_path = _resolve_markdown_path(file_path, allow_create=True)
    if full_path.exists():
        _fail(f"file already exists: {file_path}")

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
    rel_path, git = _persist(
        full_path, _render_markdown(frontmatter, body_text), "vault_create_handoff"
    )
    return f"Created handoff: {rel_path}{git.suffix()}"


@_tool(writes=True)
def vault_append_digest(file_path: str, heading: str, content: str) -> str:
    """Append a digest entry as a new H2 section."""
    full_path = _resolve_markdown_path(file_path, allow_create=True)

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
    rel_path, git = _persist(
        full_path, _render_markdown(frontmatter, new_body), "vault_append_digest"
    )
    return f"Appended digest: {rel_path}{git.suffix()}"


@_tool()
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

    # Search index health. A vault whose files are all present but whose index
    # is empty looks perfect in every section above and answers nothing.
    lines_out.append("\n## Search Index\n")
    rebuild = indexer.rebuild_status()
    lines_out.append(f"  - ready: {indexer.is_ready}")
    lines_out.append(f"  - chunks indexed: {indexer.chunk_count()}")
    lines_out.append(f"  - rebuild state: {rebuild['state']}")
    if rebuild["state"] == "running":
        lines_out.append(
            f"  - rebuild progress: {rebuild.get('indexed', 0)}/"
            f"{rebuild.get('files') or '?'} files (previous index still serving)"
        )
    elif rebuild["state"] == "failed":
        lines_out.append(f"  - rebuild error: {rebuild.get('error')}")

    return "\n".join(lines_out)


@_tool()
def vault_recent_activity(hours: int = 24, limit: int = 20, path_prefix: str = "") -> str:
    """What changed in the vault recently — for an agent resuming cold.

    Reads git history (author, commit, and the writing tool when the write came
    through this server), plus uncommitted working-tree changes and mtimes for
    vault subtrees that git does not cover.

    Args:
        hours: size of the look-back window in hours (default 24).
        limit: maximum number of files to report (default 20).
        path_prefix: optional vault path prefix filter, e.g. "shared/".
    """
    hours = min(max(int(hours), 1), 24 * 90)
    limit = min(max(int(limit), 1), 200)
    entries = recent_activity(VAULT_PATH, hours=hours, limit=limit, path_prefix=path_prefix)
    return format_activity(entries, hours=hours, limit=limit)


@_tool()
def vault_links(file_path: str, direction: str = "both", limit: int = 50) -> str:
    """Backlinks and forward links for one vault file.

    Args:
        file_path: vault path or alias, e.g. "shared/glossary.md" or "glossary.md".
        direction: "both" (default), "forward", or "backlinks".
        limit: maximum links listed per section.
    """
    normalized = file_path.strip().lstrip("./")
    if not normalized.endswith(".md"):
        normalized += ".md"
    inventory = build_inventory(VAULT_PATH)
    canonical = inventory.resolve(normalized)
    if canonical is None:
        _fail(
            f"file not found: {file_path}\n"
            "(vault_links needs an existing file; use vault_dangling_links to see "
            "links pointing at files that do not exist.)"
        )
    graph = build_link_graph(VAULT_PATH)
    return format_links(graph, file_path, canonical, direction=direction, limit=min(max(limit, 1), 500))


@_tool()
def vault_dangling_links(kind: str = "wiki", limit: int = 50, source_prefix: str = "") -> str:
    """Links whose target file does not exist — stale renames or unwritten notes.

    Args:
        kind: "wiki" (default), "markdown", or "all".
        limit: maximum missing targets listed.
        source_prefix: only report links written in files under this vault prefix.
    """
    graph = build_link_graph(VAULT_PATH)
    return format_dangling(
        graph, kind=kind, limit=min(max(limit, 1), 500), source_prefix=source_prefix
    )


def _is_thin_chunk(text: str) -> bool:
    """True when a chunk is little more than its heading line(s)."""
    body = " ".join(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return len(body) < 40


@_tool()
async def vault_search(query: str, top_k: int = 10) -> str:
    """Semantic search across the vault.

    Returns up to top_k DIFFERENT files, best-matching section of each, with
    its text. That text is often enough to answer; when a file has further
    matching sections they are named, and vault_read gives the whole file.
    """
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    top_k = min(max(top_k, 1), 20)
    # One hit per file (2026-10-13). Chunk-level top_k let a single long file
    # fill every slot (5/5 hits from one transcript), so callers saw one source
    # and fell back to reading files. Over-fetch, keep each file's best chunk
    # (Chroma returns best-first), and name the file's other matching sections.
    # A heading-only chunk ("# 3. On nyx, restore from backup") scores high
    # because it is short, but carries no answer; it may rank the file, never
    # stand in for it while the same file has a chunk with real text.
    candidates = await indexer.search(query, min(top_k * 5, 100))
    by_path: dict[str, list[dict]] = {}
    for cand in candidates:
        by_path.setdefault(cand["path"], []).append(cand)
    results: list[dict] = []
    also: dict[str, list[str]] = {}
    for path, chunks in list(by_path.items())[:top_k]:
        pick = next((c for c in chunks if not _is_thin_chunk(c["content"])), chunks[0])
        results.append({**pick, "score": chunks[0]["score"]})
        also[path] = [c["section"] or "preamble" for c in chunks if c is not pick]
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
        more = list(dict.fromkeys(also[result["path"]]))
        if more:
            shown = ", ".join(more[:4]) + (f" (+{len(more) - 4} more)" if len(more) > 4 else "")
            parts.append(f"_Other matching sections in this file: {shown}_")
        parts.append("")

    return "\n".join(parts)


def _format_index_status(status: dict) -> str:
    """Render the forced-rebuild state machine as text a caller can act on."""
    state = status.get("state", "idle")
    lines = [f"Index rebuild state: {state}"]
    if status.get("handle"):
        lines.append(f"Handle: {status['handle']}")
    if state == "running":
        if status.get("phase"):
            lines.append(f"Phase: {status['phase']}")
        if status.get("files"):
            lines.append(
                f"Progress: {status.get('indexed', 0)}/{status['files']} files"
            )
        lines.append(f"Started: {status.get('started_at')}")
        lines.append("The previous index is still serving searches.")
    elif state == "succeeded":
        lines.append(
            f"Files: {status.get('files')}  Chunks: {status.get('chunks')}  "
            f"Time: {status.get('elapsed', 0.0):.1f}s"
        )
        lines.append(f"Finished: {status.get('finished_at')}")
    elif state == "failed":
        lines.append(f"Error: {status.get('error')}")
        lines.append(f"Finished: {status.get('finished_at')}")
        lines.append("The previous index was kept and is still serving searches.")
    else:
        lines.append("No rebuild has run since this server started.")
    lines.append(f"Live chunks: {indexer.chunk_count()}")
    return "\n".join(lines)


@_tool()
async def vault_reindex(force: bool = False) -> str:
    """Re-index the vault for semantic search. Incremental by DEFAULT.

    Args:
        force: leave False (the default) for the incremental pass — it re-embeds
            only files whose mtime changed, drops chunks for files that no longer
            exist, and returns in seconds. That is what dropping or editing a
            handful of files needs.
            Set True only to rebuild every file from scratch (changed embedding
            model, suspected index corruption). The full rebuild takes minutes,
            so it runs in the BACKGROUND: this returns a handle immediately and
            you poll `vault_index_status`. Searches keep hitting the previous
            index the whole time, and it is replaced only once the new one is
            built and verified non-empty.
    """
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."

    if not force:
        stats = await indexer.index_all(force=False)
        return (
            "Re-index complete (mode: INCREMENTAL).\n"
            f"Files re-embedded: {stats['files']}\n"
            f"Files skipped (unchanged): {stats['skipped']}\n"
            f"Files pruned (deleted from vault): {stats['deleted']}\n"
            f"Chunks now in index: {stats['chunks']}\n"
            f"Time: {stats['elapsed']:.1f}s\n"
            "Pass force=true only for a full rebuild from scratch."
        )

    status = await indexer.start_rebuild()
    if status.get("already_running"):
        return (
            "A full rebuild is ALREADY RUNNING (mode: FULL REBUILD).\n"
            + _format_index_status(status)
        )
    return (
        "Full rebuild STARTED in the background (mode: FULL REBUILD).\n"
        f"Handle: {status['handle']}\n"
        "This returns immediately: a full rebuild takes minutes and would blow "
        "past the MCP idle timeout.\n"
        "Poll `vault_index_status` for running / succeeded / failed.\n"
        "Searches keep using the existing index until the new one is complete "
        "and verified; a failed or interrupted rebuild changes nothing."
    )


@_tool()
def vault_index_status() -> str:
    """Progress and outcome of the last/current full index rebuild.

    Tells running from succeeded from failed, so "no response" is never the
    signal. Also reports how many chunks the live index currently holds.
    """
    if not indexer.is_ready:
        return "Indexing in progress. Please try again in a few seconds."
    return _format_index_status(indexer.rebuild_status())


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

    # --- OAuth discovery compatibility shims -------------------------------
    # Clients differ on WHERE they look for RFC 9728 protected-resource
    # metadata, and getting it wrong is indistinguishable from "server broken":
    #   * the SDK registers ONLY the path-aware route for a resource with a
    #     path, so /.well-known/oauth-protected-resource (root) 404s;
    #   * our issuer normalises to a TRAILING SLASH, so a client that
    #     concatenates authorization_servers[0] + "/.well-known/..." requests
    #     //.well-known/... and gets a 404.
    # Serve every spelling. Cheap, and removes a whole class of silent failure.
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Route as _Route

    def _resource_metadata(request):
        return _JSONResponse({
            "resource": _resource_url,
            "authorization_servers": [OAUTH_ISSUER_URL.rstrip("/")],
            "bearer_methods_supported": ["header"],
        })

    for _p in ("/.well-known/oauth-protected-resource",
               "//.well-known/oauth-protected-resource",
               "//.well-known/oauth-protected-resource/mcp"):
        mcp._custom_starlette_routes.append(
            _Route(_p, endpoint=_resource_metadata, methods=["GET"])
        )

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
        global _MAIN_LOOP
        # Worker threads hand post-write reindex coroutines back to this loop.
        _MAIN_LOOP = asyncio.get_running_loop()
        logger.info(
            "tool dispatch: %s (write slots=%d)",
            "threaded" if VAULT_THREADED else "inline (event loop)",
            VAULT_WRITE_SLOTS,
        )
        async with original_lifespan(app_instance) as state:
            asyncio.create_task(indexer.start())
            try:
                yield state
            finally:
                # Commits are debounced; a clean stop must not leave a write on
                # disk but out of git. gitstore also registers an atexit hook, so
                # an unclean exit still gets one attempt.
                pending = gitstore.pending_summary()
                if pending:
                    logger.info("flushing pending vault commits at shutdown: %s", pending)
                await to_thread.run_sync(gitstore.flush_all)

    app.router.lifespan_context = lifespan_with_index
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    main()
