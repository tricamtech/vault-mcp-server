# vault-mcp-server

Give Claude Code persistent memory with a Markdown vault.

An MCP server that turns any folder of Markdown files into a searchable, structured knowledge base. Claude Code can list, read, write, edit, search, and summarize your vault — across sessions, across machines.

## What it does

- **7 tools** exposed via MCP: `vault_list`, `vault_read`, `vault_write`, `vault_edit`, `vault_summary`, `vault_search`, `vault_reindex`
- **Semantic search** powered by ChromaDB + sentence-transformers — ask questions in natural language, get relevant chunks back instead of full files
- **Metadata parsing** — extracts status, dates, and custom fields from `**Key:** Value` patterns in your Markdown
- **Multilingual** — search and metadata work in English and Spanish out of the box
- **Incremental indexing** — only re-embeds files that changed since last startup

## Why Markdown

Markdown is the only format that is simultaneously human-readable, version-controllable, and LLM-friendly. No database to manage, no migration scripts. Your vault is just files — back them up with git, sync them with Nextcloud, edit them in Obsidian.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│  Claude Code         │────>│  vault-mcp-server     │
│  (any machine)       │<────│  (FastMCP + HTTP)     │
└─────────────────────┘     └──────┬───────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Markdown Vault  │
                          │  (any folder)    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  ChromaDB        │
                          │  (semantic index)│
                          └─────────────────┘
```

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/thebackpackdevorg/vault-mcp-server.git
cd vault-mcp-server
```

Edit `docker-compose.yml` to point to your Markdown folder:

```yaml
volumes:
  - /path/to/your/markdown-vault:/vault
```

### 2. Run

```bash
docker compose up -d
```

The server starts on port **8091** with Streamable HTTP transport.

### 3. Register in Claude Code

```bash
claude mcp add vault http://localhost:8091/mcp -t http
```

Or add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "vault": {
      "type": "http",
      "url": "http://localhost:8091/mcp"
    }
  }
}
```

### 4. Use it

```
You: "What do I have documented about Docker networking?"

Claude: *calls vault_search("Docker networking")* → returns relevant chunks

You: "Update the status in homeserver-backlog.md to Done"

Claude: *calls vault_edit("homeserver/homeserver-backlog.md", "**Status:** Active", "**Status:** Done")*
```

## Remote deployment

To access the vault from other machines, expose it through a reverse proxy. The server supports an optional OAuth layer for authentication when deployed remotely.

### With Cloudflare Tunnel + Access

1. Create a Cloudflare Tunnel pointing to `localhost:8091`
2. Add a Cloudflare Access policy with a Service Token
3. Configure Claude Code on remote machines:

```json
{
  "mcpServers": {
    "vault": {
      "type": "http",
      "url": "https://vault.yourdomain.com/mcp",
      "headers": {
        "CF-Access-Client-Id": "${VAULT_CF_CLIENT_ID}",
        "CF-Access-Client-Secret": "${VAULT_CF_CLIENT_SECRET}"
      }
    }
  }
}
```

### With built-in OAuth

Set `OAUTH_ISSUER_URL` and optionally `OAUTH_PIN` in your environment to enable the built-in OAuth 2.1 provider with a PIN-based approval page.

## The SSE vs Streamable HTTP gotcha

If you're deploying behind Cloudflare (or any proxy that buffers SSE), you may hit a silent failure where Claude Code connects but never gets responses.

**The fix:** This server uses Streamable HTTP (not SSE). Make sure your proxy passes through the `Accept: application/json, text/event-stream` header correctly. The server includes a middleware that injects this header if missing.

## Tools reference

| Tool | Description |
|------|-------------|
| `vault_list` | List files with optional domain/status filters |
| `vault_read` | Read a file with parsed metadata, sections, and cross-references |
| `vault_write` | Create or overwrite a file |
| `vault_edit` | String replacement edit (like Claude's Edit tool) |
| `vault_summary` | Dashboard: file counts by domain, active projects, recent changes |
| `vault_search` | Semantic search — returns relevant chunks, not full files |
| `vault_reindex` | Force full re-index after external changes |

## Configuration

### config.yaml

```yaml
vault:
  path: "/vault"

server:
  host: "0.0.0.0"
  port: 8080

search:
  chroma_path: "/app/chroma_data"
  model: "paraphrase-multilingual-MiniLM-L12-v2"
```

### Environment variables (override config.yaml)

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_PATH` | `/vault` | Path to your Markdown vault inside the container |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8080` | Internal port |
| `CHROMA_PATH` | `/app/chroma_data` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model |
| `OAUTH_ISSUER_URL` | *(empty)* | Set to enable OAuth (e.g. `https://vault.yourdomain.com`) |
| `OAUTH_PIN` | *(empty)* | Optional PIN for the OAuth approval page |

### Git-backed writes (optional)

If your vault is a git repo, the server can commit every `vault_write` and `vault_edit`, so a write made by an MCP client always has history and an undo, and a later `git checkout`/`reset` can't silently discard it. It is off by default.

- The file is always written to disk first; a failed commit never loses the write. The tool's reply ends with `[git <sha>]`, `[git: queued]`, or `[git: NOT COMMITTED: <reason>]`.
- Commits are debounced: a burst of writes becomes one commit once writes stop for `VAULT_GIT_DEBOUNCE_SECONDS`. Pending commits are flushed on shutdown.
- Only the written files are committed, even if other changes are staged. The server never pushes.
- Files outside any git repo, and gitignored files, are written but not versioned. Repos are found from each file's real path, so subdirectories symlinked into other repos commit into those repos.
- Commits to one repo are serialized with a `flock` on Linux and macOS, which also covers other processes (for example a second server instance on the same vault). On Windows there is no `flock`, so commits are serialized only within the server process, and a warning is logged once. git's own `index.lock` still prevents index corruption there; a commit that collides is retried.
- `git` must be installed where the server runs (the provided Docker image does not include it). If the vault is owned by a different user than the server, git may also need `safe.directory` configured for it.

| Variable | Default | Description |
|----------|---------|-------------|
| `VAULT_GIT_COMMIT` | `false` | Set to `true` to commit vault writes to git |
| `VAULT_GIT_DEBOUNCE_SECONDS` | `30` | Quiet period before a batch of writes is committed; `0` commits every write immediately |
| `VAULT_GIT_DEBOUNCE_MAX_SECONDS` | `300` | Longest a batch may wait while writes keep arriving; `0` = no limit |
| `VAULT_GIT_NAME` / `VAULT_GIT_EMAIL` | `vault-mcp` / `vault-mcp@localhost` | Author and committer of the server's commits |
| `VAULT_GIT_TIMEOUT` | `30` | Timeout, in seconds, for each git command |
| `VAULT_GIT_LOCK_TIMEOUT` | `30` | How long a commit waits for another commit to the same repo to finish |
| `VAULT_GIT_INDEX_RETRIES` | `5` | Attempts when git reports its `index.lock` is held by another process |

## Metadata conventions

The server parses `**Key:** Value` patterns in the first 15 lines of each file:

```markdown
# My Project

**Status:** Active
**Created:** 2026-01-15
**Last Updated:** 2026-03-01
```

Recognized status values: `Active`, `Pending`, `Paused`, `Done` (and Spanish equivalents).

## Stack

- **Python 3.12** + FastMCP
- **ChromaDB** for vector storage
- **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings
- **Docker** + `uv` for fast builds

## License

MIT
