# vault-mcp-server

Give Claude Code persistent memory with a Markdown vault.

An MCP server that turns any folder of Markdown files into a searchable, structured knowledge base. Claude Code can list, read, write, edit, search, and summarize your vault — across sessions, across machines.

## What it does

- **9 tools** exposed via MCP: `vault_list`, `vault_read`, `vault_write`, `vault_edit`, `vault_summary`, `vault_search`, `vault_reindex`, `vault_links`, `vault_dangling_links`
- **Semantic search** powered by ChromaDB + sentence-transformers — ask questions in natural language, get relevant chunks back instead of full files
- **Metadata parsing** — extracts status, dates, and custom fields from `**Key:** Value` patterns in your Markdown
- **Multilingual** — search and metadata work in English and Spanish out of the box
- **Incremental indexing** — only re-embeds files that changed since last startup
- **Link graph** — Obsidian-style `[[wiki-links]]` and markdown links become a queryable graph: backlinks, forward links, and links to notes that don't exist

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
| `vault_read` | Read a file with parsed metadata, sections, cross-references, and backlinks |
| `vault_write` | Create or overwrite a file |
| `vault_edit` | String replacement edit (like Claude's Edit tool) |
| `vault_summary` | Dashboard: file counts by domain, active projects, recent changes |
| `vault_search` | Semantic search — returns relevant chunks, not full files |
| `vault_reindex` | Force full re-index after external changes |
| `vault_links` | Forward links and backlinks for one file (`direction`: `both`, `forward`, `backlinks`) |
| `vault_dangling_links` | Links whose target file doesn't exist, grouped by missing target (`kind`: `wiki`, `markdown`, `all`; optional `source_prefix`) |

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
| `VAULT_READ_BACKLINKS` | `true` | Show a **Referenced by:** line in `vault_read`. Set to `false`/`0`/`no`/`off` to disable |

## Metadata conventions

The server parses `**Key:** Value` patterns in the first 15 lines of each file:

```markdown
# My Project

**Status:** Active
**Created:** 2026-01-15
**Last Updated:** 2026-03-01
```

Recognized status values: `Active`, `Pending`, `Paused`, `Done` (and Spanish equivalents).

## Wiki links

The server understands Obsidian-style links:

| Written as | Resolves to |
|------------|-------------|
| `[[note]]`, `[[note.md]]` | the file named `note.md` anywhere in the vault |
| `[[projects/note]]` | `projects/note.md`, or any file whose path ends in `projects/note.md` |
| `[[note\|label]]`, `[[note#Heading]]`, `![[note]]` | `note` (alias, heading and embed marker are ignored) |
| `[text](../other.md)` | `other.md`, relative to the linking file |

If several files share a basename, the shallowest path wins (then alphabetical), so a bare `[[note]]` always resolves the same way; use a path-style link to reach a specific one. Matching is exact first, then case-insensitive, and nothing fuzzier: `[[my_note]]` does not match `my-note.md`.

Double brackets that are not links are ignored: anything inside code fences or inline code, shell tests like `[[ -f file ]]`, `[[ ! -d dir ]]` or `[[ a -lt b ]]`, and citation/JSON markup such as `[[^]]` or `[["a","b"]]`. Padding alone does not make a shell test: `[[ note ]]` is still a link to `note.md`.

The graph is built on first use and cached until any `.md` file in the vault changes. `vault_read` never waits for a rebuild: if the cache is stale it omits the backlinks line and rebuilds in a background thread. Writes through `vault_write`/`vault_edit` trigger that rebuild immediately, so backlinks reappear without anyone having to call `vault_links`.

## Stack

- **Python 3.12** + FastMCP
- **ChromaDB** for vector storage
- **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) for embeddings
- **Docker** + `uv` for fast builds

## License

MIT
