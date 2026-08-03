"""Recent-activity view over the vault: "what changed since I was last here".

Sources, in order of trust:

1. **git log** of every repo the vault is mounted on -- authoritative, carries
   who/what/when and (for writes made through this server) the exact tool via the
   ``Vault-Tool`` commit trailer written by :mod:`vault_mcp.gitstore`.
2. **git status** of those repos -- files changed on disk but NOT committed. For a
   cold-resuming agent this is the highest-signal line in the report: it is either
   a human editing in place, or a write whose commit failed.
3. **mtime** for vault subtrees that live in no repo at all (the curated tree has
   some), where git can tell us nothing.

The vault is a *curated* tree: its subdirectories are symlinks into one or more
real repos. So this module first builds the mount table (vault prefix -> real
directory -> owning repo) and maps every git path back through it. Anything in a
repo but not exposed by the vault is deliberately dropped: this tool answers
"what changed in the vault", not "what changed in the repo".
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import gitstore

logger = logging.getLogger(__name__)

_REC = "\x1e"  # record separator between commits
_FLD = "\x1f"  # field separator inside the commit header

_STATUS_WORDS = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "T": "typechange",
}

#: Hard cap on commits parsed per repo, so a huge window cannot wedge a request.
MAX_COMMITS_PER_REPO = 500
#: Hard cap on uncommitted-working-tree entries reported per repo.
MAX_DIRTY_PER_REPO = 50


@dataclass(frozen=True)
class Mount:
    """One vault subtree and the real directory (and repo) behind it."""

    vault_prefix: str  # "" for the vault root, else e.g. "shared" or "bob/memory"
    real_root: Path
    repo: Path | None = None


@dataclass
class ActivityEntry:
    path: str  # vault-relative path
    when: datetime
    change: str  # added | modified | deleted | renamed | uncommitted | touched
    source: str  # commit | worktree | mtime
    tool: str | None = None  # Vault-Tool trailer, when the write came through us
    author: str | None = None
    commit: str | None = None
    subject: str | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# mount table
# --------------------------------------------------------------------------


def build_mounts(vault_path: Path) -> list[Mount]:
    """Map the curated vault onto the real directories (and repos) behind it.

    Symlinked directories at ANY depth become mounts (``bob/memory`` is a symlink
    two levels down in the live layout), which is why this walks with
    ``followlinks=False``: os.walk then reports symlinked dirs without descending
    into them, i.e. it enumerates exactly the mount points.
    """
    seen: dict[str, Mount] = {}
    root_real = Path(os.path.realpath(vault_path))
    seen[str(root_real)] = Mount("", root_real, gitstore.find_repo(root_real, strict=False))

    for root, dirs, _files in os.walk(vault_path, followlinks=False):
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("_"))]
        for name in list(dirs):
            candidate = Path(root) / name
            if not candidate.is_symlink():
                continue
            real = Path(os.path.realpath(candidate))
            if not real.is_dir():
                continue
            prefix = str(candidate.relative_to(vault_path))
            seen.setdefault(str(real), Mount(prefix, real, gitstore.find_repo(real, strict=False)))

    # Longest real path first so nested mounts win over their parent.
    return sorted(seen.values(), key=lambda m: len(str(m.real_root)), reverse=True)


def to_vault_path(mounts: list[Mount], abs_path: Path | str) -> str | None:
    """Map a real absolute path back to its vault-relative path, or None."""
    target = str(abs_path)
    for mount in mounts:
        root = str(mount.real_root)
        if target == root:
            return mount.vault_prefix or "."
        if target.startswith(root + "/"):
            remainder = target[len(root) + 1:]
            return f"{mount.vault_prefix}/{remainder}" if mount.vault_prefix else remainder
    return None


# --------------------------------------------------------------------------
# git sources
# --------------------------------------------------------------------------


def _git(repo: Path, args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def _parse_git_time(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw.strip())
    except ValueError:
        return datetime.now(timezone.utc)


def _commit_entries(repo: Path, mounts: list[Mount], since: datetime) -> list[ActivityEntry]:
    fmt = (
        f"{_REC}%H{_FLD}%aI{_FLD}%an{_FLD}%s{_FLD}"
        "%(trailers:key=Vault-Tool,valueonly,separator=%x2c)"
    )
    proc = _git(
        repo,
        [
            "log",
            f"--since={since.isoformat()}",
            f"--max-count={MAX_COMMITS_PER_REPO}",
            "--name-status",
            "--no-renames",
            f"--format={fmt}",
        ],
    )
    if proc.returncode != 0:
        logger.warning("git log failed in %s: %s", repo, (proc.stderr or "").strip()[:200])
        return []

    entries: list[ActivityEntry] = []
    for block in proc.stdout.split(_REC):
        block = block.strip("\n")
        if not block:
            continue
        header, _, body = block.partition("\n")
        fields = header.split(_FLD)
        if len(fields) < 5:
            continue
        sha, when_raw, author, subject, tool = fields[:5]
        when = _parse_git_time(when_raw)
        for line in body.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0].strip()[:1]
            rel_in_repo = parts[-1].strip()
            vault_rel = to_vault_path(mounts, repo / rel_in_repo)
            if vault_rel is None:
                continue
            entries.append(
                ActivityEntry(
                    path=vault_rel,
                    when=when,
                    change=_STATUS_WORDS.get(status, status.lower() or "changed"),
                    source="commit",
                    tool=tool.strip() or None,
                    author=author.strip() or None,
                    commit=sha[:9],
                    subject=subject.strip() or None,
                )
            )
    return entries


def _dirty_entries(repo: Path, mounts: list[Mount], since: datetime) -> list[ActivityEntry]:
    """Files changed on disk but not committed -- the 'a write didn't land' signal."""
    proc = _git(repo, ["status", "--porcelain", "--untracked-files=all"])
    if proc.returncode != 0:
        return []
    entries: list[ActivityEntry] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        rel_in_repo = line[3:].strip().strip('"')
        if " -> " in rel_in_repo:  # rename
            rel_in_repo = rel_in_repo.split(" -> ", 1)[1]
        if not rel_in_repo.endswith(".md"):
            continue
        vault_rel = to_vault_path(mounts, repo / rel_in_repo)
        if vault_rel is None:
            continue
        abs_path = repo / rel_in_repo
        try:
            when = datetime.fromtimestamp(abs_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if when < since:
            continue
        entries.append(
            ActivityEntry(
                path=vault_rel,
                when=when,
                change="uncommitted",
                source="worktree",
                extra={"git_status": code.strip() or "??"},
            )
        )
        if len(entries) >= MAX_DIRTY_PER_REPO:
            break
    return entries


def _mtime_entries(mount: Mount, vault_path: Path, since: datetime) -> list[ActivityEntry]:
    """Fallback for vault subtrees that no git repo covers."""
    entries: list[ActivityEntry] = []
    for root, dirs, files in os.walk(mount.real_root, followlinks=False):
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("_"))]
        for name in files:
            if not name.endswith(".md") or name.startswith((".", "_")):
                continue
            abs_path = Path(root) / name
            try:
                when = datetime.fromtimestamp(abs_path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if when < since:
                continue
            vault_rel = to_vault_path([mount], abs_path)
            if vault_rel is None:
                continue
            entries.append(
                ActivityEntry(path=vault_rel, when=when, change="touched", source="mtime")
            )
    return entries


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def recent_activity(
    vault_path: Path,
    hours: float = 24.0,
    limit: int = 20,
    path_prefix: str = "",
    include_uncommitted: bool = True,
) -> list[ActivityEntry]:
    """Return the most recent change per vault file inside the time window."""
    hours = max(float(hours), 0.01)
    limit = max(int(limit), 1)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    mounts = build_mounts(vault_path)
    entries: list[ActivityEntry] = []

    repos: dict[str, Path] = {}
    for mount in mounts:
        if mount.repo is not None:
            repos.setdefault(str(mount.repo), mount.repo)
        else:
            entries.extend(_mtime_entries(mount, vault_path, since))

    for repo in repos.values():
        try:
            entries.extend(_commit_entries(repo, mounts, since))
            if include_uncommitted:
                entries.extend(_dirty_entries(repo, mounts, since))
        except subprocess.SubprocessError as exc:
            logger.warning("activity scan failed for %s: %s", repo, exc)

    if path_prefix:
        normalized = path_prefix.strip().lstrip("./")
        entries = [e for e in entries if e.path.startswith(normalized)]

    # One row per file: the newest event wins, but an uncommitted working-tree
    # change always outranks an older commit for the same path because it is the
    # state actually on disk right now.
    best: dict[str, ActivityEntry] = {}
    for entry in entries:
        current = best.get(entry.path)
        if current is None:
            best[entry.path] = entry
            continue
        if entry.source == "worktree" and current.source != "worktree":
            best[entry.path] = entry
        elif current.source == "worktree" and entry.source != "worktree":
            continue
        elif entry.when > current.when:
            best[entry.path] = entry

    ordered = sorted(best.values(), key=lambda e: e.when, reverse=True)
    return ordered[:limit]


def _humanize(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds < 90:
        return f"{max(seconds, 0)}s ago"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def format_activity(entries: list[ActivityEntry], hours: float, limit: int) -> str:
    """Render activity entries as markdown for an MCP client."""
    window = f"{hours:g}h" if hours < 48 else f"{hours / 24:g}d"
    if not entries:
        return f"No vault activity in the last {window}."

    now = datetime.now(timezone.utc)
    lines = [f"# Vault activity — last {window} ({len(entries)} file(s), limit {limit})", ""]
    for entry in entries:
        when = entry.when.astimezone().strftime("%Y-%m-%d %H:%M")
        bits = [f"**{entry.path}**", f"{entry.change}", f"{when} ({_humanize(now - entry.when)})"]
        lines.append("- " + "  |  ".join(bits))
        detail: list[str] = []
        if entry.tool:
            detail.append(f"tool: `{entry.tool}`")
        if entry.author:
            detail.append(f"by: {entry.author}")
        if entry.commit:
            detail.append(f"commit: `{entry.commit}`")
        if entry.source == "worktree":
            detail.append(
                f"NOT COMMITTED (git status `{entry.extra.get('git_status', '??')}`)"
            )
        if entry.source == "mtime":
            detail.append("mtime only (path not under git)")
        if entry.subject and not entry.tool:
            detail.append(f"“{entry.subject[:80]}”")
        if detail:
            lines.append("    " + " · ".join(detail))
    return "\n".join(lines)
