"""Git durability layer for vault writes.

WHY THIS EXISTS
---------------
Every vault write tool used to be a bare ``Path.write_text``: the bytes landed on
disk and nothing else happened. The corpus is a git repo that five hosts sync, so
an unrecorded write is a write that the next ``git checkout``/``reset`` can silently
eat -- which is exactly the data-loss bug this module closes.

DESIGN RULES (do not "simplify" these away)
-------------------------------------------
1. **Disk first, git second.** A git failure must never cost the user their write.
   Callers write the file, *then* call :func:`commit_write`. A failed commit is
   reported loudly in the tool's return string and logged at ERROR, but the file
   is already durable on disk.
2. **One choke point.** Nothing in this codebase should call ``git commit`` other
   than :func:`commit_write`. The 7 write tools funnel through
   ``server._persist`` which funnels here. If you find yourself adding a second
   commit call, you are re-creating the class of bug this repo keeps getting bitten
   by (a fix applied to one path and missed on the other).
3. **Locked.** Concurrent ``git add``/``git commit`` in one repo corrupts the
   index (``.git/index.lock`` races, half-staged trees). All git mutation happens
   under an exclusive ``flock`` on a per-repo lock file, so concurrent writers --
   threads in this process *and* other processes on the box -- serialize.
4. **Never push.** The deploy key on this host is read-only, and pushing from a
   write tool would turn a note edit into a fleet-wide event. Sync stays the job
   of the sync tooling. There is deliberately no push code path here.
5. **Repo discovery follows symlinks.** VAULT_PATH is a *curated* tree whose
   subdirectories are symlinks into the real corpus repo (and some of which are
   plain local directories in no repo at all). The repo that owns a write is
   therefore discovered from the file's REAL path, not from VAULT_PATH.
6. **Commits are debounced, writes are not.** One commit per write turned out to
   be the wrong granularity: the corpus takes ~228 commits/day and nothing
   downstream consumes per-write history (content files carry their own
   ``valid_from`` dates, the vault index works off file state, transcripts live in
   another repo). Writes therefore land on disk synchronously and the *commit* is
   deferred until a quiet period passes -- so six agents writing in a flurry
   produce one commit rather than forty. The durability contract is unchanged in
   substance: every write still ends up in some commit. What changed is that it is
   no longer necessarily its own. See :data:`GIT_DEBOUNCE_SECONDS`;
   ``VAULT_GIT_DEBOUNCE_SECONDS=0`` restores one-commit-per-write with no code
   edit.
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("false", "0", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


#: Master switch. Off means "behave like the old code": write to disk, no commit.
GIT_COMMIT_ENABLED = _env_flag("VAULT_GIT_COMMIT", True)
#: Hard timeout for any single git subprocess.
GIT_TIMEOUT = _env_float("VAULT_GIT_TIMEOUT", 30.0)
#: How long a writer waits for the per-repo commit lock before giving up.
GIT_LOCK_TIMEOUT = _env_float("VAULT_GIT_LOCK_TIMEOUT", 30.0)
#: Retries when git itself reports a transient index.lock collision (another
#: process -- sync tooling, a human shell -- held the real git index lock).
GIT_INDEX_RETRIES = int(_env_float("VAULT_GIT_INDEX_RETRIES", 5))

#: Quiet period, in seconds, after the LAST write of a burst before its batch is
#: committed. 0 commits immediately -- the one-commit-per-write behaviour this
#: module shipped with, kept as an env-selectable escape hatch.
GIT_DEBOUNCE_SECONDS = _env_float("VAULT_GIT_DEBOUNCE_SECONDS", 30.0)
#: Ceiling on how long a batch may be deferred while writes keep arriving. A pure
#: quiet-period debounce never fires under a sustained write stream, which would
#: leave hours of work uncommitted; this bounds it. 0 disables the ceiling.
GIT_DEBOUNCE_MAX_SECONDS = _env_float("VAULT_GIT_DEBOUNCE_MAX_SECONDS", 300.0)

LOCK_FILENAME = "vault-mcp-commit.lock"

_STATUS_OK = {"committed", "queued", "nochange", "untracked", "disabled"}


@dataclass(frozen=True)
class CommitResult:
    """Outcome of the git half of a write. Never raised -- always returned."""

    status: str  # committed | nochange | untracked | disabled | failed
    repo: Path | None = None
    commit: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in _STATUS_OK

    @property
    def committed(self) -> bool:
        return self.status == "committed"

    def suffix(self) -> str:
        """Short human marker appended to a write tool's return string."""
        if self.status == "committed":
            return f" [git {self.commit}]"
        if self.status == "nochange":
            return " [git: no change]"
        if self.status == "queued":
            # Deliberately NOT a sha: there is no commit yet. Reporting one would
            # hand the caller an identifier that does not exist.
            return " [git: queued]"
        if self.status == "untracked":
            return " [git: path not in a repo — not versioned]"
        if self.status == "disabled":
            return ""
        return f" [git: NOT COMMITTED — {self.detail or 'unknown error'}]"


# --------------------------------------------------------------------------
# repo discovery
# --------------------------------------------------------------------------

_repo_cache: dict[str, Path | None] = {}
_repo_cache_lock = threading.Lock()


def _git(repo: Path | None, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run git with a sanitized environment. Never inherits an editor or prompt."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("GIT_AUTHOR_NAME", env.get("VAULT_GIT_NAME", "vault-mcp"))
    env.setdefault("GIT_AUTHOR_EMAIL", env.get("VAULT_GIT_EMAIL", "vault-mcp@localhost"))
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    working = str(cwd or repo)
    return subprocess.run(
        ["git", "-C", working, *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
        env=env,
        check=False,
    )


class GitUnavailable(RuntimeError):
    """git itself could not be run (missing binary, timeout, OS error).

    Deliberately distinct from "this path is in no repo". Conflating them would
    report a broken git as the benign, quiet ``untracked`` status and every write
    would go unversioned without anyone noticing -- the exact silent-failure shape
    this whole feature exists to remove.
    """


def find_repo(path: Path, strict: bool = True) -> Path | None:
    """Return the git worktree root owning ``path``, or None if there is none.

    ``path`` may not exist yet (a create). We resolve the nearest existing
    ancestor, following symlinks, because the curated vault reaches the real
    corpus through symlinked subtrees.

    Raises :class:`GitUnavailable` when git could not be executed at all, unless
    ``strict=False`` (read-only callers such as the activity scan, where a missing
    repo and a missing git are equally "no history available").
    """
    probe = Path(os.path.realpath(path))
    if not probe.is_dir():
        probe = probe.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    key = str(probe)
    with _repo_cache_lock:
        if key in _repo_cache:
            return _repo_cache[key]

    root: Path | None = None
    try:
        proc = _git(None, ["rev-parse", "--show-toplevel"], cwd=probe)
        if proc.returncode == 0 and proc.stdout.strip():
            root = Path(proc.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:  # git missing, timeout
        logger.error("git could not be executed for %s: %s", probe, exc)
        if strict:
            raise GitUnavailable(f"git could not be executed: {exc}") from exc
        return None  # not cached: a broken git is usually transient

    with _repo_cache_lock:
        _repo_cache[key] = root
    return root


def clear_repo_cache():
    """Drop memoized repo lookups (tests, or after the vault layout changes)."""
    with _repo_cache_lock:
        _repo_cache.clear()


# --------------------------------------------------------------------------
# locking
# --------------------------------------------------------------------------

_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock_for(repo: Path) -> threading.Lock:
    with _thread_locks_guard:
        return _thread_locks.setdefault(str(repo), threading.Lock())


class _RepoLock:
    """Exclusive per-repo lock: in-process mutex + cross-process flock.

    flock alone would do for correctness, but the mutex keeps the common
    same-process case off the syscall path and makes the wait fair-ish.
    """

    def __init__(self, repo: Path, timeout: float):
        self.repo = repo
        self.timeout = timeout
        self._thread_lock = _thread_lock_for(repo)
        self._fd: int | None = None

    def __enter__(self) -> "_RepoLock":
        deadline = time.monotonic() + self.timeout
        if not self._thread_lock.acquire(timeout=self.timeout):
            raise TimeoutError(f"timed out waiting for in-process commit lock on {self.repo}")
        try:
            lock_dir = self.repo / ".git"
            lock_path = lock_dir / LOCK_FILENAME if lock_dir.is_dir() else self.repo / f".{LOCK_FILENAME}"
            self._fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for commit flock on {self.repo}"
                        )
                    time.sleep(0.05)
        except BaseException:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, *exc):
        try:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None
        finally:
            self._thread_lock.release()
        return False


# --------------------------------------------------------------------------
# the commit choke point
# --------------------------------------------------------------------------

_TRANSIENT_MARKERS = ("index.lock", "unable to create", "another git process")


def _resolve_agent(agent: str | None = None) -> str:
    return agent or os.environ.get("VAULT_AGENT") or os.environ.get("AGENT_NAME") or ""


def build_batch_message(
    tools: list[str], rel_paths: list[str], writes: int, agent: str | None = None
) -> str:
    """Commit message for a coalesced batch.

    Shape matches ct123's existing ``bills-vault-sync-memory`` commits
    (``vault: local writes from ct123 (N path(s))``) rather than inventing a third
    convention for the same repo. Per-file ``Vault-Path`` and per-tool
    ``Vault-Tool`` trailers are preserved, so ``vault_recent_activity`` (which
    reads ``--name-status`` plus the ``Vault-Tool`` trailer) keeps working
    unchanged and still reports one row per file.
    """
    subject = f"vault: {writes} write(s) via {', '.join(tools) or 'unknown'}"
    if len(subject) > 100:
        subject = subject[:97] + "..."
    trailers = [f"Vault-Tool: {tool}" for tool in tools]
    trailers.append(f"Vault-Batch: {writes} write(s), {len(rel_paths)} path(s)")
    trailers.extend(f"Vault-Path: {rel}" for rel in rel_paths)
    agent = _resolve_agent(agent)
    if agent:
        trailers.append(f"Vault-Agent: {agent}")
    return subject + "\n\n" + "\n".join(trailers) + "\n"


def build_message(tool: str, rel_paths: list[str], agent: str | None = None) -> str:
    """Commit message: identifies the tool and the file(s), machine-parseable."""
    head = rel_paths[0] if rel_paths else "(no path)"
    if len(rel_paths) > 1:
        head = f"{head} (+{len(rel_paths) - 1} more)"
    subject = f"vault({tool}): {head}"
    if len(subject) > 100:
        subject = subject[:97] + "..."
    agent = _resolve_agent(agent)
    trailers = [f"Vault-Tool: {tool}"]
    for rel in rel_paths:
        trailers.append(f"Vault-Path: {rel}")
    if agent:
        trailers.append(f"Vault-Agent: {agent}")
    return subject + "\n\n" + "\n".join(trailers) + "\n"


def commit_write(
    full_path: Path,
    tool: str,
    vault_rel_path: str | None = None,
    agent: str | None = None,
) -> CommitResult:
    """Commit one already-written file. THE only commit entry point.

    Returns a :class:`CommitResult`; raises nothing. Callers must have written
    the file to disk before calling.
    """
    if not GIT_COMMIT_ENABLED:
        return CommitResult(status="disabled")

    try:
        if GIT_DEBOUNCE_SECONDS > 0:
            return _queue_write(full_path, tool, vault_rel_path, agent)
        return _commit_write_inner(full_path, tool, vault_rel_path, agent)
    except Exception as exc:  # noqa: BLE001 - a commit must never lose a write
        logger.exception("vault git commit failed for %s (%s)", full_path, tool)
        return CommitResult(status="failed", detail=f"{type(exc).__name__}: {exc}")


def _commit_write_inner(
    full_path: Path,
    tool: str,
    vault_rel_path: str | None,
    agent: str | None,
) -> CommitResult:
    real_path = Path(os.path.realpath(full_path))
    repo = find_repo(real_path)
    if repo is None:
        logger.warning("vault write outside any git repo, not versioned: %s", full_path)
        return CommitResult(status="untracked", detail=str(full_path))

    try:
        repo_rel = str(real_path.relative_to(Path(os.path.realpath(repo))))
    except ValueError:
        return CommitResult(
            status="failed", repo=repo, detail=f"{real_path} is not inside {repo}"
        )

    message = build_message(tool, [vault_rel_path or repo_rel], agent=agent)
    last_error = ""

    for attempt in range(1, GIT_INDEX_RETRIES + 1):
        try:
            with _RepoLock(repo, GIT_LOCK_TIMEOUT):
                add = _git(repo, ["add", "--", repo_rel])
                if add.returncode != 0:
                    last_error = (add.stderr or add.stdout).strip()
                    if _is_transient(last_error):
                        time.sleep(0.1 * attempt)
                        continue
                    return CommitResult(status="failed", repo=repo, detail=last_error[:400])

                staged = _git(repo, ["diff", "--cached", "--quiet", "--", repo_rel])
                if staged.returncode == 0:
                    # Byte-identical rewrite (or an ignored path): nothing to record.
                    return CommitResult(status="nochange", repo=repo)

                # Pathspec-scoped commit: records ONLY our file even if another
                # agent left unrelated content staged in this shared repo.
                commit = _git(repo, ["commit", "-m", message, "--", repo_rel])
                if commit.returncode != 0:
                    last_error = (commit.stderr or commit.stdout).strip()
                    if _is_transient(last_error):
                        time.sleep(0.1 * attempt)
                        continue
                    if "nothing to commit" in last_error.lower():
                        return CommitResult(status="nochange", repo=repo)
                    logger.error(
                        "vault git commit refused (%s, %s): %s", tool, repo_rel, last_error[:400]
                    )
                    return CommitResult(status="failed", repo=repo, detail=last_error[:400])

                sha = _git(repo, ["rev-parse", "--short", "HEAD"]).stdout.strip() or None
                logger.info("vault commit %s %s (%s)", sha, repo_rel, tool)
                return CommitResult(status="committed", repo=repo, commit=sha)
        except TimeoutError as exc:
            last_error = str(exc)
            logger.error("vault git lock timeout: %s", exc)
            return CommitResult(status="failed", repo=repo, detail=last_error)
        except subprocess.TimeoutExpired as exc:
            last_error = f"git timed out after {GIT_TIMEOUT}s: {exc.cmd}"
            logger.error(last_error)
            return CommitResult(status="failed", repo=repo, detail=last_error)

    return CommitResult(
        status="failed",
        repo=repo,
        detail=f"git stayed locked after {GIT_INDEX_RETRIES} attempts: {last_error[:300]}",
    )


def _is_transient(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


# --------------------------------------------------------------------------
# debounced batching
# --------------------------------------------------------------------------
#
# THE CONTRACT, unchanged in substance:
#   * the file is already on disk before anything here runs (server._persist
#     writes atomically first), so a crash inside the debounce window costs a
#     COMMIT, never a write;
#   * every queued write ends up in some commit -- the batch is retried on
#     failure, re-armed by the next write, and flushed at interpreter exit.
# What changed is only the granularity: a burst becomes one commit.


@dataclass
class _Batch:
    """Writes waiting to be committed into one repo."""

    repo: Path
    first_queued_at: float
    #: repo-relative path -> vault-relative path (insertion ordered)
    paths: dict[str, str] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    agent: str = ""
    writes: int = 0
    attempts: int = 0
    timer: threading.Timer | None = None


_batches: dict[str, _Batch] = {}
_batch_lock = threading.RLock()


def _queue_write(
    full_path: Path,
    tool: str,
    vault_rel_path: str | None,
    agent: str | None,
) -> CommitResult:
    """Add one already-written file to its repo's pending batch."""
    real_path = Path(os.path.realpath(full_path))
    repo = find_repo(real_path)
    if repo is None:
        logger.warning("vault write outside any git repo, not versioned: %s", full_path)
        return CommitResult(status="untracked", detail=str(full_path))

    try:
        repo_rel = str(real_path.relative_to(Path(os.path.realpath(repo))))
    except ValueError:
        return CommitResult(
            status="failed", repo=repo, detail=f"{real_path} is not inside {repo}"
        )

    key = str(repo)
    with _batch_lock:
        batch = _batches.get(key)
        if batch is None:
            batch = _Batch(repo=repo, first_queued_at=time.monotonic())
            _batches[key] = batch
        batch.paths[repo_rel] = vault_rel_path or repo_rel
        if tool not in batch.tools:
            batch.tools.append(tool)
        batch.agent = batch.agent or _resolve_agent(agent)
        batch.writes += 1
        pending = len(batch.paths)
        _rearm(batch)

    return CommitResult(
        status="queued", repo=repo, detail=f"{pending} path(s) pending commit"
    )


def _rearm(batch: _Batch, delay: float | None = None):
    """(Re)start the quiet-period timer. Caller holds ``_batch_lock``."""
    if batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None

    if delay is None:
        delay = GIT_DEBOUNCE_SECONDS
        if GIT_DEBOUNCE_MAX_SECONDS > 0:
            remaining = GIT_DEBOUNCE_MAX_SECONDS - (time.monotonic() - batch.first_queued_at)
            delay = max(0.0, min(delay, remaining))

    timer = threading.Timer(delay, _on_timer, args=(str(batch.repo),))
    # Daemon so a pending window cannot hold the process open for 30 s on
    # shutdown; the atexit hook below is what guarantees the batch still lands.
    timer.daemon = True
    timer.name = f"vault-git-debounce-{batch.repo.name}"
    batch.timer = timer
    timer.start()


def _on_timer(key: str):
    """Timer callback. An exception here must never strand a batch."""
    try:
        _flush_key(key)
    except Exception:  # noqa: BLE001 - a stranded batch is a durability bug
        logger.exception("vault git debounce flush crashed for %s", key)
        with _batch_lock:
            batch = _batches.get(key)
            if batch is not None and batch.paths:
                batch.attempts += 1
                if batch.attempts <= GIT_INDEX_RETRIES:
                    _rearm(batch, delay=min(5.0, 0.5 * batch.attempts))


def _flush_key(key: str) -> CommitResult | None:
    """Commit one repo's pending batch. Returns None when there is nothing to do."""
    with _batch_lock:
        batch = _batches.get(key)
        if batch is None or not batch.paths:
            return None
        if batch.timer is not None:
            batch.timer.cancel()
            batch.timer = None
        # Take the batch out so writes arriving during the commit start a fresh
        # one instead of being silently folded into a commit that is already
        # staged (and therefore would not include them).
        del _batches[key]
        repo = batch.repo
        paths = dict(batch.paths)
        tools = list(batch.tools)
        agent = batch.agent
        writes = batch.writes
        attempts = batch.attempts

    try:
        result = _commit_batch(repo, paths, tools, writes, agent)
    except Exception as exc:  # noqa: BLE001 - the batch is already out of the map
        logger.exception("vault git batch commit raised for %s", key)
        result = CommitResult(
            status="failed", repo=repo, detail=f"{type(exc).__name__}: {exc}"
        )

    if not result.ok:
        # NEVER drop the work. Put it back and let the next write, the retry timer
        # or the shutdown flush try again.
        with _batch_lock:
            pending = _batches.get(key)
            if pending is None:
                pending = _Batch(repo=repo, first_queued_at=time.monotonic())
                _batches[key] = pending
            for repo_rel, vault_rel in paths.items():
                pending.paths.setdefault(repo_rel, vault_rel)
            for tool in tools:
                if tool not in pending.tools:
                    pending.tools.append(tool)
            pending.agent = pending.agent or agent
            pending.writes += writes
            pending.attempts = attempts + 1
            if pending.attempts <= GIT_INDEX_RETRIES:
                _rearm(pending, delay=min(5.0, 0.5 * pending.attempts))
            else:
                logger.error(
                    "vault git batch for %s STILL UNCOMMITTED after %d attempts "
                    "(%d path(s) on disk but not in git): %s",
                    key, pending.attempts, len(pending.paths), result.detail,
                )
    return result


def _commit_batch(
    repo: Path,
    paths: dict[str, str],
    tools: list[str],
    writes: int,
    agent: str,
) -> CommitResult:
    """One `git add` + one `git commit` for every path in the batch."""
    repo_rels = sorted(paths)
    message = build_batch_message(tools, [paths[rel] for rel in repo_rels], writes, agent)
    last_error = ""

    for attempt in range(1, GIT_INDEX_RETRIES + 1):
        try:
            with _RepoLock(repo, GIT_LOCK_TIMEOUT):
                # --all so a path deleted since it was queued is staged as a
                # deletion rather than failing the whole batch on "pathspec did
                # not match".
                add = _git(repo, ["add", "--all", "--", *repo_rels])
                if add.returncode != 0:
                    last_error = (add.stderr or add.stdout).strip()
                    if _is_transient(last_error):
                        time.sleep(0.1 * attempt)
                        continue
                    return CommitResult(status="failed", repo=repo, detail=last_error[:400])

                staged = _git(repo, ["diff", "--cached", "--quiet", "--", *repo_rels])
                if staged.returncode == 0:
                    return CommitResult(status="nochange", repo=repo)

                commit = _git(repo, ["commit", "-m", message, "--", *repo_rels])
                if commit.returncode != 0:
                    last_error = (commit.stderr or commit.stdout).strip()
                    if _is_transient(last_error):
                        time.sleep(0.1 * attempt)
                        continue
                    if "nothing to commit" in last_error.lower():
                        return CommitResult(status="nochange", repo=repo)
                    logger.error(
                        "vault git batch commit refused (%d paths): %s",
                        len(repo_rels), last_error[:400],
                    )
                    return CommitResult(status="failed", repo=repo, detail=last_error[:400])

                sha = _git(repo, ["rev-parse", "--short", "HEAD"]).stdout.strip() or None
                logger.info(
                    "vault batch commit %s — %d write(s), %d path(s) via %s",
                    sha, writes, len(repo_rels), ", ".join(tools),
                )
                return CommitResult(status="committed", repo=repo, commit=sha)
        except TimeoutError as exc:
            logger.error("vault git lock timeout: %s", exc)
            return CommitResult(status="failed", repo=repo, detail=str(exc))
        except subprocess.TimeoutExpired as exc:
            last_error = f"git timed out after {GIT_TIMEOUT}s: {exc.cmd}"
            logger.error(last_error)
            return CommitResult(status="failed", repo=repo, detail=last_error)

    return CommitResult(
        status="failed",
        repo=repo,
        detail=f"git stayed locked after {GIT_INDEX_RETRIES} attempts: {last_error[:300]}",
    )


def flush_all() -> list[CommitResult]:
    """Commit every pending batch NOW.

    Called from the atexit hook and from the server's shutdown path, so a clean
    stop can never leave a write on disk but out of git. Safe to call when there
    is nothing pending.
    """
    with _batch_lock:
        keys = list(_batches)
    results = []
    for key in keys:
        try:
            result = _flush_key(key)
        except Exception:  # noqa: BLE001
            logger.exception("vault git shutdown flush failed for %s", key)
            continue
        if result is not None:
            results.append(result)
    return results


def pending_summary() -> dict[str, int]:
    """Paths currently queued per repo. Diagnostics and tests."""
    with _batch_lock:
        return {key: len(batch.paths) for key, batch in _batches.items() if batch.paths}


def reset_batches():
    """Drop all pending batches WITHOUT committing. Tests only."""
    with _batch_lock:
        for batch in _batches.values():
            if batch.timer is not None:
                batch.timer.cancel()
        _batches.clear()


atexit.register(flush_all)
