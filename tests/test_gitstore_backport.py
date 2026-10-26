"""Backported gitstore fixes: gitignored writes, and the lock file's location.

Every repo here is a throwaway under tmp_path. Never point these at a real vault.
"""
import fcntl
import logging
import os
import subprocess
from pathlib import Path

import pytest

from vault_mcp import gitstore


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "protocol.file.allow=always", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "README.md").write_text("seed\n")
    git(path, "add", "README.md")
    git(path, "commit", "-qm", "seed")
    return path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def commit_count(repo: Path) -> int:
    return int(git(repo, "rev-list", "--count", "HEAD"))


def stray_locks(root: Path) -> list[str]:
    """Commit lock files outside any git dir, i.e. visible in a work tree."""
    return sorted(
        str(p.relative_to(root)) for p in root.rglob(f"*{gitstore.LOCK_FILENAME}")
        if ".git" not in p.relative_to(root).parts
    )


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_COMMIT_ENABLED", True)
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(gitstore, "GIT_LOCK_TIMEOUT", 5.0)
    monkeypatch.delenv("VAULT_AGENT", raising=False)
    gitstore.clear_repo_cache()
    gitstore.reset_batches()
    for name in ("_lock_paths", "_ignored_logged"):
        if hasattr(gitstore, name):
            getattr(gitstore, name).clear()
    yield gitstore
    gitstore.reset_batches()
    gitstore.clear_repo_cache()


@pytest.fixture
def ignoring_repo(tmp_path):
    repo = init_repo(tmp_path / "vault")
    write(repo / ".gitignore", "scratch/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "ignore")
    return repo


# --------------------------------------------------------------------------
# fix 2: a gitignored path is "ignored", not a failed commit
# --------------------------------------------------------------------------


def test_gitignored_path_is_ignored_not_failed(store, ignoring_repo):
    note = write(ignoring_repo / "scratch" / "a.md", "x\n")
    result = store.commit_write(note, "vault_write")
    assert result.status == "ignored"
    assert result.ok
    assert "NOT COMMITTED" not in result.suffix()
    assert commit_count(ignoring_repo) == 2


def test_gitignored_path_is_never_queued(store, ignoring_repo, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 30.0)
    note = write(ignoring_repo / "scratch" / "a.md", "x\n")
    result = store.commit_write(note, "vault_write")
    assert result.status == "ignored"
    assert store.pending_summary() == {}


def test_gitignored_path_does_not_poison_its_batch(store, ignoring_repo, monkeypatch):
    # Before the fix the ignored path was queued with the real one, `git add`
    # refused the whole batch, and the real note was retried and never committed.
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 30.0)
    real = write(ignoring_repo / "notes" / "real.md", "keep\n")
    junk = write(ignoring_repo / "scratch" / "junk.md", "x\n")
    assert store.commit_write(real, "vault_write").status == "queued"
    store.commit_write(junk, "vault_write")
    results = store.flush_all()
    assert [r.status for r in results] == ["committed"]
    assert git(ignoring_repo, "ls-files", "notes/real.md") == "notes/real.md"
    assert store.pending_summary() == {}


def test_gitignored_path_logged_once_at_info(store, ignoring_repo, caplog):
    a = write(ignoring_repo / "scratch" / "a.md", "x\n")
    b = write(ignoring_repo / "scratch" / "b.md", "x\n")
    with caplog.at_level(logging.DEBUG, logger=gitstore.__name__):
        for _ in range(3):
            store.commit_write(a, "vault_write")
        store.commit_write(b, "vault_write")
    records = [r for r in caplog.records if "ignored" in r.getMessage().lower()]
    assert [r.levelno for r in records] == [logging.INFO, logging.INFO]
    assert "scratch/a.md" in records[0].getMessage()
    assert "scratch/b.md" in records[1].getMessage()


def test_tracked_file_matching_an_ignore_pattern_still_commits(store, ignoring_repo):
    # check-ignore does not report tracked files, so a force-added file keeps
    # being versioned. (A file pattern, not "dir/": under an ignored directory
    # git 2.43's `git add` exits 1 even for a tracked file, a separate issue.)
    write(ignoring_repo / ".gitignore", "scratch/\n*.draft.md\n")
    note = write(ignoring_repo / "kept.draft.md", "v1\n")
    git(ignoring_repo, "add", "-f", ".gitignore", "kept.draft.md")
    git(ignoring_repo, "commit", "-qm", "force-add")
    note.write_text("v2\n")
    assert store.commit_write(note, "vault_write").status == "committed"


# --------------------------------------------------------------------------
# fix 3: the commit lock lives in the git dir that git reports
# --------------------------------------------------------------------------


def test_lock_lives_in_git_dir_of_plain_repo(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "a.md", "x\n")
    assert store.commit_write(note, "vault_write").status == "committed"
    assert (repo / ".git" / gitstore.LOCK_FILENAME).exists()
    assert stray_locks(repo) == []


def test_lock_lives_in_git_dir_of_linked_worktree(store, tmp_path):
    main = init_repo(tmp_path / "main")
    tree = tmp_path / "tree"
    git(main, "worktree", "add", "-q", str(tree))
    assert (tree / ".git").is_file()

    note = write(tree / "a.md", "x\n")
    assert store.commit_write(note, "vault_write").status == "committed"

    git_dir = Path(git(tree, "rev-parse", "--absolute-git-dir"))
    assert (git_dir / gitstore.LOCK_FILENAME).exists()
    assert stray_locks(tree) == []
    assert git(tree, "status", "--porcelain") == ""


def test_lock_lives_in_git_dir_of_submodule(store, tmp_path):
    upstream = init_repo(tmp_path / "upstream")
    parent = init_repo(tmp_path / "parent")
    git(parent, "submodule", "add", "-q", str(upstream), "sub")
    git(parent, "commit", "-qm", "add sub")
    sub = parent / "sub"
    assert (sub / ".git").is_file()

    note = write(sub / "a.md", "x\n")
    result = store.commit_write(note, "vault_write")
    assert result.status == "committed"
    assert Path(os.path.realpath(result.repo)) == Path(os.path.realpath(sub))

    git_dir = Path(git(sub, "rev-parse", "--absolute-git-dir"))
    assert (git_dir / gitstore.LOCK_FILENAME).exists()
    assert stray_locks(parent) == []


def test_lock_in_git_dir_still_excludes_other_processes(store, tmp_path, monkeypatch):
    # The lock moved, its semantics did not: another holder of the flock on the
    # git-dir lock file makes a writer wait and then time out.
    main = init_repo(tmp_path / "main")
    tree = tmp_path / "tree"
    git(main, "worktree", "add", "-q", str(tree))
    lock_path = Path(git(tree, "rev-parse", "--absolute-git-dir")) / gitstore.LOCK_FILENAME
    monkeypatch.setattr(gitstore, "GIT_LOCK_TIMEOUT", 0.3)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        note = write(tree / "a.md", "x\n")
        result = store.commit_write(note, "vault_write")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert result.status == "failed"
    assert "timed out" in result.detail
