import asyncio
import os
import subprocess
import sys
import textwrap
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from vault_mcp import gitstore

SRC = Path(__file__).resolve().parent.parent / "src"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "config", "user.name", "test")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("seed\n")
    git(path, "add", "README.md")
    git(path, "commit", "-qm", "seed")
    return path


def commit_count(repo) -> int:
    return int(git(repo, "rev-list", "--count", "HEAD"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def store(monkeypatch, tmp_path):
    """gitstore enabled, committing immediately; tests opt in to debouncing."""
    # Keep git from finding a repo above the test's temp dir.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(gitstore, "GIT_COMMIT_ENABLED", True)
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_MAX_SECONDS", 300.0)
    gitstore.reset_batches()
    gitstore.clear_repo_cache()
    yield gitstore
    gitstore.reset_batches()
    gitstore.clear_repo_cache()


# ---------------------------------------------------------------------------
# commit_write
# ---------------------------------------------------------------------------


def test_write_is_committed(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "notes" / "a.md", "hello\n")

    result = store.commit_write(note, "vault_write", vault_rel_path="notes/a.md")

    assert result.status == "committed"
    assert result.commit == git(repo, "rev-parse", "--short", "HEAD")
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "show", "HEAD:notes/a.md") == "hello"
    message = git(repo, "log", "-1", "--format=%B")
    assert message.startswith("vault(vault_write): notes/a.md")
    assert "Vault-Path: notes/a.md" in message
    assert git(repo, "log", "-1", "--format=%an <%ae>") == "vault-mcp <vault-mcp@localhost>"


def test_rewrite_with_same_bytes_is_nochange(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "a.md", "same\n")
    assert store.commit_write(note, "vault_write").committed
    before = commit_count(repo)

    write(note, "same\n")
    result = store.commit_write(note, "vault_write")

    assert result.status == "nochange"
    assert commit_count(repo) == before


def test_commit_leaves_unrelated_staged_changes_alone(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    write(repo / "README.md", "someone else's half-finished edit\n")
    git(repo, "add", "README.md")

    note = write(repo / "a.md", "mine\n")
    assert store.commit_write(note, "vault_write").committed

    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "a.md"
    assert git(repo, "diff", "--cached", "--name-only") == "README.md"


def test_non_git_vault_is_a_harmless_noop(store, tmp_path):
    vault = tmp_path / "plain"
    note = write(vault / "a.md", "not versioned\n")

    result = store.commit_write(note, "vault_write")

    assert result.status == "untracked"
    assert result.ok
    assert note.read_text() == "not versioned\n"
    assert sorted(p.name for p in vault.iterdir()) == ["a.md"]  # no .git, no lock file


def test_gitignored_path_is_not_a_failure(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    write(repo / ".gitignore", "scratch/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "ignore")
    note = write(repo / "scratch" / "a.md", "x\n")

    result = store.commit_write(note, "vault_write")

    assert result.status == "ignored"
    assert result.ok


def test_disabled_by_default_does_nothing(store, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_COMMIT_ENABLED", False)
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "a.md", "x\n")

    assert store.commit_write(note, "vault_write").status == "disabled"
    assert commit_count(repo) == 1


def test_master_switch_defaults_off():
    code = "from vault_mcp import gitstore; print(gitstore.GIT_COMMIT_ENABLED)"
    env = {k: v for k, v in os.environ.items() if k != "VAULT_GIT_COMMIT"}
    env["PYTHONPATH"] = str(SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "False"


def test_failed_commit_keeps_the_write_and_says_so(store, tmp_path):
    repo = init_repo(tmp_path / "vault")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho refused by hook >&2\nexit 1\n")
    hook.chmod(0o755)
    note = write(repo / "a.md", "precious\n")

    result = store.commit_write(note, "vault_write")

    assert result.status == "failed"
    assert not result.ok
    assert "NOT COMMITTED" in result.suffix()
    assert note.read_text() == "precious\n"
    assert commit_count(repo) == 1


def test_commit_works_without_fcntl(store, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gitstore, "fcntl", None)
    monkeypatch.setattr(gitstore, "_warned_no_flock", False)
    repo = init_repo(tmp_path / "vault")

    for name in ("a.md", "b.md"):
        note = write(repo / name, f"{name}\n")
        assert store.commit_write(note, "vault_write").committed

    assert commit_count(repo) == 3
    warnings = [r for r in caplog.records if "fcntl is unavailable" in r.getMessage()]
    assert len(warnings) == 1


def test_symlinked_subtree_commits_into_the_real_repo(store, tmp_path):
    corpus = init_repo(tmp_path / "corpus")
    (corpus / "shared").mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "shared").symlink_to(corpus / "shared")

    note = write(vault / "shared" / "a.md", "via symlink\n")
    result = store.commit_write(note, "vault_write", vault_rel_path="shared/a.md")

    assert result.committed
    assert git(corpus, "show", "HEAD:shared/a.md") == "via symlink"


# ---------------------------------------------------------------------------
# debouncing
# ---------------------------------------------------------------------------


def test_debounced_writes_batch_into_one_commit_on_flush_all(store, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 60.0)
    repo = init_repo(tmp_path / "vault")
    a = write(repo / "a.md", "a\n")
    b = write(repo / "b.md", "b\n")

    assert store.commit_write(a, "vault_write").status == "queued"
    assert store.commit_write(b, "vault_edit").status == "queued"
    assert store.pending_summary() == {str(repo): 2}
    assert commit_count(repo) == 1  # nothing committed yet

    results = store.flush_all()

    assert [r.status for r in results] == ["committed"]
    assert store.pending_summary() == {}
    assert commit_count(repo) == 2
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["a.md", "b.md"]
    message = git(repo, "log", "-1", "--format=%B")
    assert message.startswith("vault: 2 write(s) via vault_write, vault_edit")
    assert "Vault-Path: a.md" in message and "Vault-Path: b.md" in message


def test_debounce_timer_commits_after_quiet_period(store, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 0.2)
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "a.md", "a\n")

    assert store.commit_write(note, "vault_write").status == "queued"

    deadline = time.monotonic() + 10
    while commit_count(repo) == 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert commit_count(repo) == 2
    assert store.pending_summary() == {}


def test_failed_batch_is_requeued_not_dropped(store, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 60.0)
    repo = init_repo(tmp_path / "vault")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    note = write(repo / "a.md", "a\n")
    store.commit_write(note, "vault_write")

    assert [r.status for r in store.flush_all()] == ["failed"]
    assert store.pending_summary() == {str(repo): 1}

    hook.unlink()
    assert [r.status for r in store.flush_all()] == ["committed"]
    assert git(repo, "show", "HEAD:a.md") == "a"


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


def test_pending_commits_flush_at_interpreter_exit(tmp_path):
    repo = init_repo(tmp_path / "vault")
    note = write(repo / "a.md", "written just before exit\n")
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from vault_mcp import gitstore
        r = gitstore.commit_write(Path({str(note)!r}), "vault_write")
        assert r.status == "queued", r
        """
    )
    env = dict(os.environ, PYTHONPATH=str(SRC), VAULT_GIT_COMMIT="true",
               VAULT_GIT_DEBOUNCE_SECONDS="3600")

    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=60)

    assert commit_count(repo) == 2
    assert git(repo, "show", "HEAD:a.md") == "written just before exit"


# ---------------------------------------------------------------------------
# server wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def git_server(server, store, tmp_path):
    init_repo(tmp_path)
    return server


def test_vault_write_tool_commits(git_server, tmp_path):
    out = git_server.vault_write("notes/new.md", "# New\n")

    sha = git(tmp_path, "rev-parse", "--short", "HEAD")
    assert out == f"Written: notes/new.md (6 chars) [git {sha}]"
    assert git(tmp_path, "show", "HEAD:notes/new.md") == "# New"


def test_vault_edit_tool_commits(git_server, tmp_path):
    git_server.vault_write("a.md", "status: draft\n")

    out = git_server.vault_edit("a.md", "draft", "done")

    sha = git(tmp_path, "rev-parse", "--short", "HEAD")
    assert out == f"Edited: a.md (replaced 1 occurrence) [git {sha}]"
    assert git(tmp_path, "show", "HEAD:a.md") == "status: done"
    assert git(tmp_path, "log", "-1", "--format=%s") == "vault(vault_edit): a.md"


def test_write_tools_still_work_in_a_non_git_vault(server, store, tmp_path):
    out = server.vault_write("a.md", "x\n")

    assert out.startswith("Written: a.md (2 chars)")
    assert "not versioned" in out
    assert (tmp_path / "a.md").read_text() == "x\n"


def test_lifespan_flushes_pending_commits_on_shutdown(git_server, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 3600.0)

    @asynccontextmanager
    async def inner(app):
        yield {}

    async def run():
        async with git_server._with_git_flush(inner)(None):
            assert git_server.vault_write("a.md", "x\n").endswith("[git: queued]")
            assert commit_count(tmp_path) == 1

    asyncio.run(run())

    assert gitstore.pending_summary() == {}
    assert commit_count(tmp_path) == 2


def test_main_app_flushes_pending_commits_on_shutdown(git_server, tmp_path, monkeypatch):
    monkeypatch.setattr(gitstore, "GIT_DEBOUNCE_SECONDS", 3600.0)
    apps = []
    monkeypatch.setattr(git_server.uvicorn, "run", lambda app, **kw: apps.append(app))

    async def no_index():
        pass

    monkeypatch.setattr(git_server.indexer, "start", no_index)
    git_server.main()
    app = apps[0]

    async def run():
        async with app.router.lifespan_context(app):
            assert git_server.vault_write("a.md", "x\n").endswith("[git: queued]")

    asyncio.run(run())

    assert commit_count(tmp_path) == 2
