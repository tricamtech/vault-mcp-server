import importlib, os, tempfile
import pytest


@pytest.fixture
def server(tmp_path, monkeypatch):
    # Never point a test at the live vault or its Chroma DB.
    (tmp_path / "vault").mkdir()
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("VAULT_QUERY_LOG", str(tmp_path / "query-log.jsonl"))
    monkeypatch.setenv("VAULT_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    from vault_mcp import server as mod
    return importlib.reload(mod)
