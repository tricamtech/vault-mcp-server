import importlib
import sys

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Import vault_mcp.server fresh, configured against a temp vault."""
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.delenv("OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    # server.py reads its configuration at import time
    sys.modules.pop("vault_mcp.server", None)
    yield importlib.import_module("vault_mcp.server")
    sys.modules.pop("vault_mcp.server", None)
