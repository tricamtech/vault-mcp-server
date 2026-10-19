"""OAuth protected-resource metadata must be discoverable where clients probe."""

import importlib
import sys

import pytest
from starlette.testclient import TestClient

ISSUER = "https://vault.example.com"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("OAUTH_ISSUER_URL", ISSUER)
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.delenv("OAUTH_STATE_PATH", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    # server.py reads its configuration at import time
    sys.modules.pop("vault_mcp.server", None)
    server = importlib.import_module("vault_mcp.server")
    # No lifespan: the background indexer is not needed for routing tests
    yield TestClient(server.create_app())
    sys.modules.pop("vault_mcp.server", None)


@pytest.mark.parametrize(
    "path",
    [
        # RFC 9728 path-aware location, probed first by MCP clients
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
        # issuer with trailing slash + "/.well-known/..." concatenation
        "//.well-known/oauth-protected-resource",
        "//.well-known/oauth-protected-resource/mcp",
    ],
)
def test_protected_resource_metadata_is_served(client, path):
    resp = client.get(f"http://testserver{path}")
    assert resp.status_code == 200, path
    body = resp.json()
    assert body["resource"] == f"{ISSUER}/mcp"
    assert [s.rstrip("/") for s in body["authorization_servers"]] == [ISSUER]


def test_unauthenticated_mcp_points_at_path_aware_metadata(client):
    resp = client.post("http://testserver/mcp", json={})
    assert resp.status_code == 401
    assert (
        f"{ISSUER}/.well-known/oauth-protected-resource/mcp"
        in resp.headers["www-authenticate"]
    )
