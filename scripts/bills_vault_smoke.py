#!/usr/bin/env python3
"""Smoke tests for Bill's Vault."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path("/root/vault-mcp-server-v3")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vault_mcp.catalog import build_inventory  # noqa: E402
from vault_mcp.parser import is_semantic_index_blocked, parse_document  # noqa: E402


DEFAULT_VAULT = Path("/root/claude-memory")
DEFAULT_NEW_CHROMA = Path("/var/lib/vault-mcp/chroma_repo_local")
DEFAULT_PUBLIC_CHROMA = Path("/var/lib/vault-mcp/chroma_repo_public")
DEFAULT_OLD_CHROMA = Path("/var/lib/vault-mcp/chroma_local")
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def curl_status(url: str, resolve: str | None = None) -> tuple[int | None, str]:
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", url]
    if resolve:
        cmd[1:1] = ["--resolve", resolve]
    proc = run_command(cmd)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or f"curl failed with {proc.returncode}"
    try:
        return int(proc.stdout.strip()), ""
    except ValueError:
        return None, f"unexpected curl output: {proc.stdout.strip()!r}"


def dig_first_ip(hostname: str) -> tuple[str | None, str]:
    proc = run_command(["dig", "@1.1.1.1", "+short", hostname])
    if proc.returncode != 0:
        return None, proc.stderr.strip() or f"dig failed with {proc.returncode}"
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line and ":" not in line:
            return line, ""
    return None, "no IPv4 answer from dig"


def get_collection_count(path: Path) -> int | None:
    if not path.exists():
        return None
    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_collection("vault_chunks")
    except Exception:
        return None
    return collection.count()


def path_chunk_count(path: Path, doc_path: str) -> int:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection("vault_chunks")
    result = collection.get(where={"doc_path": doc_path}, include=[])
    return len(result.get("ids", []))


def semantic_top_paths(path: Path, model: SentenceTransformer, query: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection("vault_chunks")
    embedding = model.encode(query).tolist()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    rows: list[dict] = []
    for idx, meta in enumerate(result["metadatas"][0]):
        distance = result["distances"][0][idx]
        rows.append(
            {
                "path": meta.get("doc_path", ""),
                "title": meta.get("doc_title", ""),
                "section": meta.get("section_header", ""),
                "score": round(1.0 - (distance / 2.0), 4),
                "sensitivity": meta.get("sensitivity", ""),
            }
        )
    return rows


def collect_metrics(vault_root: Path) -> dict:
    inventory = build_inventory(vault_root)
    repo_md_files = sum(1 for _ in vault_root.rglob("*.md"))
    duplicates = sum(1 for rel, canonical in inventory.duplicate_map.items() if rel != canonical)
    return {
        "repo_md_files": repo_md_files,
        "canonical_files": len(inventory.canonical_files),
        "alias_count": len(inventory.alias_map),
        "duplicate_suppressed": duplicates,
    }


def build_report(args: argparse.Namespace) -> dict:
    checks: list[CheckResult] = []
    metrics = collect_metrics(args.vault_root)
    inventory = build_inventory(args.vault_root)

    local_status, local_error = curl_status(args.local_url)
    checks.append(
        CheckResult(
            "local_mcp",
            local_status == 400,
            local_error or f"expected 400 from {args.local_url}, got {local_status}",
        )
    )

    oauth_local_status, oauth_local_error = curl_status(args.public_local_url)
    checks.append(
        CheckResult(
            "public_local_oauth",
            oauth_local_status == 401,
            oauth_local_error or f"expected 401 from {args.public_local_url}, got {oauth_local_status}",
        )
    )

    external_ip, external_ip_error = dig_first_ip(args.public_host)
    if external_ip:
        external_status, external_error = curl_status(
            f"https://{args.public_host}/mcp",
            resolve=f"{args.public_host}:443:{external_ip}",
        )
        checks.append(
            CheckResult(
                "public_external_oauth",
                external_status == 401,
                external_error or f"expected 401 from external {args.public_host}, got {external_status}",
            )
        )
    else:
        checks.append(CheckResult("public_external_oauth", False, external_ip_error))

    alias_expectations = {
        "tour-registry.md": "shared/reference/tour-registry.md",
        "HANDOFF-BB.md": "shared/HANDOFF-BB.md",
        "people.cameron.md": "shared/people.cameron.md",
    }
    alias_report = {}
    for probe, expected in alias_expectations.items():
        actual = inventory.resolve(probe)
        alias_report[probe] = actual
        checks.append(
            CheckResult(
                f"alias:{probe}",
                actual == expected,
                f"expected {expected}, got {actual}",
            )
        )

    metadata_expectations = {
        "HANDOFF-BB.md": {"scope": "tour", "doc_type": "handoff", "tour": "bb"},
        "people.cameron.md": {"scope": "personal", "sensitivity": "restricted"},
        "bob/MEMORY.md": {"scope": "ops", "doc_type": "mission-control", "sensitivity": "secret"},
    }
    metadata_report = {}
    for probe, expected in metadata_expectations.items():
        rel_path = inventory.resolve(probe)
        document = parse_document(args.vault_root, rel_path)
        actual = asdict(document.metadata)
        metadata_report[probe] = actual
        ok = all(actual.get(key) == value for key, value in expected.items())
        checks.append(
            CheckResult(
                f"metadata:{probe}",
                ok,
                f"expected subset {expected}, got scope={actual.get('scope')} type={actual.get('doc_type')} "
                f"tour={actual.get('tour')} sensitivity={actual.get('sensitivity')}",
            )
        )

    old_chunks = get_collection_count(args.old_chroma)
    new_chunks = get_collection_count(args.new_chroma)
    public_chunks = get_collection_count(args.public_chroma)

    blocked_docs = []
    for rel_path in inventory.canonical_files:
        document = parse_document(args.vault_root, rel_path)
        if is_semantic_index_blocked(document.metadata.sensitivity):
            blocked_docs.append(rel_path)

    blocked_chunk_report = {}
    for rel_path in blocked_docs:
        count = path_chunk_count(args.new_chroma, rel_path)
        blocked_chunk_report[rel_path] = count
    blocked_zero = all(count == 0 for count in blocked_chunk_report.values())
    checks.append(
        CheckResult(
            "blocked_docs_not_indexed",
            blocked_zero,
            f"expected 0 chunks for blocked docs, got {blocked_chunk_report}",
        )
    )

    semantic_report = {}
    if not args.quick:
        model = SentenceTransformer(args.model, local_files_only=True)
        semantic_tests = [
            {
                "name": "argus_audit_precision",
                "query": "porch light control dead motion branch automation audit",
                "expect_new": {"argus/memory/automation-audit-2026-03-17.md"},
            },
            {
                "name": "bob_vault_docs_precision",
                "query": "vault mcp cross-device memory setup cloudflare tunnel oauth bob",
                "expect_new": {
                    "bob/memory/vault-mcp-detailed-setup.md",
                    "bob/auto-memory/project_vault_mcp_setup.md",
                },
            },
            {
                "name": "restricted_personal_not_returned",
                "query": "Cameron passport number",
                "forbid_new_suffixes": {"people.cameron.md", "bob/MEMORY.md", "air/MEMORY.md", "mbp/MEMORY.md"},
            },
        ]

        for test in semantic_tests:
            new_rows = semantic_top_paths(args.new_chroma, model, test["query"])
            old_rows = semantic_top_paths(args.old_chroma, model, test["query"]) if args.old_chroma.exists() else []
            semantic_report[test["name"]] = {
                "query": test["query"],
                "old": old_rows,
                "new": new_rows,
            }

            if "expect_new" in test:
                new_paths = {row["path"] for row in new_rows}
                ok = bool(new_paths & test["expect_new"])
                checks.append(
                    CheckResult(
                        f"semantic:{test['name']}",
                        ok,
                        f"expected one of {sorted(test['expect_new'])}, got {sorted(new_paths)}",
                    )
                )

            if "forbid_new_suffixes" in test:
                forbidden = [
                    row["path"]
                    for row in new_rows
                    if any(row["path"].endswith(suffix) for suffix in test["forbid_new_suffixes"])
                ]
                checks.append(
                    CheckResult(
                        f"semantic:{test['name']}",
                        not forbidden,
                        f"forbidden sensitive hits present: {forbidden}",
                    )
                )

    overall_ok = all(check.ok for check in checks)
    return {
        "ok": overall_ok,
        "checks": [asdict(check) for check in checks],
        "metrics": metrics,
        "chunk_counts": {
            "old_shared_local": old_chunks,
            "new_repo_local": new_chunks,
            "new_repo_public": public_chunks,
        },
        "aliases": alias_report,
        "metadata": metadata_report,
        "blocked_docs": blocked_docs,
        "blocked_chunk_counts": blocked_chunk_report,
        "semantic": semantic_report,
    }


def print_human(report: dict):
    print("Bill's Vault smoke test")
    print(f"Overall: {'PASS' if report['ok'] else 'FAIL'}")
    print()

    print("Metrics")
    for key, value in report["metrics"].items():
        print(f"- {key}: {value}")
    for key, value in report["chunk_counts"].items():
        print(f"- {key}: {value}")
    print()

    print("Checks")
    for check in report["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        print(f"- {status} {check['name']}: {check['detail']}")
    print()

    if report["semantic"]:
        print("Semantic")
        for name, payload in report["semantic"].items():
            print(f"- {name}: {payload['query']}")
            for row in payload["new"]:
                print(
                    f"  new -> {row['path']} [{row['score']}]"
                    + (f" sensitivity={row['sensitivity']}" if row["sensitivity"] else "")
                )
            if payload["old"]:
                for row in payload["old"]:
                    print(
                        f"  old -> {row['path']} [{row['score']}]"
                        + (f" sensitivity={row['sensitivity']}" if row["sensitivity"] else "")
                    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke tests for Bill's Vault.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")
    parser.add_argument("--quick", action="store_true", help="Skip semantic retrieval tests.")
    parser.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--new-chroma", type=Path, default=DEFAULT_NEW_CHROMA)
    parser.add_argument("--public-chroma", type=Path, default=DEFAULT_PUBLIC_CHROMA)
    parser.add_argument("--old-chroma", type=Path, default=DEFAULT_OLD_CHROMA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--public-host", default="mem.tricamhome.com")
    parser.add_argument("--local-url", default="http://127.0.0.1:8092/mcp")
    parser.add_argument("--public-local-url", default="http://127.0.0.1:8091/mcp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
