"""The filesystem watcher and the full walk agree on what belongs in the index."""

import asyncio

import pytest

from vault_mcp.indexer import VaultIndexer, is_indexable


@pytest.mark.parametrize(
    "rel_path, expected",
    [
        ("notes.md", True),
        ("projects/plan.md", True),
        ("deep/nested/dir/page.md", True),
        (".obsidian/workspace.md", False),
        ("_review/draft.md", False),
        ("_inbox.md", False),
        ("shared/_scratch/x.md", False),
        ("dir/.md", False),
        ("NOTES.MD", False),
        ("Mixed.Md", False),
        ("notes.txt", False),
        ("notes.md.bak", False),
        ("", False),
    ],
)
def test_is_indexable(rel_path, expected):
    assert is_indexable(rel_path) is expected


TREE = [
    "notes.md",
    "projects/plan.md",
    "NOTES.MD",
    "projects/Upper.Md",
    ".obsidian/workspace.md",
    "_review/draft.md",
    "_inbox.md",
    "shared/_scratch/x.md",
    "readme.txt",
]


def make_indexer(tmp_path, **kwargs):
    vault = tmp_path / "vault"
    for rel in TREE:
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text("# x\n", encoding="utf-8")
    return VaultIndexer(vault, tmp_path / "chroma", "unused-model", **kwargs)


def test_walk_indexes_exactly_the_expected_files(tmp_path):
    indexer = make_indexer(tmp_path, watch_files=False)
    assert indexer._list_vault_files() == ["notes.md", "projects/plan.md"]


def test_predicate_matches_walk_for_every_file(tmp_path):
    # If the watcher accepts a path the walk does not list, the watcher indexes it
    # and the next index_all treats it as deleted (and vice versa).
    indexer = make_indexer(tmp_path, watch_files=False)
    walked = set(indexer._list_vault_files())
    for p in indexer.vault_path.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(indexer.vault_path))
            assert is_indexable(rel) == (rel in walked), rel


def test_stop_cancels_the_watcher(tmp_path):
    indexer = make_indexer(tmp_path)

    async def go():
        indexer._watch_task = asyncio.create_task(indexer._watch_vault())
        task = indexer._watch_task
        await asyncio.sleep(0.1)
        await indexer.stop()
        assert task.cancelled()
        assert indexer._watch_task is None
        await indexer.stop()  # idempotent

    asyncio.run(go())
