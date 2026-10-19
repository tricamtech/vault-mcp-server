"""Keyword channel: identifiers are found, plain-English nonsense stays quiet."""

import pytest

from vault_mcp import lexical
from vault_mcp.lexical import is_identifier

DOCS = [
    "Vendor entity Zorbatek Holdings LLC (subcontractor)",
    "Backup of vm/204 is 14 days stale on the storage node",
    "database migration 00042 applied",
    "we sleep late after the show",
    "penguin plush in the merch bin",
    "ran deploy-web.sh which picked the wrong release",
]


@pytest.fixture
def index():
    k = lexical.KeywordIndex()
    metas = [{"doc_path": f"d{i}.md", "doc_title": "", "section_header": ""} for i in range(len(DOCS))]
    k.build([str(i) for i in range(len(DOCS))], DOCS, metas)
    return k


def paths(index, query):
    return [r["path"] for r in index.search(query)]


def test_made_up_name_is_found(index):
    assert paths(index, "Zorbatek") == ["d0.md"]


def test_identifiers_are_found(index):
    assert paths(index, "vm/204") == ["d1.md"]
    assert paths(index, "00042") == ["d2.md"]
    assert paths(index, "migration 00042") == ["d2.md"]


def test_part_of_a_compound_token_matches(index):
    assert "d5.md" in paths(index, "deploy")


def test_partial_plain_english_overlap_is_gated_out(index):
    # "sleep" and "penguin" each appear, but no chunk holds every term
    assert paths(index, "how do penguins sleep") == []
    assert paths(index, "breeding cycle of Antarctic emperor penguins") == []


def test_title_and_path_are_searchable():
    k = lexical.KeywordIndex()
    k.build(["0"], ["body text without the name"], [{"doc_path": "projects/zorbatek.md", "doc_title": "Zorbatek"}])
    assert [r["path"] for r in k.search("zorbatek")] == ["projects/zorbatek.md"]


@pytest.mark.parametrize("term", ["00042", "a3f9c2e1", "vm/204", "host12", "release-notes.json"])
def test_identifier_shapes(term):
    assert is_identifier(term)


@pytest.mark.parametrize("term", ["1998", "17", "1929", "penguin"])
def test_years_small_numbers_and_words_are_not_identifiers(term):
    assert not is_identifier(term)


def test_years_and_small_numbers_do_not_open_the_gate():
    k = lexical.KeywordIndex()
    k.build(["0", "1"], ["bought the 1998 tour van", "group 17 rehearsal"], [{"doc_path": "a.md"}, {"doc_path": "b.md"}])
    assert k.search("replace a timing belt on a 1998 Honda Civic") == []
    assert k.search("the periodic table group 17") == []


def test_refresh_from_collection():
    class Collection:
        def get(self, include):
            return {"ids": ["0"], "documents": ["see ticket 00042"], "metadatas": [{"doc_path": "a.md"}]}

    k = lexical.KeywordIndex()
    assert not k.ready
    k.refresh_from(Collection())
    assert k.ready and not k.stale()
    assert [r["path"] for r in k.search("00042")] == ["a.md"]
