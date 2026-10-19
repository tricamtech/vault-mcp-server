"""vault_search returns one hit per file, never a heading-only chunk."""

import asyncio


class FakeIndexer:
    is_ready = True

    def __init__(self, hits):
        self.hits = hits
        self.requested = None

    async def search(self, query, top_k):
        self.requested = top_k
        return self.hits[:top_k]


def hit(path, section, content, score):
    return {
        "path": path,
        "title": path.removesuffix(".md"),
        "section": section,
        "content": content,
        "score": score,
        "start_line": 1,
    }


BODY = "A paragraph with enough real text in it to answer the question asked."

HITS = [
    hit("long.md", "Steps", "## Steps", 0.91),
    hit("long.md", "Details", f"## Details\n\n{BODY}", 0.90),
    hit("long.md", "More", f"## More\n\n{BODY}", 0.89),
    hit("long.md", "Even more", f"## Even more\n\n{BODY}", 0.88),
    hit("other.md", "", f"# Other\n\n{BODY}", 0.80),
    hit("third.md", "Notes", f"## Notes\n\n{BODY}", 0.70),
]


def run_search(server, monkeypatch, hits, top_k):
    fake = FakeIndexer(hits)
    monkeypatch.setattr(server, "indexer", fake)
    return asyncio.run(server.vault_search("question", top_k=top_k)), fake


def test_each_file_appears_once(server, monkeypatch):
    out, _ = run_search(server, monkeypatch, HITS, top_k=3)
    assert out.count("**File:** long.md") == 1
    assert "**File:** other.md" in out
    assert "**File:** third.md" in out


def test_files_rank_by_best_chunk_score(server, monkeypatch):
    out, _ = run_search(server, monkeypatch, HITS, top_k=3)
    assert out.index("long.md") < out.index("other.md") < out.index("third.md")
    # The file keeps its best score even though a lower chunk is shown
    assert "**Score:** 0.910" in out


def test_heading_only_chunk_never_stands_in_for_its_file(server, monkeypatch):
    out, _ = run_search(server, monkeypatch, HITS, top_k=3)
    assert "### 1. long > Details" in out
    assert "```markdown\n## Steps\n```" not in out


def test_other_matching_sections_are_named(server, monkeypatch):
    out, _ = run_search(server, monkeypatch, HITS, top_k=3)
    assert "_Other matching sections in this file: Steps, More, Even more_" in out


def test_top_k_counts_files_not_chunks(server, monkeypatch):
    out, fake = run_search(server, monkeypatch, HITS, top_k=2)
    assert fake.requested > 2  # over-fetches chunks
    assert "(2 matches)" in out
    assert "third.md" not in out


def test_file_with_only_thin_chunks_still_shows(server, monkeypatch):
    out, _ = run_search(server, monkeypatch, [hit("a.md", "Title", "## Title", 0.5)], top_k=5)
    assert "**File:** a.md" in out
