"""vault_search fuses the semantic channel with the keyword channel."""

import asyncio


def hit(path, section, content, score, start_line=1):
    return {
        "path": path,
        "title": path.removesuffix(".md"),
        "section": section,
        "content": content,
        "score": score,
        "start_line": start_line,
    }


class FakeCollection:
    def __init__(self, chunks):
        self.chunks = chunks

    def get(self, include):
        return {
            "ids": [str(i) for i in range(len(self.chunks))],
            "documents": [c["content"] for c in self.chunks],
            "metadatas": [
                {"doc_path": c["path"], "doc_title": c["title"], "section_header": c["section"],
                 "start_line": c["start_line"]}
                for c in self.chunks
            ],
        }


class FakeIndexer:
    """Semantic channel that returns fixed hits, over a fixed chunk store."""

    is_ready = True

    def __init__(self, chunks, semantic):
        self.collection = FakeCollection(chunks)
        self.semantic = semantic

    async def search(self, query, top_k):
        return self.semantic[:top_k]


NOISE = [
    hit("weather.md", "", "Notes about the weather this spring and summer.", 0.62),
    hit("recipes.md", "Soup", "## Soup\n\nA soup recipe with leeks and potatoes.", 0.60),
]
CODENAME = hit("vendors.md", "Zorbatek", "## Zorbatek\n\nSubcontractor, invoices quarterly.", None, start_line=5)


def search(server, monkeypatch, chunks, semantic, query, top_k=5):
    monkeypatch.setattr(server, "indexer", FakeIndexer(chunks, semantic))
    return asyncio.run(server.vault_search(query, top_k=top_k))


def test_exact_term_missed_by_embeddings_is_found_and_ranked_first(server, monkeypatch):
    # The semantic channel only returns noise for a made-up name
    out = search(server, monkeypatch, NOISE + [CODENAME], NOISE, "Zorbatek")
    assert "**File:** vendors.md" in out
    assert out.index("vendors.md") < out.index("weather.md")
    assert "**Score:** exact-term match  |  **Matched by:** keyword" in out


def test_chunk_found_by_both_channels_is_merged(server, monkeypatch):
    both = hit("vendors.md", "Zorbatek", CODENAME["content"], 0.58, start_line=5)
    out = search(server, monkeypatch, NOISE + [CODENAME], NOISE + [both], "Zorbatek")
    assert out.count("vendors.md") == 1
    assert "**Score:** 0.580  |  **Matched by:** meaning+keyword" in out
    assert out.index("vendors.md") < out.index("weather.md")


def test_without_keyword_hits_semantic_order_is_unchanged(server, monkeypatch):
    out = search(server, monkeypatch, NOISE + [CODENAME], NOISE, "how do penguins sleep")
    assert "vendors.md" not in out
    assert out.index("weather.md") < out.index("recipes.md")
    assert "**Matched by:** meaning" in out


def test_keyword_weight_zero_is_semantic_only(server, monkeypatch):
    monkeypatch.setattr(server, "KEYWORD_WEIGHT", 0.0)
    out = search(server, monkeypatch, NOISE + [CODENAME], NOISE, "Zorbatek")
    assert "vendors.md" not in out


def test_keyword_channel_failure_falls_back_to_semantic(server, monkeypatch):
    class Broken:
        def get(self, include):
            raise RuntimeError("boom")

    fake = FakeIndexer(NOISE, NOISE)
    fake.collection = Broken()
    monkeypatch.setattr(server, "indexer", fake)
    monkeypatch.setattr(server, "_keyword_index", server.lexical.KeywordIndex())
    out = asyncio.run(server.vault_search("Zorbatek", top_k=5))
    assert "weather.md" in out
