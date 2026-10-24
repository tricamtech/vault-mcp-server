"""Wiki-link extraction, the link graph, and the vault_links / vault_dangling_links tools."""

import threading
import time

import pytest

from vault_mcp import linkgraph
from vault_mcp.linkgraph import NoteResolver, build_link_graph, cached_link_graph
from vault_mcp.parser import extract_wiki_links


@pytest.fixture(autouse=True)
def fresh_graph_cache():
    linkgraph.clear_cache()
    yield
    linkgraph.clear_cache()


def write(root, rel_path, text):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def vault(tmp_path):
    write(tmp_path, "index.md", "# Index\n\nSee [[alpha]], [[projects/beta]] and [[missing-note]].\n")
    write(tmp_path, "notes/alpha.md", "# Alpha\n\nBack to [[index|home]] and [[gamma#Details]].\n")
    write(tmp_path, "projects/beta.md", "# Beta\n\nRelated: [[alpha.md]] and [Gamma](../notes/gamma.md).\n")
    write(tmp_path, "notes/gamma.md", "# Gamma\n\n## Details\n\nNo links here.\n")
    return tmp_path


def wait_for_refresh():
    for thread in threading.enumerate():
        if thread.name == "linkgraph-refresh":
            thread.join(timeout=10)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_extracts_targets_with_alias_and_heading_stripped():
    text = "see [[alpha]] and [[beta|shown text]] and [[gamma#Heading]] and ![[delta]]"
    assert extract_wiki_links(text) == ["alpha", "beta", "gamma", "delta"]


def test_same_document_anchor_and_urls_are_not_links():
    assert extract_wiki_links("jump to [[#Heading]] or [[https://example.com]]") == []


def test_duplicates_collapse_in_order():
    assert extract_wiki_links("[[b]] [[a]] [[b|again]]") == ["b", "a"]


def test_bash_tests_and_citation_markup_are_not_links():
    text = (
        'if [[ "$(uname -r | cut -d. -f1)" -ge 13 ]]; then\n'
        "if [[ ${answer,,} =~ ^(y|yes)$ ]]; then\n"
        "[[ -f config.yaml ]] && echo found\n"
        '[["citation0","Some Product"]]\n'
        "footnote [[^]] marker\n"
        "[[ a -lt b ]] && [[ ! -f x ]]\n"
    )
    assert extract_wiki_links(text) == []


def test_padded_brackets_are_still_links():
    assert extract_wiki_links("see [[ note ]] and [[ my note ]]") == ["note", "my note"]


def test_code_fences_and_inline_code_are_not_links():
    text = "```bash\n[[ -d dir ]] && [[real]]\n```\n\nuse `[[name]]` syntax, then [[actual]]\n"
    assert extract_wiki_links(text) == ["actual"]


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_resolution_forms():
    resolver = NoteResolver(["index.md", "notes/alpha.md", "projects/beta.md", "work/projects/beta.md"])
    assert resolver.resolve("alpha") == "notes/alpha.md"  # basename
    assert resolver.resolve("alpha.md") == "notes/alpha.md"  # .md suffix optional
    assert resolver.resolve("notes/alpha") == "notes/alpha.md"  # full path
    assert resolver.resolve("notes/alpha.md") == "notes/alpha.md"
    assert resolver.resolve("work/projects/beta") == "work/projects/beta.md"
    assert resolver.resolve("Alpha") == "notes/alpha.md"  # case-insensitive fallback


def test_shared_basename_resolves_deterministically_to_shallowest():
    resolver = NoteResolver(["work/projects/beta.md", "projects/beta.md"])
    assert resolver.resolve("beta") == "projects/beta.md"
    # A path-style target that names the deeper file still reaches it.
    assert resolver.resolve("work/projects/beta") == "work/projects/beta.md"


def test_path_style_target_with_wrong_directory_is_dangling():
    resolver = NoteResolver(["notes/alpha.md"])
    assert resolver.resolve("archive/alpha") is None


def test_markdown_links_resolve_relative_to_source():
    resolver = NoteResolver(["notes/gamma.md", "projects/beta.md"])
    assert resolver.resolve("../notes/gamma.md", "projects/beta.md", linkgraph.MARKDOWN) == "notes/gamma.md"


def test_no_fuzzy_matching_across_separators():
    resolver = NoteResolver(["notes/my-note.md"])
    assert resolver.resolve("my_note") is None


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def test_forward_links(vault):
    graph, _ = build_link_graph(vault)
    targets = {(link.target_raw, link.target) for link in graph.forward_links("index.md")}
    assert targets == {
        ("alpha", "notes/alpha.md"),
        ("projects/beta", "projects/beta.md"),
        ("missing-note", None),
    }


def test_backlinks(vault):
    graph, _ = build_link_graph(vault)
    assert sorted(link.source for link in graph.backlinks_for("notes/alpha.md")) == [
        "index.md",
        "projects/beta.md",
    ]
    # Wiki link with a heading and a relative markdown link both count.
    assert sorted(link.source for link in graph.backlinks_for("notes/gamma.md")) == [
        "notes/alpha.md",
        "projects/beta.md",
    ]
    assert [link.source for link in graph.backlinks_for("index.md")] == ["notes/alpha.md"]


def test_dangling_links(vault):
    graph, _ = build_link_graph(vault)
    assert [(link.source, link.target_raw) for link in graph.dangling] == [("index.md", "missing-note")]


def test_padded_links_resolve(vault):
    write(vault, "notes/my note.md", "# My note\n")
    write(vault, "padded.md", "# Padded\n\n[[ alpha ]] and [[ my note ]]\n")
    graph, _ = build_link_graph(vault)
    assert [link.target for link in graph.forward_links("padded.md")] == [
        "notes/alpha.md",
        "notes/my note.md",
    ]


def test_bash_brackets_in_a_note_do_not_become_dangling_links(vault):
    write(vault, "scripts.md", '# Scripts\n\nif [[ "$x" -ge 13 ]]; then\n[[ -f x ]] && run\n[[ a -lt b ]]\n')
    graph, _ = build_link_graph(vault)
    assert [link.target_raw for link in graph.dangling] == ["missing-note"]


def test_cache_is_invalidated_by_a_change(vault):
    graph, _ = build_link_graph(vault)
    assert cached_link_graph(vault) is graph
    time.sleep(0.01)
    write(vault, "missing-note.md", "# Now it exists\n")
    assert cached_link_graph(vault) is None
    rebuilt, _ = build_link_graph(vault)
    assert rebuilt.dangling == []


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_vault_links_tool(server, tmp_path):
    write(tmp_path, "a.md", "# A\n\n[[b]] and [[nowhere]]\n")
    write(tmp_path, "sub/b.md", "# B\n\n[[a]]\n")
    out = server.vault_links("b")
    assert "# Links for sub/b.md" in out
    assert "## Forward links (1)\n- a.md" in out
    assert "## Backlinks (1)\n- a.md" in out

    out = server.vault_links("a.md", direction="forward")
    assert "- sub/b.md" in out
    assert "DANGLING `nowhere`" in out
    assert "Backlinks" not in out

    assert server.vault_links("nope").startswith("Error: file not found")


def test_vault_dangling_links_tool(server, tmp_path):
    write(tmp_path, "a.md", "# A\n\n[[ghost]] [[ghost|again]] [x](gone.md)\n")
    write(tmp_path, "sub/b.md", "# B\n\n[[ghost]]\n")
    out = server.vault_dangling_links()
    assert "2 reference(s) to 1 missing target(s), wiki" in out
    assert "- `ghost` — 2 ref(s) from: a.md, sub/b.md" in out
    assert "gone.md" not in out
    assert "gone.md" in server.vault_dangling_links(kind="markdown")
    assert "from: sub/b.md" in server.vault_dangling_links(source_prefix="sub/")


def test_vault_read_shows_backlinks_once_graph_is_warm(server, tmp_path):
    write(tmp_path, "a.md", "# A\n\n[[b]]\n")
    write(tmp_path, "b.md", "# B\n")
    # Cold cache: the read never waits for a build, it schedules one instead.
    assert "Referenced by" not in server.vault_read("b.md")
    wait_for_refresh()
    assert "**Referenced by:** a.md" in server.vault_read("b.md")


def test_vault_read_backlinks_kill_switch(server, tmp_path, monkeypatch):
    write(tmp_path, "a.md", "# A\n\n[[b]]\n")
    write(tmp_path, "b.md", "# B\n")
    build_link_graph(tmp_path)
    monkeypatch.setattr(server, "VAULT_READ_BACKLINKS", False)
    assert "Referenced by" not in server.vault_read("b.md")


@pytest.mark.parametrize("tool", ["write", "edit"])
def test_graph_refreshes_in_background_after_a_write(server, tmp_path, tool):
    write(tmp_path, "a.md", "# A\n\nplaceholder\n")
    write(tmp_path, "target.md", "# Target\n")
    build_link_graph(tmp_path)
    time.sleep(0.01)

    if tool == "write":
        server.vault_write("a.md", "# A\n\nnow links to [[target]]\n")
    else:
        server.vault_edit("a.md", "placeholder", "now links to [[target]]")
    wait_for_refresh()

    # No reader has asked for the graph since the write, yet it is warm again
    # and already reflects the new link.
    graph = cached_link_graph(tmp_path)
    assert graph is not None
    assert [link.source for link in graph.backlinks_for("target.md")] == ["a.md"]
