import tempfile
from pathlib import Path
from vault_mcp.parser import extract_wiki_links
from vault_mcp.linkgraph import build_link_graph
def test_real_links_kept():
    assert extract_wiki_links("see [[jk-handoff]] and [[feedback-expense-sheets|expenses]] and [[people.cameron#Phone]]") == ["jk-handoff", "feedback-expense-sheets", "people.cameron"]
def test_bash_and_citation_brackets_ignored():
    text = 'if [[ "$(get_os_version_major)" -ge 13 ]]; then\nif [[ ${prompt,,} =~ ^(y|yes)$ ]]; then\n[["turn0product1","Souper Cubes"]]\nx [[^]] y'
    assert extract_wiki_links(text) == []
def test_shell_tests_ignored():
    # `[[ a || b ]]` would otherwise split on the alias bar and yield "a".
    assert extract_wiki_links("[[ -f x ]] && [[ ! -d y ]]\n[[ a -lt b ]]\n[[ a || b ]]") == []
def test_padded_links_kept():
    assert extract_wiki_links("see [[ note ]] and [[ my note ]]") == ["note", "my note"]
def test_link_graph_uses_the_same_rule():
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp)
        (vault / "note.md").write_text("# Note\n")
        (vault / "a.md").write_text("# A\n\n[[ -f x ]] [[ ! -d y ]] [[ a -lt b ]] [[ a || b ]] [[ note ]] [[missing]]\n")
        graph = build_link_graph(vault, use_cache=False)
        assert [link.target_raw for link in graph.forward_links("a.md")] == ["note", "missing"]
        assert [link.target_raw for link in graph.dangling] == ["missing"]
if __name__ == "__main__":
    test_real_links_kept(); test_bash_and_citation_brackets_ignored(); test_shell_tests_ignored(); test_padded_links_kept(); test_link_graph_uses_the_same_rule(); print("WIKILINK TESTS PASSED")
