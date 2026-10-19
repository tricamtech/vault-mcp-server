from vault_mcp.parser import extract_wiki_links
def test_real_links_kept():
    assert extract_wiki_links("see [[jk-handoff]] and [[feedback-expense-sheets|expenses]] and [[people.cameron#Phone]]") == ["jk-handoff", "feedback-expense-sheets", "people.cameron"]
def test_bash_and_citation_brackets_ignored():
    text = 'if [[ "$(get_os_version_major)" -ge 13 ]]; then\nif [[ ${prompt,,} =~ ^(y|yes)$ ]]; then\n[["turn0product1","Souper Cubes"]]\nx [[^]] y'
    assert extract_wiki_links(text) == []
if __name__ == "__main__":
    test_real_links_kept(); test_bash_and_citation_brackets_ignored(); print("WIKILINK TESTS PASSED")
