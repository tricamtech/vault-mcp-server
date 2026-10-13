"""Checks for the 2026-10-13 vault changes. Run: PYTHONPATH=src python tests/test_vault_changes.py"""
import tempfile, textwrap
from pathlib import Path
from vault_mcp.chunker import chunk_document, _iter_heading_sections
from vault_mcp import lexical

# chunker: comments in fences are not headings
body = textwrap.dedent("""\
    # Nyx upgrade
    Some intro text about the upgrade plan for the node.

    ## Steps
    ```bash
    # 3. On nyx, restore from backup
    qmrestore /mnt/backup/vzdump-qemu-101.vma.zst 101
    # 4. Start it
    qm start 101
    ```

    ## Empty heading

    ## Real section
    This section has actual words in it that answer something.
    """)
secs = _iter_heading_sections(body.splitlines())
names = [h for h, _, _ in secs]
assert not any("On nyx, restore" in n for n in names), names
assert not any(n.endswith("Empty heading") for n in names), names
assert any(n.endswith("Steps") for n in names) and any(n.endswith("Real section") for n in names), names
steps = next(l for h, _, l in secs if h.endswith("Steps"))
assert any("qmrestore" in l for l in steps) and any("# 4. Start it" in l for l in steps)

with tempfile.TemporaryDirectory() as d:
    Path(d, "n.md").write_text(body)
    chunks = chunk_document(Path(d), "n.md")
    assert all(c.content.strip().splitlines()[-1].strip() for c in chunks)
    assert not any(c.content.strip().startswith("#") and len(c.content.strip().splitlines()) == 1 for c in chunks), [c.content for c in chunks]

# lexical: identifiers found, nonsense quiet
k = lexical.KeywordIndex()
docs = ["Artist entity Forosoco Music LLC (loanout)", "PBS ct/106 backup 14 days stale on koiva",
        "migration 00117 applied", "we sleep late after the show", "penguin plush in the merch bin",
        "ran ct122-respawn.sh which picked the wrong transcript"]
metas = [{"doc_path": f"d{i}.md", "doc_title": "", "section_header": ""} for i in range(len(docs))]
k.build([str(i) for i in range(len(docs))], docs, metas)
hit = lambda q: [r["path"] for r in k.search(q)]
assert hit("Forosoco") == ["d0.md"]
assert hit("ct/106") == ["d1.md"] and hit("koiva") == ["d1.md"]
assert hit("migration 00117") == ["d2.md"] and hit("00117") == ["d2.md"]
assert hit("how do penguins sleep") == [], hit("how do penguins sleep")
assert hit("breeding cycle of Antarctic emperor penguins") == []
assert "d5.md" in hit("respawn")
print("ALL CHECKS PASSED")

# years / small numbers are not identifiers (2026-10-13 junk leak)
from vault_mcp.lexical import is_identifier
assert not is_identifier("1998") and not is_identifier("17") and not is_identifier("1929")
assert is_identifier("00117") and is_identifier("15604df1") and is_identifier("ct/106") and is_identifier("ct122")
k2 = lexical.KeywordIndex()
k2.build(["0", "1"], ["bought the 1998 tour van", "group 17 rehearsal"], [{"doc_path": "a.md"}, {"doc_path": "b.md"}])
assert k2.search("replace a timing belt on a 1998 Honda Civic") == []
assert k2.search("the periodic table group 17") == []
print("NUMBER CHECKS PASSED")
