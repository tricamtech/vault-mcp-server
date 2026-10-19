"""Chunker: H2 sectioning, code fences, and heading-only sections."""

from vault_mcp.chunker import chunk_document


def _chunks(tmp_path, text):
    (tmp_path / "doc.md").write_text(text, encoding="utf-8")
    return chunk_document(tmp_path, "doc.md")


DOC_WITH_FENCE = """\
# Restore runbook

Intro paragraph that explains what this runbook is for.

## Restore steps

Run these on the backup host:

```bash
## 1. stop the service first
systemctl stop app
## 2. restore from backup
restore --latest /srv/app
```

## Verify

Check that the service answers on its health endpoint.
"""


def test_h2_inside_code_fence_is_not_a_heading(tmp_path):
    chunks = _chunks(tmp_path, DOC_WITH_FENCE)
    headers = [c.section_header for c in chunks]
    assert headers == ["preamble", "Restore steps", "Verify"]
    steps = chunks[1]
    # The whole code block stays inside its real section
    assert "## 1. stop the service first" in steps.content
    assert "restore --latest /srv/app" in steps.content
    assert "```" in steps.content


def test_tilde_fences_are_recognised(tmp_path):
    chunks = _chunks(
        tmp_path,
        "## Example\n\nA sample document:\n\n~~~markdown\n## Not a heading\nbody\n~~~\n",
    )
    assert [c.section_header for c in chunks] == ["Example"]


def test_start_lines_point_at_the_heading(tmp_path):
    chunks = _chunks(tmp_path, DOC_WITH_FENCE)
    lines = DOC_WITH_FENCE.splitlines()
    for c in chunks[1:]:
        assert lines[c.start_line - 1] == f"## {c.section_header}"


def test_heading_only_section_is_skipped(tmp_path):
    chunks = _chunks(
        tmp_path,
        "# Title\n\n"
        "Some preamble text that is long enough to keep.\n\n"
        "## A parent heading with nothing under it\n\n"
        "## Child\n\n"
        "The actual content lives here, under the child heading.\n",
    )
    assert [c.section_header for c in chunks] == ["preamble", "Child"]


def test_section_with_only_subheadings_is_skipped(tmp_path):
    chunks = _chunks(
        tmp_path,
        "## Overview of everything\n### Details to follow\n\n"
        "## Real section\n\nWith a body that is long enough to index.\n",
    )
    assert [c.section_header for c in chunks] == ["Real section"]


def test_file_without_h2_is_one_chunk(tmp_path):
    chunks = _chunks(
        tmp_path,
        "# Notes\n\nJust a paragraph.\n\n```\n## inside a fence\n```\n",
    )
    assert len(chunks) == 1
    assert chunks[0].section_header == ""
