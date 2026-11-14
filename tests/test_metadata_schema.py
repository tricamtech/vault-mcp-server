"""Frontmatter schema 2026-11-13: lifecycle `status`, `progress`, `subject`, `as_of`.

Each test here was written against the pre-change parser and WATCHED FAIL before
the fix landed (see the branch's commit message for the run). They pin the
behaviours the vault audit found wrong on 2026-11-13:
  * a body `**Status:**` line overwrote the YAML status (16 files),
  * substring normalisation turned `draft` and `inactive` into Active,
  * invalid YAML frontmatter was dropped silently (63 tour-cache files),
  * the tour hint table named the wrong artists,
  * `historical` was not treated as stale by ranking.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from vault_mcp import parser
from vault_mcp.parser import (
    frontmatter_error,
    infer_subject,
    normalize_status,
    parse_document,
    parse_metadata,
    split_frontmatter,
)


def _parse(tmp_path: Path, rel: str, text: str):
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text, encoding="utf-8")
    return parse_document(tmp_path, rel)


# ----------------------------------------------------------------- status


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("current", "current"),
        ("Current", "current"),
        ("draft", "draft"),
        ("superseded", "superseded"),
        ("SUPERSEDED", "superseded"),
        ("archive", "archived"),
        ("archived", "archived"),
        ("historical", "historical"),
        ("active", "Active"),
        ("completed", "Done"),
        ("pending", "Pending"),
    ],
)
def test_normalize_status_exact_vocabulary(raw, expected):
    assert normalize_status(raw) == expected


@pytest.mark.parametrize("raw", ["inactive", "draft", "BUILT + SHIPPED", "done-ish, pending review"])
def test_normalize_status_no_substring_matching(raw):
    """`inactive` is not Active, `draft` is not Active, prose stays prose."""
    out = normalize_status(raw)
    assert out != "Active"
    if raw == "draft":
        assert out == "draft"
    elif raw == "inactive":
        assert out == "inactive"
    else:
        assert out == raw  # verbatim pass-through, counted as non-standard by vault_summary


def test_yaml_status_wins_over_body_status_line(tmp_path):
    doc = _parse(
        tmp_path,
        "shared/fleet/thing.md",
        "---\nstatus: current\n---\n# Thing\n\n**Status:** BUILT + SHIPPED 2026-10-01\n\nbody\n",
    )
    assert doc.metadata.status == "current"
    assert doc.metadata.progress == "BUILT + SHIPPED 2026-10-01"


def test_body_status_without_yaml_known_value_is_status(tmp_path):
    doc = _parse(tmp_path, "shared/x.md", "# X\n\n**Status:** completed\n")
    assert doc.metadata.status == "Done"
    assert doc.metadata.progress is None


def test_body_status_without_yaml_prose_is_progress(tmp_path):
    doc = _parse(tmp_path, "shared/x.md", "# X\n\n**Status:** waiting on Cameron for the DK budget\n")
    assert doc.metadata.status is None
    assert doc.metadata.progress == "waiting on Cameron for the DK budget"


def test_draft_handoff_is_draft_not_active(tmp_path):
    doc = _parse(
        tmp_path,
        "shared/handoffs/bob-vault-organization-remediation-2026-11-13.md",
        "---\nstatus: draft\ntype: handoff\n---\n# Handoff\n",
    )
    assert doc.metadata.status == "draft"


def test_explicit_progress_field_kept(tmp_path):
    doc = _parse(
        tmp_path,
        "shared/fleet/a.md",
        "---\nstatus: current\nprogress: phase 2 of 4\n---\n# A\n\n**Status:** ignored body prose\n",
    )
    assert doc.metadata.status == "current"
    assert doc.metadata.progress == "phase 2 of 4"


# ---------------------------------------------------------- new fields


def test_subject_as_of_superseded_by_parsed(tmp_path):
    doc = _parse(
        tmp_path,
        "shared/archive/old.md",
        "---\nstatus: superseded\nsubject: MTassist\nas_of: 2026-11-01\nsuperseded_by: shared/mastertour/eventric-api.md\n---\n# Old\n",
    )
    assert doc.metadata.subject == "mtassist"
    assert doc.metadata.as_of == "2026-11-01"
    assert doc.metadata.superseded_by == "shared/mastertour/eventric-api.md"


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("shared/tours/jk-handoff.md", "touring"),
        ("shared/tour-cache/JK/tour-state-JK.md", "touring"),
        ("shared/fleet/openclaw.md", "fleet"),
        ("shared/homelab/homelab.md", "homelab"),
        ("shared/mtassist/x.md", "mtassist"),
        ("shared/mastertour/eventric-api.md", "mtassist"),
        ("shared/business/shesynergy.md", "business"),
        ("shared/personal/actual-budget.md", "personal"),
        ("shared/comms/gmail-management.md", "comms"),
        ("shared/services.md", None),
        ("bob/admin-auto-memory/feedback_x.md", None),
    ],
)
def test_subject_directory_default(rel, expected):
    assert infer_subject(rel) == expected


def test_frontmatter_subject_beats_directory_default(tmp_path):
    doc = _parse(tmp_path, "shared/fleet/budget.md", "---\nsubject: business\n---\n# B\n")
    assert doc.metadata.subject == "business"


def test_mail_subject_frontmatter_is_not_a_vault_subject(tmp_path):
    doc = _parse(
        tmp_path,
        "shared/tour-cache/JK/tour-email-thread-JK-abc.md",
        "---\nsubject: Monitor engineer for Sept 6th show\n---\n# Thread\n",
    )
    assert doc.metadata.subject == "touring"  # directory default, not the mail subject
    assert doc.metadata.raw["subject"] == "Monitor engineer for Sept 6th show"


def test_subject_defaulted_on_parse(tmp_path):
    doc = _parse(tmp_path, "shared/tours/oa-va-handoff.md", "---\nstatus: historical\n---\n# OA\n")
    assert doc.metadata.subject == "touring"


# ---------------------------------------------------- invalid frontmatter


_BAD_YAML = (
    "---\nname: JK Email Thread 1a11\ndescription: Cached email thread summary for JK / Re: Hotel: Salt Lake\ntype: reference\n---\n# JK Email Thread\n"
)


def test_invalid_yaml_frontmatter_is_reported(tmp_path):
    assert split_frontmatter(_BAD_YAML)[0] == {}  # unchanged: parser still degrades gracefully
    err = frontmatter_error(_BAD_YAML)
    assert err and err.startswith("yaml:")
    doc = _parse(tmp_path, "shared/tour-cache/JK/tour-email-thread-JK-1a11.md", _BAD_YAML)
    assert doc.frontmatter_error == err


def test_valid_frontmatter_has_no_error(tmp_path):
    doc = _parse(tmp_path, "shared/x.md", "---\nstatus: current\n---\n# X\n")
    assert doc.frontmatter_error is None
    assert frontmatter_error("# no frontmatter at all\n") is None


def test_html_comment_before_frontmatter_is_not_frontmatter():
    text = "<!-- RETIRED -->\n---\nstatus: current\n---\n# OA\n"
    assert split_frontmatter(text)[0] == {}
    assert frontmatter_error(text) is None  # not malformed, just absent: the fix is in the file


# ------------------------------------------------------------- tour hints


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("shared/tours/jk-handoff.md", "jk"),
        ("shared/tours/jonah-kagen-notes.md", "jk"),
        ("shared/tours/oliver-anthony-va-run.md", "oa"),
        ("shared/tours/lisa-loeb-japan.md", "ll"),
        ("shared/tours/madeleine-peyroux-canada.md", "mp"),
        ("shared/tours/john-kim-notes.md", None),
        ("shared/tours/celtic-thunder.md", None),
    ],
)
def test_tour_hints_match_registry(rel, expected):
    assert parser._infer_tour(rel) == expected


# ---------------------------------------------------------------- ranking


def test_historical_is_stale_and_current_state_boost(server):
    base = server._freshness_weight("shared/tours/x.md", "current")
    assert server._freshness_weight("shared/tours/x.md", "historical") == pytest.approx(base * 0.8)
    assert server._freshness_weight("shared/tours/x.md", "superseded") == pytest.approx(base * 0.8)

    today = date.today().isoformat()
    old = (date.today() - timedelta(days=90)).isoformat()
    fresh = server._freshness_weight("shared/tours/jk-current-state.md", "current", "current-state", today)
    assert fresh == pytest.approx(base * 1.15)
    stale = server._freshness_weight("shared/tours/jk-current-state.md", "current", "current-state", old)
    assert stale == pytest.approx(base)
    # Only the current-state TYPE gets it; a heading or other doc type never does.
    assert server._freshness_weight("shared/tours/jk-handoff.md", "current", "handoff", today) == pytest.approx(base)


def test_create_handoff_default_status_is_lifecycle(server):
    import inspect

    sig = inspect.signature(server.vault_create_handoff)
    assert sig.parameters["status"].default == "current"


def test_vault_list_accepts_subject(server):
    import inspect

    assert "subject" in inspect.signature(server.vault_list).parameters
