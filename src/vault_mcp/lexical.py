"""Keyword (BM25) channel for vault search, fused with the semantic channel.

Why (2026-10-13): embeddings carry almost no signal for made-up words and IDs.
"Forosoco" is in 31 files and semantic search returned NOTHING; "15604df1",
"00117", "koiva", "Club Quattro" all missed. Names, hosts, venues and IDs are
the most common thing agents look up, so exact matching runs beside meaning.

Kept deliberately small and dependency-free: an in-memory inverted index over
the chunk texts already stored in Chroma, rebuilt from the collection every
REFRESH_SECONDS (the semantic index is the source of truth; this mirrors it).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, defaultdict

REFRESH_SECONDS = 120
_K1, _B = 1.2, 0.75

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._\-/]*[a-z0-9]|[a-z0-9]")
_STOP = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is it its
me my no not of on or our so than that the their them then there these they this to
was we were what when where which who why will with you your about all any did get got
just more most much now should some up use used using via vs out over also only per
""".split())


def _norm(token: str) -> str:
    token = token.strip("._-/")
    if token.isalpha() and len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Whole tokens (keeps `ct122-respawn.sh`, `sensor.argus_report`, `ct/106`)
    plus their parts, so `respawn` also finds `ct122-respawn.sh`."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tok = _norm(raw)
        if not tok:
            continue
        out.append(tok)
        if re.search(r"[._\-/]", tok):
            out.extend(_norm(p) for p in re.split(r"[._\-/]+", tok) if p and _norm(p) != tok)
    return out


def query_terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for tok in tokenize(query):
        if tok and tok not in _STOP:
            seen.setdefault(tok, None)
    return list(seen)


def is_identifier(term: str) -> bool:
    """Looks like a name/ID rather than an English word or an everyday number.

    Yes: punctuation inside (`ct/106`, `known-issues.json`), letters mixed with
    digits (`15604df1`, `ct122`), long or zero-padded numbers (`00117`).
    No: years and small numbers (`1998`, `17`) -- these made "replace a timing
    belt on a 1998 Honda Civic" match any chunk mentioning 1998.
    """
    if re.search(r"[._\-/]", term):
        return True
    has_digit, has_alpha = bool(re.search(r"\d", term)), bool(re.search(r"[a-z]", term))
    if has_digit and has_alpha:
        return True
    return term.isdigit() and (len(term) >= 5 or (term.startswith("0") and len(term) >= 3))


class KeywordIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._building = False
        self._built_at = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._avg_len = 1.0
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []

    @property
    def ready(self) -> bool:
        return bool(self._ids)

    def stale(self) -> bool:
        return time.monotonic() - self._built_at > REFRESH_SECONDS

    def build(self, ids: list[str], docs: list[str], metas: list[dict]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for idx, (doc, meta) in enumerate(zip(docs, metas)):
            # Title and section path count as text too: a file named after the
            # thing you search for should match even if the body never repeats it.
            toks = tokenize(f"{meta.get('doc_title', '')} {meta.get('section_header', '')} {meta.get('doc_path', '')} {doc}")
            lengths.append(len(toks))
            for tok, tf in Counter(toks).items():
                postings[tok].append((idx, tf))
        with self._lock:
            self._postings = dict(postings)
            self._lengths = lengths
            self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
            self._ids, self._docs, self._metas = ids, docs, metas
            self._built_at = time.monotonic()

    def refresh_from(self, collection) -> None:
        """Rebuild from a Chroma collection. Safe to call from a worker thread;
        concurrent calls collapse into one."""
        with self._lock:
            if self._building:
                return
            self._building = True
        try:
            got = collection.get(include=["documents", "metadatas"])
            self.build(got["ids"], got["documents"], got["metadatas"])
        finally:
            with self._lock:
                self._building = False

    def search(self, query: str, limit: int = 100) -> list[dict]:
        """BM25 over chunks, GATED so nonsense stays quiet.

        A chunk qualifies only if it contains EVERY query term, or contains an
        identifier-shaped term from the query (digits/punctuation inside, e.g.
        `00117`, `ct/106`, `known-issues.json`). Plain-English queries that only
        brush against a word or two are left to the semantic channel and its
        relevance floor; without this gate "how do penguins sleep" would surface
        any chunk that says "sleep".
        """
        terms = query_terms(query)
        with self._lock:
            postings, lengths, avg = self._postings, self._lengths, self._avg_len
            ids, docs, metas = self._ids, self._docs, self._metas
        if not terms or not ids:
            return []
        n = len(ids)
        present = [t for t in terms if t in postings]
        if not present:
            return []
        ident_terms = {t for t in present if is_identifier(t)}
        matched: dict[int, set[str]] = defaultdict(set)
        scores: dict[int, float] = defaultdict(float)
        for term in present:
            plist = postings[term]
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for idx, tf in plist:
                denom = tf + _K1 * (1 - _B + _B * lengths[idx] / avg)
                scores[idx] += idf * tf * (_K1 + 1) / denom
                matched[idx].add(term)
        need = set(terms)
        out = []
        for idx, score in scores.items():
            hit_terms = matched[idx]
            if hit_terms >= need or (hit_terms & ident_terms):
                meta = metas[idx]
                out.append({
                    "id": ids[idx],
                    "path": meta.get("doc_path", ""),
                    "title": meta.get("doc_title", ""),
                    "section": meta.get("section_header", ""),
                    "content": docs[idx],
                    "bm25": score,
                    "start_line": meta.get("start_line", 0),
                    "sensitivity": meta.get("sensitivity", ""),
                    "status": meta.get("status", ""),
                })
        out.sort(key=lambda r: r["bm25"], reverse=True)
        return out[:limit]
