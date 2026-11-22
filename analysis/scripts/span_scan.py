#!/usr/bin/env python3
"""Position-aware token scanning, shared by check_numbers.py and its
regression tests (test_make_tables.py).

A plain "does this superseded value appear anywhere in the file" exception
is too permissive: once a token is whitelisted for a file, it could also
be reintroduced into unrelated primary-result prose without the scanner
noticing. find_tokens_outside_spans() instead checks that every match of a
token is physically located inside one of a given list of allowed
(start, end) character-offset spans, so the token is still flagged if it
ever appears outside the specific descriptive-audit passage it was
whitelisted for.
"""
import re


def token_pattern(token):
    """Match TOKEN as a standalone number, not as part of a longer one."""
    return r"(?<![\d.])" + re.escape(token) + r"(?!\d)"


def span_of(text, start_marker, end_marker, search_from=0):
    """Return the (start, end) character offsets of the passage that runs
    from start_marker up to (but not including) end_marker. Raises
    ValueError via str.index if either marker is not found, so a marker
    that drifts out of sync with the document fails loudly rather than
    silently allowing nothing (or everything)."""
    i = text.index(start_marker, search_from)
    j = text.index(end_marker, i)
    return (i, j)


def find_tokens_outside_spans(text, token, spans):
    """Return the list of match objects for TOKEN in TEXT whose start
    position does not fall inside any of the given (start, end) spans.
    An empty return means every occurrence of TOKEN is accounted for by
    one of the allowed spans (including the case where TOKEN does not
    appear in TEXT at all)."""
    bad = []
    for m in re.finditer(token_pattern(token), text):
        if not any(s <= m.start() < e for s, e in spans):
            bad.append(m)
    return bad


def count_tokens_in_spans(text, token, spans):
    """Count occurrences of TOKEN in TEXT whose start position falls
    inside at least one of the given spans (used only for reporting)."""
    n = 0
    for m in re.finditer(token_pattern(token), text):
        if any(s <= m.start() < e for s, e in spans):
            n += 1
    return n
