"""The single tokenizer spec, shared by Python training and the Go service.

Rule: lowercase ASCII letters only (A-Z), treat every byte that is not [a-z0-9] as a separator,
keep tokens of length >= 2. Unigrams and bigrams (two adjacent tokens joined by one space).
Deliberately ASCII-only so Go and Python agree exactly (Unicode lowercasing and \\w differ).
"""
from __future__ import annotations

_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


def tokens(text: str) -> list[str]:
    data = text.encode("utf-8").translate(bytes.maketrans(
        bytes(range(ord("A"), ord("Z") + 1)), bytes(range(ord("a"), ord("z") + 1))))
    out, cur = [], bytearray()
    for b in data:
        if 97 <= b <= 122 or 48 <= b <= 57:
            cur.append(b)
        else:
            if len(cur) >= 2:
                out.append(cur.decode())
            cur = bytearray()
    if len(cur) >= 2:
        out.append(cur.decode())
    return out


def analyze(text: str) -> list[str]:
    t = tokens(text)
    return t + [f"{a} {b}" for a, b in zip(t, t[1:])]
