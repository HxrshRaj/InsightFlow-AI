"""Binary model file (MDL1) with an explicit, self-describing byte order.

Why: IBM Z (s390x) is big-endian while x86 and ARM are little-endian, so a model file written
with the writer's native order silently corrupts when read on the other. This format writes
a marker so the reader detects the order instead of assuming it.

Layout (all integers/floats in the file's declared byte order):
  magic      4 bytes  b"MDL1"
  marker     u32      0x01020304 as written; a reader that sees 0x04030201 knows to swap
  version    u32      1
  n_features u32
  bias       f64
  per term:  u16 byte-length, UTF-8 bytes          (n_features times, in column order)
  idf        f64 * n_features
  coef       f64 * n_features
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

MAGIC = b"MDL1"
MARKER = 0x01020304
VERSION = 1


@dataclass
class LinearTextModel:
    terms: list[str]
    idf: list[float]
    coef: list[float]
    bias: float


def write(model: LinearTextModel, byteorder: str = "little") -> bytes:
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be 'little' or 'big'")
    p = "<" if byteorder == "little" else ">"
    n = len(model.terms)
    if not (len(model.idf) == len(model.coef) == n):
        raise ValueError("terms, idf and coef must have the same length")
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack(p + "III", MARKER, VERSION, n))
    buf.write(struct.pack(p + "d", model.bias))
    for t in model.terms:
        b = t.encode("utf-8")
        buf.write(struct.pack(p + "H", len(b)))
        buf.write(b)
    buf.write(struct.pack(f"{p}{n}d", *model.idf))
    buf.write(struct.pack(f"{p}{n}d", *model.coef))
    return buf.getvalue()


def read(data: bytes) -> LinearTextModel:
    if data[:4] != MAGIC:
        raise ValueError("not an MDL1 file")
    marker_le = struct.unpack("<I", data[4:8])[0]
    if marker_le == MARKER:
        p = "<"
    elif marker_le == 0x04030201:
        p = ">"
    else:
        raise ValueError("corrupt byte-order marker")
    version, n = struct.unpack(p + "II", data[8:16])
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    (bias,) = struct.unpack(p + "d", data[16:24])
    pos, terms = 24, []
    for _ in range(n):
        (ln,) = struct.unpack(p + "H", data[pos:pos + 2])
        pos += 2
        terms.append(data[pos:pos + ln].decode("utf-8"))
        pos += ln
    idf = list(struct.unpack(f"{p}{n}d", data[pos:pos + 8 * n]))
    pos += 8 * n
    coef = list(struct.unpack(f"{p}{n}d", data[pos:pos + 8 * n]))
    if pos + 8 * n != len(data):
        raise ValueError("trailing or missing bytes")
    return LinearTextModel(terms, idf, coef, bias)
