"""
structdist.formats — Convert between real-world data and algebraic types.

Supported conversions:
    • Python objects (dict, list, str, int, float, bool, None) ↔ AVal
    • JSON strings ↔ AVal
    • Character-level string → ASeq of AAtom (for Levenshtein equivalence)
"""

import json
from typing import Any

from .core import AAtom, AMap, ASeq, ATagged, AVal


# ═══════════════════════════════════════════════════════════════════
#  PYTHON OBJECTS ↔ ALGEBRAIC VALUES
# ═══════════════════════════════════════════════════════════════════

def from_python(obj: Any) -> AVal:
    """
    Convert a Python object to an algebraic value.

    Mapping:
        str        → AAtom(str)
        int/float  → AAtom(number)
        bool       → AAtom(bool)
        None       → AAtom(None)
        list/tuple → ASeq(...)
        dict       → AMap(...)

    Nested structures are converted recursively.
    """
    if obj is None:
        return AAtom(None)
    if isinstance(obj, bool):  # Must check before int (bool is subclass of int)
        return AAtom(obj)
    if isinstance(obj, (int, float)):
        return AAtom(obj)
    if isinstance(obj, str):
        return AAtom(obj)
    if isinstance(obj, bytes):
        return AAtom(obj)
    if isinstance(obj, (list, tuple)):
        return ASeq(tuple(from_python(item) for item in obj))
    if isinstance(obj, dict):
        return AMap({str(k): from_python(v) for k, v in obj.items()})

    # Fallback: convert to string representation
    return AAtom(str(obj))


def to_python(val: AVal) -> Any:
    """
    Convert an algebraic value back to a plain Python object.

    Inverse of from_python:
        to_python(from_python(obj)) == obj
    for JSON-compatible objects.
    """
    if isinstance(val, AAtom):
        return val.val
    if isinstance(val, ASeq):
        return [to_python(item) for item in val.items]
    if isinstance(val, AMap):
        return {k: to_python(v) for k, v in val.entries.items()}
    if isinstance(val, ATagged):
        return {"__tag__": val.tag, "__value__": to_python(val.inner)}
    raise TypeError(f"Unknown AVal type: {type(val)}")


# ═══════════════════════════════════════════════════════════════════
#  JSON STRINGS ↔ ALGEBRAIC VALUES
# ═══════════════════════════════════════════════════════════════════

def from_json(text: str) -> AVal:
    """Parse a JSON string into an algebraic value."""
    return from_python(json.loads(text))


def to_json(val: AVal, **kwargs) -> str:
    """Convert an algebraic value to a JSON string."""
    return json.dumps(to_python(val), **kwargs)


# ═══════════════════════════════════════════════════════════════════
#  CHARACTER-LEVEL STRING CONVERSION  (for Levenshtein equivalence)
# ═══════════════════════════════════════════════════════════════════

def string_to_seq(s: str) -> ASeq:
    """
    Convert a string into a Seq of single-character Atoms.

    This is the bridge that proves AED reduces to Levenshtein:

        distance(string_to_seq("kitten"), string_to_seq("sitting"))

    computes exactly the Levenshtein distance (with unit costs).
    """
    return ASeq(tuple(AAtom(c) for c in s))


def seq_to_string(val: ASeq) -> str:
    """
    Convert a Seq of single-character Atoms back to a string.

    Inverse of string_to_seq.
    """
    return "".join(
        item.val if isinstance(item, AAtom) and isinstance(item.val, str) else "?"
        for item in val.items
    )
