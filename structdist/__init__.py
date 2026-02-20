"""
Algebraic Edit Distance (AED)
=============================

A universal, mathematically rigorous distance metric on structured data.

    d("hello", "hallo")           → 1.0     (reduces to Levenshtein)
    d([1, 2, 3], [1, 3])          → 1.0     (sequence deletion)
    d({"a": 1}, {"a": 2, "b": 3}) → 2.0     (map modification + insertion)

AED extends Levenshtein's 1965 insight — that the right distance between
strings is the minimum cost to transform one into the other — to ALL
structured data: sequences, mappings, tagged values, and any nesting
of these.

The result is a single metric that is:
  • A true metric (d≥0, d(x,x)=0, symmetric, triangle inequality)
  • Universal (any JSON/YAML/XML/AST/protobuf/config is representable)
  • Levenshtein-compatible (reduces exactly to edit distance on strings)
  • Actionable (produces minimum edit scripts, not just a number)
"""

from structdist.core import (
    # Types
    AVal,
    AAtom,
    ASeq,
    AMap,
    ATagged,
    # Distance
    distance,
    normalized_distance,
    # Edit operations
    EditOp,
    diff,
    patch,
)
from structdist.formats import (
    from_json, to_json, from_python, to_python, string_to_seq, seq_to_string,
)
from structdist.merge import merge, MergeResult, ConflictEntry

__version__ = "0.1.0"
__all__ = [
    "AVal", "AAtom", "ASeq", "AMap", "ATagged",
    "distance", "normalized_distance",
    "EditOp", "diff", "patch",
    "from_json", "to_json", "from_python", "to_python",
    "string_to_seq", "seq_to_string",
    "merge", "MergeResult", "ConflictEntry",
]
