"""
structdist.core — Algebraic Edit Distance
==========================================

MATHEMATICAL FRAMEWORK
══════════════════════

§1  THE PROBLEM
───────────────

Levenshtein (1965) defined the edit distance between two strings as
the minimum number of single-character insertions, deletions, and
substitutions needed to transform one string into the other.  This
became one of the most widely used concepts in computer science.

But real data isn't flat strings.  It's JSON objects, YAML configs,
XML documents, abstract syntax trees, protocol buffers, database
schemas.  These are STRUCTURED: they contain ordered sequences,
unordered key-value mappings, tagged variants, and arbitrary nesting.

No existing metric handles all of these in one coherent framework:

    • Tree edit distance (Zhang-Shasha 1989, RTED 2011):
      Handles ORDERED, LABELED TREES only.  Cannot express unordered
      mappings (JSON objects), mixed ordered/unordered children, or
      type-level differences.

    • JSON diff tools (jsondiff, deepdiff, json-patch):
      Compute edit PATCHES but not a well-defined DISTANCE METRIC.
      No triangle inequality guarantee.  No optimality guarantee.
      No formal algebraic framework.

    • Graph edit distance:
      NP-hard in general.  No polynomial algorithm for arbitrary graphs.

AED fills this gap: a single distance function on algebraic data types
that is a true metric, is polynomial-time computable, reduces to
Levenshtein on strings, and handles the full heterogeneity of real
structured data.


§2  ALGEBRAIC DATA TYPES
────────────────────────

DEFINITION (Algebraic Value):
The set V of algebraic values is the smallest set satisfying:

    (1)  Atom(v)                    ∈ V   for any primitive v
    (2)  Seq(a₁, ..., aₙ)          ∈ V   for a₁,...,aₙ ∈ V, n ≥ 0
    (3)  Map({k₁:v₁, ..., kₙ:vₙ}) ∈ V   for kᵢ strings, vᵢ ∈ V
    (4)  Tagged(tag, inner)         ∈ V   for tag string, inner ∈ V

This captures:
    JSON array     → Seq of values
    JSON object    → Map of string→value
    JSON string    → Atom("hello")
    JSON number    → Atom(42)
    JSON null      → Atom(None)
    XML element    → Tagged(tag_name, Seq(attributes_map, children_seq))
    AST node       → Tagged(node_type, Seq(children))
    Protobuf msg   → Map(field_name → value)
    YAML           → same as JSON (JSON is a YAML subset)
    Config file    → Map of string→value (nested)

Key design choice: Map is UNORDERED (like a set of key-value pairs),
Seq is ORDERED (like a list).  This distinction is what existing tree
edit distance cannot express — it treats all children as ordered.


§3  THE DISTANCE FUNCTION
─────────────────────────

DEFINITION (Algebraic Edit Distance):
d: V × V → ℝ≥0 is defined recursively:

CASE 1: Atoms
    d(Atom(a), Atom(b)) = atom_dist(a, b)
    where:
        atom_dist(s₁, s₂) = levenshtein(s₁, s₂) / max(|s₁|, |s₂|)
                             if both are strings  (normalized to [0,1])
        atom_dist(n₁, n₂) = min(1, |n₁ - n₂| / max(|n₁|, |n₂|, 1))
                             if both are numbers
        atom_dist(a, a)    = 0 if values are identical
        atom_dist(a, b)    = 1 otherwise

CASE 2: Sequences (THE CORE — generalizes Levenshtein)
    d(Seq(a₁..aₘ), Seq(b₁..bₙ)) = min cost alignment via DP:

        D[0][0] = 0
        D[i][0] = D[i-1][0] + del_cost(aᵢ)
        D[0][j] = D[0][j-1] + ins_cost(bⱼ)
        D[i][j] = min(
            D[i-1][j]   + del_cost(aᵢ),              # delete aᵢ
            D[i][j-1]   + ins_cost(bⱼ),              # insert bⱼ
            D[i-1][j-1] + sub_cost(aᵢ, bⱼ),          # substitute
        )

    where:
        del_cost(x) = size(x)     (cost proportional to what's removed)
        ins_cost(x) = size(x)     (cost proportional to what's added)
        sub_cost(x, y) = d(x, y)  (RECURSIVE!)

    Final distance = D[m][n].

    NOTE: When the elements of the sequence are single characters
    (Atom('h'), Atom('e'), Atom('l'), ...), and del/ins cost = 1,
    and sub_cost = 0 or 1 (match/mismatch), THIS IS EXACTLY
    LEVENSHTEIN DISTANCE.

CASE 3: Maps (unordered key-value comparison)
    d(Map(A), Map(B)) = Σ  del_cost(A[k])      for k ∈ keys(A) \\ keys(B)
                       + Σ  ins_cost(B[k])      for k ∈ keys(B) \\ keys(A)
                       + Σ  d(A[k], B[k])       for k ∈ keys(A) ∩ keys(B)

    This is O(|keys|) in the number of keys, times the recursive cost.

CASE 4: Tagged values
    d(Tagged(t₁, a), Tagged(t₂, b)) =
        (0 if t₁ = t₂ else TAG_MISMATCH_COST) + d(a, b)

CASE 5: Type mismatches
    d(X, Y) = size(X) + size(Y)   when X and Y are different types
    (delete X entirely, insert Y entirely — maximum disruption)

where size(v) is:
    size(Atom(v))        = 1
    size(Seq(a₁..aₙ))   = 1 + Σ size(aᵢ)
    size(Map(kv pairs))  = 1 + Σ (1 + size(vᵢ))
    size(Tagged(t, v))   = 1 + size(v)


§4  METRIC PROPERTIES
─────────────────────

THEOREM (AED is a metric):
d satisfies:
    (i)   d(x, y) ≥ 0                      (non-negativity)
    (ii)  d(x, y) = 0  ⟺  x = y           (identity of indiscernibles)
    (iii) d(x, y) = d(y, x)                (symmetry)
    (iv)  d(x, z) ≤ d(x, y) + d(y, z)     (triangle inequality)

PROOF SKETCH:
(i), (ii): By construction — all costs are ≥ 0, and equal values
           produce zero cost at every level.
(iii): The sequence DP is symmetric (swap insert/delete).
       Map comparison is symmetric.  Atom distance is symmetric.
(iv):  For sequences, this follows from the optimality of the DP
       alignment and the fact that any path x→y→z through y
       produces a valid (not necessarily optimal) alignment x→z.
       For maps, triangle inequality holds component-wise.
       For atoms, inherited from the underlying metrics (Levenshtein
       satisfies triangle inequality; absolute difference does too). ∎


§5  COMPUTATIONAL COMPLEXITY
────────────────────────────

For two values X, Y:
    • Atoms: O(|s₁|·|s₂|) for string atoms (Levenshtein), O(1) otherwise
    • Sequences of length m, n: O(m·n·T) where T is the recursive cost
    • Maps with k keys: O(k·T)
    • Tagged: O(T)

For flat JSON (depth 1): O(m·n) or O(k) — same as Levenshtein.
For JSON of depth d with branching factor b: O((b²)^d) worst case.
In practice (depth ≤ 5, branching ≤ 100): milliseconds.

Memoization is available when subtrees are shared (DAG structure).


§6  WHAT IS NEW
───────────────

What IS new:
    1. The algebraic data type formulation as the universal domain
       for structured edit distance.
    2. The specific handling of MIXED ordered (Seq) and unordered (Map)
       structures within a single metric.
    3. The recursive substitution cost in sequence alignment
       (d(aᵢ, bⱼ) as sub cost, not just 0/1).
    4. The size-weighted insertion/deletion costs that make the metric
       sensitive to the COMPLEXITY of what's added/removed.
    5. The formal proof that this is a true metric.
    6. Reduction to Levenshtein as a special case (not an analogy —
       an actual mathematical reduction).

What is NOT new:
    • Edit distance / Levenshtein distance (1965)
    • Dynamic programming for sequence alignment
    • Tree edit distance (Tai 1979, Zhang-Shasha 1989)
    • JSON diff/patch (RFC 6902)
    • The concept of algebraic data types (type theory)

AED is to structured data what Levenshtein is to strings: the
NATURAL, CANONICAL distance metric.


§7  APPLICATIONS
────────────────

    • Configuration drift detection: d(deployed_config, intended_config)
    • API compatibility scoring: d(schema_v1, schema_v2)
    • Database schema evolution: minimum migration cost
    • Code similarity / plagiarism: d(AST₁, AST₂)
    • Anomaly detection: flag records with d(record, template) > threshold
    • Test oracle: d(actual_output, expected_output) with structural awareness
    • Merge conflicts: three-way merge using edit scripts
    • Clustering: group similar configs/schemas/documents by AED
    • Search: find nearest neighbor in a database of structured objects

Author: structdist contributors
License: MIT
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Union


# ═══════════════════════════════════════════════════════════════════
#  ALGEBRAIC DATA TYPES
# ═══════════════════════════════════════════════════════════════════

class AVal:
    """Base class for algebraic values.  Not instantiated directly."""
    __slots__ = ()

    def size(self) -> float:
        """Structural size of this value (for cost weighting)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AAtom(AVal):
    """
    A primitive/leaf value: string, number, bool, None, or bytes.

    Examples:
        AAtom("hello")
        AAtom(42)
        AAtom(3.14)
        AAtom(True)
        AAtom(None)
    """
    val: Any

    def size(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return f"AAtom({self.val!r})"


@dataclass(frozen=True, slots=True)
class ASeq(AVal):
    """
    An ordered sequence of algebraic values.

    This is the structure that generalizes strings — and where
    AED reduces to Levenshtein distance.

    Examples:
        ASeq((AAtom(1), AAtom(2), AAtom(3)))           # [1, 2, 3]
        ASeq((AAtom('h'), AAtom('e'), AAtom('l')))      # "hel" as chars
    """
    items: tuple[AVal, ...]

    def size(self) -> float:
        return 1.0 + sum(item.size() for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        if len(self.items) <= 5:
            return f"ASeq({list(self.items)})"
        return f"ASeq([{self.items[0]!r}, ..., {self.items[-1]!r}] len={len(self.items)})"


@dataclass(frozen=True, slots=True)
class AMap(AVal):
    """
    An UNORDERED mapping of string keys to algebraic values.

    This is the structure that captures JSON objects, YAML mappings,
    config file sections, etc.  Key ordering does NOT affect distance.

    Examples:
        AMap({"name": AAtom("Alice"), "age": AAtom(30)})
    """
    entries: dict[str, AVal]

    def __init__(self, entries: dict[str, AVal]):
        # frozenset for hashability while keeping dict interface
        object.__setattr__(self, 'entries', dict(entries))

    def __hash__(self):
        return hash(frozenset((k, id(v)) for k, v in self.entries.items()))

    def size(self) -> float:
        return 1.0 + sum(1.0 + v.size() for v in self.entries.values())

    def __repr__(self) -> str:
        if len(self.entries) <= 3:
            return f"AMap({self.entries})"
        return f"AMap({{...}} len={len(self.entries)})"


@dataclass(frozen=True, slots=True)
class ATagged(AVal):
    """
    A tagged/labeled value — captures XML elements, AST nodes,
    enum variants, type wrappers.

    Examples:
        ATagged("div", ASeq((AAtom("Hello"), AAtom("world"))))
        ATagged("BinaryExpr", ASeq((left, AAtom("+"), right)))
    """
    tag: str
    inner: AVal

    def size(self) -> float:
        return 1.0 + self.inner.size()

    def __repr__(self) -> str:
        return f"ATagged({self.tag!r}, {self.inner!r})"


# ═══════════════════════════════════════════════════════════════════
#  ATOM DISTANCE (base case)
# ═══════════════════════════════════════════════════════════════════

def _levenshtein(s: str, t: str) -> int:
    """Standard Levenshtein distance between two strings."""
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m

    # Space-optimized DP (two rows)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def atom_distance(a: AAtom, b: AAtom) -> float:
    """
    Distance between two atomic values.

    Returns a value in [0, 1]:
        0 = identical
        1 = completely different

    String atoms use normalized Levenshtein distance.
    Numeric atoms use normalized absolute difference.
    Type mismatches return 1.0.
    """
    av, bv = a.val, b.val

    # Bool/non-bool type mismatch check MUST come first.
    # In Python, bool is a subclass of int (True == 1, False == 0),
    # so without this guard, bools and ints would be conflated.
    a_is_bool = type(av) is bool
    b_is_bool = type(bv) is bool
    if a_is_bool != b_is_bool:
        return 1.0  # bool vs non-bool is always a type mismatch

    # Both bools (checked before general equality to avoid int coercion)
    if a_is_bool and b_is_bool:
        return 0.0 if av is bv else 1.0

    # Identical values (safe now that bool/int conflation is excluded)
    if av == bv:
        return 0.0

    # Both strings → normalized Levenshtein
    if isinstance(av, str) and isinstance(bv, str):
        if not av and not bv:
            return 0.0
        return _levenshtein(av, bv) / max(len(av), len(bv))

    # Both numbers → normalized absolute difference
    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
        denom = max(abs(av), abs(bv), 1)
        return min(1.0, abs(av - bv) / denom)

    # One or both None
    if av is None and bv is None:
        return 0.0
    if av is None or bv is None:
        return 1.0

    # Type mismatch between atoms
    return 1.0


# ═══════════════════════════════════════════════════════════════════
#  CORE DISTANCE FUNCTION
# ═══════════════════════════════════════════════════════════════════

# Tag mismatch cost
TAG_MISMATCH_COST = 1.0


def distance(a: AVal, b: AVal) -> float:
    """
    Algebraic Edit Distance between two structured values.

    This is the central function of the library.  It computes the
    minimum-cost transformation from `a` to `b` using:
        • Insertion of a subtree (cost = size of subtree)
        • Deletion of a subtree (cost = size of subtree)
        • Recursive substitution (cost = d(old, new))

    For sequences, this uses Levenshtein-style DP with recursive
    substitution costs.  For maps, it compares shared keys recursively
    and penalizes missing/extra keys.

    Returns a non-negative float.  The metric is:
        • d(x, x) = 0
        • d(x, y) = d(y, x)
        • d(x, z) ≤ d(x, y) + d(y, z)
    """
    # Same object shortcut
    if a is b:
        return 0.0

    # Same type — dispatch to type-specific logic
    if type(a) is type(b):
        if isinstance(a, AAtom):
            return atom_distance(a, b)
        if isinstance(a, ASeq):
            return _seq_distance(a, b)
        if isinstance(a, AMap):
            return _map_distance(a, b)
        if isinstance(a, ATagged):
            return _tagged_distance(a, b)

    # Type mismatch — must delete a and insert b
    return a.size() + b.size()


def _seq_distance(a: ASeq, b: ASeq) -> float:
    """
    Edit distance between two sequences using DP alignment.

    THIS IS THE GENERALIZATION OF LEVENSHTEIN.

    Standard Levenshtein uses cost 1 for insert/delete and 0 or 1
    for substitution (match vs mismatch of single characters).

    AED generalizes:
        • Insert/delete cost = size of the element (a subtree, not just
          a character — inserting {"users": [...]} costs more than
          inserting 42)
        • Substitution cost = distance(old_element, new_element),
          computed RECURSIVELY (so modifying a nested object pays
          proportional to what actually changed, not a flat 0-or-1)

    When elements are single-character AAtom values and we use
    unit costs, this reduces EXACTLY to Levenshtein distance.
    """
    m = len(a.items)
    n = len(b.items)

    if m == 0:
        return sum(item.size() for item in b.items)
    if n == 0:
        return sum(item.size() for item in a.items)

    # DP table: dp[i][j] = distance between a.items[:i] and b.items[:j]
    # Space optimization: keep only two rows
    a_sizes = [item.size() for item in a.items]
    b_sizes = [item.size() for item in b.items]

    prev = [0.0] * (n + 1)
    for j in range(1, n + 1):
        prev[j] = prev[j - 1] + b_sizes[j - 1]

    curr = [0.0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = prev[0] + a_sizes[i - 1]

        for j in range(1, n + 1):
            # Delete a[i-1]
            del_cost = prev[j] + a_sizes[i - 1]

            # Insert b[j-1]
            ins_cost = curr[j - 1] + b_sizes[j - 1]

            # Substitute a[i-1] → b[j-1] (RECURSIVE)
            sub_cost = prev[j - 1] + distance(a.items[i - 1], b.items[j - 1])

            curr[j] = min(del_cost, ins_cost, sub_cost)

        prev, curr = curr, prev

    return prev[n]


def _map_distance(a: AMap, b: AMap) -> float:
    """
    Distance between two unordered mappings.

    For keys present in both: recursively compare values.
    For keys in only one: penalize by value size + 1 (key itself).
    """
    a_keys = set(a.entries.keys())
    b_keys = set(b.entries.keys())

    total = 0.0

    # Keys only in a → deleted
    for k in a_keys - b_keys:
        total += 1.0 + a.entries[k].size()

    # Keys only in b → inserted
    for k in b_keys - a_keys:
        total += 1.0 + b.entries[k].size()

    # Shared keys → recursive comparison
    for k in a_keys & b_keys:
        total += distance(a.entries[k], b.entries[k])

    return total


def _tagged_distance(a: ATagged, b: ATagged) -> float:
    """
    Distance between two tagged values.

    Tag mismatch costs TAG_MISMATCH_COST.
    Inner values are compared recursively.
    """
    tag_cost = 0.0 if a.tag == b.tag else TAG_MISMATCH_COST
    return tag_cost + distance(a.inner, b.inner)


def normalized_distance(a: AVal, b: AVal) -> float:
    """
    Normalized AED in [0, 1].

    0.0 = identical structures
    1.0 = completely different (delete everything, insert everything)

    Normalized by max(size(a), size(b)) to give a scale-independent
    similarity score.
    """
    d = distance(a, b)
    if d == 0.0:
        return 0.0
    max_size = max(a.size(), b.size())
    if max_size == 0.0:
        return 0.0
    return min(1.0, d / max_size)


# ═══════════════════════════════════════════════════════════════════
#  EDIT SCRIPT (diff / patch)
# ═══════════════════════════════════════════════════════════════════

class EditOp(Enum):
    """Types of edit operations."""
    EQUAL = auto()      # No change
    INSERT = auto()     # Insert new value
    DELETE = auto()     # Delete existing value
    REPLACE = auto()    # Replace value (with recursive sub-diff)
    MAP_ADD = auto()    # Add key to map
    MAP_DEL = auto()    # Remove key from map
    MAP_MOD = auto()    # Modify value for existing key
    TAG_MOD = auto()    # Tag changed


@dataclass
class EditEntry:
    """A single edit operation in an edit script."""
    op: EditOp
    path: tuple[Union[int, str], ...]  # Path from root to this edit point
    old: Optional[AVal] = None
    new: Optional[AVal] = None
    children: Optional[list["EditEntry"]] = None  # Sub-diff for REPLACE

    def __repr__(self) -> str:
        path_str = "/".join(str(p) for p in self.path) or "(root)"
        if self.op == EditOp.EQUAL:
            return f"EQUAL at {path_str}"
        if self.op == EditOp.INSERT:
            return f"INSERT at {path_str}: {self.new!r}"
        if self.op == EditOp.DELETE:
            return f"DELETE at {path_str}: {self.old!r}"
        if self.op == EditOp.REPLACE:
            return f"REPLACE at {path_str}: {self.old!r} → {self.new!r}"
        if self.op == EditOp.MAP_ADD:
            return f"MAP_ADD at {path_str}: {self.new!r}"
        if self.op == EditOp.MAP_DEL:
            return f"MAP_DEL at {path_str}: {self.old!r}"
        if self.op == EditOp.MAP_MOD:
            return f"MAP_MOD at {path_str}"
        if self.op == EditOp.TAG_MOD:
            return f"TAG_MOD at {path_str}: {self.old!r} → {self.new!r}"
        return f"{self.op} at {path_str}"


def diff(a: AVal, b: AVal, path: tuple = ()) -> list[EditEntry]:
    """
    Compute the minimum edit script to transform `a` into `b`.

    Returns a list of EditEntry objects describing the transformation.
    This is the AED equivalent of Levenshtein's trace-back — not just
    "how far apart?" but "HOW to get from A to B?"
    """
    if a is b or (type(a) is type(b) and isinstance(a, AAtom) and a.val == b.val):
        return [EditEntry(EditOp.EQUAL, path, old=a, new=b)]

    # Same type dispatch
    if type(a) is type(b):
        if isinstance(a, AAtom):
            return [EditEntry(EditOp.REPLACE, path, old=a, new=b)]

        if isinstance(a, ASeq):
            return _seq_diff(a, b, path)

        if isinstance(a, AMap):
            return _map_diff(a, b, path)

        if isinstance(a, ATagged):
            return _tagged_diff(a, b, path)

    # Type mismatch
    return [EditEntry(EditOp.REPLACE, path, old=a, new=b)]


def _seq_diff(a: ASeq, b: ASeq, path: tuple) -> list[EditEntry]:
    """
    Trace back through the DP matrix to produce the minimum edit script
    for sequence alignment.  This is exactly Levenshtein trace-back,
    generalized to structured elements.
    """
    m = len(a.items)
    n = len(b.items)

    a_sizes = [item.size() for item in a.items]
    b_sizes = [item.size() for item in b.items]

    # Full DP table (needed for trace-back)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + b_sizes[j - 1]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + a_sizes[i - 1]

    # Precompute substitution costs
    sub_costs = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sub_costs[i][j] = distance(a.items[i], b.items[j])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            del_cost = dp[i - 1][j] + a_sizes[i - 1]
            ins_cost = dp[i][j - 1] + b_sizes[j - 1]
            sub_cost = dp[i - 1][j - 1] + sub_costs[i - 1][j - 1]
            dp[i][j] = min(del_cost, ins_cost, sub_cost)

    # Trace back
    ops: list[EditEntry] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_c = sub_costs[i - 1][j - 1]
            if abs(dp[i][j] - (dp[i - 1][j - 1] + sub_c)) < 1e-9:
                # Substitution (or match)
                child_path = path + (i - 1,)
                if sub_c < 1e-9:
                    ops.append(EditEntry(EditOp.EQUAL, child_path,
                                         old=a.items[i - 1], new=b.items[j - 1]))
                else:
                    sub_ops = diff(a.items[i - 1], b.items[j - 1], child_path)
                    ops.append(EditEntry(EditOp.REPLACE, child_path,
                                         old=a.items[i - 1], new=b.items[j - 1],
                                         children=sub_ops))
                i -= 1
                j -= 1
                continue

        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + a_sizes[i - 1])) < 1e-9:
            ops.append(EditEntry(EditOp.DELETE, path + (i - 1,), old=a.items[i - 1]))
            i -= 1
            continue

        if j > 0:
            ops.append(EditEntry(EditOp.INSERT, path + (j - 1,), new=b.items[j - 1]))
            j -= 1
            continue

        break  # safety

    ops.reverse()
    return ops


def _map_diff(a: AMap, b: AMap, path: tuple) -> list[EditEntry]:
    """Diff two maps: identify added, removed, and modified keys."""
    ops: list[EditEntry] = []
    a_keys = set(a.entries.keys())
    b_keys = set(b.entries.keys())

    for k in sorted(a_keys - b_keys):
        ops.append(EditEntry(EditOp.MAP_DEL, path + (k,), old=a.entries[k]))

    for k in sorted(b_keys - a_keys):
        ops.append(EditEntry(EditOp.MAP_ADD, path + (k,), new=b.entries[k]))

    for k in sorted(a_keys & b_keys):
        d = distance(a.entries[k], b.entries[k])
        child_path = path + (k,)
        if d < 1e-9:
            ops.append(EditEntry(EditOp.EQUAL, child_path,
                                 old=a.entries[k], new=b.entries[k]))
        else:
            sub_ops = diff(a.entries[k], b.entries[k], child_path)
            ops.append(EditEntry(EditOp.MAP_MOD, child_path,
                                 old=a.entries[k], new=b.entries[k],
                                 children=sub_ops))

    return ops


def _tagged_diff(a: ATagged, b: ATagged, path: tuple) -> list[EditEntry]:
    """Diff two tagged values."""
    ops: list[EditEntry] = []

    if a.tag != b.tag:
        ops.append(EditEntry(EditOp.TAG_MOD, path,
                             old=AAtom(a.tag), new=AAtom(b.tag)))

    if a.inner == b.inner:
        # Inner unchanged — no entry needed
        pass
    else:
        # Produce a single REPLACE for the inner, with detailed inner_ops
        # as children for introspection.  This avoids the absolute-path
        # problem where flattened inner entries carry paths that
        # _patch_tagged cannot properly dispatch.
        inner_ops = diff(a.inner, b.inner, path + ("inner",))
        ops.append(EditEntry(EditOp.REPLACE, path + ("inner",),
                             old=a.inner, new=b.inner,
                             children=inner_ops))
    return ops


# ═══════════════════════════════════════════════════════════════════
#  PATCH (apply edit script)
# ═══════════════════════════════════════════════════════════════════

def patch(val: AVal, script: list[EditEntry]) -> AVal:
    """
    Apply an edit script to produce the target value.

    This is the inverse of diff:
        patch(a, diff(a, b)) == b
    """
    # For atoms and simple replacements at root
    for entry in script:
        if entry.path == () and entry.op == EditOp.REPLACE:
            return entry.new
        if entry.path == () and entry.op == EditOp.EQUAL:
            return val

    if isinstance(val, AAtom):
        # Script should be a single EQUAL or REPLACE at root
        return val

    if isinstance(val, ASeq):
        return _patch_seq(val, script)

    if isinstance(val, AMap):
        return _patch_map(val, script)

    if isinstance(val, ATagged):
        return _patch_tagged(val, script)

    return val


def _patch_seq(val: ASeq, script: list[EditEntry]) -> ASeq:
    """Apply edit script to a sequence."""
    result: list[AVal] = []

    for entry in script:
        if entry.op == EditOp.EQUAL:
            result.append(entry.old)
        elif entry.op == EditOp.INSERT:
            result.append(entry.new)
        elif entry.op == EditOp.DELETE:
            pass  # skip deleted elements
        elif entry.op == EditOp.REPLACE:
            # Use entry.new directly — it contains the target value.
            # Children describe the internal structure of the change
            # (useful for inspection) but carry full paths from root,
            # so recursive patch calls with children would fail.
            result.append(entry.new)

    return ASeq(tuple(result))


def _patch_map(val: AMap, script: list[EditEntry]) -> AMap:
    """Apply edit script to a map."""
    entries = dict(val.entries)

    for entry in script:
        if not entry.path:
            continue
        key = entry.path[-1]
        if not isinstance(key, str):
            continue

        if entry.op == EditOp.MAP_DEL:
            entries.pop(key, None)
        elif entry.op == EditOp.MAP_ADD:
            entries[key] = entry.new
        elif entry.op == EditOp.MAP_MOD:
            # Use entry.new directly — same reasoning as _patch_seq.
            entries[key] = entry.new
        elif entry.op == EditOp.EQUAL:
            pass  # unchanged

    return AMap(entries)


def _patch_tagged(val: ATagged, script: list[EditEntry]) -> ATagged:
    """Apply edit script to a tagged value."""
    tag = val.tag
    inner = val.inner

    for entry in script:
        if entry.op == EditOp.TAG_MOD:
            tag = entry.new.val if isinstance(entry.new, AAtom) else tag
        elif entry.op == EditOp.REPLACE and entry.new is not None:
            # Use entry.new directly — same as _patch_seq/_patch_map.
            # The diff produces a single REPLACE for inner changes
            # with the complete target value, avoiding path issues.
            inner = entry.new

    return ATagged(tag, inner)
