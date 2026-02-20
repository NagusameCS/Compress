"""
Comprehensive test suite for structdist — Algebraic Edit Distance.

Tests are organized to demonstrate the mathematical claims:
    §1  Levenshtein reduction (THE critical proof)
    §2  Metric properties (d≥0, identity, symmetry, triangle inequality)
    §3  Sequence distance (the Levenshtein generalization)
    §4  Map distance (unordered key-value)
    §5  Tagged values
    §6  Type mismatches
    §7  Diff / Patch round-trip
    §8  Format converters (JSON, Python)
    §9  Three-way merge
    §10 Practical use cases (real JSON configs)
"""

import sys
import os
import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structdist.core import (
    AAtom, ASeq, AMap, ATagged,
    distance, normalized_distance,
    atom_distance, _levenshtein,
    EditOp, EditEntry, diff, patch,
)
from structdist.formats import (
    from_python, to_python, from_json, to_json,
    string_to_seq, seq_to_string,
)
from structdist.merge import merge, MergeResult, ConflictEntry


# ═══════════════════════════════════════════════════════════════════
#  §1  LEVENSHTEIN REDUCTION
#      This is the most important test: it proves AED is a true
#      generalization of Levenshtein, not just "inspired by."
# ═══════════════════════════════════════════════════════════════════

class TestLevenshteinReduction:
    """
    AED on character sequences must produce EXACTLY the same distance
    as Levenshtein on the corresponding strings.

    If AED(string_to_seq(s1), string_to_seq(s2)) == levenshtein(s1, s2)
    for all s1, s2, then AED is a strict generalization.
    """

    @pytest.mark.parametrize("s1,s2,expected", [
        ("", "", 0),
        ("a", "", 1),
        ("", "a", 1),
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("abc", "axc", 1),
        ("abc", "abcd", 1),
        ("abcd", "abc", 1),
        ("a", "b", 1),
        ("ab", "ba", 2),
        ("intention", "execution", 5),
        ("flaw", "lawn", 2),
    ])
    def test_levenshtein_equivalence(self, s1, s2, expected):
        """AED on char sequences must equal Levenshtein distance."""
        # Verify our Levenshtein implementation
        assert _levenshtein(s1, s2) == expected

        # The key test: AED must match
        seq1 = string_to_seq(s1)
        seq2 = string_to_seq(s2)
        aed = distance(seq1, seq2)
        assert aed == expected, (
            f"AED({s1!r}, {s2!r}) = {aed}, expected {expected} (Levenshtein)"
        )

    def test_levenshtein_empty_strings(self):
        """Empty string → empty sequence → distance 0."""
        seq1 = string_to_seq("")
        seq2 = string_to_seq("")
        assert distance(seq1, seq2) == 0.0

    def test_levenshtein_single_char_diff(self):
        """Single character substitution = distance 1."""
        assert distance(string_to_seq("cat"), string_to_seq("bat")) == 1.0

    def test_levenshtein_symmetry(self):
        """d(s1, s2) == d(s2, s1) for strings."""
        pairs = [("kitten", "sitting"), ("abc", "xyz"), ("hello", ""),
                 ("ab", "ba"), ("intention", "execution")]
        for s1, s2 in pairs:
            d12 = distance(string_to_seq(s1), string_to_seq(s2))
            d21 = distance(string_to_seq(s2), string_to_seq(s1))
            assert d12 == d21, f"Asymmetric: d({s1!r},{s2!r})={d12} ≠ d({s2!r},{s1!r})={d21}"


# ═══════════════════════════════════════════════════════════════════
#  §2  METRIC PROPERTIES
# ═══════════════════════════════════════════════════════════════════

class TestMetricProperties:
    """Verify that AED is a true metric on the space of algebraic values."""

    # Test values of various types and complexities
    VALUES = [
        AAtom(None),
        AAtom(0),
        AAtom(42),
        AAtom("hello"),
        AAtom("world"),
        ASeq(()),
        ASeq((AAtom(1), AAtom(2), AAtom(3))),
        ASeq((AAtom("a"), AAtom("b"))),
        AMap({}),
        AMap({"x": AAtom(1)}),
        AMap({"x": AAtom(1), "y": AAtom(2)}),
        ATagged("div", AAtom("hello")),
        ATagged("span", AAtom("world")),
        # Nested
        AMap({"list": ASeq((AAtom(1), AAtom(2))), "name": AAtom("test")}),
    ]

    def test_non_negativity(self):
        """d(x, y) ≥ 0 for all x, y."""
        for x in self.VALUES:
            for y in self.VALUES:
                d = distance(x, y)
                assert d >= 0, f"d({x!r}, {y!r}) = {d} < 0"

    def test_identity_of_indiscernibles(self):
        """d(x, x) = 0 for all x."""
        for x in self.VALUES:
            d = distance(x, x)
            assert d == 0.0, f"d({x!r}, {x!r}) = {d} ≠ 0"

    def test_symmetry(self):
        """d(x, y) = d(y, x) for all x, y."""
        for x in self.VALUES:
            for y in self.VALUES:
                dxy = distance(x, y)
                dyx = distance(y, x)
                assert abs(dxy - dyx) < 1e-9, (
                    f"Asymmetric: d({x!r}, {y!r}) = {dxy} ≠ d({y!r}, {x!r}) = {dyx}"
                )

    def test_triangle_inequality(self):
        """d(x, z) ≤ d(x, y) + d(y, z) for all x, y, z."""
        # Test a diverse subset to keep runtime reasonable
        subset = self.VALUES[:8]
        for x in subset:
            for y in subset:
                for z in subset:
                    dxz = distance(x, z)
                    dxy = distance(x, y)
                    dyz = distance(y, z)
                    assert dxz <= dxy + dyz + 1e-9, (
                        f"Triangle inequality violated: "
                        f"d({x!r},{z!r})={dxz} > "
                        f"d({x!r},{y!r})={dxy} + d({y!r},{z!r})={dyz} = {dxy+dyz}"
                    )

    def test_positive_definiteness(self):
        """d(x, y) > 0 when x ≠ y (for distinguishable values)."""
        assert distance(AAtom(1), AAtom(2)) > 0
        assert distance(AAtom("a"), AAtom("b")) > 0
        assert distance(ASeq((AAtom(1),)), ASeq((AAtom(2),))) > 0
        assert distance(AMap({"a": AAtom(1)}), AMap({"b": AAtom(1)})) > 0


# ═══════════════════════════════════════════════════════════════════
#  §3  SEQUENCE DISTANCE
# ═══════════════════════════════════════════════════════════════════

class TestSequenceDistance:
    """Test the generalized Levenshtein on sequences."""

    def test_empty_sequences(self):
        assert distance(ASeq(()), ASeq(())) == 0.0

    def test_insert_single(self):
        a = ASeq(())
        b = ASeq((AAtom(1),))
        assert distance(a, b) == 1.0  # Insert one atom (size=1)

    def test_delete_single(self):
        a = ASeq((AAtom(1),))
        b = ASeq(())
        assert distance(a, b) == 1.0  # Delete one atom (size=1)

    def test_substitute_atom(self):
        a = ASeq((AAtom(1),))
        b = ASeq((AAtom(2),))
        # sub_cost = atom_distance(AAtom(1), AAtom(2)) = min(1, |1-2|/max(1,2,1)) = 0.5
        d = distance(a, b)
        assert d == pytest.approx(0.5, abs=1e-9)

    def test_nested_sequence_edit(self):
        """Editing a nested sequence should recurse properly."""
        inner1 = ASeq((AAtom(1), AAtom(2)))
        inner2 = ASeq((AAtom(1), AAtom(3)))
        a = ASeq((inner1,))
        b = ASeq((inner2,))
        # sub_cost = d(inner1, inner2) = cost of substituting 2→3
        # which is atom_distance(2, 3) = min(1, |2-3|/3) ≈ 0.333
        d = distance(a, b)
        expected_inner = distance(inner1, inner2)
        assert d == pytest.approx(expected_inner, abs=1e-9)

    def test_size_weighted_insert(self):
        """Inserting a complex value costs more than inserting a simple one."""
        a = ASeq(())
        b_simple = ASeq((AAtom(1),))
        b_complex = ASeq((AMap({"x": AAtom(1), "y": AAtom(2)}),))
        d_simple = distance(a, b_simple)
        d_complex = distance(a, b_complex)
        assert d_complex > d_simple, "Complex insertions should cost more"


# ═══════════════════════════════════════════════════════════════════
#  §4  MAP DISTANCE
# ═══════════════════════════════════════════════════════════════════

class TestMapDistance:
    """Test unordered map comparison."""

    def test_identical_maps(self):
        m = AMap({"a": AAtom(1), "b": AAtom(2)})
        assert distance(m, m) == 0.0

    def test_empty_maps(self):
        assert distance(AMap({}), AMap({})) == 0.0

    def test_add_key(self):
        a = AMap({})
        b = AMap({"x": AAtom(42)})
        # Insert key "x" → cost = 1 (key) + 1 (atom value) = 2
        assert distance(a, b) == 2.0

    def test_remove_key(self):
        a = AMap({"x": AAtom(42)})
        b = AMap({})
        assert distance(a, b) == 2.0  # Symmetric with add

    def test_modify_value(self):
        a = AMap({"x": AAtom("hello")})
        b = AMap({"x": AAtom("hallo")})
        # d = atom_distance("hello", "hallo") = 1/5 = 0.2
        d = distance(a, b)
        assert d == pytest.approx(0.2, abs=1e-9)

    def test_key_order_irrelevant(self):
        """Map distance should NOT depend on key insertion order."""
        a = AMap({"a": AAtom(1), "b": AAtom(2), "c": AAtom(3)})
        b = AMap({"c": AAtom(3), "a": AAtom(1), "b": AAtom(2)})
        assert distance(a, b) == 0.0

    def test_nested_map_modification(self):
        a = AMap({"config": AMap({"debug": AAtom(True), "port": AAtom(8080)})})
        b = AMap({"config": AMap({"debug": AAtom(False), "port": AAtom(8080)})})
        d = distance(a, b)
        assert d > 0  # debug changed
        assert d == pytest.approx(1.0, abs=1e-9)  # bool flip = 1.0


# ═══════════════════════════════════════════════════════════════════
#  §5  TAGGED VALUES
# ═══════════════════════════════════════════════════════════════════

class TestTaggedDistance:

    def test_same_tag_same_inner(self):
        a = ATagged("div", AAtom("hello"))
        assert distance(a, a) == 0.0

    def test_same_tag_diff_inner(self):
        a = ATagged("div", AAtom("hello"))
        b = ATagged("div", AAtom("world"))
        d = distance(a, b)
        assert d > 0
        # Should equal atom_distance("hello", "world")
        assert d == pytest.approx(atom_distance(AAtom("hello"), AAtom("world")), abs=1e-9)

    def test_diff_tag_same_inner(self):
        a = ATagged("div", AAtom("hello"))
        b = ATagged("span", AAtom("hello"))
        d = distance(a, b)
        assert d == pytest.approx(1.0, abs=1e-9)  # TAG_MISMATCH_COST


# ═══════════════════════════════════════════════════════════════════
#  §6  TYPE MISMATCHES
# ═══════════════════════════════════════════════════════════════════

class TestTypeMismatch:

    def test_atom_vs_seq(self):
        a = AAtom(1)
        b = ASeq((AAtom(1), AAtom(2)))
        d = distance(a, b)
        assert d == a.size() + b.size()  # Full delete + insert

    def test_seq_vs_map(self):
        a = ASeq((AAtom(1),))
        b = AMap({"x": AAtom(1)})
        d = distance(a, b)
        assert d == a.size() + b.size()

    def test_atom_vs_map(self):
        a = AAtom("hello")
        b = AMap({"greeting": AAtom("hello")})
        d = distance(a, b)
        assert d == a.size() + b.size()


# ═══════════════════════════════════════════════════════════════════
#  §7  DIFF / PATCH ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════

class TestDiffPatch:
    """
    The critical property: patch(a, diff(a, b)) should produce
    a value equivalent to b.
    """

    def _assert_round_trip(self, a: "AVal", b: "AVal"):
        """Verify diff/patch round-trip."""
        script = diff(a, b)
        result = patch(a, script)
        d = distance(result, b)
        assert d < 1e-9, (
            f"Round-trip failed:\n"
            f"  a = {a!r}\n"
            f"  b = {b!r}\n"
            f"  result = {result!r}\n"
            f"  d(result, b) = {d}"
        )

    def test_atom_replace(self):
        self._assert_round_trip(AAtom("hello"), AAtom("world"))

    def test_atom_equal(self):
        self._assert_round_trip(AAtom(42), AAtom(42))

    def test_seq_insert(self):
        self._assert_round_trip(
            ASeq((AAtom(1), AAtom(2))),
            ASeq((AAtom(1), AAtom(2), AAtom(3)))
        )

    def test_seq_delete(self):
        self._assert_round_trip(
            ASeq((AAtom(1), AAtom(2), AAtom(3))),
            ASeq((AAtom(1), AAtom(3)))
        )

    def test_seq_substitute(self):
        self._assert_round_trip(
            ASeq((AAtom("a"), AAtom("b"), AAtom("c"))),
            ASeq((AAtom("a"), AAtom("x"), AAtom("c")))
        )

    def test_map_add_key(self):
        self._assert_round_trip(
            AMap({"a": AAtom(1)}),
            AMap({"a": AAtom(1), "b": AAtom(2)})
        )

    def test_map_remove_key(self):
        self._assert_round_trip(
            AMap({"a": AAtom(1), "b": AAtom(2)}),
            AMap({"a": AAtom(1)})
        )

    def test_map_modify_value(self):
        self._assert_round_trip(
            AMap({"x": AAtom("old")}),
            AMap({"x": AAtom("new")})
        )

    def test_nested_round_trip(self):
        """Complex nested structure diff/patch."""
        a = from_python({"users": [{"name": "Alice", "age": 30}], "count": 1})
        b = from_python({"users": [{"name": "Alice", "age": 31}], "count": 1})
        self._assert_round_trip(a, b)

    def test_empty_to_complex(self):
        self._assert_round_trip(ASeq(()), ASeq((AAtom(1), AAtom(2), AAtom(3))))

    def test_complex_json_round_trip(self):
        """Real-world JSON diff/patch round-trip."""
        a = from_python({
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb",
                "ssl": False
            },
            "cache": {
                "ttl": 300,
                "max_size": 1000
            }
        })
        b = from_python({
            "database": {
                "host": "prod.example.com",
                "port": 5432,
                "name": "mydb",
                "ssl": True
            },
            "cache": {
                "ttl": 600,
                "max_size": 1000
            },
            "logging": {
                "level": "INFO"
            }
        })
        self._assert_round_trip(a, b)


# ═══════════════════════════════════════════════════════════════════
#  §8  FORMAT CONVERTERS
# ═══════════════════════════════════════════════════════════════════

class TestFormats:

    def test_python_primitives(self):
        assert to_python(from_python(None)) is None
        assert to_python(from_python(42)) == 42
        assert to_python(from_python(3.14)) == 3.14
        assert to_python(from_python("hello")) == "hello"
        assert to_python(from_python(True)) is True
        assert to_python(from_python(False)) is False

    def test_python_list(self):
        obj = [1, 2, 3]
        assert to_python(from_python(obj)) == obj

    def test_python_dict(self):
        obj = {"a": 1, "b": "hello", "c": [1, 2]}
        assert to_python(from_python(obj)) == obj

    def test_python_nested(self):
        obj = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "meta": {"version": 2, "active": True}
        }
        assert to_python(from_python(obj)) == obj

    def test_json_round_trip(self):
        json_str = '{"key": [1, 2, 3], "flag": true, "nested": {"x": null}}'
        val = from_json(json_str)
        result = to_json(val, sort_keys=True)
        expected = to_json(from_json(json_str), sort_keys=True)
        assert result == expected

    def test_string_to_seq_round_trip(self):
        s = "hello world"
        seq = string_to_seq(s)
        assert seq_to_string(seq) == s

    def test_string_to_seq_empty(self):
        assert seq_to_string(string_to_seq("")) == ""


# ═══════════════════════════════════════════════════════════════════
#  §9  THREE-WAY MERGE
# ═══════════════════════════════════════════════════════════════════

class TestMerge:

    def test_no_changes(self):
        base = from_python({"a": 1, "b": 2})
        result = merge(base, base, base)
        assert not result.has_conflicts
        assert distance(result.merged, base) < 1e-9

    def test_only_ours_changed(self):
        base = from_python({"a": 1, "b": 2})
        ours = from_python({"a": 99, "b": 2})
        theirs = from_python({"a": 1, "b": 2})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 99, "b": 2})
        assert distance(result.merged, expected) < 1e-9

    def test_only_theirs_changed(self):
        base = from_python({"a": 1, "b": 2})
        ours = from_python({"a": 1, "b": 2})
        theirs = from_python({"a": 1, "b": 99})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 1, "b": 99})
        assert distance(result.merged, expected) < 1e-9

    def test_both_changed_different_keys(self):
        """Non-overlapping changes should merge cleanly."""
        base = from_python({"a": 1, "b": 2, "c": 3})
        ours = from_python({"a": 99, "b": 2, "c": 3})  # Changed a
        theirs = from_python({"a": 1, "b": 2, "c": 99})  # Changed c
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 99, "b": 2, "c": 99})
        assert distance(result.merged, expected) < 1e-9

    def test_both_changed_same_key_same_value(self):
        """Identical changes on both sides should not conflict."""
        base = from_python({"a": 1})
        ours = from_python({"a": 42})
        theirs = from_python({"a": 42})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 42})
        assert distance(result.merged, expected) < 1e-9

    def test_both_changed_same_key_diff_value_conflict(self):
        """Different changes to same key → conflict."""
        base = from_python({"a": 1})
        ours = from_python({"a": 42})
        theirs = from_python({"a": 99})
        result = merge(base, ours, theirs)
        assert result.has_conflicts
        assert len(result.conflicts) >= 1

    def test_one_added_key(self):
        base = from_python({"a": 1})
        ours = from_python({"a": 1, "b": 2})  # Added b
        theirs = from_python({"a": 1})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 1, "b": 2})
        assert distance(result.merged, expected) < 1e-9

    def test_one_deleted_key(self):
        base = from_python({"a": 1, "b": 2})
        ours = from_python({"a": 1})  # Deleted b
        theirs = from_python({"a": 1, "b": 2})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"a": 1})
        assert distance(result.merged, expected) < 1e-9

    def test_nested_merge(self):
        """Merge should work recursively on nested structures."""
        base = from_python({"config": {"debug": False, "port": 8080}})
        ours = from_python({"config": {"debug": True, "port": 8080}})
        theirs = from_python({"config": {"debug": False, "port": 9090}})
        result = merge(base, ours, theirs)
        assert not result.has_conflicts
        expected = from_python({"config": {"debug": True, "port": 9090}})
        assert distance(result.merged, expected) < 1e-9


# ═══════════════════════════════════════════════════════════════════
#  §10  PRACTICAL USE CASES
# ═══════════════════════════════════════════════════════════════════

class TestPracticalUseCases:

    def test_config_drift_detection(self):
        """Detect configuration drift between deployed and intended configs."""
        intended = from_python({
            "server": {"host": "0.0.0.0", "port": 443, "tls": True},
            "database": {"host": "db.internal", "pool_size": 10},
            "logging": {"level": "WARN", "format": "json"}
        })
        deployed = from_python({
            "server": {"host": "0.0.0.0", "port": 443, "tls": True},
            "database": {"host": "db.internal", "pool_size": 5},  # Drifted!
            "logging": {"level": "DEBUG", "format": "json"}       # Drifted!
        })
        d = distance(intended, deployed)
        assert d > 0, "Should detect drift"

        nd = normalized_distance(intended, deployed)
        assert 0 < nd < 1, "Normalized distance should be between 0 and 1"

        # Diff should identify exactly what changed
        script = diff(intended, deployed)
        changes = [e for e in script if e.op not in (EditOp.EQUAL,)]
        # Everything at root level passes through MAP_MOD
        assert any(e.op == EditOp.MAP_MOD for e in changes), "Should detect map modification"

    def test_api_schema_evolution(self):
        """Measure how much an API schema changed between versions."""
        v1 = from_python({
            "endpoints": [
                {"path": "/users", "method": "GET", "params": ["page", "limit"]},
                {"path": "/users", "method": "POST", "params": ["name", "email"]},
            ]
        })
        v2 = from_python({
            "endpoints": [
                {"path": "/users", "method": "GET", "params": ["page", "limit", "sort"]},
                {"path": "/users", "method": "POST", "params": ["name", "email", "role"]},
                {"path": "/users", "method": "DELETE", "params": ["id"]},
            ]
        })
        d = distance(v1, v2)
        assert d > 0

    def test_json_similarity_search(self):
        """Find the most similar config among candidates."""
        query = from_python({"type": "postgres", "host": "db.local", "port": 5432})
        candidates = [
            from_python({"type": "postgres", "host": "db.local", "port": 5432}),  # exact match
            from_python({"type": "postgres", "host": "db.prod", "port": 5432}),   # close
            from_python({"type": "mysql", "host": "mysql.local", "port": 3306}),  # far
            from_python({"type": "redis", "port": 6379}),                         # very far
        ]
        distances = [distance(query, c) for c in candidates]

        # Exact match should be 0
        assert distances[0] == 0.0
        # Close should be smaller than far
        assert distances[1] < distances[2]
        # Far should be smaller than very far
        assert distances[2] < distances[3]


# ═══════════════════════════════════════════════════════════════════
#  §11  ATOM DISTANCE DETAILS
# ═══════════════════════════════════════════════════════════════════

class TestAtomDistance:

    def test_string_identity(self):
        assert atom_distance(AAtom("hello"), AAtom("hello")) == 0.0

    def test_string_one_edit(self):
        # "cat" → "bat" = 1 edit, length 3
        assert atom_distance(AAtom("cat"), AAtom("bat")) == pytest.approx(1/3)

    def test_number_identity(self):
        assert atom_distance(AAtom(42), AAtom(42)) == 0.0

    def test_number_distance(self):
        # |10 - 20| / max(10, 20) = 10/20 = 0.5
        assert atom_distance(AAtom(10), AAtom(20)) == pytest.approx(0.5)

    def test_bool_same(self):
        assert atom_distance(AAtom(True), AAtom(True)) == 0.0

    def test_bool_diff(self):
        assert atom_distance(AAtom(True), AAtom(False)) == 1.0

    def test_none_same(self):
        assert atom_distance(AAtom(None), AAtom(None)) == 0.0

    def test_type_mismatch(self):
        assert atom_distance(AAtom("42"), AAtom(42)) == 1.0


# ═══════════════════════════════════════════════════════════════════
#  §12  SIZE FUNCTION
# ═══════════════════════════════════════════════════════════════════

class TestSize:

    def test_atom_size(self):
        assert AAtom(42).size() == 1.0

    def test_empty_seq_size(self):
        assert ASeq(()).size() == 1.0  # Just the seq container

    def test_seq_size(self):
        # 1 (container) + 3 (atoms)
        assert ASeq((AAtom(1), AAtom(2), AAtom(3))).size() == 4.0

    def test_empty_map_size(self):
        assert AMap({}).size() == 1.0

    def test_map_size(self):
        # 1 (container) + 2*(1 key + 1 atom value) = 5
        assert AMap({"a": AAtom(1), "b": AAtom(2)}).size() == 5.0

    def test_tagged_size(self):
        # 1 (tag) + 1 (inner atom) = 2
        assert ATagged("x", AAtom(1)).size() == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
