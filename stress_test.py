"""
Stress tests / adversarial evaluation of structdist.

This script attempts to BREAK the claimed properties:
  1. Metric properties (especially triangle inequality)
  2. Levenshtein reduction
  3. Identity of indiscernibles (d=0 iff equal)
  4. Normalized distance semantics
  5. Diff/patch round-trip correctness
  6. Edge cases that might expose design flaws
"""

import sys, os, random, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structdist.core import (
    AAtom, ASeq, AMap, ATagged,
    distance, normalized_distance, atom_distance, _levenshtein,
    diff, patch,
)
from structdist.formats import from_python, to_python, string_to_seq
from structdist.merge import merge


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ═══════════════════════════════════════════════════════════════
#  §1  LEVENSHTEIN REDUCTION — exhaustive small cases
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  §1  LEVENSHTEIN REDUCTION — exhaustive check")
print("=" * 70)

# Generate all strings of length ≤ 4 over alphabet {a, b, c}
alphabet = "abc"
all_strings = [""]
for length in range(1, 5):
    for combo in itertools.product(alphabet, repeat=length):
        all_strings.append("".join(combo))

# Test a random sample of pairs (full cross-product is 10K+ pairs)
random.seed(42)
sample_pairs = random.sample(
    [(s1, s2) for s1 in all_strings for s2 in all_strings],
    min(2000, len(all_strings)**2)
)

lev_mismatches = 0
for s1, s2 in sample_pairs:
    expected = _levenshtein(s1, s2)
    aed = distance(string_to_seq(s1), string_to_seq(s2))
    if abs(aed - expected) > 1e-9:
        lev_mismatches += 1
        if lev_mismatches <= 5:
            print(f"    MISMATCH: d(\"{s1}\", \"{s2}\") = {aed}, Levenshtein = {expected}")

test("Levenshtein reduction (2000 random pairs, len≤4)",
     lev_mismatches == 0,
     f"{lev_mismatches} mismatches")

# ═══════════════════════════════════════════════════════════════
#  §2  TRIANGLE INEQUALITY — random structured data
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §2  TRIANGLE INEQUALITY — random structures")
print("=" * 70)

def random_aval(depth=0, max_depth=3):
    """Generate a random algebraic value."""
    if depth >= max_depth:
        return AAtom(random.choice([1, 2, 3, "a", "b", None, True, False]))
    
    kind = random.choice(["atom", "seq", "map", "tagged"])
    if kind == "atom":
        return AAtom(random.choice([42, "hello", "world", 3.14, None, True, 0]))
    elif kind == "seq":
        n = random.randint(0, 4)
        return ASeq(tuple(random_aval(depth+1, max_depth) for _ in range(n)))
    elif kind == "map":
        n = random.randint(0, 3)
        keys = random.sample(["a", "b", "c", "d", "e", "x", "y"], min(n, 7))
        return AMap({k: random_aval(depth+1, max_depth) for k in keys})
    else:
        tag = random.choice(["div", "span", "p", "node"])
        return ATagged(tag, random_aval(depth+1, max_depth))

random.seed(123)
values = [random_aval() for _ in range(30)]

tri_violations = 0
tri_checks = 0
for i in range(len(values)):
    for j in range(len(values)):
        for k in range(len(values)):
            dik = distance(values[i], values[k])
            dij = distance(values[i], values[j])
            djk = distance(values[j], values[k])
            tri_checks += 1
            if dik > dij + djk + 1e-9:
                tri_violations += 1
                if tri_violations <= 3:
                    print(f"    VIOLATION: d(v{i},v{k})={dik:.4f} > d(v{i},v{j})={dij:.4f} + d(v{j},v{k})={djk:.4f}")

test(f"Triangle inequality ({tri_checks} triples, 30 random structures)",
     tri_violations == 0,
     f"{tri_violations} violations")

# Symmetry
sym_violations = 0
for i in range(len(values)):
    for j in range(i+1, len(values)):
        dij = distance(values[i], values[j])
        dji = distance(values[j], values[i])
        if abs(dij - dji) > 1e-9:
            sym_violations += 1
test(f"Symmetry ({len(values)*(len(values)-1)//2} pairs)",
     sym_violations == 0,
     f"{sym_violations} violations")

# Identity
id_violations = 0
for v in values:
    if distance(v, v) > 1e-9:
        id_violations += 1
test(f"Identity d(x,x)=0 ({len(values)} values)",
     id_violations == 0)


# ═══════════════════════════════════════════════════════════════
#  §3  ATOM DISTANCE EDGE CASES
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §3  ATOM DISTANCE EDGE CASES")
print("=" * 70)

# Does atom_distance for numbers satisfy triangle inequality?
nums = [AAtom(n) for n in [0, 1, 2, 5, 10, 100, -1, -50, 0.001, 999999]]
num_tri_violations = 0
for a in nums:
    for b in nums:
        for c in nums:
            dac = atom_distance(a, c)
            dab = atom_distance(a, b)
            dbc = atom_distance(b, c)
            if dac > dab + dbc + 1e-9:
                num_tri_violations += 1
                if num_tri_violations <= 5:
                    print(f"    VIOLATION: d({a.val},{c.val})={dac:.4f} > "
                          f"d({a.val},{b.val})={dab:.4f} + d({b.val},{c.val})={dbc:.4f}")

test(f"Atom numeric triangle inequality ({len(nums)**3} triples)",
     num_tri_violations == 0,
     f"{num_tri_violations} violations")

# bool vs int edge case (Python: True == 1, False == 0)
test("bool True vs int 1",
     atom_distance(AAtom(True), AAtom(1)) == 1.0,
     f"got {atom_distance(AAtom(True), AAtom(1))}")

test("bool False vs int 0", 
     atom_distance(AAtom(False), AAtom(0)) == 1.0,
     f"got {atom_distance(AAtom(False), AAtom(0))}")

# Are (True == 1) creating issues? In Python, True == 1 is True
test("AAtom(True) == AAtom(1) identity",
     distance(AAtom(True), AAtom(1)) == 1.0,
     "Should be 1.0 (type mismatch) — is it?")


# ═══════════════════════════════════════════════════════════════
#  §4  THE CRITICAL FLAW CHECK: does atom normalization 
#      break Levenshtein reduction?
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §4  ATOM NORMALIZATION vs LEVENSHTEIN REDUCTION")
print("=" * 70)

# atom_distance normalizes string distance to [0,1].
# But in _seq_distance, sub_cost = distance(AAtom(c1), AAtom(c2))
#   = atom_distance(c1, c2) = levenshtein(c1,c2)/max(len,len)
# For single chars: levenshtein("a","b") = 1, max(1,1) = 1, so 1/1 = 1.0
# For insertion: size(AAtom(c)) = 1.0
# For deletion: size(AAtom(c)) = 1.0
# So sub_cost = 1.0 when chars differ, 0.0 when same
# del_cost = 1.0 (atom size = 1)
# ins_cost = 1.0 (atom size = 1)
# This gives EXACTLY Levenshtein costs. The reduction works BECAUSE
# single-char atoms have normalized distance = 0 or 1, and size = 1.

# But what about multi-char string atoms in a sequence?
# d(AAtom("cat"), AAtom("bat")) = 1/3 ≈ 0.333
# This is NOT a standard Levenshtein cost — it's a fractional substitution.
# That's intentional design, but worth noting.

test("Single char sub cost = 1.0 (match/mismatch)",
     atom_distance(AAtom("a"), AAtom("b")) == 1.0,
     f"got {atom_distance(AAtom('a'), AAtom('b'))}")

test("Single char match cost = 0.0",
     atom_distance(AAtom("a"), AAtom("a")) == 0.0)

# Does this still work with unicode?
test("Unicode: é vs e",
     distance(string_to_seq("café"), string_to_seq("cafe")) == 1.0,
     f"got {distance(string_to_seq('café'), string_to_seq('cafe'))}")


# ═══════════════════════════════════════════════════════════════
#  §5  DIFF/PATCH ROUND-TRIP — random structures
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §5  DIFF/PATCH ROUND-TRIP — random structures")
print("=" * 70)

random.seed(456)
roundtrip_failures = 0
roundtrip_tests = 0
for _ in range(200):
    a = random_aval(max_depth=2)
    b = random_aval(max_depth=2)
    
    script = diff(a, b)
    result = patch(a, script)
    d = distance(result, b)
    roundtrip_tests += 1
    
    if d > 1e-9:
        roundtrip_failures += 1
        if roundtrip_failures <= 3:
            print(f"    FAIL: d(patch(a, diff(a,b)), b) = {d:.4f}")
            print(f"      a = {a!r}")
            print(f"      b = {b!r}")
            print(f"      result = {result!r}")

test(f"Diff/patch round-trip ({roundtrip_tests} random pairs)",
     roundtrip_failures == 0,
     f"{roundtrip_failures} failures")


# ═══════════════════════════════════════════════════════════════
#  §6  NORMALIZED DISTANCE SANITY
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §6  NORMALIZED DISTANCE BOUNDS [0, 1]")
print("=" * 70)

norm_violations = 0
for v1 in values[:15]:
    for v2 in values[:15]:
        nd = normalized_distance(v1, v2)
        if nd < -1e-9 or nd > 1.0 + 1e-9:
            norm_violations += 1
            if norm_violations <= 3:
                print(f"    OUT OF BOUNDS: nd = {nd}")

test(f"Normalized distance in [0,1] ({15*15} pairs)",
     norm_violations == 0,
     f"{norm_violations} violations")

# Can normalized distance exceed 1 for type mismatches?
nd_type_mismatch = normalized_distance(AAtom(1), ASeq((AAtom(1), AAtom(2), AAtom(3))))
test(f"Type mismatch normalized distance ≤ 1",
     nd_type_mismatch <= 1.0 + 1e-9,
     f"got {nd_type_mismatch:.4f}")

# ═══════════════════════════════════════════════════════════════
#  §7  IDENTITY OF INDISCERNIBLES — can d=0 for unequal values?
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §7  IDENTITY OF INDISCERNIBLES")
print("=" * 70)

# AAtom(0) vs AAtom(0.0) — in Python, 0 == 0.0 is True
d_int_float = distance(AAtom(0), AAtom(0.0))
test("AAtom(0) vs AAtom(0.0)",
     d_int_float == 0.0,
     f"d = {d_int_float} (0 == 0.0 in Python, so this is correct)")

# AAtom(1) vs AAtom(True) — in Python, True == 1 is True!
d_bool_int = distance(AAtom(True), AAtom(1))
test("AAtom(True) vs AAtom(1) — Python True==1 trap",
     True,  # Just report
     f"d = {d_bool_int:.4f} — atom_distance checks isinstance before ==")

# Empty seq vs empty map
d_empty_types = distance(ASeq(()), AMap({}))
test("Empty seq vs empty map (type mismatch)",
     d_empty_types > 0,
     f"d = {d_empty_types}")

# Map key ordering
m1 = AMap({"z": AAtom(1), "a": AAtom(2)})
m2 = AMap({"a": AAtom(2), "z": AAtom(1)})
test("Map key order independence",
     distance(m1, m2) == 0.0,
     f"d = {distance(m1, m2)}")


# ═══════════════════════════════════════════════════════════════
#  §8  PERFORMANCE / COMPLEXITY
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §8  PERFORMANCE (wall-clock)")
print("=" * 70)

# Sequence DP is O(m*n) — does it blow up?
for n in [100, 200, 500, 1000]:
    a = ASeq(tuple(AAtom(i) for i in range(n)))
    b = ASeq(tuple(AAtom(i+1) for i in range(n)))
    t0 = time.perf_counter()
    d = distance(a, b)
    dt = time.perf_counter() - t0
    print(f"  Seq({n}) vs Seq({n}): {dt*1000:.1f}ms  d={d:.2f}")

# Deeply nested structures
def make_deep(depth):
    v = AAtom("leaf")
    for i in range(depth):
        v = AMap({"child": v, "level": AAtom(i)})
    return v

for depth in [5, 10, 20, 50]:
    a = make_deep(depth)
    b = make_deep(depth)  # same structure but different objects
    t0 = time.perf_counter()
    d = distance(a, b)
    dt = time.perf_counter() - t0
    print(f"  Depth {depth}: {dt*1000:.3f}ms  d={d:.2f}")


# ═══════════════════════════════════════════════════════════════
#  §9  MERGE EDGE CASES
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  §9  MERGE EDGE CASES")
print("=" * 70)

# Same change both sides — should not conflict
base = from_python({"a": 1, "b": 2})
ours = from_python({"a": 42, "b": 2})
theirs = from_python({"a": 42, "b": 2})
result = merge(base, ours, theirs)
test("Same change both sides → no conflict",
     not result.has_conflicts)

# Both add same new key with same value
base = from_python({"a": 1})
ours = from_python({"a": 1, "new": "value"})
theirs = from_python({"a": 1, "new": "value"})
result = merge(base, ours, theirs)
test("Both add same key/value → no conflict",
     not result.has_conflicts)

# Both add same key, different values → conflict
base = from_python({"a": 1})
ours = from_python({"a": 1, "new": "ours"})
theirs = from_python({"a": 1, "new": "theirs"})
result = merge(base, ours, theirs)
test("Both add same key, diff value → conflict",
     result.has_conflicts)


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  STRESS TEST SUMMARY")
print("=" * 70)
print("  If you see FAIL above, there's a bug.")
print("  If everything is PASS, the implementation is correct")
print("  for the tested cases (not a proof, but high confidence).")
