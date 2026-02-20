# structdist — Honest Evaluation

## Methodology

This evaluation follows the same framework applied to the DensePDF algorithms:
stress-test correctness, research prior art exhaustively, then judge novelty
honestly. The goal is to determine whether AED (Algebraic Edit Distance) is a
genuine contribution to computer science — and specifically, whether it's
anywhere near "Levenshtein-level."

---

## 1. Correctness

### What Works

- **86/86 tests pass.** Unit test suite is thorough.
- **Metric properties hold empirically.** Stress-tested with:
  - 2,000 random string pairs: Levenshtein reduction exact in all cases.
  - 27,000 random triples: Zero triangle inequality violations.
  - 435 random pairs: Zero symmetry violations.
  - 30 random values: Identity d(x,x)=0 holds.
  - 1,000 numeric triples: Atom triangle inequality holds.
- **Normalization bounds [0,1] hold** on atom distances.
- **Map key-order independence** confirmed.
- **Merge** passes all edge cases.
- **Unicode** handled correctly.

### What's Broken

**Bug 1: Bool/int conflation.**
`AAtom(True)` vs `AAtom(1)` returns distance 0.0. This is caused by two
compounding issues:
1. `atom_distance` starts with `if av == bv: return 0.0`. In Python,
   `True == 1` evaluates to `True`, so this fast path fires before any
   type checking.
2. Even if you removed the fast path, `bool` is a subclass of `int` in
   Python, so `isinstance(True, (int, float))` is `True`. The numeric
   branch would compute `abs(True - 1) / max(abs(True), abs(1), 1)` = 0.0.
3. The `isinstance(av, bool)` check exists but is unreachable — it comes
   after the numeric branch.

This means the metric **cannot distinguish** `True` from `1`, `False` from
`0`. In a JSON-oriented tool, this is a real bug — JSON `true` and `1` are
semantically distinct.

**Fix:** Check `isinstance(av, bool) != isinstance(bv, bool)` BEFORE the
equality check. If one is bool and the other isn't, return 1.0.

**Bug 2: Diff/patch round-trip failure for ATagged with inner type changes.**
11 out of 200 random diff/patch round-trips fail. The failing cases all
involve `ATagged` where the inner value changes type (e.g.,
`ATagged('div', AAtom(3.14))` → `ATagged('div', AMap({...}))`).

Root cause: `_patch_tagged` passes child entries with path `("inner",)` to
`patch()`, but `patch()` expects root-level entries to have path `()`. The
entry doesn't match any root condition, so the original value is returned
unchanged. This is the *same class of bug* that was previously fixed for
`_patch_seq` and `_patch_map` — but the fix was never applied to
`_patch_tagged`.

**Fix:** In `_patch_tagged`, when an entry targets `("inner",)`, use
`entry.new` directly instead of recursing with `patch(inner, [entry])`.

### Performance

- `Seq(1000)` vs `Seq(1000)`: 634ms (O(n²) DP — expected)
- `Depth 50` nested maps: 0.038ms (O(k) per level — fast)
- Practical JSON configs: < 1ms

Performance is adequate. No algorithmic surprises.

---

## 2. Prior Art Analysis

### What the docstring claims is new

The docstring (§6) claims six novelties:

> 1. The algebraic data type formulation as the universal domain
> 2. Mixed ordered (Seq) and unordered (Map) in a single metric
> 3. Recursive substitution cost in sequence alignment
> 4. Size-weighted insertion/deletion costs
> 5. Formal proof that this is a true metric
> 6. Reduction to Levenshtein as a special case

Let's evaluate each honestly.

### Claim 1: "Algebraic data type formulation as universal domain"

**Verdict: Not new.**

Algebraic data types have been the standard foundation in type theory since
the 1970s (ML, Haskell, Coq). The idea of representing structured data as
sums and products of algebraic types is textbook. The specific choice of
{Atom, Seq, Map, Tagged} is a practical design decision, not a mathematical
contribution. Any developer who has worked with JSON schema, Protocol Buffers,
or Haskell's data types would arrive at a similar decomposition.

### Claim 2: "Mixed ordered and unordered in a single metric"

**Verdict: Modest practical contribution, not a theoretical one.**

The "insight" is: use Levenshtein-style DP for sequences, key-based matching
for maps. This is the obvious thing to do, and it works because **dictionary
keys provide a natural correspondence** between elements.

This is worth contrasting with the actual hard problem: *unordered* tree edit
distance (where children have NO keys) is NP-hard (Zhang 1996). AED sidesteps
this entirely by requiring that all unordered children be accessed by string
keys. This is not solving the hard problem — it's defining it away by
restricting the input domain.

This is a reasonable engineering choice. It is not a breakthrough.

### Claim 3: "Recursive substitution cost in sequence alignment"

**Verdict: Well-known.**

Using recursive substitution costs in tree/sequence alignment has been studied
extensively:
- **Jiang, Wang, Zhang (1995)** — "Alignment of trees — an alternative to
  tree edit distance." *Journal of Algorithms*, 19(3):460-481.
  Explicitly defines alignment with recursive substitution costs.
- **Bille (2005)** — "A survey on tree edit distance and related problems."
  *Theoretical Computer Science*, 337(1-3):22-34.
  Covers alignment with non-unit costs.
- **Wagner-Fischer (1974)** — The original DP algorithm for string edit
  distance already allows arbitrary substitution costs. The generalization
  from unit costs to function-valued costs is immediate.

The idea that you can replace the 0/1 substitution cost with a recursive
call `d(a_i, b_j)` in the DP recurrence is not a new concept. It's exactly
what tree alignment algorithms do.

### Claim 4: "Size-weighted insertion/deletion costs"

**Verdict: Standard variant.**

Using non-unit insertion/deletion costs proportional to element size is a
standard technique in tree edit distance and sequence alignment. The Zhang-
Shasha algorithm supports arbitrary cost functions. The choice of `size(x)`
as the ins/del cost is natural and obvious — but it's a parameter choice
within existing frameworks, not a new framework.

### Claim 5: "Formal proof that this is a true metric"

**Verdict: Trivially true, well-known.**

It has been known since Wagner & Fischer (1974) that edit distance with
non-negative costs satisfying the inverse property yields a metric. The
proof that AED is a metric follows directly from:
1. Atom distances are metrics (Levenshtein is a metric; absolute difference
   is a metric; boolean 0/1 distance is a metric).
2. DP sequence alignment with metric substitution costs and non-negative
   ins/del costs is a metric.
3. Key-wise map comparison with metric component distances is a metric
   (product metric).
4. Tagged comparison with metric inner distance is a metric.
5. The type-mismatch cost `size(X) + size(Y)` satisfies triangle inequality
   (since `size(X) ≥ 0` for all X).

This is a routine verification, not a theorem. No referee would consider
this a mathematical contribution.

### Claim 6: "Reduction to Levenshtein as a special case"

**Verdict: Correct but almost tautological.**

The claim: represent string "abc" as `ASeq((AAtom('a'), AAtom('b'),
AAtom('c')))`. Then AED on these sequences equals Levenshtein distance on
the original strings.

This is true because:
- Each `AAtom('x')` has `size() = 1`, so ins/del costs are 1.
- `atom_distance(AAtom('x'), AAtom('y'))` = Levenshtein('x','y')/max(1,1)
  = 0 if equal, 1 if not.
- The DP recurrence is then *identical* to Levenshtein's recurrence.

This is correct. It is also *tautological*: you constructed the DP to have
the same recurrence as Levenshtein, with single-character atoms having unit
costs. Of course it reduces to Levenshtein — it IS Levenshtein with the
parameters chosen to make it so. The "reduction" is a special case by
construction, not a deep mathematical relationship.

For comparison: when we say "Hamming distance is a special case of
Levenshtein distance (restricted to equal-length strings with substitutions
only)" — that is a genuine structural insight. Saying "Levenshtein is a
special case of a generalization I built by parameterizing Levenshtein" is
circular.

---

## 3. What Already Exists

| Tool / Paper | What it does | How AED compares |
|---|---|---|
| **Zhang-Shasha (1989)** | Edit distance on ordered labeled trees, O(n²-n⁴) | AED for Seq-only structures is essentially this with normalized costs |
| **RTED / APTED** | Optimal ordered tree edit distance | More efficient than AED for the same problem |
| **zss** (Python package) | Zhang-Shasha in Python | Exists, maintained, handles ordered trees |
| **deepdiff** (Python package) | Deep comparison of Python objects | NOT a metric (no triangle inequality), but has diff/patch/delta. 8.6.1, mature, 4M downloads/month |
| **jsondiff** | JSON diff | Patches, not a metric |
| **dictdiffer** | Dict comparison | Patches, not a metric |
| **Jiang et al. (1995)** | Tree alignment with recursive costs | Exactly the technique AED uses for sequences |
| **Graph edit distance** | Metric on arbitrary graphs | True generalization, but NP-hard |

### The Gap AED Claims to Fill

The docstring argues: "Tree edit distance handles ordered trees only. JSON
diff tools don't produce a metric. Graph edit distance is NP-hard."

This is accurate. There IS a gap: no existing widely-used tool computes a
**polynomial-time metric** on **mixed ordered/unordered** structured data.

But the reason this gap exists is that it's **trivially fillable** once you
make the key design decision (use DP for ordered children, key-matching for
unordered children). The gap wasn't a hard open problem — it was a gap nobody
bothered to package, because the solution is straightforward.

---

## 4. The Honest Verdict

### What structdist IS

- A **clean, well-documented Python library** for computing a distance metric
  on structured data (JSON, configs, ASTs, etc.).
- A **correct metric** (modulo the two bugs found) with verified triangle
  inequality, symmetry, and identity.
- A **useful engineering tool** for config diffing, schema comparison, and
  structured merging.
- **Better than deepdiff** for use cases requiring a scalar metric (deepdiff
  gives you a diff object, not a number).
- A **nice API** with `from_python()`, `to_python()`, `diff()`, `patch()`,
  `merge()`.

### What structdist is NOT

- **Not a new algorithm.** The DP sequence alignment, key-based map matching,
  recursive substitution costs, and size-weighted ins/del costs are all
  standard techniques from the tree edit distance and sequence alignment
  literature.
- **Not a new mathematical concept.** "Apply edit distance to algebraic data
  types" is a natural engineering idea, not a contribution to mathematics.
- **Not solving a hard problem.** Unordered tree edit distance is NP-hard.
  AED sidesteps this by requiring string keys, which gives a trivial
  polynomial-time solution. This is avoiding the hard problem, not solving it.
- **Not Levenshtein-level.** Levenshtein (1965) *invented* the concept of edit
  distance. AED *applies* a generalization of that concept using known
  techniques. The distance between "inventing a concept" and "packaging known
  concepts" is the distance between structdist and Levenshtein.

### The Same Pattern as DensePDF

This is the same pattern we identified in RED/AXIOM/GEM:

> Competent engineering. Well-implemented. Useful in practice.
> But not a fundamental contribution to computer science.

Each component of AED — DP alignment, key matching, recursive costs, size
weighting, metric verification — is individually well-known. The combination
is natural and useful. The implementation is clean. But the act of combining
known techniques into a library, no matter how clean, is engineering, not
science.

### Distance from Levenshtein-Level

**Very far.**

| Criterion | Levenshtein (1965) | AED / structdist |
|---|---|---|
| Novelty of concept | Invented edit distance | Applies edit distance to structured data |
| Mathematical depth | Defined a new metric space on sequences | Routine verification of known metric properties |
| Problem difficulty | No prior solution existed | Gap is trivially fillable |
| Influence potential | Spawned entire fields (bioinformatics, NLP, spell-checking, coding theory) | Useful niche library |
| Conceptual surprise | High — nobody had formalized string similarity this way | Low — "generalize Levenshtein to trees" is the obvious next step |
| Year of first related work | 1965 (the concept itself) | 1979 (Tai's tree edit distance) — so the generalization has been studied for 47 years |

---

## 5. Bugs to Fix

### Bug 1: Bool/int conflation
```python
# In atom_distance(), BEFORE the `if av == bv` check:
if isinstance(av, bool) != isinstance(bv, bool):
    return 1.0  # bool vs non-bool is a type mismatch
if type(av) is bool and type(bv) is bool:
    return 0.0 if av == bv else 1.0
```

### Bug 2: _patch_tagged inner type change
```python
# In _patch_tagged(), when entry.path[-1] == "inner":
# Replace: patch(inner, [entry])
# With:    entry.new  (use the diff's stored new value directly)
```

---

## 6. What Would Be Levenshtein-Level?

To be clear about the bar: a Levenshtein-level contribution would need to
**invent a concept that doesn't exist yet** and that becomes **fundamental
to how we think** about some class of problems. Some directions:

1. **Solve unordered tree edit distance in polynomial time** (or prove a
   tighter hardness result). This is an actual open problem.

2. **Define a metric on programs/computations** (not just their ASTs, but
   their *behavior*). Edit distance on λ-calculus terms modulo β-reduction,
   for instance. This doesn't exist and would be genuinely new mathematics.

3. **Compression distance that actually works** — Normalized Compression
   Distance (NCD, Li & Vitányi 2004) is theoretically beautiful but
   computationally vacuous because real compressors aren't Kolmogorov-optimal.
   A practical, polynomial-time, metric-satisfying compression distance would
   be a real contribution.

4. **A distance metric on TYPE SYSTEMS** — not on values, but on the types
   themselves. How far apart are two type systems? This connects to the
   geometry of type spaces, which is largely unexplored.

None of these are easy. That's why Levenshtein-level contributions are rare.

---

## 7. Recommendation

**structdist is worth publishing as a library.** It fills a real (if modest)
gap. The API is clean. The metric properties are correct. Fix the two bugs,
write good docs, push to PyPI.

But **do not claim** it's a new contribution to computer science or
mathematics. The docstring's §6 ("What is New") significantly overstates the
novelty. The phrase "AED is to structured data what Levenshtein is to strings:
the NATURAL, CANONICAL distance metric" is marketing, not mathematics. Tree
edit distance has been that for 47 years.

**Call it what it is:** a useful, well-engineered Python library that packages
known algorithms into a clean API for structured data comparison. That's a
good thing to build. It's not a Levenshtein.
