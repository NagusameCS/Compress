"""
structdist.merge — Three-way merge using algebraic edit distance.

Given a common ancestor (base) and two diverged versions (ours, theirs),
compute a merged result by applying non-conflicting edits from both sides.

This is structurally analogous to `git merge` but operates on algebraic
data types instead of text lines — so it can merge JSON objects, configs,
ASTs, etc. with structural awareness.

ALGORITHM:
    1. Compute diff(base, ours)  → edit_script_A
    2. Compute diff(base, theirs) → edit_script_B
    3. For each path:
       • Changed only in A → accept A's change
       • Changed only in B → accept B's change
       • Changed in both A and B identically → accept (no conflict)
       • Changed in both A and B differently → CONFLICT
    4. Apply merged edits to base
"""

from dataclasses import dataclass
from typing import Optional

from .core import (
    AAtom, AMap, ASeq, ATagged, AVal,
    EditEntry, EditOp,
    diff, distance, patch,
)


@dataclass
class ConflictEntry:
    """A merge conflict — the same path was edited differently in both sides."""
    path: tuple
    ours: Optional[AVal]
    theirs: Optional[AVal]
    base: Optional[AVal]

    def __repr__(self) -> str:
        path_str = "/".join(str(p) for p in self.path) or "(root)"
        return f"CONFLICT at {path_str}: ours={self.ours!r}, theirs={self.theirs!r}"


@dataclass
class MergeResult:
    """Result of a three-way merge."""
    merged: AVal
    conflicts: list[ConflictEntry]
    has_conflicts: bool

    def __repr__(self) -> str:
        if self.has_conflicts:
            return f"MergeResult(CONFLICTS: {len(self.conflicts)})"
        return f"MergeResult(clean merge)"


def merge(base: AVal, ours: AVal, theirs: AVal) -> MergeResult:
    """
    Three-way merge of structured data.

    Arguments:
        base:   The common ancestor
        ours:   Our modified version
        theirs: Their modified version

    Returns a MergeResult with the merged value and any conflicts.

    For maps (JSON objects), this merges key-by-key:
        • Keys changed only on one side → accept that change
        • Keys changed on both sides identically → accept
        • Keys changed on both sides differently → conflict

    For sequences, if both sides made changes, we fall back to
    accepting ours (with conflict noted) since sequence merge is
    inherently ambiguous without line-level heuristics.

    For atoms, if both changed differently → conflict, pick ours.
    """
    conflicts: list[ConflictEntry] = []
    merged = _merge_recursive(base, ours, theirs, (), conflicts)
    return MergeResult(merged=merged, conflicts=conflicts,
                       has_conflicts=len(conflicts) > 0)


def _vals_equal(a: AVal, b: AVal) -> bool:
    """Check structural equality of two values."""
    return distance(a, b) < 1e-9


def _merge_recursive(
    base: AVal, ours: AVal, theirs: AVal,
    path: tuple, conflicts: list[ConflictEntry]
) -> AVal:
    """Recursive three-way merge."""

    base_ours_eq = _vals_equal(base, ours)
    base_theirs_eq = _vals_equal(base, theirs)

    # Neither side changed → keep base
    if base_ours_eq and base_theirs_eq:
        return base

    # Only we changed → accept ours
    if base_theirs_eq and not base_ours_eq:
        return ours

    # Only they changed → accept theirs
    if base_ours_eq and not base_theirs_eq:
        return theirs

    # Both changed — check if identically
    if _vals_equal(ours, theirs):
        return ours

    # Both changed differently — type-specific merge
    if isinstance(base, AMap) and isinstance(ours, AMap) and isinstance(theirs, AMap):
        return _merge_maps(base, ours, theirs, path, conflicts)

    if isinstance(base, ASeq) and isinstance(ours, ASeq) and isinstance(theirs, ASeq):
        return _merge_seqs(base, ours, theirs, path, conflicts)

    if isinstance(base, ATagged) and isinstance(ours, ATagged) and isinstance(theirs, ATagged):
        return _merge_tagged(base, ours, theirs, path, conflicts)

    # Atom or type mismatch conflict — pick ours, record conflict
    conflicts.append(ConflictEntry(path=path, ours=ours, theirs=theirs, base=base))
    return ours


def _merge_maps(
    base: AMap, ours: AMap, theirs: AMap,
    path: tuple, conflicts: list[ConflictEntry]
) -> AMap:
    """Merge two maps key-by-key against a common base."""
    all_keys = set(base.entries.keys()) | set(ours.entries.keys()) | set(theirs.entries.keys())

    merged_entries: dict[str, AVal] = {}

    for key in sorted(all_keys):
        child_path = path + (key,)

        in_base = key in base.entries
        in_ours = key in ours.entries
        in_theirs = key in theirs.entries

        b_val = base.entries.get(key)
        o_val = ours.entries.get(key)
        t_val = theirs.entries.get(key)

        if in_base and in_ours and in_theirs:
            # Key exists in all three — recursive merge
            merged_entries[key] = _merge_recursive(b_val, o_val, t_val, child_path, conflicts)

        elif in_base and in_ours and not in_theirs:
            # They deleted this key
            if _vals_equal(b_val, o_val):
                # We didn't change it, accept their deletion
                pass
            else:
                # We modified, they deleted → conflict, keep ours
                conflicts.append(ConflictEntry(path=child_path, ours=o_val, theirs=None, base=b_val))
                merged_entries[key] = o_val

        elif in_base and not in_ours and in_theirs:
            # We deleted this key
            if _vals_equal(b_val, t_val):
                # They didn't change it, accept our deletion
                pass
            else:
                # They modified, we deleted → conflict, accept deletion (ours)
                conflicts.append(ConflictEntry(path=child_path, ours=None, theirs=t_val, base=b_val))

        elif in_base and not in_ours and not in_theirs:
            # Both deleted → accept deletion
            pass

        elif not in_base and in_ours and in_theirs:
            # Both added — check if same value
            if _vals_equal(o_val, t_val):
                merged_entries[key] = o_val
            else:
                conflicts.append(ConflictEntry(path=child_path, ours=o_val, theirs=t_val, base=None))
                merged_entries[key] = o_val

        elif not in_base and in_ours and not in_theirs:
            # Only we added → accept
            merged_entries[key] = o_val

        elif not in_base and not in_ours and in_theirs:
            # Only they added → accept
            merged_entries[key] = t_val

    return AMap(merged_entries)


def _merge_seqs(
    base: ASeq, ours: ASeq, theirs: ASeq,
    path: tuple, conflicts: list[ConflictEntry]
) -> ASeq:
    """
    Merge two sequences against a common base.

    Sequence merge is inherently harder than map merge because there's
    no natural "key" to align elements.  We use a simple strategy:

    If both modified and the changes are at different positions (non-overlapping),
    try to integrate both.  Otherwise, conflict → pick ours.
    """
    # Simple element-wise merge when sequences have the same length
    if len(base.items) == len(ours.items) == len(theirs.items):
        merged_items = []
        has_conflict = False
        for i, (b_item, o_item, t_item) in enumerate(zip(base.items, ours.items, theirs.items)):
            child_path = path + (i,)
            result = _merge_recursive(b_item, o_item, t_item,
                                       child_path, conflicts)
            merged_items.append(result)
        return ASeq(tuple(merged_items))

    # Different lengths — accept ours with conflict
    conflicts.append(ConflictEntry(path=path, ours=ours, theirs=theirs, base=base))
    return ours


def _merge_tagged(
    base: ATagged, ours: ATagged, theirs: ATagged,
    path: tuple, conflicts: list[ConflictEntry]
) -> ATagged:
    """Merge two tagged values against a common base."""
    # Merge tags
    if ours.tag == theirs.tag:
        tag = ours.tag
    elif ours.tag == base.tag:
        tag = theirs.tag  # They changed the tag
    elif theirs.tag == base.tag:
        tag = ours.tag  # We changed the tag
    else:
        # Both changed tag differently — conflict, use ours
        conflicts.append(ConflictEntry(
            path=path + ("tag",),
            ours=AAtom(ours.tag), theirs=AAtom(theirs.tag),
            base=AAtom(base.tag)
        ))
        tag = ours.tag

    # Merge inner
    inner = _merge_recursive(base.inner, ours.inner, theirs.inner,
                              path + ("inner",), conflicts)

    return ATagged(tag, inner)
