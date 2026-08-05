# SPARK-58598 - Design Document

Extend `verify_return_type` for `Iterable`/`Collection`, reuse it across the
scattered pandas UDF return-type checks, unify the emitted error message, and
loosen the pinned test assertions.

JIRA: https://issues.apache.org/jira/browse/SPARK-58598 (Improvement, PySpark)

## Overview

`verify_return_type(result, expected_type)` (`python/pyspark/worker.py:248`)
already centralizes the `UDF_RETURN_TYPE` check for `Iterator[T]` and plain `T`,
and `mapInArrow` already reuses it via `verify_return_type(result,
Iterator[pa.RecordBatch])`. But several pandas UDF paths still open-code the same
`raise PySparkTypeError(errorClass="UDF_RETURN_TYPE", ...)` with hand-written
`isinstance` / `hasattr` checks and hand-written `expected` strings. This causes
duplicated logic and inconsistent wording (`iterator of` vs `iterable of` vs a
pre-computed `iter_type_label`).

This change extends the helper to cover two more type shapes, converts the
open-coded sites to reuse it, lets the emitted `expected` text be built
automatically from the type expression (one wording rule), and relaxes the tests
that pinned the old exact wording.

## Architecture

### Component 1: extend `verify_return_type`

Add two dispatch branches. The `expected` label is always auto-built from the
type expression; there is no per-site custom label parameter.

| Type expression | dispatch (`get_origin`) | runtime check | auto label |
|---|---|---|---|
| `Iterator[X]` (existing) | `collections.abc.Iterator` | `isinstance(result, abc.Iterator)` + per-element | `iterator of pkg.X` |
| `Iterable[X]` (new) | `collections.abc.Iterable` | `isinstance(result, abc.Iterable)` + per-element | `iterable of pkg.X` |
| `Collection[X]` (new) | `collections.abc.Collection` | `isinstance(result, abc.Collection)` (no per-element) | `pkg.X` |
| `X` (existing) | (none) | `isinstance(result, X)` | `pkg.X` |

- `Iterator` and `Iterable` share the same lazy per-element `check_element`
  logic; the only differences are the outer ABC checked and the label prefix
  (`iterator of` vs `iterable of`). Factor the shared element-checking into a
  small helper to avoid duplication.
- `Collection[X]` does NOT check elements. It replaces the array-like
  `hasattr(result, "__len__")` gate; element type is not meaningful there, so
  the label is the bare `pkg.X` (e.g. `pandas.Series`), matching what those
  sites emit today.
- `abc.Collection` = `Sized & Iterable & Container`. Verified that
  `pd.Series` / `pd.DataFrame` / `np.ndarray` / `list` all satisfy it.
  `Sized[T]` is not subscriptable and `Sequence[T]` would false-reject
  pandas/numpy, so `Collection[T]` is the only correct subscriptable choice.

### Import change

Today `Iterator` comes from `collections.abc` (used for `isinstance`) while
`Iterable` comes from `typing` (annotation-only). Move `Iterable` to the
`collections.abc` import and add `Collection` there, so both work with
`isinstance` and with `get_origin`. (`typing.Iterable[X].__origin__` is also
`collections.abc.Iterable`, so callers may write either; the runtime check needs
the abc class.)

### Component 2: convert 5 sites to reuse the helper

| Site (approx line) | current | change to | wording effect |
|---|---|---|---|
| mapInPandas outer (~2705) | hand `isinstance(Iterator) or hasattr(__iter__)`, `"iterator of {iter_type_label}"` | `verify_return_type(result, Iterable[elem_type])` | `iterator of` -> `iterable of` |
| mapInPandas element (~2714) | hand per-element loop | absorbed by the `Iterable` branch's per-element check | same as above |
| scalar pandas UDF (~3025) | hand `hasattr(__len__)`, dynamic `pandas.DataFrame`/`Series` | `verify_return_type(result, Collection[elem_type])` | wording unchanged; SEMANTIC TIGHTENING (below) |
| grouped-map `verify_element` (~3616) | hand `isinstance(pd.DataFrame)` | reuse via `Iterator[pd.DataFrame]` element check | wording unchanged |
| applyInPandasWithState iter() (~3670) | hand `iter()` + `TypeError`, label `"iterable"` | `verify_return_type(result, Iterable[pd.DataFrame])` | `iterable` -> `iterable of pandas.DataFrame` |

Also delete the now-obsolete comment (~2701) that says `verify_return_type` is
"intentionally not reused here" because it requires an `Iterator`.

EXCLUDE the reverse-logic site (~3662): `isinstance(result_iter, pd.DataFrame)
-> raise` ("must NOT be a DataFrame") is incompatible with the forward-match
model of `verify_return_type`; leave it hand-written.

### Component 3: loosen tests (match the `arrow_map` style)

`arrow_map` already asserts with token-anchored regexes (e.g.
`r"iterator of pyarrow\.RecordBatch.*\bint\b"`). Bring the pandas-side tests to
the same style: the regex anchors only the semantic tokens (expected type name,
actual type name), NOT the fixed `Return type of ... should be ... but is ...`
sentence frame and NOT the `iterator` vs `iterable` prefix word. Future wording
tweaks then won't break tests.

Files to update (verify exact lines before editing):

- `pandas/test_pandas_map.py` (~191, ~198) - exact strings -> token regex; also
  absorbs `iterator` -> `iterable`.
- `pandas/test_pandas_cogrouped_map.py` (~167) - already `error_message_regex`;
  loosen anchor.
- `pandas/test_pandas_grouped_map.py` (~325) - exact string -> token regex.
- applyInPandasWithState tests asserting the `"iterable"` wording - re-grep and
  loosen.
- `arrow/test_arrow_map.py`, `arrow/test_arrow_grouped_map.py` - already loose;
  confirm no change needed.

## Design Decisions

1. **Fully auto-generated, unified label.** Wording is determined solely by the
   type shape (`Iterator[X]` / `Iterable[X]` / `Collection[X]` / `X`). No
   per-site `label=` override. Sites that need different wording change the type
   expression, not a string.
2. **`Collection[X]` uses the bare `pkg.X` label** (no prefix), because it
   replaces sites that already emit the bare type name and check `__len__`.
3. **Scalar pandas UDF (~3025) is a user-facing semantic tightening, not a
   no-op.** Moving from `hasattr("__len__")` to `abc.Collection` newly rejects
   an object that has `__len__` but no `__iter__`. In practice pandas/numpy
   returns are full Collections and a len-only non-iterable would crash
   downstream anyway, but this MUST be called out explicitly in the PR
   description for reviewer sign-off, not presented as a pure refactor.
4. **Single PR / single JIRA (SPARK-58598).** Extend + reuse + unify message +
   loosen tests ship together so the reviewer sees the full cause/effect. The
   message unification and test loosening are the direct consequence of reuse.
5. **grouped-map (~3616) reuses `Iterator[pd.DataFrame]` element semantics**,
   not `Iterable`: the outer iterable-ness is already validated at ~3670;
   `verify_element` only checks that each element is a DataFrame.

## Rules & Conventions

- PR title: `[SPARK-58598][PYTHON] ...`; concise description.
- Line length <= 100; ASCII only in code/comments.
- New worktree `~/spark-SPARK-58598`; never develop in the main repo.
- Push to personal fork; open PR against upstream `master`.
- External ops (push / open PR) require user approval first.
- Verify: `build/sbt -Phive package`, then run the pandas + arrow map / grouped /
  cogrouped / applyInPandasWithState suites; fix all pinned assertions.

## References

- Prior ticket: SPARK-56612 (Resolved) - introduced `verify_return_type`.
- Helper: `python/pyspark/worker.py:248` `verify_return_type`.
- Existing reuse precedent: `worker.py:2455` (mapInArrow).
