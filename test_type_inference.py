#!/usr/bin/env python3
"""
Test type inference logic for iterator Arrow grouped agg UDF (no Spark required)
"""
import sys
sys.path.insert(0, 'python')

print("Step 1: Importing modules...")
from pyspark.sql.pandas.typehints import infer_arrow_eval_type
from pyspark.sql.pandas.functions import ArrowUDFType
from typing import Iterator, Tuple, get_type_hints
from inspect import signature
import pyarrow as pa

print("\nStep 2: Testing type inference for Iterator[pa.Array] -> float (single column)")

def arrow_sum_iter(it: Iterator[pa.Array]) -> float:
    total = 0.0
    for v in it:
        total += pa.compute.sum(v).as_py()
    return total

sig = signature(arrow_sum_iter)
type_hints = get_type_hints(arrow_sum_iter)
print(f"  Function signature: {sig}")
print(f"  Type hints: {type_hints}")

result = infer_arrow_eval_type(sig, type_hints)
print(f"  Inferred eval type: {result}")
print(f"  Expected: {ArrowUDFType.GROUPED_AGG_ITER}")

assert result == ArrowUDFType.GROUPED_AGG_ITER, \
    f"Expected {ArrowUDFType.GROUPED_AGG_ITER}, got {result}"
print("  ✓ Single column iterator type inference is correct!")

print("\nStep 3: Testing type inference for Iterator[Tuple[pa.Array, pa.Array]] -> float (multiple columns)")

def arrow_weighted_sum_iter(it: Iterator[Tuple[pa.Array, pa.Array]]) -> float:
    total = 0.0
    for v, w in it:
        total += pa.compute.sum(pa.compute.multiply(v, w)).as_py()
    return total

sig = signature(arrow_weighted_sum_iter)
type_hints = get_type_hints(arrow_weighted_sum_iter)
print(f"  Function signature: {sig}")
print(f"  Type hints: {type_hints}")

result = infer_arrow_eval_type(sig, type_hints)
print(f"  Inferred eval type: {result}")
print(f"  Expected: {ArrowUDFType.GROUPED_AGG_ITER}")

assert result == ArrowUDFType.GROUPED_AGG_ITER, \
    f"Expected {ArrowUDFType.GROUPED_AGG_ITER}, got {result}"
print("  ✓ Multiple columns iterator type inference is correct!")

print("\nStep 4: Testing regular (non-iterator) Arrow grouped agg for comparison")

def arrow_sum_regular(v: pa.Array) -> float:
    return pa.compute.sum(v).as_py()

sig = signature(arrow_sum_regular)
type_hints = get_type_hints(arrow_sum_regular)
print(f"  Function signature: {sig}")
print(f"  Type hints: {type_hints}")

result = infer_arrow_eval_type(sig, type_hints)
print(f"  Inferred eval type: {result}")
print(f"  Expected: {ArrowUDFType.GROUPED_AGG}")

assert result == ArrowUDFType.GROUPED_AGG, \
    f"Expected {ArrowUDFType.GROUPED_AGG}, got {result}"
print("  ✓ Regular (non-iterator) type inference is correct!")

print("\nStep 5: Testing scalar iterator (should NOT be grouped agg iter)")

def arrow_scalar_iter(it: Iterator[pa.Array]) -> Iterator[pa.Array]:
    for v in it:
        yield pa.compute.add(v, 1)

sig = signature(arrow_scalar_iter)
type_hints = get_type_hints(arrow_scalar_iter)
print(f"  Function signature: {sig}")
print(f"  Type hints: {type_hints}")

result = infer_arrow_eval_type(sig, type_hints)
print(f"  Inferred eval type: {result}")
print(f"  Expected: {ArrowUDFType.SCALAR_ITER}")

assert result == ArrowUDFType.SCALAR_ITER, \
    f"Expected {ArrowUDFType.SCALAR_ITER}, got {result}"
print("  ✓ Scalar iterator type inference is correct (not confused with grouped agg iter)!")

print("\n" + "="*60)
print("✅ All type inference tests passed!")
print("="*60)

