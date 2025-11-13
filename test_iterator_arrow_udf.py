#!/usr/bin/env python3
"""
Simple test script for iterator Arrow grouped agg UDF
"""
import sys
sys.path.insert(0, 'python')

print("Step 1: Importing modules...")
from pyspark.sql.functions import arrow_udf
from pyspark.util import PythonEvalType
from typing import Iterator
import pyarrow as pa

print("Step 2: Testing type inference for Iterator[pa.Array] -> float")

# Test 1: Single column iterator
@arrow_udf("double")
def arrow_sum_iter(it: Iterator[pa.Array]) -> float:
    print("  Inside arrow_sum_iter UDF")
    total = 0.0
    for v in it:
        total += pa.compute.sum(v).as_py()
    return total

print(f"  Eval type: {arrow_sum_iter.evalType}")
print(f"  Expected: {PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF}")
assert arrow_sum_iter.evalType == PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF, \
    f"Expected {PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF}, got {arrow_sum_iter.evalType}"
print("  ✓ Single column iterator eval type is correct!")

print("\nStep 3: Testing type inference for Iterator[Tuple[pa.Array, pa.Array]] -> float")

# Test 2: Multiple columns iterator
from typing import Tuple

@arrow_udf("double")
def arrow_weighted_sum_iter(it: Iterator[Tuple[pa.Array, pa.Array]]) -> float:
    print("  Inside arrow_weighted_sum_iter UDF")
    total = 0.0
    for v, w in it:
        total += pa.compute.sum(pa.compute.multiply(v, w)).as_py()
    return total

print(f"  Eval type: {arrow_weighted_sum_iter.evalType}")
print(f"  Expected: {PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF}")
assert arrow_weighted_sum_iter.evalType == PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF, \
    f"Expected {PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF}, got {arrow_weighted_sum_iter.evalType}"
print("  ✓ Multiple columns iterator eval type is correct!")

print("\nStep 4: Testing regular (non-iterator) Arrow grouped agg for comparison")

@arrow_udf("double")
def arrow_sum_regular(v: pa.Array) -> float:
    return pa.compute.sum(v).as_py()

print(f"  Eval type: {arrow_sum_regular.evalType}")
print(f"  Expected: {PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF}")
assert arrow_sum_regular.evalType == PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF, \
    f"Expected {PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF}, got {arrow_sum_regular.evalType}"
print("  ✓ Regular (non-iterator) eval type is correct!")

print("\n" + "="*60)
print("✅ All basic type inference tests passed!")
print("="*60)

