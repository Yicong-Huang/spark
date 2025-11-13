#!/usr/bin/env python3
"""
Test worker wrapping logic for iterator Arrow grouped agg UDF (no Spark required)
"""
import sys
sys.path.insert(0, 'python')

print("Step 1: Importing modules...")
from pyspark.worker import wrap_grouped_agg_arrow_iter_udf
from pyspark.sql.types import DoubleType
from typing import Iterator
import pyarrow as pa
import numpy as np

print("\nStep 2: Testing single column iterator wrapper")

def arrow_mean_iter(it: Iterator[pa.Array]) -> float:
    print("    Inside arrow_mean_iter")
    sum_val = 0.0
    cnt = 0
    for v in it:
        print(f"      Processing batch with {len(v)} elements")
        sum_val += pa.compute.sum(v).as_py()
        cnt += len(v)
    result = sum_val / cnt if cnt > 0 else 0.0
    print(f"    Returning: {result}")
    return result

# Wrap the function
runner_conf = {}
args_offsets = [0]
kwargs_offsets = {}
return_type = DoubleType()

print("  Wrapping function...")
wrapped_offsets, wrapped_func = wrap_grouped_agg_arrow_iter_udf(
    arrow_mean_iter, args_offsets, kwargs_offsets, return_type, runner_conf
)

print(f"  Wrapped offsets: {wrapped_offsets}")

# Create test data - iterator of arrays
print("\n  Creating test data...")
test_arrays = [
    pa.array([1.0, 2.0, 3.0]),
    pa.array([4.0, 5.0, 6.0]),
    pa.array([7.0, 8.0, 9.0])
]

print("  Calling wrapped function...")
result_array, result_type = wrapped_func(iter(test_arrays))

print(f"  Result array: {result_array}")
print(f"  Result type: {result_type}")
print(f"  Result value: {result_array[0].as_py()}")

expected = 5.0  # mean of 1-9
actual = result_array[0].as_py()
assert abs(actual - expected) < 0.001, f"Expected ~{expected}, got {actual}"
print(f"  ✓ Single column iterator wrapper works correctly! (mean = {actual})")

print("\nStep 3: Testing multiple columns iterator wrapper")

def arrow_weighted_mean_iter(it: Iterator) -> float:
    print("    Inside arrow_weighted_mean_iter")
    weighted_sum = 0.0
    weight = 0.0
    for v, w in it:
        print(f"      Processing batch with {len(v)} elements")
        weighted_sum += np.dot(v.to_numpy(), w.to_numpy())
        weight += pa.compute.sum(w).as_py()
    result = weighted_sum / weight if weight > 0 else 0.0
    print(f"    Returning: {result}")
    return result

# Wrap the function
args_offsets = [0, 1]
kwargs_offsets = {}

print("  Wrapping function...")
wrapped_offsets, wrapped_func = wrap_grouped_agg_arrow_iter_udf(
    arrow_weighted_mean_iter, args_offsets, kwargs_offsets, return_type, runner_conf
)

print(f"  Wrapped offsets: {wrapped_offsets}")

# Create test data - two iterators
print("\n  Creating test data...")
values = [
    pa.array([1.0, 2.0, 3.0]),
    pa.array([4.0, 5.0])
]
weights = [
    pa.array([1.0, 1.0, 1.0]),
    pa.array([1.0, 1.0])
]

print("  Calling wrapped function...")
result_array, result_type = wrapped_func(iter(values), iter(weights))

print(f"  Result array: {result_array}")
print(f"  Result type: {result_type}")
print(f"  Result value: {result_array[0].as_py()}")

# Weighted mean with equal weights = regular mean
expected = 3.0  # mean of [1, 2, 3, 4, 5]
actual = result_array[0].as_py()
assert abs(actual - expected) < 0.001, f"Expected ~{expected}, got {actual}"
print(f"  ✓ Multiple columns iterator wrapper works correctly! (weighted mean = {actual})")

print("\n" + "="*60)
print("✅ All worker wrapping tests passed!")
print("="*60)

