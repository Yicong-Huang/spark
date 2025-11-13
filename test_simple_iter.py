#!/usr/bin/env python3
"""
Test the Python-side iterator logic without needing Spark
"""
import sys
sys.path.insert(0, 'python')

print("Step 1: Testing serializer ColumnIterators class...")

# Test the ColumnIterators class
import pyarrow as pa

class ColumnIterators:
    def __init__(self, batches, num_cols):
        self._batches = batches
        self._num_cols = num_cols
    
    def __getitem__(self, col_idx):
        print(f"  Getting iterator for column {col_idx}")
        return (batch.column(col_idx) for batch in self._batches)
    
    def __len__(self):
        return self._num_cols

# Create test batches
batch1 = pa.RecordBatch.from_arrays(
    [pa.array([1.0, 2.0])],
    names=['v']
)
batch2 = pa.RecordBatch.from_arrays(
    [pa.array([3.0])],
    names=['v']
)

batches = [batch1, batch2]
col_iters_obj = ColumnIterators(batches, 1)

print(f"  ColumnIterators created with {len(batches)} batches")
print(f"  Number of columns: {len(col_iters_obj)}")

# Test accessing column 0
print("\nStep 2: Testing column iterator access...")
col_0_iter = col_iters_obj[0]
print(f"  Got iterator for column 0: {col_0_iter}")

arrays = list(col_0_iter)
print(f"  Consumed iterator, got {len(arrays)} arrays")
for i, arr in enumerate(arrays):
    print(f"    Array {i}: {arr.to_pylist()}")

# Test with the wrapper function
print("\nStep 3: Testing wrapper function...")
from typing import Iterator

def arrow_mean_iter(it: Iterator[pa.Array]) -> float:
    print("  Inside arrow_mean_iter")
    sum_val = 0.0
    cnt = 0
    for i, v in enumerate(it):
        print(f"    Processing array {i}: {v.to_pylist()}")
        sum_val += pa.compute.sum(v).as_py()
        cnt += len(v)
    result = sum_val / cnt if cnt > 0 else 0.0
    print(f"  Returning: {result}")
    return result

# Simulate what the mapper does: col_iters_obj[0] for column 0
col_iters_obj2 = ColumnIterators(batches, 1)
result = arrow_mean_iter(col_iters_obj2[0])
print(f"\nStep 4: Result = {result}")
print(f"Expected: {(1.0 + 2.0 + 3.0) / 3} = 2.0")

if abs(result - 2.0) < 0.001:
    print("\n✅ Test passed!")
else:
    print(f"\n❌ Test failed! Expected 2.0, got {result}")

