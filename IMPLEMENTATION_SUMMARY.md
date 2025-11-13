# Iterator API for Arrow Grouped Aggregation UDF - Implementation Summary

## JIRA Ticket
SPARK-53615: Introduce iterator API for arrow grouped agg UDF

## Overview
This implementation adds iterator API support for Arrow grouped aggregation UDFs in PySpark, allowing users to process grouped data in batches using iterators instead of receiving all data at once.

## API Examples

### Single Column
```python
import pyarrow as pa
from typing import Iterator
from pyspark.sql.functions import arrow_udf

@arrow_udf("double")
def arrow_mean(it: Iterator[pa.Array]) -> float:
    sum_val = 0.0
    cnt = 0
    for v in it:
        assert isinstance(v, pa.Array)
        sum_val += pa.compute.sum(v).as_py()
        cnt += len(v)
    return sum_val / cnt
```

### Multiple Columns
```python
import pyarrow as pa
import numpy as np
from typing import Iterator, Tuple
from pyspark.sql.functions import arrow_udf

@arrow_udf("double")
def arrow_weighted_mean(it: Iterator[Tuple[pa.Array, pa.Array]]) -> float:
    weighted_sum = 0.0
    weight = 0.0
    for v, w in it:
        assert isinstance(v, pa.Array)
        assert isinstance(w, pa.Array)
        weighted_sum += np.dot(v, w)
        weight += pa.compute.sum(w).as_py()
    return weighted_sum / weight
```

## Changes Made

### 1. Python Eval Type Definition
**Files Modified:**
- `python/pyspark/util.py`
- `python/pyspark/sql/pandas/_typing/__init__.pyi`

**Changes:**
- Added new eval type: `SQL_GROUPED_AGG_ARROW_ITER_UDF = 254`
- Added type annotation: `ArrowGroupedAggIterUDFType = Literal[254]`

### 2. Type Hint Inference
**File Modified:** `python/pyspark/sql/pandas/typehints.py`

**Changes:**
- Extended `infer_arrow_eval_type()` to recognize:
  - `Iterator[pa.Array] -> Any` (single column)
  - `Iterator[Tuple[pa.Array, ...]] -> Any` (multiple columns)
- Returns `ArrowUDFType.GROUPED_AGG_ITER` for these patterns

### 3. Arrow UDF Functions
**File Modified:** `python/pyspark/sql/pandas/functions.py`

**Changes:**
- Added `GROUPED_AGG_ITER` to `ArrowUDFType` class
- Updated validation logic to include the new eval type
- Added documentation with examples for iterator-based aggregations

### 4. Python Worker
**File Modified:** `python/pyspark/worker.py`

**Changes:**
- Added `wrap_grouped_agg_arrow_iter_udf()` function to wrap iterator UDFs
- Updated eval type handling in `read_single_udf()` 
- Added support for named arguments for the new eval type
- Imports `ArrowStreamAggArrowIterUDFSerializer`

**Key Implementation:**
```python
def wrap_grouped_agg_arrow_iter_udf(f, args_offsets, kwargs_offsets, return_type, runner_conf):
    func, args_kwargs_offsets = wrap_kwargs_support(f, args_offsets, kwargs_offsets)
    arrow_return_type = to_arrow_type(return_type, prefers_large_types=use_large_var_types(runner_conf))
    
    def wrapped(*col_iters):
        import pyarrow as pa
        
        if len(col_iters) == 1:
            # Single column: Iterator[pa.Array] -> Any
            result = func(col_iters[0])
        else:
            # Multiple columns: Iterator[Tuple[pa.Array, ...]] -> Any
            result = func(zip(*col_iters))
        
        return pa.array([result])
    
    return (args_kwargs_offsets, lambda *a: (wrapped(*a), arrow_return_type))
```

### 5. Serializer
**File Modified:** `python/pyspark/sql/pandas/serializers.py`

**Changes:**
- Added `ArrowStreamAggArrowIterUDFSerializer` class
- Yields column iterators instead of concatenating batches
- Uses custom `ColumnIterators` class to provide indexable access to column iterators

**Key Implementation:**
```python
class ArrowStreamAggArrowIterUDFSerializer(ArrowStreamArrowUDFSerializer):
    def load_stream(self, stream):
        # ... reads batches for each group ...
        
        class ColumnIterators:
            def __init__(self, batches, num_cols):
                self._batches = batches
                self._num_cols = num_cols
            
            def __getitem__(self, col_idx):
                return (batch.column(col_idx) for batch in self._batches)
            
            def __len__(self):
                return self._num_cols
        
        yield ColumnIterators(batches, num_cols)
```

### 6. Scala-side Support
**Files Modified:**
- `core/src/main/scala/org/apache/spark/api/python/PythonRunner.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/python/UserDefinedPythonFunction.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/python/ArrowAggregatePythonExec.scala`

**Changes:**
- Added `SQL_GROUPED_AGG_ARROW_ITER_UDF = 254` constant
- Updated `toString()` method to include the new eval type
- Added eval type to `NamedParametersSupport` conditions
- Added eval type to `PythonUDAF` creation logic
- Added eval type to `supportedPythonEvalTypes` in `ArrowAggregatePythonExec`

### 7. Tests
**File Added:** `python/pyspark/sql/tests/arrow/test_arrow_iter_minimal.py`

**Tests Included:**
- `test_eval_type_inference`: Verifies correct eval type inference
- `test_single_column_basic`: Tests single column iterator aggregation

**Additional Test Files Created (for development):**
- `test_type_inference.py`: Standalone type inference validation (✅ PASSED)
- `test_worker_wrap.py`: Worker wrapper logic validation (✅ PASSED)
- `test_simple_iter.py`: Python iterator logic validation (✅ PASSED)

## Architecture

### Data Flow
1. **JVM → Python Worker**: Arrow record batches sent for each group
2. **Serializer**: `ArrowStreamAggArrowIterUDFSerializer` yields `ColumnIterators` object
3. **Mapper**: Indexes `ColumnIterators` by column offset to get iterators
4. **Wrapper**: Passes column iterator(s) to user function
   - Single column: passes one iterator
   - Multiple columns: zips iterators into tuples
5. **Result**: User function returns scalar, wrapped as Arrow array

### Key Design Decisions
1. **Separate Serializer**: Created new serializer instead of modifying existing one to avoid breaking regular grouped agg UDFs
2. **ColumnIterators Class**: Provides indexable access to column iterators, matching the mapper's expectations
3. **Generator Expressions**: Used to avoid materializing all data at once
4. **Worker Wrapper**: Handles both single and multiple column cases transparently

## Testing Status

### Unit Tests (Standalone - No Spark Required)
- ✅ Type inference logic
- ✅ Worker wrapper logic  
- ✅ Python iterator mechanics

### Integration Tests (Requires Compiled Spark)
- ⏳ Pending: Full end-to-end test with Spark session
- ⚠️ Build system cache issues preventing compilation
- 📝 Solution: Clean rebuild in progress

## Build Commands

### Standard Build (as per workspace rules)
```bash
git checkout master
git pull upstream master
git checkout -b SPARK-53615/feat/add-iterator-arrow-grouped-agg-support

# For compilation issues, clean caches:
rm -rf ~/.m2 ~/.ivy2/
build/sbt -Phive clean package
```

### Testing
```bash
# Run specific test
SPARK_TESTING=1 python/run-tests --testnames 'pyspark.sql.tests.arrow.test_arrow_iter_minimal'

# Or with pytest
/path/to/venv/bin/python3 -m pytest python/pyspark/sql/tests/arrow/test_arrow_iter_minimal.py -v
```

## Files Changed

### Python Files
- `python/pyspark/util.py`
- `python/pyspark/sql/pandas/_typing/__init__.pyi`
- `python/pyspark/sql/pandas/typehints.py`
- `python/pyspark/sql/pandas/functions.py`
- `python/pyspark/worker.py`
- `python/pyspark/sql/pandas/serializers.py`
- `python/pyspark/sql/tests/arrow/test_arrow_iter_minimal.py` (new)

### Scala Files
- `core/src/main/scala/org/apache/spark/api/python/PythonRunner.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/python/UserDefinedPythonFunction.scala`
- `sql/core/src/main/scala/org/apache/spark/sql/execution/python/ArrowAggregatePythonExec.scala`

## Next Steps

1. ✅ Complete implementation (DONE)
2. ⏳ Resolve build cache issues and compile
3. ⏳ Run full integration tests
4. ⏳ Run existing Arrow grouped agg tests to ensure no regression
5. ⏳ Format changed lines with scalafmt
6. ⏳ Run scalastyle verification
7. ⏳ Create PR with description

## Notes

- The Python-side implementation is complete and verified with standalone tests
- The Scala-side changes are minimal (just adding the new eval type to existing lists)
- The serializer follows the same pattern as existing iterator-based UDFs (e.g., `SQL_GROUPED_MAP_ARROW_ITER_UDF`)
- Type inference correctly distinguishes between:
  - Regular Arrow grouped agg: `pa.Array -> Any`
  - Iterator Arrow grouped agg: `Iterator[pa.Array] -> Any`
  - Scalar iterator: `Iterator[pa.Array] -> Iterator[pa.Array]`

