#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest

from pyspark.sql.functions import arrow_udf
from pyspark.util import PythonEvalType
from pyspark.testing.utils import have_pyarrow, pyarrow_requirement_message
from pyspark.testing.sqlutils import ReusedSQLTestCase


@unittest.skipIf(not have_pyarrow, pyarrow_requirement_message)
class ArrowIterMinimalTests(ReusedSQLTestCase):
    """Minimal tests for arrow iterator grouped agg UDF"""

    def test_eval_type_inference(self):
        """Test that the eval type is correctly inferred for iterator grouped agg UDFs."""
        import pyarrow as pa
        from typing import Iterator

        print("\n[TEST] Testing eval type inference...")

        @arrow_udf("double")
        def arrow_sum_iter(it: Iterator[pa.Array]) -> float:
            total = 0.0
            for v in it:
                total += pa.compute.sum(v).as_py()
            return total

        print(f"[TEST] Eval type: {arrow_sum_iter.evalType}")
        print(f"[TEST] Expected: {PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF}")

        self.assertEqual(arrow_sum_iter.evalType, PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF)
        print("[TEST] ✓ Eval type inference passed!")

    def test_single_column_basic(self):
        """Test iterator API for grouped aggregation with single column - basic case."""
        import pyarrow as pa
        from typing import Iterator
        from pyspark.sql import functions as sf
        import sys

        print("\n" + "=" * 80, flush=True)
        print("[TEST] Testing single column iterator...", flush=True)
        print("=" * 80, flush=True)

        @arrow_udf("double")
        def arrow_mean_iter(it: Iterator[pa.Array]) -> float:
            print(f"[UDF] arrow_mean_iter called", file=sys.stderr, flush=True)
            print(f"[UDF] arrow_mean_iter called", flush=True)
            sum_val = 0.0
            cnt = 0
            batch_count = 0
            for v in it:
                batch_count += 1
                print(
                    f"[UDF] Processing batch {batch_count} with {len(v)} elements",
                    file=sys.stderr,
                    flush=True,
                )
                print(f"[UDF] Processing batch {batch_count} with {len(v)} elements", flush=True)
                sum_val += pa.compute.sum(v).as_py()
                cnt += len(v)
            result = sum_val / cnt if cnt > 0 else 0.0
            print(f"[UDF] Returning: {result}", file=sys.stderr, flush=True)
            print(f"[UDF] Returning: {result}", flush=True)
            return result

        # Enable faulthandler for better error messages
        self.spark.conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")

        # Create a small test dataframe
        print("[TEST] Creating test data...", flush=True)
        df = self.spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0)], ("id", "v"))
        print("[TEST] Showing dataframe...", flush=True)
        df.show()

        print("[TEST] Running groupby aggregation...", flush=True)
        print("[TEST] About to call df.groupby('id').agg()...", flush=True)
        result = df.groupby("id").agg(arrow_mean_iter(df["v"]).alias("mean")).sort("id")

        print("[TEST] About to collect results...", flush=True)
        result_data = result.collect()

        print(f"[TEST] Result: {result_data}", flush=True)

        # Verify results
        self.assertEqual(len(result_data), 2)
        self.assertAlmostEqual(result_data[0]["mean"], 1.5, places=5)  # (1+2)/2
        self.assertAlmostEqual(result_data[1]["mean"], 4.0, places=5)  # (3+5)/2

        print("[TEST] ✓ Single column iterator test passed!", flush=True)


if __name__ == "__main__":
    from pyspark.sql.tests.arrow.test_arrow_iter_minimal import *  # noqa: F401

    try:
        import xmlrunner

        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
