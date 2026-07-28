import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvaluationValidationTests(unittest.TestCase):
    def test_unknown_expected_pair_id_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump([{"query": "test", "expected_pair_id": "not_an_active_pair"}], handle)
            golden = Path(handle.name)
        try:
            result = subprocess.run(
                [sys.executable, "scripts/evaluate.py", "--golden", str(golden)],
                cwd=root, capture_output=True, text=True,
            )
        finally:
            golden.unlink(missing_ok=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not active in this index", result.stderr)


if __name__ == "__main__":
    unittest.main()
