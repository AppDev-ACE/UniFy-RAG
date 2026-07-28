import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import app.config as config

class CalibrationTests(unittest.TestCase):
    def test_valid_calibration_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"tau_low": .4, "tau_high": .8}))
            with patch.object(config, "CALIBRATION_PATH", path):
                self.assertEqual(config.calibrated_thresholds(), (.4, .8))

    def test_invalid_or_missing_calibration_disables_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(config, "CALIBRATION_PATH", Path(tmp) / "missing.json"):
                # A low default permits clarification; the unreachable high
                # threshold still disables automatic answering.
                self.assertEqual(config.calibrated_thresholds(), (.15, 1.1))

if __name__ == "__main__": unittest.main()
