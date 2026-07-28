from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.json"
INDEX_DIR = ROOT / "data" / "index"
LOG_DIR = ROOT / "data" / "logs"
CALIBRATION_PATH = ROOT / "data" / "calibration.json"

# Deliberately disables automatic answers until scripts/evaluate.py calibrates a
# real golden set. Setting an uncalibrated threshold is unsafe.
def calibrated_thresholds() -> tuple[float, float]:
    """Use audited thresholds when present; otherwise disable auto-answering."""
    try:
        value = json.loads(CALIBRATION_PATH.read_text())
        low, high = float(value["tau_low"]), float(value["tau_high"])
        if 0 <= low < high <= 1:
            return low, high
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Before a real golden set exists, allow a low-evidence *clarification*
    # rather than throwing away useful verified candidates. tau_high remains
    # unreachable, so this never enables automatic answers.
    return 0.15, 1.10

TAU_LOW, TAU_HIGH = calibrated_thresholds()
TOP_K = 20
RRF_K = 60
