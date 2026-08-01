from pathlib import Path
import json, os

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

# Same VPS hosts Ollama, so the default points at its local API, not the
# public IP. Override via env for other deployments.
# Retrieval evidence strong enough that a 3B model's "this record doesn't
# answer the question" must not be allowed to discard it.
#
# The clarify tier asks the LLM which candidate answers the question, and an
# all-NOT_FOUND verdict abstains. That gives the model an absolute veto over
# retrieval, and it was spending it on questions the corpus plainly answers:
# "Documents required for inauguration" retrieves the reviewed admission-
# documents record at 0.592 -- 2.5x its nearest rival -- and the model
# refused it because the record says "admission" where the student said
# "inauguration". The student got "I don't have an answer" for a question
# with a good, human-reviewed answer sitting in the index.
#
# Above this score the record is answered anyway (see
# app.llm.disambiguate_and_answer). This is the same principle the
# past-TAU_HIGH path already applies -- there, retrieval decides relevance
# and the LLM only phrases -- extended to evidence that is strong and
# unambiguous but short of that gate.
#
# Chosen by measuring the two populations rather than by feel. Queries the
# corpus answers but the model vetoed score 0.53-0.99 (inauguration
# documents 0.592, hostel rules for boys 0.845, girls' outing rules 0.904).
# Off-corpus questions -- swimming pool, US visas, driving licences, stock
# prices, yesterday's cricket -- top out at 0.407 ("can I keep a pet dog in
# the hostel" reaching the maggi-noodles record). It is a heuristic standing
# in for a real calibration, and should be replaced the moment
# data/golden_set.json has consented labelled fresher questions in it;
# unlike TAU_HIGH it cannot cause a *fabricated* answer, only the serving of
# a strongly-matched reviewed record, so it is not the unsafe kind of
# uncalibrated threshold.
#
# Lowered from 0.5 to 0.42 after two more real vetoed-but-correct cases
# surfaced in production: "whether stationery shop is available" (0.463,
# correct record was legacy_159) and "whether buses to Thanjavur and Trichy
# is frequently available" (0.444, correct record was community_002) -- both
# below the old 0.5 floor. 0.42 keeps the same gap logic as before: it stays
# above the known off-corpus ceiling of 0.407 (with margin, so those queries
# still abstain) while sitting below both new confirmed-correct scores.
LLM_VETO_OVERRIDE_SCORE = float(os.environ.get("LLM_VETO_OVERRIDE_SCORE", "0.42"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "8"))
LLM_MAX_CONTEXT_RECORDS = 3
