#!/usr/bin/env python3
"""Evaluate retrieval and sweep the conservative confidence thresholds."""
from __future__ import annotations
import argparse, json, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.retrieval import Retriever

def metrics(rows, retriever, low, high):
    hit1 = hit3 = confident_wrong = answered = abstain_expected = abstain_correct = abstained = 0
    for row in rows:
        expected = row["expected_pair_id"]
        matches = retriever.search(row["query"])
        ids = [m["record"]["id"] for m in matches]; score = matches[0]["score"] if matches else 0
        # The abstain/clarify gate in app/main.py and scripts/chat.py checks
        # the best score anywhere in the retrieved pool, not just position 0
        # (a weak RRF top pick shouldn't hide a strong candidate at rank 2-3).
        # Mirror that here so a swept/written threshold matches deployed
        # behaviour. The auto-answer check stays on matches[0] specifically,
        # since that is still the record that would actually be returned.
        best_score = max((m["score"] for m in matches), default=0)
        if expected == "SHOULD_ABSTAIN":
            abstain_expected += 1
            if best_score < low: abstain_correct += 1
            if best_score < low: abstained += 1
            continue
        hit1 += bool(ids and ids[0] == expected); hit3 += expected in ids
        if score >= high:
            answered += 1
            confident_wrong += not (ids and ids[0] == expected)
        elif best_score < low: abstained += 1
    total_answerable = len(rows) - abstain_expected
    return {"samples": len(rows), "hit_at_1": hit1 / total_answerable if total_answerable else None, "hit_at_3": hit3 / total_answerable if total_answerable else None, "wrong_answer_rate": confident_wrong / max(answered, 1), "answered_coverage": answered / max(total_answerable, 1), "abstention_precision": abstain_correct / max(abstained, 1), "abstention_recall": abstain_correct / max(abstain_expected, 1)}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--golden", default=ROOT / "data/golden_set.json", type=Path); parser.add_argument("--index", default=ROOT / "data/index", type=Path); parser.add_argument("--sweep", action="store_true"); parser.add_argument("--write-calibration", action="store_true")
    args = parser.parse_args(); rows = json.loads(args.golden.read_text())
    if not rows: raise SystemExit("Golden set is empty. Add consented, labelled fresher questions before calibration.")
    required = {"query", "expected_pair_id"}
    if any(not required <= row.keys() for row in rows): raise SystemExit("Each golden row needs query and expected_pair_id (or SHOULD_ABSTAIN).")
    r = Retriever(args.index)
    active_ids = {record["id"] for record in r.records}
    unknown = sorted({row["expected_pair_id"] for row in rows if row["expected_pair_id"] != "SHOULD_ABSTAIN"} - active_ids)
    if unknown:
        raise SystemExit(
            "Golden set refers to pair IDs that are not active in this index: " + ", ".join(unknown)
        )
    candidates = [(low, high) for low in (.25,.35,.45,.50,.55,.60) for high in (.60,.65,.70,.75,.80,.85,.90) if low < high] if args.sweep else [(.50,.78)]
    results = [{"tau_low": low, "tau_high": high, **metrics(rows,r,low,high)} for low,high in candidates]
    # Release rule: maximize coverage while keeping confident wrong answers below 2%.
    safe = [x for x in results if x["wrong_answer_rate"] < .02]
    chosen = max(safe, key=lambda x:x["answered_coverage"]) if safe else results[0]
    if args.write_calibration:
        if chosen["wrong_answer_rate"] >= .02:
            raise SystemExit("No threshold satisfies the <2% wrong-answer release gate; calibration was not written.")
        output = {"calibrated_on": date.today().isoformat(), "tau_low": chosen["tau_low"], "tau_high": chosen["tau_high"], "metrics": chosen}
        (ROOT / "data" / "calibration.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(chosen, indent=2))

if __name__ == "__main__": main()
