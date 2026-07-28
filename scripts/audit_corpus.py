#!/usr/bin/env python3
"""Report corpus quality issues without modifying any records."""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.json"
REPORT = ROOT / "data" / "corpus_audit.json"

def norm(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()

def main():
    rows = json.loads(CORPUS.read_text())
    answers = defaultdict(list)
    indexable_answers = defaultdict(list)
    issues = []
    for row in rows:
        answers[norm(row["answer"])].append(row["id"])
        if row["status"] in {"active", "needs_review"}:
            indexable_answers[norm(row["answer"])].append(row["id"])
        if row["status"] == "active":
            required = ("source", "source_url", "last_verified")
            missing = [field for field in required if not row.get(field)]
            if missing:
                issues.append({"severity": "error", "id": row["id"], "issue": "active record missing " + ", ".join(missing)})
        if row["status"] == "needs_review" and len(row["questions"]) < 4:
            issues.append({"severity": "warning", "id": row["id"], "issue": "fewer than 4 question phrasings"})
    historical_duplicate_answers = [ids for ids in answers.values() if len(ids) > 1]
    duplicate_answers = [ids for ids in indexable_answers.values() if len(ids) > 1]
    report = {
        "records": len(rows),
        "active": sum(x["status"] == "active" for x in rows),
        "needs_review": sum(x["status"] == "needs_review" for x in rows),
        "superseded": sum(x["status"] == "superseded" for x in rows),
        "duplicate_answer_groups": duplicate_answers,
        "historical_duplicate_answer_groups": historical_duplicate_answers,
        "issues": issues,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['records']} records; {len(duplicate_answers)} indexable duplicate-answer groups; {len(issues)} review issues. Report: {REPORT.relative_to(ROOT)}")

if __name__ == "__main__": main()
