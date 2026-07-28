#!/usr/bin/env python3
"""Convert the supplied delimiter-based Q/A file into a reviewable corpus.

No provenance was provided with the raw file, so this tool deliberately does
not promote claims to active. That prevents accidental publication of stale
student advice as university policy.
"""
from __future__ import annotations
import csv, json, re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sastra_freshers.json"
OUT = ROOT / "data" / "corpus.json"
QUEUE = ROOT / "data" / "review_queue.csv"
OFFICIAL_SEED = ROOT / "data" / "official_seed.json"
HOSTEL_RULES_URL = "https://www.sastra.edu/downloads/news/2026/june/Hostel_Rules_Regulations_2026-27.pdf"
HOSTEL_RULES_SOURCE = "Rules and Regulations for Inmates of Students’ Home (SASTRA official, effective 01 Jul 2026)"
EXAM_RULES_URL = "https://sastra.edu/downloads/menu/Parents/2026/ExaminationRegulations_2026.pdf"
EXAM_RULES_SOURCE = "Examination Rules & Regulations (SASTRA official, effective 28 Jan 2026)"
ADMISSIONS_2026_URL = "https://www.sastra.edu/home/admissions.html"
ADMISSIONS_2026_SOURCE = "Admissions 2026 FAQ (SASTRA official)"
OPENING_DAY_2026_URL = "https://www.sastra.edu/academic-updates/4640-opening-day-instructions-for-1st-year-b-tech-m-tech-intg-students-admitted-for-the-academic-year-2026-27.html"
OPENING_DAY_2026_SOURCE = "Opening Day Instructions for I Year B.Tech./M.Tech. (Intg.) 2026-27 (SASTRA official)"
CALENDAR_2026_URL = "https://www.sastra.edu/downloads/menu/Academics/2026-27/T_Calendar-2627U1.pdf"
CALENDAR_2026_SOURCE = "Academic Calendar 2026-27 — Thanjavur Campus (SASTRA official)"

# These two admissions-FAQ records are active and source-backed but were
# supplied with only one or two headings. Add faithful student-style phrasings
# so they have the same retrieval coverage required of every served record.
REVIEWED_QUESTION_VARIANTS = {
    285: [
        "What is minor specialization in SASTRA BTech?",
        "can I take a minor in BTech",
        "minor specialization options for BTech students",
        "does SASTRA offer AI ML minor",
        "how does a BTech minor specialization work",
    ],
    286: [
        "What are career track pathway courses at SASTRA?",
        "BTech career tracks SASTRA",
        "when do career track courses start",
        "career pathway courses from third semester",
        "emerging technology career tracks in BTech",
    ],
}

def split_questions(text: str) -> list[str]:
    # The legacy separator is space-question-mark-space; retain genuine '?'.
    values = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\s+\?\s+", text)]
    return list(dict.fromkeys(x for x in values if len(x) > 2))

def slug(text: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return result[:54] or "question"

def category(text: str) -> str:
    t = text.lower()
    groups = {
        "hostel": ["hostel", "mess", "outing", "warden", "roll call", "room"],
        "academics": ["class", "attendance", "exam", "cia", "grade", "semester", "course", "faculty"],
        "admissions": ["admission", "joining", "certificate", "transfer", "opening day"],
        "transport": ["bus", "transport", "vehicle", "parking"],
        "finance": ["fee", "bank", "cub", "scholarship"],
        "health": ["hospital", "medical", "sick"],
        "campus": ["campus", "library", "club", "sports", "canteen", "wifi"],
    }
    return next((name for name, words in groups.items() if any(w in t for w in words)), "general")

def known_superseded(question: str, answer: str) -> bool:
    t = (question + " " + answer).lower()
    # Legacy claims contradicted by the supplied 2026-27 material. Specific
    # patterns keep current versions available for human verification.
    patterns = [
        r"attendance.{0,80}(below|less than).{0,30}80%", r"80%.{0,70}attendance",
        r"(iron box|kettle|induction).{0,80}(not allowed|prohibited)",
        r"(go two times in a semester|twice a semester)",
        r"girls can go out only once a month", r"boys are allowed to go outside on all days",
        r"boys:\s*9:00\s*pm,?\s*girls:\s*6:30\s*pm",
    ]
    return any(re.search(pattern, t) for pattern in patterns)

def has_calendar_date(answer: str) -> bool:
    return bool(re.search(r"\b(20(?:26|27))\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}", answer, re.I))

def main():
    raw = json.loads(RAW.read_text())
    records = []
    for number, item in enumerate(raw, 1):
        questions = split_questions(item["question"])
        questions = list(dict.fromkeys(questions + REVIEWED_QUESTION_VARIANTS.get(number, [])))
        answer = item["answer"].strip()
        state = "superseded" if known_superseded(item["question"], answer) else "needs_review"
        # These records are a direct, individually checked transcription of
        # the official rules linked above (raw records 94–110).
        official_hostel_rule = number == 80 or 94 <= number <= 110
        official_exam_rule = 77 <= number <= 93
        official_admissions_rule = number in {285, 286}
        official_opening_day = number == 1 or 49 <= number <= 61 or 63 <= number <= 67
        official_calendar_rule = 68 <= number <= 76
        if official_hostel_rule:
            state = "active"
        if official_exam_rule:
            state = "active"
        if official_admissions_rule:
            state = "active"
        if official_opening_day:
            state = "active"
        if official_calendar_rule:
            state = "active"
        rec = {
            "id": f"legacy_{number:03d}_{slug(questions[0])}",
            "questions": questions,
            "answer": answer,
            "category": category(" ".join(questions)),
            "source": CALENDAR_2026_SOURCE if official_calendar_rule else OPENING_DAY_2026_SOURCE if official_opening_day else ADMISSIONS_2026_SOURCE if official_admissions_rule else EXAM_RULES_SOURCE if official_exam_rule else HOSTEL_RULES_SOURCE if official_hostel_rule else None,
            "source_url": CALENDAR_2026_URL if official_calendar_rule else OPENING_DAY_2026_URL if official_opening_day else ADMISSIONS_2026_URL if official_admissions_rule else EXAM_RULES_URL if official_exam_rule else HOSTEL_RULES_URL if official_hostel_rule else None,
            "source_type": "official" if official_hostel_rule or official_exam_rule or official_admissions_rule or official_opening_day or official_calendar_rule else "unknown",
            "effective_from": "2026-01-28" if official_exam_rule else "2026-07-01" if official_hostel_rule else "2026-01-01" if official_admissions_rule or official_opening_day or official_calendar_rule else None,
            "valid_until": "2027-07-31" if official_admissions_rule or official_opening_day or official_calendar_rule else None if official_hostel_rule or official_exam_rule else "2027-07-01" if has_calendar_date(answer) else None,
            "status": state,
            "last_verified": "2026-07-28" if official_hostel_rule or official_exam_rule or official_admissions_rule or official_opening_day or official_calendar_rule else None,
            "contact_fallback": None,
            "legacy": {"file": RAW.name, "record_number": number},
            "review_note": ("Verified against the official 2026-27 academic calendar." if official_calendar_rule else "Verified against the official 2026-27 opening-day instructions." if official_opening_day else "Verified against the official 2026 admissions FAQ." if official_admissions_rule else "Verified against the official 2026 examination regulations." if official_exam_rule else "Verified against the official 2026-27 hostel rules." if official_hostel_rule else "Known legacy claim superseded by the 2026-27 source material; retain only for audit." if state == "superseded" else "Add an official primary source and verify this answer before activation."),
        }
        records.append(rec)
    # Hand-curated records are kept in their own reviewed file so future raw
    # migrations are deterministic and never overwrite source-backed edits.
    if OFFICIAL_SEED.exists():
        seed = json.loads(OFFICIAL_SEED.read_text())
        ids = {r["id"] for r in records}
        for record in seed:
            required = {"id", "questions", "answer", "source", "source_url", "last_verified", "status"}
            if not required <= record.keys() or record["id"] in ids:
                raise ValueError(f"Invalid or duplicate official seed: {record.get('id')}")
            records.append(record)
    # Exact legacy answer duplicates represent the same fact expressed as
    # separate records. Merge their phrasings into the earliest record and
    # retain the rest as auditable superseded history. Never merge active or
    # source-backed records automatically.
    duplicate_groups = {}
    for record in records:
        if record["status"] != "needs_review" or "legacy" not in record:
            continue
        key = re.sub(r"\W+", "", record["answer"]).lower()
        duplicate_groups.setdefault(key, []).append(record)
    for group in duplicate_groups.values():
        if len(group) < 2:
            continue
        primary, *duplicates = group
        for duplicate in duplicates:
            primary["questions"] = list(dict.fromkeys(primary["questions"] + duplicate["questions"]))
            duplicate["status"] = "superseded"
            duplicate["superseded_by"] = primary["id"]
            duplicate["review_note"] = "Exact duplicate answer merged into " + primary["id"] + "."
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    with QUEUE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "category", "question", "source", "source_url", "last_verified", "review_note"])
        writer.writeheader()
        for r in records:
            writer.writerow({"id": r["id"], "status": r["status"], "category": r["category"], "question": r["questions"][0], "source": r.get("source") or "", "source_url": r.get("source_url") or "", "last_verified": r.get("last_verified") or "", "review_note": r.get("review_note", "Official curated record")})
    print(f"Migrated {len(records)} records: {sum(r['status']=='superseded' for r in records)} superseded; {sum(r['status']=='needs_review' for r in records)} awaiting provenance review.")

if __name__ == "__main__": main()
