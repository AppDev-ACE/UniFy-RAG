# SASTRA Freshers Assistant

A safety-first question-matching service for Unify. It does **not** generate
answers: it returns a verified stored answer, asks the student to choose among
close matches, or routes them to a human.

## Data safety workflow

`sastra_freshers.json` is the original collected material and is never edited.
Run `python scripts/migrate_corpus.py` to create these review artifacts:

- `data/corpus.json` — canonical records; all imported records start as
  `needs_review` because the supplied data contains no source provenance.
- `data/review_queue.csv` — a compact checklist for a SASTRA reviewer.

Before a record can be served it must have `status: "active"`, a named source,
a traceable source URL, and a `last_verified` date. SASTRA policy requires an
official `https://` URL. Product-owner statements about UniFy may use a
checked-in `local://` source that records who supplied the claim and when. This
is enforced by the index builder. Do not mark experiential claims as policy;
use `source_type: "student_opinion"` instead.

Run `python scripts/audit_corpus.py` after migration to find duplicate answers,
records with too few question phrasings, and invalid active-record metadata.

The migration also flags known outdated/contradictory legacy answers as
`superseded` (attendance, old outing rules, manual roll-call, and appliance
rules). They can never enter the index.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python scripts/migrate_corpus.py
# Review and activate sourced records in data/corpus.json.
.venv/bin/python scripts/build_index.py
.venv/bin/uvicorn app.main:app --reload
```

`POST /ask` accepts `{ "query": "when do classes start", "session_id": "optional" }`.
It returns one of `answered`, `clarify`, or `abstained`.
When a student taps a `clarify` suggestion, call `GET /pairs/{pair_id}` to return
that exact verified record without another fuzzy lookup.

`POST /feedback` accepts `{ "rating": "up" | "down", "pair_id": "...", "session_id": "optional", "note": "optional" }`.
Repeated downvotes should trigger a source review; feedback never changes an
answer automatically.

## Evaluation

Add real, consented fresher wording to `data/golden_set.json` (never personal
records). Then run `python scripts/evaluate.py --sweep --write-calibration`.
The command writes a local `data/calibration.json` only if the wrong-answer
release gate is below 2%. Without that file, automatic answering is disabled
and the API only offers clarification or human routing.

## Key rules

- Never use an LLM to rewrite policy answers in v1.
- Never activate an answer with no primary source and verification date.
- Rebuild the index after every corpus edit. The builder excludes superseded
  and inactive entries; expired date-specific records are returned with a
  visible stale-information warning.
- The service always routes personal academic/fee/disciplinary cases, medical
  or mental-health distress, ragging reports, and contradictions with staff.
