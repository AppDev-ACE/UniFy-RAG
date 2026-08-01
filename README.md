# SASTRA Freshers Assistant

A safety-first question-matching service for Unify. It returns an answer
grounded in a verified stored record — optionally rephrased, or in the
close-match case, optionally chosen, by a locally hosted LLM strictly
grounded in that stored text (see "LLM answer synthesis" below) — or routes
the student to a human. The LLM never supplies a fact of its own, and every
answer it produces is traceable to exactly one stored record. The student is
never shown a raw list of topics to pick from: below `TAU_HIGH` the LLM
either confirms one candidate record (answered, rephrased or verbatim) or
the query is routed to a human — there is no third "choose one" state.

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
# The default production build includes all trusted, non-superseded records.
.venv/bin/python scripts/build_index.py
.venv/bin/uvicorn app.main:app --reload
```

`POST /ask` accepts `{ "query": "when do classes start", "session_id": "optional" }`.
It returns one of `answered` or `abstained`. `GET /pairs/{pair_id}` still
exists for direct lookups by ID but nothing in the live `/ask` flow returns a
suggestion list to call it from anymore (see "LLM answer synthesis" below).

`POST /feedback` accepts `{ "rating": "up" | "down", "pair_id": "...", "session_id": "optional", "note": "optional" }`.
Repeated downvotes should trigger a source review; feedback never changes an
answer automatically.

## LLM answer synthesis

`POST /ask` sends top RAG matches to a locally hosted Ollama model (default
`qwen2.5:3b`, same VPS, no public exposure needed) so the final response can
come from the LLM rather than a bare record dump. It runs in two places,
with two different levels of trust in the LLM:

- **High-confidence path** (exact question match, or `matches[0]` past
  `TAU_HIGH`): retrieval alone already decided the record is correct. The
  LLM only rephrases that one record's text for tone — `answer_mode:
  "llm_synthesized"`.
- **Clarify-tier path** (below `TAU_HIGH`, 2-3 topically close candidates):
  here retrieval couldn't separate the candidates, so the LLM is given all
  of them and must pick the ONE that actually answers the question and
  rephrase only that record — it may never blend facts across candidates.
  The student never sees the other candidates as a pick-list. Success is
  `answer_mode: "llm_disambiguated"`; the response's `pair_id`, `source`,
  and `raw_answer` all come from whichever record the LLM chose, never from
  `matches[0]`.

Either way, the LLM is given only the candidate records' stored answer text
as context and instructed to add nothing beyond it. If it can't ground the
question in any candidate, it says so explicitly (a fixed `NOT_FOUND`
sentinel on the single-record path, an `INDEX: NONE` verdict on the
clarify-tier path) rather than guessing — and that becomes an `abstained`
response, not a guess and not a list of topics to choose from.

Every LLM reply is checked before use: call failure, timeout, an explicit
"not grounded" verdict, an unparseable or out-of-range candidate choice, or
any number in the reply not exactly matching the set of numbers in the
chosen record's source text (a fare, a time, a date silently drifting, word
forms like "two" and digit forms like "2" treated as equivalent) all fall
back. A reply that's just a copy of the source sentences is rejected too and
retried once with an explicit "don't copy, rephrase" nudge — a personalised
answer that still reads like the RAG record dumped verbatim isn't what this
path is for — and a reply that dropped a number gets one retry naming the
exact figure it missed. On the high-confidence path, a reply that still
fails after retry falls back to the verbatim record. On the clarify-tier
path, the outcome depends on *what* failed: if the LLM confidently picked a
candidate and only its rephrase couldn't pass grounding, that candidate's
record is still served, verbatim (`answer_mode: "verbatim"`) — the topic was
resolved, only the wording wasn't usable. If the LLM never confirmed a
candidate at all (a `NONE` verdict, an unparseable or out-of-range index, or
the call failed), the query is routed to a human instead — never a guess,
and never a raw list of topics. Neither path ever reaches mandatory-routed
queries (ragging, medical, personal record, staff contradiction) — those
never reach retrieval at all.

This does trade away one thing deliberately: the previous design let a
human pick between close topics below `TAU_HIGH`; now that choice is made
by the LLM (or the query is routed to a human outright) so the end user is
never shown a "choose one" list. There is no longer a human check on which
candidate record is correct in this tier.

The API always reports what happened via `"answer_mode"`
(`"llm_synthesized"`, `"llm_disambiguated"`, or `"verbatim"`) and, on any
`answered` response, always includes `"raw_answer"` alongside `"answer"` for
audit. Configure with `OLLAMA_HOST`, `OLLAMA_MODEL`, `LLM_ENABLED=0` to
disable, and `LLM_TIMEOUT_SECONDS`.

## Evaluation

Add real, consented fresher wording to `data/golden_set.json` (never personal
records). Then run `python scripts/evaluate.py --sweep --write-calibration`.
The command writes a local `data/calibration.json` only if the wrong-answer
release gate is below 2%. Without that file, automatic answering is disabled
and the API only offers an LLM-confirmed answer or human routing.

## Key rules

- The LLM never supplies a fact of its own: on the high-confidence path it
  only rephrases an already-accepted answer; on the clarify-tier path it may
  only choose among human-clarify-quality candidates and rephrase the one it
  picked. Any reply that fails the grounding check (see above) falls back to
  the verbatim record if a candidate was confirmed, or to human routing if
  none was. It is never used on mandatory-route (ragging/medical/personal-record)
  responses.
- Never activate an answer with no primary source and verification date.
- Rebuild the index after every corpus edit. The builder excludes superseded
  and inactive entries; expired date-specific records are returned with a
  visible stale-information warning.
- The service always routes personal academic/fee/disciplinary cases, medical
  or mental-health distress, ragging reports, and contradictions with staff.
