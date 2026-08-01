from __future__ import annotations
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.config import INDEX_DIR, LOG_DIR, TAU_HIGH, TAU_LOW
from app.llm import disambiguate_and_answer, synthesize_answer
from app.retrieval import Retriever, normalize
from app.safety import abstention, mandatory_route

app = FastAPI(title="SASTRA Freshers Assistant", version="0.1.0")
_retriever = None

# A large fresher intake asks a small set of questions in near-identical
# phrasing (mess menu, bus fare, hostel rules), and each of those can cost
# several sequential LLM calls on this CPU-only deployment. Caching by exact
# normalised query text turns every repeat of the same phrasing into a
# free hit instead of re-running retrieval and grounding from scratch. Keyed
# on the same `normalize` the retriever itself uses (no synonym expansion,
# so the key stays literal), so "What is the bus fare to Thanjavur?" and
# "what is the bus fare to thanjavur" collide on purpose. A fresh process
# (every deploy or index rebuild restarts it) clears the cache, so a corpus
# update can never serve a stale cached answer.
_response_cache: dict[str, dict] = {}
_CACHE_MAX_ENTRIES = 5000

class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    session_id: str | None = Field(default=None, max_length=200)

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    pair_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=200)
    note: str | None = Field(default=None, max_length=1000)

def retriever():
    global _retriever
    if _retriever is None:
        if not (INDEX_DIR / "records.json").exists():
            raise HTTPException(503, "Index unavailable. Run scripts/build_index.py after activating verified records.")
        _retriever = Retriever(INDEX_DIR)
    return _retriever

def log(payload: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "queries.jsonl").open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def verification_status(record: dict) -> str:
    if record.get("test_only_unverified"):
        return "unverified_test_only"
    if record.get("trusted_legacy"):
        return "trusted_legacy"
    return "verified"

def answer_payload(record: dict, confidence: float | None = None, llm_answer: str | None = None) -> dict:
    response = {
        "status": "answered", "answer": llm_answer if llm_answer is not None else record["answer"],
        "raw_answer": record["answer"], "source": record["source"],
        "source_url": record["source_url"], "last_verified": record["last_verified"],
        "pair_id": record["id"], "verification_status": verification_status(record),
        "warning": record.get("stale_warning") or record.get("test_warning") or record.get("trusted_notice"),
        "answer_mode": "llm_synthesized" if llm_answer is not None else "verbatim",
    }
    if confidence is not None:
        response["confidence"] = round(confidence, 3)
    return response

@app.get("/health")
def health():
    return {"ok": True, "index_ready": (INDEX_DIR / "records.json").exists(),
             "dense_retrieval": (INDEX_DIR / "vectors.npy").exists()}

@app.post("/ask")
def ask(request: AskRequest):
    started = time.perf_counter(); forced = mandatory_route(request.query)
    if forced:
        response = abstention(*forced)
    else:
        cache_key = normalize(request.query, expand_synonyms=False)
        cached = _response_cache.get(cache_key)
        if cached is not None:
            response = cached
            log({"event_id": str(uuid.uuid4()), "at": datetime.now(timezone.utc).isoformat(), "session_id": request.session_id, "query": request.query, "outcome": response["status"], "pair_id": response.get("pair_id"), "answer_mode": response.get("answer_mode"), "latency_ms": round((time.perf_counter()-started)*1000, 2), "cache_hit": True})
            return response
        response = _resolve(request)
        if len(_response_cache) < _CACHE_MAX_ENTRIES:
            _response_cache[cache_key] = response
    log({"event_id": str(uuid.uuid4()), "at": datetime.now(timezone.utc).isoformat(), "session_id": request.session_id, "query": request.query, "outcome": response["status"], "pair_id": response.get("pair_id"), "answer_mode": response.get("answer_mode"), "latency_ms": round((time.perf_counter()-started)*1000, 2)})
    return response

def _resolve(request: AskRequest) -> dict:
    searcher = retriever()
    # Terminal testing lets a student select a matching stored question.
    # Resolve an unambiguous exact phrasing here too, so POST /ask has the
    # same direct-answer behaviour without relaxing fuzzy-match safety.
    exact = searcher.exact_match(request.query)
    if exact:
        llm_answer = synthesize_answer(request.query, {"record": exact["record"]})
        return answer_payload(exact["record"], 1.0, llm_answer)
    matches = searcher.search(request.query)
    # Gate on the best evidence anywhere in the retrieved pool, not
    # only whichever record RRF ranked first. RRF's ordering can put
    # a weak or unrelated record in position 0 while a strong match
    # -- one whose wording only overlaps the record's answer text,
    # not any reviewed question -- sits at position 2 or 3. Basing
    # the abstain decision on position 0 alone threw those away.
    best_score = max((m["score"] for m in matches), default=0.0)
    if not matches or best_score < TAU_LOW:
        category = matches[0]["record"].get("category") if matches else None
        return abstention(category)
    if matches[0]["score"] < TAU_HIGH:
        # The cutoff is deliberately anchored to matches[0], not
        # best_score: matches[0] is what the suggestion list is
        # built from and displayed first, so a cutoff derived from a
        # different (higher-scoring) record could exclude matches[0]
        # from its own suggestion list. Do not pad a clarification
        # with weakly related records merely to reach three
        # suggestions.
        minimum_suggestion_score = max(0.12, matches[0]["score"] * 0.35)
        candidates = [x for x in matches[:3] if x["score"] >= minimum_suggestion_score]
        # The LLM gets a vote here only on WHICH of these
        # human-clarify-tier candidates answers the question -- it
        # confirms exactly one and is grounded against only that
        # one record's raw answer (see app.llm.disambiguate_and_answer).
        # The confirmed record is NOT always the one it picked first:
        # a pick whose own rephrase answers NOT_FOUND is disowned and
        # the next candidate tried, so every field below must be read
        # off the returned record, never off matches[0] or
        # candidates[0].
        # The end user is never shown a raw pick-list: once the LLM
        # has confirmed a record, that record's topic is settled --
        # (chosen, None) still means "this is the right record, the
        # rephrase just didn't pass grounding," so it's served
        # verbatim, not turned into a menu. Only a fully-unconfirmed
        # pick (None) routes to a human.
        disambiguated = disambiguate_and_answer(request.query, candidates)
        if not disambiguated:
            return abstention(matches[0]["record"].get("category"))
        chosen, llm_answer = disambiguated
        response = answer_payload(chosen["record"], chosen["score"], llm_answer)
        if llm_answer is not None:
            response["answer_mode"] = "llm_disambiguated"
        return response
    x = matches[0]
    # LLM synthesis only ever runs once the confidence gate above
    # has already accepted x as the answer; it rephrases, it does
    # not get a vote on which record is correct. Only x is passed:
    # the rest of the pool is not evidence for this answer, and
    # grounding against it makes the completeness check
    # unsatisfiable (see synthesize_answer's docstring).
    llm_answer = synthesize_answer(request.query, x)
    return answer_payload(x["record"], x["score"], llm_answer)

@app.get("/pairs/{pair_id}")
def get_selected_pair(pair_id: str):
    """Return a verified pair selected from a previous clarify suggestion.

    Flutter must call this endpoint when a student taps a suggestion. It avoids
    turning the tap into a new fuzzy-search decision.
    """
    match = next((record for record in retriever().records if record["id"] == pair_id), None)
    if match is None:
        raise HTTPException(404, "Indexed pair not found")
    forced = mandatory_route(" ".join(match.get("questions", [])))
    if forced:
        return abstention(*forced)
    return answer_payload(match)

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """Store a lightweight answer-quality signal for the monthly review loop."""
    if request.pair_id and not any(r["id"] == request.pair_id for r in retriever().records):
        raise HTTPException(404, "Indexed pair not found")
    log({
        "event_id": str(uuid.uuid4()), "at": datetime.now(timezone.utc).isoformat(),
        "event": "feedback", "session_id": request.session_id, "pair_id": request.pair_id,
        "rating": request.rating, "note": request.note,
    })
    return {"ok": True}
