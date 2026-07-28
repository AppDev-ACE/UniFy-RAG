from __future__ import annotations
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.config import INDEX_DIR, LOG_DIR, TAU_HIGH, TAU_LOW
from app.retrieval import Retriever
from app.safety import abstention, mandatory_route

app = FastAPI(title="SASTRA Freshers Assistant", version="0.1.0")
_retriever = None

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

def answer_payload(record: dict, confidence: float | None = None) -> dict:
    response = {
        "status": "answered", "answer": record["answer"], "source": record["source"],
        "source_url": record["source_url"], "last_verified": record["last_verified"],
        "pair_id": record["id"], "verification_status": verification_status(record),
        "warning": record.get("stale_warning") or record.get("test_warning") or record.get("trusted_notice"),
    }
    if confidence is not None:
        response["confidence"] = round(confidence, 3)
    return response

@app.get("/health")
def health():
    return {"ok": True, "index_ready": (INDEX_DIR / "records.json").exists()}

@app.post("/ask")
def ask(request: AskRequest):
    started = time.perf_counter(); forced = mandatory_route(request.query)
    if forced:
        response = abstention(*forced)
    else:
        matches = retriever().search(request.query)
        if not matches or matches[0]["score"] < TAU_LOW:
            category = matches[0]["record"].get("category") if matches else None
            response = abstention(category)
        elif matches[0]["score"] < TAU_HIGH:
            # Do not pad a clarification with weakly related records merely
            # to reach three suggestions.
            minimum_suggestion_score = max(0.12, matches[0]["score"] * 0.35)
            response = {"status": "clarify", "suggestions": [
                {"pair_id": x["record"]["id"], "question": x["question"],
                 "verification_status": verification_status(x["record"])}
                for x in matches[:3] if x["score"] >= minimum_suggestion_score
            ]}
        else:
            x = matches[0]
            response = answer_payload(x["record"], x["score"])
    log({"event_id": str(uuid.uuid4()), "at": datetime.now(timezone.utc).isoformat(), "session_id": request.session_id, "query": request.query, "outcome": response["status"], "pair_id": response.get("pair_id"), "latency_ms": round((time.perf_counter()-started)*1000, 2)})
    return response

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
