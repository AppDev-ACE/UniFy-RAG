#!/usr/bin/env python3
"""Interactive local terminal for trying the Freshers Assistant safely."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import INDEX_DIR, LLM_ENABLED, OLLAMA_MODEL, TAU_HIGH, TAU_LOW
from app.llm import disambiguate_and_answer, synthesize_answer
from app.retrieval import Retriever
from app.safety import abstention, mandatory_route


def show_answer(record: dict, confidence: float | None = None, llm_answer: str | None = None) -> None:
    if llm_answer is not None:
        print(f"\nAnswer (LLM-phrased, grounded in the stored record below):\n{llm_answer}")
        print(f"\nStored record it was grounded in:\n{record['answer']}")
    else:
        print(f"\nAnswer:\n{record['answer']}")
    print(f"\nSource: {record['source']}")
    print(f"Last verified: {record['last_verified']}")
    if record.get("warning") or record.get("stale_warning") or record.get("test_warning") or record.get("trusted_notice"):
        print(f"Warning: {record.get('warning') or record.get('stale_warning') or record.get('test_warning') or record['trusted_notice']}")
    if confidence is not None:
        print(f"Confidence: {confidence:.3f}")


def show_abstention(response: dict) -> None:
    print(f"\n{response['message']}")
    for contact in response["contacts"]:
        details = ", ".join(f"{key}: {value}" for key, value in contact.items() if key != "label")
        print(f"- {contact['label']}{': ' + details if details else ''}")


def respond(retriever: Retriever, query: str) -> None:
    forced = mandatory_route(query)
    if forced:
        show_abstention(abstention(*forced))
        return
    # Mirrors app/main.py's /ask handler exactly -- see that file's
    # comments for what each branch is and isn't allowed to trust the LLM
    # with.
    exact = retriever.exact_match(query)
    if exact:
        llm_answer = synthesize_answer(query, {"record": exact["record"]})
        show_answer(exact["record"], 1.0, llm_answer)
        return
    matches = retriever.search(query)
    best_score = max((m["score"] for m in matches), default=0.0)
    if not matches or best_score < TAU_LOW:
        category = matches[0]["record"].get("category") if matches else None
        show_abstention(abstention(category))
        return
    if matches[0]["score"] >= TAU_HIGH:
        # Only matches[0] -- the record the gate just accepted. The rest of
        # the pool is not evidence for this answer; see app/main.py.
        llm_answer = synthesize_answer(query, matches[0])
        show_answer(matches[0]["record"], matches[0]["score"], llm_answer)
        return
    # Anchored to matches[0], not best_score -- see app/main.py: a cutoff
    # derived from a higher-scoring pool record could otherwise exclude
    # matches[0] from its own suggestion list.
    minimum = max(0.12, matches[0]["score"] * 0.35)
    candidates = [match for match in matches[:3] if match["score"] >= minimum]
    # Same disambiguation step as app/main.py's clarify branch: the LLM
    # confirms the one candidate that actually answers the question -- not
    # necessarily the one it picked first, since a pick whose own rephrase
    # answers NOT_FOUND is disowned and the next candidate tried. The end
    # user never sees a raw pick-list -- once a record is confirmed,
    # (chosen, None) means the topic is settled but the rephrase didn't pass
    # grounding, so it's shown verbatim, not turned into a menu. Only a
    # fully-unconfirmed pick (None) routes to a human.
    disambiguated = disambiguate_and_answer(query, candidates)
    if disambiguated:
        chosen, llm_answer = disambiguated
        if llm_answer is not None:
            print("\n(LLM picked this from a few close topics -- see below for the others it considered.)")
        show_answer(chosen["record"], chosen["score"], llm_answer)
    else:
        show_abstention(abstention(matches[0]["record"].get("category")))


def main() -> None:
    if not (INDEX_DIR / "records.json").exists():
        raise SystemExit("Index unavailable. Run .venv/bin/python scripts/build_index.py first.")
    print("SASTRA Freshers Assistant — local terminal test")
    llm_status = f"enabled ({OLLAMA_MODEL})" if LLM_ENABLED else "disabled (LLM_ENABLED=0)"
    print(f"LLM synthesis: {llm_status}. Falls back to the verbatim stored answer if Ollama is unreachable.")
    print("Type a question. Type exit or quit to stop.")
    retriever = Retriever(INDEX_DIR)
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if len(query) < 2:
            print("Please enter a longer question.")
            continue
        respond(retriever, query)


if __name__ == "__main__":
    main()
