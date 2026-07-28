#!/usr/bin/env python3
"""Interactive local terminal for trying the Freshers Assistant safely."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import INDEX_DIR, TAU_HIGH, TAU_LOW
from app.retrieval import Retriever
from app.safety import abstention, mandatory_route


def show_answer(record: dict, confidence: float | None = None) -> None:
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


def choose_match(matches: list[dict]) -> dict | None:
    print("\nI found close topics. Choose one to view its stored answer:")
    for number, match in enumerate(matches, 1):
        print(f"  {number}. {match['question']}")
    while True:
        selection = input("Choose 1-%d, or press Enter to skip: " % len(matches)).strip()
        if not selection:
            return None
        if selection.isdigit() and 1 <= int(selection) <= len(matches):
            return matches[int(selection) - 1]
        print("Please enter a listed number or press Enter.")


def respond(retriever: Retriever, query: str) -> None:
    forced = mandatory_route(query)
    if forced:
        show_abstention(abstention(*forced))
        return
    matches = retriever.search(query)
    if not matches or matches[0]["score"] < TAU_LOW:
        category = matches[0]["record"].get("category") if matches else None
        show_abstention(abstention(category))
        return
    if matches[0]["score"] >= TAU_HIGH:
        show_answer(matches[0]["record"], matches[0]["score"])
        return
    minimum = max(0.12, matches[0]["score"] * 0.35)
    selected = choose_match([match for match in matches[:3] if match["score"] >= minimum])
    if selected:
        show_answer(selected["record"], selected["score"])
    else:
        show_abstention(abstention(matches[0]["record"].get("category")))


def main() -> None:
    if not (INDEX_DIR / "records.json").exists():
        raise SystemExit("Index unavailable. Run .venv/bin/python scripts/build_index.py first.")
    print("SASTRA Freshers Assistant — local terminal test")
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
