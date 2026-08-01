#!/usr/bin/env python3
"""Build the production question index from all trusted, non-superseded records."""
from __future__ import annotations
import argparse, json, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.json"
OUT = ROOT / "data" / "index"

def valid(record: dict, today: date, include_needs_review: bool = False, include_trusted_legacy: bool = False) -> tuple[bool, str, bool]:
    required = ("id", "questions", "answer", "source", "source_url", "last_verified")
    is_legacy_record = record.get("status") == "needs_review" and (include_needs_review or include_trusted_legacy)
    missing = [x for x in required if not record.get(x)]
    if record.get("status") != "active" and not is_legacy_record: return False, "not active", False
    if missing and not is_legacy_record: return False, "missing " + ", ".join(missing), False
    if is_legacy_record:
        # This is deliberately local-test-only: source provenance has not been
        # established, so production builds must never use this path.
        return True, "", False
    source_url = str(record["source_url"])
    # Product-owner statements may be traceable to a checked-in local source.
    # University policy still requires an official HTTPS source.
    if source_url.startswith("local://"):
        if record.get("source_type") != "product_owner":
            return False, "local source requires product_owner source_type", False
        local_path = (ROOT / source_url.removeprefix("local://")).resolve()
        if ROOT not in local_path.parents or not local_path.is_file():
            return False, "local source is not a checked-in file", False
    elif not source_url.startswith("https://"):
        return False, "source_url must be https or local", False
    stale = bool(record.get("valid_until") and date.fromisoformat(record["valid_until"]) < today)
    return True, "", stale

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-needs-review", action="store_true", help="LOCAL TESTING ONLY: index unsourced legacy records with a visible warning")
    group.add_argument("--official-only", action="store_true", help="Exclude trusted legacy records and index only officially sourced active records")
    args = parser.parse_args()
    include_trusted_legacy = not args.official_only and not args.include_needs_review
    if not CORPUS.exists():
        sys.exit("Corpus missing. Run python scripts/migrate_corpus.py first.")
    records = json.loads(CORPUS.read_text()); today = date.today(); active = []; skipped = []
    for record in records:
        ok, reason, stale = valid(record, today, args.include_needs_review, include_trusted_legacy)
        if ok:
            if record.get("status") == "needs_review":
                if args.include_needs_review:
                    record = {**record, "test_only_unverified": True,
                              "source": record.get("source") or "Unverified legacy collection (test mode)",
                              "source_url": record.get("source_url") or "local://data/corpus.json",
                              "last_verified": record.get("last_verified") or "Not verified",
                              "test_warning": "TEST MODE ONLY — this legacy answer has not been verified against an official source. Do not use it as policy or publish it."}
                elif include_trusted_legacy:
                    record = {**record, "trusted_legacy": True,
                              "source": record.get("source") or "Trusted legacy corpus — approved by UniFy project owner",
                              "source_url": record.get("source_url") or "local://data/product_sources/trusted_legacy_approval_2026-07-28.md",
                              "last_verified": record.get("last_verified") or "2026-07-28",
                              "trusted_notice": "Trusted legacy content approved by the project owner. It is not an official SASTRA policy source and should be re-verified when a current circular is available."}
            if stale:
                record = {**record, "stale_warning": "This information was accurate for a previous academic period. Check the current SASTRA circular or academic calendar."}
            active.append(record)
        else:
            skipped.append({"id": record.get("id"), "reason": reason})
    if not active:
        sys.exit("No verified active records. Review data/corpus.json; the index builder refuses unsourced data.")
    questions = [{"record_index": i, "question": q} for i, r in enumerate(active) for q in r["questions"]]
    OUT.mkdir(parents=True, exist_ok=True)
    # Never let a failed dense rebuild leave embeddings for a previous corpus.
    (OUT / "vectors.npy").unlink(missing_ok=True)
    (OUT / "records.json").write_text(json.dumps(active, ensure_ascii=False))
    (OUT / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))
    dense_embeddings, dense_error = False, None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        vectors = model.encode([x["question"] for x in questions], normalize_embeddings=True, show_progress_bar=True)
        np.save(OUT / "vectors.npy", vectors)
        dense_embeddings = True
        mode = " (TEST MODE: unverified legacy records included)" if args.include_needs_review else " (TRUSTED LEGACY: owner-approved records included)" if include_trusted_legacy else ""
        print("Built hybrid dense + sparse index", len(active), "records", len(questions), "phrasings" + mode)
    except Exception as exc:
        # A missing/offline model must never prevent the conservative BM25
        # service from starting, so this stays non-fatal. But a print()
        # alone is easy to miss on a deploy and silently ships keyword-only
        # search -- this is exactly what happened in production (dense
        # retrieval was off for an unknown period with no visible signal).
        # Recording it in build_report.json makes the degraded state
        # something a health check or operator can actually catch.
        dense_error = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: Built sparse-only index; dense BGE vectors were NOT created ({dense_error}). Retrieval quality is degraded -- see data/index/build_report.json.")
    (OUT / "build_report.json").write_text(json.dumps({"built_on": today.isoformat(), "records": len(active), "question_phrasings": len(questions), "test_mode_include_needs_review": args.include_needs_review, "include_trusted_legacy": include_trusted_legacy, "dense_embeddings": dense_embeddings, "dense_embeddings_error": dense_error, "skipped": skipped}, indent=2))

if __name__ == "__main__": main()
