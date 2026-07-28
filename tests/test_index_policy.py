import unittest
import json
from datetime import date
from pathlib import Path
from scripts.build_index import valid

class IndexPolicyTests(unittest.TestCase):
    def test_expired_active_record_is_retained_with_warning_flag(self):
        record = {
            "id": "old_calendar", "questions": ["old"], "answer": "old",
            "source": "Official", "source_url": "https://www.sastra.edu/old",
            "last_verified": "2025-01-01", "status": "active", "valid_until": "2025-07-01",
        }
        self.assertEqual(valid(record, date(2026, 7, 28)), (True, "", True))

    def test_unsourced_active_record_is_rejected(self):
        record = {"id": "bad", "questions": ["x"], "answer": "x", "status": "active"}
        self.assertFalse(valid(record, date(2026, 7, 28))[0])

    def test_local_source_requires_product_owner_provenance(self):
        record = {
            "id": "bad_local", "questions": ["x"], "answer": "x", "source": "Unknown",
            "source_url": "local://data/product_sources/unify_capabilities_2026-07-28.md",
            "last_verified": "2026-07-28", "status": "active", "source_type": "official",
        }
        self.assertFalse(valid(record, date(2026, 7, 28))[0])

    def test_all_active_records_have_minimum_question_coverage(self):
        root = Path(__file__).resolve().parents[1]
        records = json.loads((root / "data" / "corpus.json").read_text())
        too_short = [record["id"] for record in records if record["status"] == "active" and len(record["questions"]) < 4]
        self.assertEqual(too_short, [])

if __name__ == "__main__": unittest.main()
