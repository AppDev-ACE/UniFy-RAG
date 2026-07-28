import json, tempfile, unittest
from pathlib import Path
from app.retrieval import Retriever, normalize
from app.safety import mandatory_route

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        records = [
            {"id":"hostel_biometric","category":"hostel","answer":"Verified answer","source":"Official","source_url":"https://www.sastra.edu/x","last_verified":"2026-07-28"},
            {"id":"cub_account","category":"finance","answer":"Verified answer","source":"Official","source_url":"https://www.sastra.edu/x","last_verified":"2026-07-28"},
        ]
        questions = [{"record_index":0,"question":"hostel biometric attendance timing"},{"record_index":1,"question":"city union bank account setup"}]
        (root/'records.json').write_text(json.dumps(records)); (root/'questions.json').write_text(json.dumps(questions)); self.r = Retriever(root)
    def tearDown(self): self.temp.cleanup()
    def test_abbreviation_is_expanded(self): self.assertIn("city union bank", normalize("CUB account"))
    def test_keyword_retrieval_returns_right_pair(self): self.assertEqual(self.r.search("biometric hostel attendance")[0]["record"]["id"], "hostel_biometric")
    def test_results_are_unique_records(self):
        self.assertEqual(len({x["record"]["id"] for x in self.r.search("hostel attendance", limit=3)}), len(self.r.search("hostel attendance", limit=3)))
    def test_ragging_is_never_retrieved(self): self.assertEqual(mandatory_route("I need to report ragging")[0], "ragging")
    def test_personal_marks_are_never_retrieved(self): self.assertEqual(mandatory_route("what are my marks")[0], "general")

if __name__ == "__main__": unittest.main()
