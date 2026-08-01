import json, tempfile, unittest
from pathlib import Path
from app.retrieval import Retriever, normalize
from app.safety import abstention, mandatory_route

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.json"

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
    def test_exact_match_uses_normalisation(self):
        match = self.r.exact_match("HOSTEL biometric attendance timing?")
        self.assertIsNotNone(match)
        self.assertEqual(match["record"]["id"], "hostel_biometric")
    def test_exact_match_rejects_unknown_question(self):
        self.assertIsNone(self.r.exact_match("tell me something unrelated"))
    def test_results_are_unique_records(self):
        self.assertEqual(len({x["record"]["id"] for x in self.r.search("hostel attendance", limit=3)}), len(self.r.search("hostel attendance", limit=3)))
    def test_ragging_is_never_retrieved(self): self.assertEqual(mandatory_route("I need to report ragging")[0], "ragging")
    def test_personal_marks_are_never_retrieved(self): self.assertEqual(mandatory_route("what are my marks")[0], "general")
    def test_unknown_question_routes_to_campus_peer_support(self):
        response = abstention("hostel")
        self.assertEqual(response["contacts"][0]["label"], "Campus seniors or fresher volunteers")
        self.assertNotIn("email", response["contacts"][0])

class DressCodeDisambiguationTests(unittest.TestCase):
    """Regression test for a real bug: a fresher asking about the lab-specific
    uniform policy ("dress code for lab") was matched to unrelated records
    ("gym dress code", general "campus dress code") that happen to share the
    literal phrase "dress code", while the correct lab-uniform record was
    filtered out of the results entirely. Fixed by making synonym expansion
    (dress <-> uniform/attire) a query-side-only bridge, never indexed into
    documents -- a record merely containing the word "dress" must not gain
    false "uniform" affinity.

    Built from the real data/corpus.json (BM25 only, no vectors.npy -- this
    discrimination was verified to hold on the sparse channel alone, so the
    test stays fast and self-sufficient without needing an index build or
    the embedding model) because the bug is specifically about how these
    three real records compete against each other; a synthetic fixture
    would not reproduce it.
    """
    @classmethod
    def setUpClass(cls):
        corpus = json.loads(CORPUS_PATH.read_text())
        active = [r for r in corpus if r.get("status") != "superseded"]
        questions = [{"record_index": i, "question": q} for i, r in enumerate(active) for q in r["questions"]]
        cls.temp = tempfile.TemporaryDirectory(); root = Path(cls.temp.name)
        (root / "records.json").write_text(json.dumps(active))
        (root / "questions.json").write_text(json.dumps(questions))
        cls.r = Retriever(root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def top_id(self, query):
        matches = self.r.search(query)
        return matches[0]["record"]["id"] if matches else None

    def test_lab_specific_query_matches_lab_uniform_record(self):
        for query in ("dress code for lab", "lab dress code", "what to wear in lab"):
            with self.subTest(query=query):
                self.assertTrue(self.top_id(query).startswith("legacy_020"))

    def test_general_campus_query_matches_general_dress_code_record(self):
        for query in ("what is the dress code in campus", "is there any dress code"):
            with self.subTest(query=query):
                self.assertTrue(self.top_id(query).startswith("legacy_155"))

    def test_gym_query_matches_gym_record(self):
        self.assertTrue(self.top_id("gym dress code").startswith("legacy_040"))

if __name__ == "__main__": unittest.main()
