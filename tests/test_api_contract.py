import unittest
from unittest.mock import patch
import app.main as main
from app.main import AskRequest, FeedbackRequest, ask, feedback, get_selected_pair, verification_status

class ApiContractTests(unittest.TestCase):
    def setUp(self):
        # /ask caches by normalised query text; several tests below reuse the
        # same query with different mocked outcomes, which the cache would
        # otherwise defeat by serving an earlier test's response.
        main._response_cache.clear()

    def test_ask_falls_back_to_verbatim_answer_when_llm_is_unreachable(self):
        # Forced (not relying on Ollama being absent from the environment --
        # a dev machine with Ollama running would make this test's outcome
        # depend on whether the LLM happened to succeed): synthesize_answer
        # returning None must still leave /ask serving the stored answer.
        with patch("app.main.log"), patch("app.main.synthesize_answer", return_value=None):
            result = ask(AskRequest(query="does SASTRA have bus service"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_mode"], "verbatim")
        self.assertEqual(result["answer"], result["raw_answer"])

    def test_ask_uses_llm_disambiguated_candidate_not_matches_zero(self):
        # "when is orientation day" is a real clarify-tier query (multiple
        # close candidates, none past TAU_HIGH). Force disambiguation to
        # pick a specific candidate to prove the response is built from the
        # LLM's chosen record throughout -- pair_id, raw_answer, and
        # confidence must all come from that record, not matches[0]. Since
        # there's no more clarify-list probe to read candidate IDs from, use
        # the retriever directly to find a real second-tier candidate.
        from app.retrieval import Retriever
        from app.config import INDEX_DIR, TAU_HIGH
        matches = Retriever(INDEX_DIR).search("when is orientation day")
        self.assertLess(matches[0]["score"], TAU_HIGH, "test assumption broke: query is no longer clarify-tier")
        second_pair_id = matches[1]["record"]["id"]

        def fake_disambiguate(query, candidates):
            chosen = next(c for c in candidates if c["record"]["id"] == second_pair_id)
            return chosen, "LLM-picked rephrasing of the second candidate."

        with patch("app.main.log"), patch("app.main.disambiguate_and_answer", side_effect=fake_disambiguate):
            result = ask(AskRequest(query="when is orientation day"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_mode"], "llm_disambiguated")
        self.assertEqual(result["pair_id"], second_pair_id)
        self.assertEqual(result["answer"], "LLM-picked rephrasing of the second candidate.")
        self.assertNotEqual(result["answer"], result["raw_answer"])

    def test_ask_routes_to_human_when_llm_never_confirms_a_candidate(self):
        # disambiguate_and_answer returning outright None means no candidate
        # was ever confirmed -- the end user must never see a raw pick-list,
        # so this must abstain, not return a suggestions array.
        with patch("app.main.log"), patch("app.main.disambiguate_and_answer", return_value=None):
            result = ask(AskRequest(query="when is orientation day"))
        self.assertEqual(result["status"], "abstained")
        self.assertNotIn("suggestions", result)

    def test_ask_serves_chosen_candidate_verbatim_when_llm_confirms_but_cant_phrase(self):
        # disambiguate_and_answer returning (chosen, None) means the LLM DID
        # confirm a candidate -- only its rephrase failed grounding. That
        # candidate must still be served verbatim, not a pick-list and not
        # human routing.
        from app.retrieval import Retriever
        from app.config import INDEX_DIR
        matches = Retriever(INDEX_DIR).search("when is orientation day")
        chosen_pair_id = matches[0]["record"]["id"]

        def fake_disambiguate(query, candidates):
            chosen = next(c for c in candidates if c["record"]["id"] == chosen_pair_id)
            return chosen, None

        with patch("app.main.log"), patch("app.main.disambiguate_and_answer", side_effect=fake_disambiguate):
            result = ask(AskRequest(query="when is orientation day"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_mode"], "verbatim")
        self.assertEqual(result["pair_id"], chosen_pair_id)
        self.assertEqual(result["answer"], result["raw_answer"])

    def test_feedback_accepts_a_verified_pair(self):
        with patch("app.main.log"):
            result = feedback(FeedbackRequest(rating="down", pair_id="legacy_095_how_often_can_a_first_year_student_go_home_on_weekends", note="test"))
        self.assertEqual(result, {"ok": True})

    def test_direct_ragging_pair_is_human_routed(self):
        result = get_selected_pair("legacy_105_what_is_the_anti_ragging_position_under_the_2026_27_ho")
        self.assertEqual(result["status"], "abstained")

    def test_unverified_test_record_has_machine_readable_status(self):
        self.assertEqual(verification_status({"test_only_unverified": True}), "unverified_test_only")
        self.assertEqual(verification_status({"trusted_legacy": True}), "trusted_legacy")
        self.assertEqual(verification_status({}), "verified")

if __name__ == "__main__": unittest.main()
