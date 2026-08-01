import unittest
from unittest.mock import Mock, patch
import requests
from app.llm import (REWORD_SYSTEM_PROMPT, _call_ollama, _echoes_correction,
                     _has_dangling_clause, _is_bare_enumeration, _names_drifted,
                     budget_spent, disambiguate_and_answer, latency_budget,
                     synthesize_answer)

MATCH = {"record": {"id": "x", "answer": "The bus fare to Thanjavur is around 16 rupees."}}
MATCH_WITH_TWO_NUMBERS = {"record": {"id": "z", "answer": "Classes run from 8:45 AM to 5:15 PM. First-year B.Tech students have an average of 6 class hours per day."}}
CANDIDATE_A = {"score": 0.84, "question": "when is orientation", "record": {
    "id": "a", "answer": "A two-day orientation is scheduled on August 3 and 4, 2026."}}
CANDIDATE_B = {"score": 0.36, "question": "parent orientation", "record": {
    "id": "b", "answer": "An optional parent orientation is held on August 3 from 11:30 a.m. to 1:00 p.m."}}
# Both below LLM_VETO_OVERRIDE_SCORE, so a walk in which every candidate
# refuses the question really does abstain. CANDIDATE_A scores 0.84 and is
# answered anyway when the model refuses it -- that is the override, tested
# on its own below. Abstention tests must use these weak fixtures, or what
# they actually assert is the override's absence rather than the behaviour
# they are named for.
WEAK_A = {"score": 0.31, "question": "when is orientation", "record": {
    "id": "wa", "answer": "A two-day orientation is scheduled on August 3 and 4, 2026."}}
WEAK_B = {"score": 0.22, "question": "parent orientation", "record": {
    "id": "wb", "answer": "An optional parent orientation is held on August 3 from 11:30 a.m. to 1:00 p.m."}}

def mock_response(content: str) -> Mock:
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"message": {"content": content}}
    return resp

def system_prompts(post: Mock) -> list[str]:
    return [call.kwargs["json"]["messages"][0]["content"] for call in post.call_args_list]

def reword_calls(post: Mock) -> int:
    """How many of `post`'s calls were the question-free reword pass rather
    than the question-driven ladder. Every ladder failure on an accepted
    record now spends one extra call here before the caller is allowed to
    fall back to verbatim, so the call counts below are ladder calls + 1."""
    return sum(1 for prompt in system_prompts(post) if prompt == REWORD_SYSTEM_PROMPT)

class LlmGroundingTests(unittest.TestCase):
    def test_grounded_reply_with_matching_numbers_is_returned(self):
        with patch("app.llm.requests.post", return_value=mock_response("It costs about 16 rupees to reach Thanjavur by bus!")):
            self.assertEqual(synthesize_answer("bus fare to thanjavur", MATCH),
                              "It costs about 16 rupees to reach Thanjavur by bus!")

    def test_connection_error_falls_back_to_none(self):
        with patch("app.llm.requests.post", side_effect=requests.ConnectionError("no ollama here")):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_timeout_falls_back_to_none(self):
        with patch("app.llm.requests.post", side_effect=requests.Timeout("slow")):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_not_found_sentinel_falls_back_to_none(self):
        with patch("app.llm.requests.post", return_value=mock_response("NOT_FOUND")):
            self.assertIsNone(synthesize_answer("something unrelated", MATCH))

    def test_dropped_number_falls_back_to_none(self):
        with patch("app.llm.requests.post", return_value=mock_response("It costs a little money to reach Thanjavur by bus.")):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_invented_number_falls_back_to_none(self):
        with patch("app.llm.requests.post", return_value=mock_response("It costs around 16 rupees, so keep 20 rupees handy just in case.")):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_empty_reply_falls_back_to_none(self):
        with patch("app.llm.requests.post", return_value=mock_response("   ")):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_disabled_flag_skips_network_call_entirely(self):
        with patch("app.llm.LLM_ENABLED", False), patch("app.llm.requests.post") as post:
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))
            post.assert_not_called()

    def test_no_match_skips_network_call_entirely(self):
        with patch("app.llm.requests.post") as post:
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", None))
            post.assert_not_called()

    def test_spelled_out_number_is_not_treated_as_drift(self):
        match = {"record": {"id": "y", "answer": "The best of 2 from the three is considered as the final internal marks."}}
        with patch("app.llm.requests.post", return_value=mock_response("The best of two out of three counts towards your final internal marks.")):
            result = synthesize_answer("how are internal marks decided", match)
        self.assertEqual(result, "The best of two out of three counts towards your final internal marks.")

    def test_dropped_number_is_recovered_by_one_retry(self):
        # First attempt silently drops "6"; second attempt (the retry, which
        # names the missing figure explicitly) includes it. This is the
        # real observed failure mode: omission, not fabrication.
        first = "Classes run from 8:45 AM to 5:15 PM."
        second = "Classes run from 8:45 AM to 5:15 PM, with 6 hours of class per day."
        with patch("app.llm.requests.post", side_effect=[mock_response(first), mock_response(second)]) as post:
            result = synthesize_answer("class timings", MATCH_WITH_TWO_NUMBERS)
        self.assertEqual(result, second)
        self.assertEqual(post.call_count, 2)

    def test_retry_still_dropping_the_number_falls_back_to_none(self):
        with patch("app.llm.requests.post", return_value=mock_response("Classes run from 8:45 AM to 5:15 PM.")) as post:
            result = synthesize_answer("class timings", MATCH_WITH_TWO_NUMBERS)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_invented_number_is_not_retried(self):
        # An invented number is a fabrication, not an omission -- retrying
        # the *answering* prompt would just invite the model to double down
        # on a different one. The one extra call here is the reword pass,
        # which is not that retry: it drops the question entirely and asks
        # only for the stored answer in different words, so it cannot double
        # down on a fabrication the question provoked.
        with patch("app.llm.requests.post", return_value=mock_response("It costs around 16 rupees to reach Thanjavur, so keep 20 rupees handy.")) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(reword_calls(post), 1)

    def test_dropped_named_place_is_recovered_by_one_retry(self):
        # The real observed bug: a whole clause -- "If you are visiting
        # Trichy, Kannappa Hotel is a great option" -- vanished from a
        # rephrase without touching a single digit. Names, not just
        # numbers, need the same drop-detection and one-shot retry.
        eatery = {"record": {"id": "e", "answer": (
            "In Thanjavur, popular spots include Barbequeen, KFC, and various "
            "Mandi restaurants. If you are visiting Trichy, Kannappa Hotel is a great option.")}}
        first = "In Thanjavur, you can enjoy meals at places like Barbequeen, KFC, and various Mandi restaurants."
        second = first + " If you're in Trichy, Kannappa Hotel is worth a visit too."
        with patch("app.llm.requests.post", side_effect=[mock_response(first), mock_response(second)]) as post:
            result = synthesize_answer("places to eat", eatery)
        self.assertEqual(result, second)
        self.assertEqual(post.call_count, 2)

    def test_name_case_difference_is_not_treated_as_drift(self):
        # "Sastra" in CONTEXT vs "SASTRA" in the rephrase is the same fact,
        # not a drop-and-reinvent -- comparison must be case-insensitive.
        match = {"record": {"id": "u", "answer": "You can visit the Sastra office anytime."}}
        with patch("app.llm.requests.post", return_value=mock_response("The SASTRA office is open for visits anytime.")):
            result = synthesize_answer("where is the office", match)
        self.assertEqual(result, "The SASTRA office is open for visits anytime.")

    def test_multiword_name_broken_by_a_dropped_word_is_still_caught(self):
        # "Sastra University" with "University" dropped entirely (not just
        # re-cased) is a genuine incomplete name, not a false positive.
        match = {"record": {"id": "u2", "answer": "You can visit the Sastra University office anytime."}}
        with patch("app.llm.requests.post", return_value=mock_response("The Sastra office is open for visits anytime.")) as post:
            result = synthesize_answer("where is the office", match)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_name_check_itself_stays_one_directional(self):
        # The name check is deliberately "missing only" -- see
        # _names_drifted's docstring: it cannot reliably tell a repositioned
        # repeat of a source name from a fabricated one, so an extra name
        # alone is not drift *by this check*. Asserted at the unit level
        # because end-to-end the fabrication below is now rejected anyway,
        # by the invented-sentence check (see the next test).
        self.assertFalse(_names_drifted("It costs 16 rupees to reach Thanjavur, via Example Road.",
                                        [MATCH["record"]["answer"]]))

    def test_invented_place_appended_to_a_grounded_sentence_is_rejected(self):
        # "via Example Road" is a road the source never mentions. Neither
        # the number check (16 is intact) nor the name check (one-directional)
        # objects; the sentence-level grounding check is what catches it.
        with patch("app.llm.requests.post", return_value=mock_response("It costs 16 rupees to reach Thanjavur, via Example Road.")):
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertIsNone(result)

    def test_leading_yes_comma_is_not_treated_as_a_dropped_name(self):
        # Real observed bug: "Yes, SASTRA has..." index-adjacent-matched
        # "Yes" and "SASTRA" as a 2-word name run (like "SASTRA DEEMED
        # UNIVERSITY"), so a rephrase that correctly dropped the RAG-ish
        # leading "Yes," was wrongly rejected as having dropped a name --
        # forcing fallback to the verbatim, un-personalised answer every
        # time. A comma after "Yes" must break the run.
        match = {"record": {"id": "c", "answer": (
            "Yes, SASTRA has multiple dining options. Popular spots are "
            "Krishna Canteen (KC), Canopy, and Vihar. All food is entirely vegetarian.")}}
        rephrase = ("The dining options available at SASTRA include Krishna Canteen (KC), "
                    "Canopy, and Vihar. All these places serve vegetarian food exclusively.")
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = synthesize_answer("what canteens are available", match)
        self.assertEqual(result, rephrase)
        self.assertEqual(post.call_count, 1)

    def test_verbatim_copy_is_retried_and_recovers_with_paraphrase(self):
        source = MATCH["record"]["answer"]
        paraphrased = "You'll pay roughly 16 rupees for the bus ride to Thanjavur."
        with patch("app.llm.requests.post", side_effect=[mock_response(source), mock_response(paraphrased)]) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertEqual(result, paraphrased)
        self.assertEqual(post.call_count, 2)

    def test_verbatim_copy_ignoring_punctuation_is_still_caught(self):
        # "within 6 PM" vs "within 6 PM." -- exactly the real case seen in
        # testing: a trailing period is not a rephrase.
        source = "Girls should be in the hostel within 6 PM"
        match = {"record": {"id": "w", "answer": source}}
        with patch("app.llm.requests.post", return_value=mock_response(source + ".")) as post:
            result = synthesize_answer("girls hostel curfew", match)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_verbatim_retry_still_copying_falls_back_to_none(self):
        source = MATCH["record"]["answer"]
        with patch("app.llm.requests.post", return_value=mock_response(source)) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_verbatim_retry_that_drifts_numbers_falls_back_to_none(self):
        source = MATCH["record"]["answer"]
        drifted_paraphrase = "You'll pay roughly 20 rupees for the bus ride to Thanjavur from SASTRA."
        with patch("app.llm.requests.post", side_effect=[
                mock_response(source), mock_response(drifted_paraphrase),
                # The reword pass, drifting the same way -- so verbatim
                # remains the outcome and the drift check still holds on the
                # last call before it.
                mock_response(drifted_paraphrase)]) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_added_solicitation_is_retried_and_recovers_with_plain_answer(self):
        # Real observed failure: the rephrase kept every fact intact but
        # tacked on an unsourced question back to the student.
        first = "It costs about 16 rupees to reach Thanjavur by bus. Do you need help planning your trip?"
        second = "It costs about 16 rupees to reach Thanjavur by bus."
        with patch("app.llm.requests.post", side_effect=[mock_response(first), mock_response(second)]) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertEqual(result, second)
        self.assertEqual(post.call_count, 2)

    def test_added_commentary_retry_still_failing_falls_back_to_none(self):
        first = "It costs about 16 rupees to reach Thanjavur by bus. That's a small amount, so keep it handy!"
        with patch("app.llm.requests.post", return_value=mock_response(first)) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_recased_source_name_is_not_treated_as_a_dropped_name(self):
        # The bug that made almost every answer fall back to verbatim. The
        # source capitalizes "Kurti"/"Boys"/"Collared" only because of its
        # label formatting; a natural rephrase lowercases them mid-sentence.
        # Re-parsing the reply with the source's own capitalization rules
        # then read those as dropped names and rejected a perfect answer.
        match = {"record": {"id": "d", "answer": "Girls: Kurti with dupatta Boys: Collared shirt with formal pants or casual jeans."}}
        rephrase = ("Girls wear a kurti with a dupatta, and boys wear a collared "
                    "shirt with either formal pants or casual jeans.")
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = synthesize_answer("what is the dress code", match)
        self.assertEqual(result, rephrase)
        self.assertEqual(post.call_count, 1)

    def test_genuinely_dropped_name_is_still_caught_after_recasing_fix(self):
        # The relaxation above must not cost the real guarantee: a name that
        # is absent from the reply in ANY casing is still a dropped fact.
        match = {"record": {"id": "d2", "answer": "Girls: Kurti with dupatta Boys: Collared shirt."}}
        with patch("app.llm.requests.post", return_value=mock_response("Girls wear a kurti with a dupatta.")) as post:
            result = synthesize_answer("what is the dress code", match)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_zero_padded_source_date_matches_an_unpadded_rephrase(self):
        match = {"record": {"id": "p", "answer": "The session commences on August 03, 2026."}}
        rephrase = "Your session gets underway on August 3, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = synthesize_answer("when does the session start", match)
        self.assertEqual(result, rephrase)
        self.assertEqual(post.call_count, 1)

    def test_filler_phrase_present_in_the_source_is_not_added_commentary(self):
        # "Make sure to" is a FILLER_MARKER, but here the source says it --
        # repeating a source instruction is not inventing commentary.
        match = {"record": {"id": "f", "answer": "Make sure to carry the original Transfer Certificate."}}
        rephrase = "You should make sure to bring your original Transfer Certificate along."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = synthesize_answer("what documents do I bring", match)
        self.assertEqual(result, rephrase)
        self.assertEqual(post.call_count, 1)

    def test_dropped_prohibition_clause_is_retried_and_recovered(self):
        # The real observed hole: "Non-vegetarian food is strictly
        # prohibited inside the campus and hostels" silently vanished from
        # the rephrase. It carries no digit and no proper noun, so neither
        # the number nor the name check could see it go -- a prohibition
        # dropped clean out of a safety-relevant answer.
        match = {"record": {"id": "h", "answer": (
            "Yes, you can bring food items to the hostel, but only vegetarian food "
            "is allowed. Non-vegetarian food is strictly prohibited inside the "
            "campus and hostels.")}}
        first = "Yes, you can bring food items to the hostel, but only vegetarian food is allowed."
        second = ("You're welcome to bring food into the hostel as long as it's vegetarian. "
                  "Non-vegetarian food is strictly prohibited anywhere inside the campus and the hostels.")
        with patch("app.llm.requests.post", side_effect=[mock_response(first), mock_response(second)]) as post:
            result = synthesize_answer("can I bring food to hostel", match)
        self.assertEqual(result, second)
        self.assertEqual(post.call_count, 2)

    def test_dropped_clause_that_is_never_restored_falls_back_to_none(self):
        match = {"record": {"id": "h2", "answer": (
            "Yes, you can bring food items to the hostel, but only vegetarian food "
            "is allowed. Non-vegetarian food is strictly prohibited inside the "
            "campus and hostels.")}}
        first = "Yes, you can bring food items to the hostel, but only vegetarian food is allowed."
        with patch("app.llm.requests.post", return_value=mock_response(first)) as post:
            result = synthesize_answer("can I bring food to hostel", match)
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

    def test_invented_advice_sentence_is_retried_and_recovers_with_plain_answer(self):
        # Observed verbatim from the model against the one-line bus-fare
        # record: every number and name intact, two whole sentences of
        # travel advice CONTEXT never said. No FILLER_MARKER matched it.
        first = ("The bus to Trichy costs around 40 rupees from Sastra University. "
                 "This is your best option for transportation as buses are usually "
                 "available and affordable.")
        second = "The bus to Trichy costs around 40 rupees from Sastra University."
        match = {"record": {"id": "t", "answer": "The bus fare to Trichy is around 40 rupees from Sastra University."}}
        with patch("app.llm.requests.post", side_effect=[mock_response(first), mock_response(second)]) as post:
            result = synthesize_answer("how do I get to Trichy", match)
        self.assertEqual(result, second)
        self.assertEqual(post.call_count, 2)

    def test_multi_sentence_rephrase_that_keeps_every_clause_is_not_flagged(self):
        # Guards the other side: heavy re-wording across several sentences
        # must not read as either a drop or an invention.
        match = {"record": {"id": "a2", "answer": (
            "A minimum of 75% attendance in each course is required to appear for "
            "the end semester examinations. The requirement is applied course by "
            "course, not on your overall average.")}}
        rephrase = ("You need at least 75% attendance in every single course to be allowed "
                    "to sit the end semester examinations. That requirement is checked course "
                    "by course rather than on your overall average.")
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = synthesize_answer("what is the attendance requirement", match)
        self.assertEqual(result, rephrase)
        self.assertEqual(post.call_count, 1)

    def test_plain_answer_with_no_filler_is_not_flagged(self):
        with patch("app.llm.requests.post", return_value=mock_response("You'll pay roughly 16 rupees for the bus ride to Thanjavur.")) as post:
            result = synthesize_answer("bus fare to thanjavur", MATCH)
        self.assertEqual(result, "You'll pay roughly 16 rupees for the bus ride to Thanjavur.")
        self.assertEqual(post.call_count, 1)

class RewordFallbackTests(unittest.TestCase):
    """The question-free last pass that keeps raw corpus text off a
    student's screen when the question-driven ladder can't phrase a record
    the caller has already accepted."""

    def test_ladder_failure_is_recovered_by_the_reword_pass(self):
        # Both ladder attempts drop the "6 class hours" figure; the reword,
        # which never sees the question and so has nothing to judge that
        # figure irrelevant to, keeps it. Before this pass existed the
        # student saw the stored record verbatim.
        dropped = "Classes run from 8:45 AM to 5:15 PM."
        reworded = "Your day runs 8:45 AM to 5:15 PM, and that works out to about 6 hours of class."
        with patch("app.llm.requests.post", side_effect=[
                mock_response(dropped), mock_response(dropped), mock_response(reworded)]) as post:
            result = synthesize_answer("class timings", MATCH_WITH_TWO_NUMBERS)
        self.assertEqual(result, reworded)
        self.assertEqual(reword_calls(post), 1)

    def test_reword_prompt_carries_the_record_but_never_the_question(self):
        # The whole reason this pass recovers what the ladder cannot: with
        # no question in the prompt, the model has nothing to reach for and
        # cannot decide a source clause is beside the point.
        with patch("app.llm.requests.post", return_value=mock_response("NOT_FOUND")) as post:
            synthesize_answer("is there an ATM on campus", MATCH)
        reword = next(call for call in post.call_args_list
                      if call.kwargs["json"]["messages"][0]["content"] == REWORD_SYSTEM_PROMPT)
        content = reword.kwargs["json"]["messages"][1]["content"]
        self.assertIn(MATCH["record"]["answer"], content)
        self.assertNotIn("ATM", content)

    def test_reworded_text_is_held_to_the_same_grounding_checks(self):
        # The reword is a cheaper prompt, not a looser one: a fare that
        # changes in transit is rejected here exactly as in the ladder, and
        # the caller falls back to verbatim.
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"),
                mock_response("The bus to Thanjavur will cost you about 20 rupees.")]):
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))

    def test_unreachable_ollama_does_not_attempt_a_reword(self):
        # UNAVAILABLE means the model never answered, so a second call is
        # just another timeout charged to the student's wait.
        with patch("app.llm.requests.post", side_effect=requests.ConnectionError("no ollama here")) as post:
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))
        self.assertEqual(post.call_count, 1)

    def test_reword_is_skipped_while_disambiguation_is_still_probing(self):
        # `reword_on_failure=False` is what lets the disambiguation loop read
        # a NOT_FOUND as "wrong record" -- rewording there would produce a
        # fluent answer from a record that doesn't answer the question.
        with patch("app.llm.requests.post", return_value=mock_response("NOT_FOUND")) as post:
            self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH, reword_on_failure=False))
        self.assertEqual(post.call_count, 1)

class DisambiguationTests(unittest.TestCase):
    """The clarify-tier walk. There is no separate picker call any more (see
    disambiguate_and_answer): the candidates are probed one at a time in
    retrieval order and the first that answers is used, so the call counts
    below are the walk's own probes plus whatever the retry ladder spends."""

    def test_first_candidate_that_answers_is_used(self):
        rephrase = "A two-day orientation runs on August 3 and 4, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            result = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIsNotNone(result)
        chosen, answer = result
        self.assertIs(chosen, CANDIDATE_A)
        self.assertEqual(answer, rephrase)
        # One probe, and CANDIDATE_B is never touched.
        self.assertEqual(post.call_count, 1)

    def test_no_pick_call_is_made(self):
        # The regression guard for the change itself: every call must be a
        # grounding call about one record, never a "choose between these"
        # call listing all the candidates at once.
        rephrase = "A two-day orientation runs on August 3 and 4, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)) as post:
            disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        for prompt in system_prompts(post):
            self.assertNotIn("INDEX:", prompt)
        for call in post.call_args_list:
            user = call.kwargs["json"]["messages"][1]["content"]
            self.assertNotIn(CANDIDATE_B["record"]["answer"], user)

    def test_probe_is_grounded_only_in_the_candidate_being_probed(self):
        # CANDIDATE_B's facts must never leak into the grounding context for
        # a rephrase of CANDIDATE_A -- that is what protects pair_id/source
        # attribution.
        with patch("app.llm.requests.post", return_value=mock_response(
                "You will have a two-day orientation on August 3 and 4, 2026.")) as post:
            disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        probe_user = post.call_args_list[0].kwargs["json"]["messages"][1]["content"]
        self.assertIn(CANDIDATE_A["record"]["answer"], probe_user)
        self.assertNotIn(CANDIDATE_B["record"]["answer"], probe_user)

    def test_number_drift_falls_back_to_chosen_with_no_text(self):
        # The record is right -- the model engaged with it rather than
        # disowning it -- and only the rephrase drifted. That is
        # "(chosen, None)", not outright None: the caller serves CANDIDATE_A
        # verbatim, not a pick-list.
        drifted = "A two-day orientation runs on August 5 and 6, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(drifted)):
            result = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIsNotNone(result)
        chosen, answer = result
        self.assertIs(chosen, CANDIDATE_A)
        self.assertIsNone(answer)

    def test_strong_retrieval_match_is_answered_despite_a_total_llm_veto(self):
        # The "even though it has a source, still the same" bug. Every
        # candidate is refused by the model, but CANDIDATE_A is a strong,
        # human-reviewed retrieval match, and one small model's opinion does
        # not get to discard it -- exactly as on the past-TAU_HIGH path,
        # where retrieval decides relevance and the LLM only phrases.
        reworded = "Orientation runs for two days, on August 3 and 4, 2026."
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND"),
                mock_response(reworded)]):
            chosen, text = disambiguate_and_answer("documents for inauguration", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_A)
        self.assertEqual(text, reworded)

    def test_override_answers_from_the_best_scoring_record(self):
        # The override exists because retrieval outranks the model here, so
        # it must serve retrieval's best record -- CANDIDATE_A at 0.84 --
        # once every candidate has refused the question.
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND"),
                mock_response("Orientation is two days, August 3 and 4, 2026.")]):
            chosen, _ = disambiguate_and_answer("when is the session", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_A)

    def test_override_still_serves_the_record_when_the_reword_fails(self):
        # (match, None) -- the record is right, only the wording is
        # unusable, so the caller shows it verbatim rather than abstaining.
        with patch("app.llm.requests.post", return_value=mock_response("NOT_FOUND")):
            chosen, text = disambiguate_and_answer("documents for inauguration", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_A)
        self.assertIsNone(text)

    def test_override_does_not_fire_on_weak_evidence(self):
        # The other half of the rule: when the model's refusal and weak
        # retrieval agree, that is a real abstention. This is what keeps
        # off-corpus questions ("where is the swimming pool") routed to a
        # human instead of answered from whatever ranked first.
        with patch("app.llm.requests.post", return_value=mock_response("NOT_FOUND")):
            self.assertIsNone(disambiguate_and_answer("where is the swimming pool", [WEAK_A, WEAK_B]))

    def test_walk_abstains_when_no_record_answers(self):
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND")]) as post:
            self.assertIsNone(disambiguate_and_answer("where is the swimming pool", [WEAK_A, WEAK_B]))
        # Both candidates probed, nothing else attempted.
        self.assertEqual(post.call_count, 2)

    def test_refusal_shaped_reply_does_not_confirm_the_record(self):
        # Observed against the live model: "can I keep a pet dog in the
        # hostel" probed against a record about storing food in the hostel
        # came back "NO" -- not the NOT_FOUND sentinel, so it read as a
        # wording failure and would have confirmed the food record and
        # served it verbatim for a question about pets. A reply made of
        # none of the record's own substance is not a near-miss rephrase of
        # it, so the walk must move on rather than settle.
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NO"),
                mock_response("Parent orientation is optional, on August 3 from 11:30 a.m. to 1:00 p.m.")]):
            chosen, text = disambiguate_and_answer("is the parent session optional", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_B)
        self.assertIsNotNone(text)

    def test_hedge_that_shares_no_source_substance_does_not_confirm_the_record(self):
        # The prose form of the same failure: fluent, not the sentinel, and
        # made of nothing the record says.
        hedge = "Exact timings cannot be provided as they differ among individual hostels."
        with patch("app.llm.requests.post", return_value=mock_response(hedge)) as post:
            self.assertIsNone(disambiguate_and_answer("what time is curfew", [WEAK_A, WEAK_B]))
        self.assertEqual(reword_calls(post), 0)

    def test_near_miss_rephrase_still_confirms_the_record(self):
        # The other side of the engagement rule: a reply built almost
        # entirely from the record's own substance, failing only because it
        # dropped one figure, IS a wording failure -- the record is right.
        # This must keep reaching the reword pass rather than being read as
        # a refusal and walked past.
        near_miss = "There is a two-day orientation for first-year students, scheduled in August."
        with patch("app.llm.requests.post", return_value=mock_response(near_miss)) as post:
            chosen, _ = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_A)
        self.assertEqual(reword_calls(post), 1)

    def test_candidate_disowned_by_its_own_probe_moves_on_to_the_next(self):
        # A NOT_FOUND is a verdict about the record, not the wording, so the
        # walk must fall through rather than serve a record its own probe
        # just disowned -- that was wrong record AND raw corpus text.
        answer = "Parent orientation is optional and runs August 3, 11:30 a.m. to 1:00 p.m."
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response(answer)]) as post:
            chosen, text = disambiguate_and_answer("when is the parent session", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_B)
        self.assertEqual(text, answer)
        self.assertEqual(post.call_count, 2)

    def test_walk_grounds_each_probe_in_that_candidate_alone(self):
        # Falling through to another candidate must not widen the grounding
        # context -- pair_id/source attribution depends on each probe seeing
        # exactly one record.
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"),
                mock_response("Parent orientation is optional, August 3, 11:30 a.m. to 1:00 p.m.")]) as post:
            disambiguate_and_answer("when is the parent session", [CANDIDATE_A, CANDIDATE_B])
        second_probe = post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        self.assertIn(CANDIDATE_B["record"]["answer"], second_probe)
        self.assertNotIn(CANDIDATE_A["record"]["answer"], second_probe)

    def test_unreachable_ollama_mid_walk_abstains_rather_than_probing_on(self):
        with patch("app.llm.requests.post", side_effect=requests.ConnectionError("ollama died")) as post:
            self.assertIsNone(disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B]))
        self.assertEqual(post.call_count, 1)

    def test_disabled_flag_skips_network_call_entirely(self):
        with patch("app.llm.LLM_ENABLED", False), patch("app.llm.requests.post") as post:
            self.assertIsNone(disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B]))
            post.assert_not_called()

    def test_verbatim_copy_is_retried_and_recovers_with_paraphrase(self):
        copy_reply = CANDIDATE_A["record"]["answer"]
        paraphrased = "You'll have two full days of orientation starting August 3rd, running into the 4th, 2026."
        with patch("app.llm.requests.post", side_effect=[
                mock_response(copy_reply), mock_response(paraphrased)]) as post:
            result = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIsNotNone(result)
        chosen, answer = result
        self.assertIs(chosen, CANDIDATE_A)
        self.assertEqual(answer, paraphrased)
        self.assertEqual(post.call_count, 2)

    def test_verbatim_copy_retry_still_copying_falls_back_to_chosen_with_no_text(self):
        # Same reasoning as the number-drift case: the record is still good,
        # only the wording never became usable -- (chosen, None), not None.
        copy_reply = CANDIDATE_A["record"]["answer"]
        with patch("app.llm.requests.post", return_value=mock_response(copy_reply)) as post:
            result = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        chosen, answer = result
        self.assertIs(chosen, CANDIDATE_A)
        self.assertIsNone(answer)
        # Probe, the anti-copy retry, then the reword pass -- and crucially
        # it stops there rather than walking on to CANDIDATE_B: the model
        # engaged with CANDIDATE_A instead of answering NOT_FOUND, so the
        # topic is settled and only the wording failed.
        self.assertEqual(post.call_count, 3)
        self.assertEqual(reword_calls(post), 1)

class LatencyBudgetTests(unittest.TestCase):
    """One question's whole allowance of Ollama time. A clarify-tier query
    spends 4-6 calls in series, the Flutter client gives up at 20s, and
    7.6% of live cache-miss queries were overrunning it -- so past the
    budget the student gets the retrieved record verbatim instead of a
    timeout."""

    def test_expired_budget_serves_the_top_record_verbatim_not_an_abstention(self):
        # The whole point: a human-reviewed record the index already holds
        # beats "I don't have an answer for that".
        with patch("app.llm.requests.post") as post:
            with latency_budget(0):
                result = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIsNotNone(result)
        chosen, text = result
        self.assertIs(chosen, CANDIDATE_A)
        self.assertIsNone(text)          # verbatim -- no LLM wording
        post.assert_not_called()         # and not one call was started

    def test_expired_budget_still_refuses_a_shape_mismatched_record(self):
        # Running out of time is no reason to reintroduce the bug this
        # module was just fixed for: a yes/no-shaped record must not be
        # served raw to a student who asked "how many". The next candidate
        # is used instead.
        names_them = {"score": 0.59, "question": "which hostels for first years",
                      "record": {"id": "h", "answer": "The boys' hostels are AHALYA and ARUNDHATHI."}}
        with patch("app.llm.requests.post"):
            with latency_budget(0):
                chosen, text = disambiguate_and_answer("How many hostels are there for boys ?",
                                                        [POLAR_CANDIDATE, names_them])
        self.assertIs(chosen, names_them)
        self.assertIsNone(text)

    def test_expired_budget_abstains_when_every_record_is_shape_mismatched(self):
        with patch("app.llm.requests.post"):
            with latency_budget(0):
                self.assertIsNone(disambiguate_and_answer("How many hostels are there for boys ?",
                                                           [POLAR_CANDIDATE]))

    def test_budget_expiring_is_not_confused_with_ollama_being_down(self):
        # Both surface as UNAVAILABLE from the ladder and they mean opposite
        # things: out of time serves the record, a dead model abstains.
        with patch("app.llm.requests.post", side_effect=requests.ConnectionError("ollama died")):
            self.assertIsNone(disambiguate_and_answer("when is orientation", [WEAK_A, WEAK_B]))

    def test_synthesize_answer_falls_back_to_verbatim_when_the_budget_is_gone(self):
        with patch("app.llm.requests.post") as post:
            with latency_budget(0):
                self.assertIsNone(synthesize_answer("bus fare to thanjavur", MATCH))
        post.assert_not_called()

    def test_a_live_budget_does_not_interfere(self):
        rephrase = "A two-day orientation runs on August 3 and 4, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)):
            with latency_budget(30):
                chosen, text = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertIs(chosen, CANDIDATE_A)
        self.assertEqual(text, rephrase)

    def test_no_budget_in_force_leaves_behaviour_unchanged(self):
        # The terminal tester and scripts/ call in without a budget.
        self.assertFalse(budget_spent())
        rephrase = "A two-day orientation runs on August 3 and 4, 2026."
        with patch("app.llm.requests.post", return_value=mock_response(rephrase)):
            chosen, text = disambiguate_and_answer("when is orientation", [CANDIDATE_A, CANDIDATE_B])
        self.assertEqual(text, rephrase)

    def test_per_call_timeout_shrinks_to_what_is_left_of_the_budget(self):
        # A call must not be able to outlive the budget it started inside.
        with patch("app.llm.requests.post", return_value=mock_response("ok")) as post:
            with latency_budget(0.5):
                _call_ollama("system", "user")
        self.assertLessEqual(post.call_args.kwargs["timeout"], 0.5)

    def test_budget_is_per_request_and_does_not_leak(self):
        with latency_budget(0):
            self.assertTrue(budget_spent())
        self.assertFalse(budget_spent())

class CorrectionEchoTests(unittest.TestCase):
    """A retry that recites the correction's own list back instead of
    rewriting the answer. Every grounding check passes it by construction --
    the correction is assembled out of the source's figures, names and
    sentences -- so it reached students as the whole answer."""

    def test_retry_that_echoes_the_correction_is_rejected(self):
        drops_a_number = "Classes run from 8:45 AM to 5:15 PM for first-year B.Tech students."
        echo = ('6 and these statements: "First-year B.Tech students have an '
                'average of 6 class hours per day."')
        with patch("app.llm.requests.post", side_effect=[
                mock_response(drops_a_number), mock_response(echo)]):
            self.assertIsNone(synthesize_answer("what are the class timings",
                                                 MATCH_WITH_TWO_NUMBERS, reword_on_failure=False))

    def test_scaffolding_phrases_and_bare_lists_are_both_caught(self):
        observed = ('3, 2, Continuous, Internal, Assessments, Three, CIAs and these '
                    'statements: "The best of 2 from the three is considered as the '
                    'final internal marks".')
        self.assertTrue(_echoes_correction(observed))
        self.assertTrue(_is_bare_enumeration(observed))
        # A real answer that happens to list things is not a bare list: its
        # first comma-separated fragment is a clause, not a token.
        self.assertFalse(_is_bare_enumeration(
            "The boys' hostels are AHALYA and ARUNDHATHI, and the girls' hostels are "
            "SANDIPANI SADAN and RAJALAKSHMI VIHAR."))
        self.assertFalse(_echoes_correction("Three CIAs are held each semester."))

class DanglingClauseTests(unittest.TestCase):
    def test_trailing_fragment_is_not_served(self):
        # "...if possible" re-punctuated into a sentence of its own. Every
        # fact survives, nothing is invented, and the result is a fragment.
        record = {"record": {"id": "s", "answer":
                  "Yes, seniors will assist and guide you to solve the problems you face, if possible."}}
        split_clause = ("Yes, seniors will assist and guide you to solve the problems "
                        "you face. If possible.")
        with patch("app.llm.requests.post", return_value=mock_response(split_clause)):
            self.assertIsNone(synthesize_answer("seniors in sastra", record, reword_on_failure=False))

    def test_ordinary_second_sentence_is_kept(self):
        self.assertFalse(_has_dangling_clause(
            "Girls should be in the hostel by 6 PM. Boys should be back by 8:30 PM."))

class NameInflectionTests(unittest.TestCase):
    """Title-cased ordinary words ("Continuous Internal Assessments") parse
    as source names, and a rephrase that inflects them ("continuously",
    "assessment") was scored as having dropped facts it kept -- the retry
    that echoed its own correction above is what that spurious retry
    produced."""

    def test_inflected_forms_count_as_present(self):
        source = ["CIA - Continuous Internal Assessments are the internal exams. "
                  "Three CIAs are conducted per semester."]
        reply = ("CIA exams run continuously through each semester, with three such "
                 "internal assessment rounds in all.")
        self.assertFalse(_names_drifted(reply, source))

    def test_a_genuinely_dropped_name_is_still_caught(self):
        source = ["Krishna Canteen, Canopy and Kannappa Hotel are all good options."]
        self.assertTrue(_names_drifted("Krishna Canteen and Canopy are good options.", source))

# A record whose stored answer is written as the reply to a yes/no question.
# Scored well above LLM_VETO_OVERRIDE_SCORE, so a shape-based abstention here
# really is the shape rule and not a weak-evidence one.
POLAR_CANDIDATE = {"score": 0.84, "question": "boys hostel outing time", "record": {
    "id": "p", "answer": "Yes, boys are allowed to go outside the hostels, but they must return before 9:30 PM."}}

class QuestionShapeGuardTests(unittest.TestCase):
    def test_override_falls_through_to_the_record_that_answers_the_question(self):
        # The reported bug in full. Both records are strong and a hair apart
        # -- the outing-time record edges it on score alone -- and the model
        # has refused both. Setting the yes/no-shaped one aside leaves the
        # record that actually names the hostels, which is the answer.
        names_the_hostels = {"score": 0.597, "question": "which hostels are allotted to first years",
                             "record": {"id": "h", "answer":
                                        "The boys' hostels are AHALYA and ARUNDHATHI."}}
        reworded = "You'll be in either AHALYA or ARUNDHATHI -- those are the boys' hostels."
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND"),
                mock_response(reworded)]):
            chosen, text = disambiguate_and_answer("What are the hostels available for boys ?",
                                                    [POLAR_CANDIDATE, names_the_hostels])
        self.assertIs(chosen, names_the_hostels)
        self.assertEqual(text, reworded)

    def test_value_question_does_not_get_a_yes_no_record_by_override(self):
        # The reported bug: "What are the hostels available for boys?" and
        # "How many hostels are there for boys?" both answered "Yes, boys are
        # allowed to go outside the hostels..." -- a sentence that names no
        # hostel and counts nothing. Every candidate refused the question, so
        # only topical overlap was carrying this record.
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND")]):
            self.assertIsNone(disambiguate_and_answer("How many hostels are there for boys ?",
                                                       [POLAR_CANDIDATE, WEAK_B]))

    def test_same_record_is_served_when_the_student_did_ask_yes_or_no(self):
        # The other half: the shape rule must not cost the override any of
        # the cases it exists for. Same record, same total veto, polar
        # question -- answered.
        reworded = "Boys may head out of the hostels as long as they are back before 9:30 PM."
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND"),
                mock_response(reworded)]):
            chosen, text = disambiguate_and_answer("Are boys allowed to go outside the hostels?",
                                                    [POLAR_CANDIDATE, WEAK_B])
        self.assertIs(chosen, POLAR_CANDIDATE)
        self.assertEqual(text, reworded)

    def test_bare_topic_query_still_gets_a_yes_no_record(self):
        # "Seniors in SASTRA" asks for no particular value, so a yes/no
        # record answers it fine and must not be discarded. Only an explicit
        # what/when/how-many question rejects on shape.
        reworded = "Boys can go out of the hostels, but they need to be back by 9:30 PM."
        with patch("app.llm.requests.post", side_effect=[
                mock_response("NOT_FOUND"), mock_response("NOT_FOUND"),
                mock_response(reworded)]):
            chosen, text = disambiguate_and_answer("boys hostel outing", [POLAR_CANDIDATE, WEAK_B])
        self.assertIs(chosen, POLAR_CANDIDATE)
        self.assertEqual(text, reworded)

    def test_unphraseable_yes_no_record_is_not_served_raw_to_a_value_question(self):
        # The other route the reported answer arrived by: the model engages
        # with the record (so the walk would settle here), the rephrase fails
        # every check, the reword fails too, and the caller shows raw corpus
        # text. Serving "Yes, ..." unrephrased to "how many" is the one case
        # where an engaged-with record is still the wrong record.
        drifted = "Boys can go outside the hostels but must be back before 10:30 PM."
        with patch("app.llm.requests.post", side_effect=[
                mock_response(drifted), mock_response("NOT_FOUND"),
                mock_response("NOT_FOUND")]):
            self.assertIsNone(disambiguate_and_answer("How many hostels are there for boys ?",
                                                       [POLAR_CANDIDATE]))

if __name__ == "__main__": unittest.main()
