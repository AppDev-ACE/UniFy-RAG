import unittest
from unittest.mock import patch
from app.main import FeedbackRequest, feedback, get_selected_pair, verification_status

class ApiContractTests(unittest.TestCase):
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
