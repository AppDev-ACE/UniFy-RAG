import unittest

from app.phrasing import asks_for_value, is_polar_answer, is_polar_question, strip_polar_prefix

class QuestionShapeTests(unittest.TestCase):
    def test_wh_questions_are_not_polar_and_ask_for_a_value(self):
        for query in ["What are the hostels available for boys ?",
                      "How many hostels are there for boys ?",
                      "What is the hostel rule for girls ?",
                      "Which hostels are allotted to first years",
                      "When are the CIA exams"]:
            with self.subTest(query=query):
                self.assertFalse(is_polar_question(query))
                self.assertTrue(asks_for_value(query))

    def test_yes_no_questions_are_polar(self):
        for query in ["Are boys allowed to go outside the hostels?",
                      "Is a laptop compulsory for first year?",
                      "Can I keep maggi noodles in my hostel room",
                      "Do seniors help freshers at SASTRA",
                      "Whether seniors are friendly ?",
                      "whether stationery shop is available"]:
            with self.subTest(query=query):
                self.assertTrue(is_polar_question(query))
                self.assertFalse(asks_for_value(query))

    def test_lead_ins_do_not_hide_the_opener(self):
        self.assertTrue(is_polar_question("So can I bring a laptop to class"))
        self.assertTrue(asks_for_value("please what are the canteens"))

    def test_bare_topic_is_neither_shape(self):
        # A topic phrase demands no particular value, so it must not be
        # allowed to reject a yes/no-shaped record -- but leading its answer
        # with "Yes" is still wording nobody asked for.
        for query in ["Seniors in SASTRA", "hostel gate closing time"]:
            with self.subTest(query=query):
                self.assertFalse(is_polar_question(query))
                self.assertFalse(asks_for_value(query))

class PolarPrefixTests(unittest.TestCase):
    def test_yes_prefix_dropped_for_a_question_that_did_not_ask_yes_or_no(self):
        self.assertEqual(
            strip_polar_prefix("Yes, boys are allowed to go outside the hostels, but they must return before 9:30 PM.",
                                "What are the hostels available for boys ?"),
            "Boys are allowed to go outside the hostels, but they must return before 9:30 PM.")

    def test_yes_prefix_kept_for_a_yes_no_question(self):
        answer = "Yes, boys are allowed to go outside the hostels, but they must return before 9:30 PM."
        self.assertEqual(strip_polar_prefix(answer, "Are boys allowed to go outside the hostels?"), answer)

    def test_no_prefix_dropped_only_when_the_clause_still_says_no(self):
        self.assertEqual(
            strip_polar_prefix("No, day scholars are not allowed to bring non-vegetarian food to campus.",
                                "What food can day scholars bring to campus"),
            "Day scholars are not allowed to bring non-vegetarian food to campus.")
        # Nothing behind the "No," repeats the verdict, so stripping it would
        # invert the answer. Left alone.
        kept = "No, it is part of the tuition fee only."
        self.assertEqual(strip_polar_prefix(kept, "What is the fee for a minor specialization"), kept)

    def test_bare_sentence_verdict_is_never_stripped(self):
        # "No." carries the whole verdict on its own -- only the comma form
        # is followed by a clause that restates it.
        answer = "No. It is part of the tuition fee only."
        self.assertEqual(strip_polar_prefix(answer, "What is the fee for a minor specialization"), answer)

    def test_answer_without_a_polar_opener_is_untouched(self):
        answer = "Girls should be in the hostel within 6 PM"
        self.assertEqual(strip_polar_prefix(answer, "What is the hostel rule for girls ?"), answer)

    def test_is_polar_answer(self):
        self.assertTrue(is_polar_answer("Yes, hot water is available in the common restrooms."))
        self.assertTrue(is_polar_answer("No. It is part of the tuition fee only."))
        self.assertFalse(is_polar_answer("Girls should be in the hostel within 6 PM"))
        self.assertFalse(is_polar_answer(""))

if __name__ == "__main__":
    unittest.main()
