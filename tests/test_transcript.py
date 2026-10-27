import unittest

from speakingt.transcript import inspect_transcript


class TranscriptTests(unittest.TestCase):
    def test_observed_fillers_repetitions_and_korean(self):
        report = inspect_transcript("Um I I think 음 it's feasible.")
        kinds = [event["kind"] for event in report["events"]]
        self.assertEqual(kinds.count("filler_candidate"), 2)
        self.assertEqual(kinds.count("adjacent_repetition_candidate"), 1)
        self.assertEqual(kinds.count("korean_text_observed"), 1)
        self.assertEqual(report["english_token_count"], 6)

    def test_no_substring_matching_or_unconditional_like_penalty(self):
        self.assertEqual(inspect_transcript("summer rhythm like that") ["events"], [])

    def test_positions_are_characters_not_time(self):
        report = inspect_transcript("I, uh, agree")
        self.assertEqual(report["events"][0]["char_start"], 3)
        self.assertNotIn("start_s", report["events"][0])


if __name__ == "__main__":
    unittest.main()
