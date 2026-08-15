"""Unit tests for the arithmetic GRPO reward contract."""

import unittest

from reward import exact_answer_reward, extract_tagged_answer


class ExtractTaggedAnswerTests(unittest.TestCase):
    def test_extracts_a_plain_tagged_integer(self) -> None:
        self.assertEqual(extract_tagged_answer("<answer>1776</answer>"), "1776")

    def test_allows_whitespace_and_surrounding_markdown(self) -> None:
        self.assertEqual(extract_tagged_answer("`<answer>  1776 </answer>`"), "1776")

    def test_rejects_missing_or_malformed_tags(self) -> None:
        self.assertIsNone(extract_tagged_answer("1776"))
        self.assertIsNone(extract_tagged_answer("<answer>one seven seven six</answer>"))
        self.assertIsNone(extract_tagged_answer("<answer>1776"))

    def test_rejects_multiple_answer_tags(self) -> None:
        completion = "<answer>1700</answer> 或 <answer>1776</answer>"
        self.assertIsNone(extract_tagged_answer(completion))


class ExactAnswerRewardTests(unittest.TestCase):
    def test_correct_answer_receives_one(self) -> None:
        self.assertEqual(exact_answer_reward("<answer>1776</answer>", "1776"), 1.0)

    def test_integer_reference_answer_is_supported(self) -> None:
        self.assertEqual(exact_answer_reward("<answer>1776</answer>", 1776), 1.0)

    def test_wrong_answer_receives_zero(self) -> None:
        self.assertEqual(exact_answer_reward("<answer>1775</answer>", "1776"), 0.0)

    def test_invalid_format_receives_zero(self) -> None:
        self.assertEqual(exact_answer_reward("答案是 1776", "1776"), 0.0)

    def test_multiple_tags_receive_zero_even_if_one_is_correct(self) -> None:
        completion = "<answer>1776</answer><answer>1775</answer>"
        self.assertEqual(exact_answer_reward(completion, "1776"), 0.0)


if __name__ == "__main__":
    unittest.main()
