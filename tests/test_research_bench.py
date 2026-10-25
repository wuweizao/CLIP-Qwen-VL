"""Pure protocol checks; no model weights, network, or GPU required."""
import unittest

from mmresearch.research_bench import answer_choice, balanced_options, _task_metrics


class ChoiceProtocolTests(unittest.TestCase):
    def test_strict_leading_choice(self):
        self.assertEqual(answer_choice("Answer: **B**."), "B")
        self.assertEqual(answer_choice("A\nThe first caption matches."), "A")
        for value in ("A/B", "A or B", "The answer is A", "perhaps B", "", None):
            self.assertEqual(answer_choice(value), "invalid")

    def test_per_type_balance_and_order_independence(self):
        rows = [{"id": "%s:%d" % (category, i), "positive": "A dog runs.",
                 "negative": "A cat runs.", "type": category}
                for category in ("replace_obj", "swap_att") for i in range(16)]
        first = balanced_options(rows, seed=123)
        second = balanced_options(list(reversed(rows)), seed=123)
        self.assertEqual({r["id"]: r["positive_option"] for r in first},
                         {r["id"]: r["positive_option"] for r in second})
        for category in ("replace_obj", "swap_att"):
            self.assertEqual(sum(r["positive_option"] == "A" for r in first if r["type"] == category), 8)
        with self.assertRaises(ValueError):
            balanced_options(rows + rows[:1])


class BenchmarkSummaryTests(unittest.TestCase):
    def test_pope_invalid_answers_are_retained(self):
        rows = [{"id": "1", "image_id": "image", "variant": "random", "answer": "Yes.", "label": "yes"},
                {"id": "2", "image_id": "image", "variant": "random", "answer": "Unsure", "label": "no"}]
        metrics = _task_metrics("pope", rows, {"model": "example"}, 0.1, None)
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["variants"]["random"]["invalid_rate"], .5)
        self.assertEqual(metrics["n_images"], 1)
        self.assertTrue(metrics["subset_only"])

    def test_sugar_counts_ties_and_invalid_separately(self):
        rows = [{"id": "1", "image_id": "image", "category": "replace_obj", "correct": True,
                 "clip_correct": False, "clip_tie": True, "parsed_answer": "A", "positive_option": "A"},
                {"id": "2", "image_id": "image2", "category": "replace_obj", "correct": False,
                 "clip_correct": True, "clip_tie": False, "parsed_answer": "invalid", "positive_option": "B"}]
        metrics = _task_metrics("sugarcrepe", rows, {"model": "example"}, 0.1, None)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["clip_accuracy"], .5)
        self.assertEqual(metrics["clip_ties"], 1)
        self.assertEqual(metrics["invalid_rate"], .5)
        self.assertEqual(metrics["types"]["replace_obj"]["positive_option_counts"], {"A": 1, "B": 1})


if __name__ == "__main__":
    unittest.main()
