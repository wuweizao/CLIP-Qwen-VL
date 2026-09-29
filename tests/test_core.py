import unittest
import numpy as np
from mmresearch.metrics import recall_at_k, token_f1, rouge_l, caption_scores, paired_bootstrap
from mmresearch.data import training_subset


class MetricsTests(unittest.TestCase):
    def test_multiple_positives(self):
        scores = [[0.1, 0.9, 0.2], [0.1, 0.2, 0.8]]
        self.assertEqual(recall_at_k(scores, [[0, 1], [0]], (1, 3)), {"R@1": .5, "R@3": 1.0})

    def test_transpose_ownership(self):
        scores = np.array([[1, .9, .1], [.1, .2, 1]])
        self.assertEqual(recall_at_k(scores.T, [[0], [0], [1]])["R@1"], 1.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            recall_at_k([[1]], [[]])
        with self.assertRaises(ValueError):
            caption_scores([], [])

    def test_text_metrics(self):
        self.assertEqual(token_f1("Dog, dog!", "dog"), 2 / 3)
        self.assertEqual(rouge_l("a b c", "c b a"), 1 / 3)
        self.assertEqual(caption_scores(["a dog"], [["a cat", "A dog!"]])["token_f1"], 1)

    def test_nested_image_subsets(self):
        rows = [{"id": str(i)} for i in range(100)]
        a = training_subset(rows, .25, 42)
        b = training_subset(rows, 1, 42)
        self.assertEqual(a, b[:25])
        self.assertEqual(a, training_subset(list(reversed(rows)), .25, 42))

    def test_paired_ci(self):
        self.assertEqual(paired_bootstrap([0, 0], [1, 1]), {"delta": 1., "ci_low": 1., "ci_high": 1.})


if __name__ == "__main__":
    unittest.main()
