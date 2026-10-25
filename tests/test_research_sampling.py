import inspect
import unittest

import numpy as np

from mmresearch.research_sampling import TAG_PROMPTS, semantic_coverage


class SemanticSamplingTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["d", "b", "a", "c", "e"]
        self.images = np.array([[1., 0., 0., 0.], [.9, .1, 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.tags = np.eye(4)

    def test_interface_is_label_free_and_vocabulary_is_fixed(self):
        parameters = list(inspect.signature(semantic_coverage).parameters)
        self.assertEqual(parameters, ["ids", "image_features", "tag_features", "count", "seed"])
        self.assertGreaterEqual(len(TAG_PROMPTS), 60)
        self.assertLessEqual(len(TAG_PROMPTS), 100)
        self.assertEqual(len(TAG_PROMPTS), len(set(TAG_PROMPTS)))
        self.assertTrue(all(isinstance(prompt, str) and prompt.startswith("A photo of ") for prompt in TAG_PROMPTS))

    def test_determinism_and_input_order_invariance(self):
        first = semantic_coverage(self.ids, self.images, self.tags, 4, seed=43)
        second = semantic_coverage(self.ids[::-1], self.images[::-1], self.tags, 4, seed=43)
        self.assertEqual(first, second)
        self.assertEqual(first, semantic_coverage(self.ids, self.images, self.tags, 4, seed=43))
        self.assertEqual(len(set(first)), 4)

    def test_subset_is_nested_and_does_not_mutate_inputs(self):
        before = self.images.copy()
        selected = semantic_coverage(self.ids, self.images, self.tags, 5)
        self.assertEqual(set(selected), set(self.ids))
        self.assertEqual(semantic_coverage(self.ids, self.images, self.tags, 2), selected[:2])
        self.assertEqual(semantic_coverage(self.ids, self.images, self.tags, 0), [])
        np.testing.assert_array_equal(self.images, before)

    def test_after_identical_tag_coverage_farthest_point_breaks_ties(self):
        # One fixed tag gives every image identical semantic coverage. The
        # second selected point must then minimize cosine with the first.
        selected = semantic_coverage(self.ids, self.images, [[1., 1., 1., 1.]], 2, seed=12)
        normalized = self.images / np.linalg.norm(self.images, axis=1, keepdims=True)
        first, second = map(self.ids.index, selected)
        values = normalized @ normalized[first]
        self.assertAlmostEqual(values[second], min(values[j] for j in range(len(self.ids)) if j != first))

    def test_rescaling_embeddings_does_not_change_selection(self):
        first = semantic_coverage(self.ids, self.images, self.tags, 4)
        scaled = self.images * np.arange(1, 6)[:, None]
        self.assertEqual(first, semantic_coverage(self.ids, scaled, self.tags * 7, 4))

    def test_rejects_invalid_counts_alignment_and_nonfinite_features(self):
        for count in (-1, 6, 1.5):
            with self.assertRaises(ValueError):
                semantic_coverage(self.ids, self.images, self.tags, count)
        with self.assertRaises(ValueError):
            semantic_coverage(["same"] * 5, self.images, self.tags, 2)
        with self.assertRaises(ValueError):
            semantic_coverage(self.ids, self.images[:4], self.tags, 2)
        with self.assertRaises(ValueError):
            semantic_coverage(self.ids, self.images, [[0., 0., 0., 0.]], 2)
        with self.assertRaises(ValueError):
            semantic_coverage(self.ids, self.images, [[float("nan"), 0., 0., 0.]], 2)


if __name__ == "__main__":
    unittest.main()
