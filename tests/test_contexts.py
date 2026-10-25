import unittest

import numpy as np

from mmresearch.contexts import (build_contexts, match_context_lengths,
                                 training_context, select_context,
                                 diversity_subset, coverage_subset)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.bank = [
            {"id": "a", "captions": ["A red dog runs beside a boy.", "Two dogs in water."]},
            {"id": "b", "captions": ["A blue cat sits.", "A cat near a person."]},
            {"id": "c", "captions": ["Three horses jump over a fence in a green field.", "A horse runs."]},
            {"id": "d", "captions": ["A girl walks in snow.", "A person wearing white."]},
        ]
        self.images = np.array([[1., 0., 0.], [.9, .1, 0.], [0., 1., 0.], [0., 0., 1.]])
        self.texts = np.repeat(self.images, 2, axis=0)
        self.owners = [0, 0, 1, 1, 2, 2, 3, 3]

    def build(self, queries=None, features=None):
        queries = queries if queries is not None else [{"id": "a", "captions": ["DO NOT USE QUERY ANSWERS"]}]
        features = features if features is not None else self.images[:1]
        return build_contexts(queries, self.bank, features, self.images, self.texts, self.owners, k=2, max_words=20)

    def test_excludes_query_and_all_associated_captions(self):
        result = self.build()
        for mode, queries in result.items():
            self.assertTrue(all(row["id"] != "a" for row in queries["a"]), mode)
            self.assertTrue(all("DO NOT USE" not in row["caption"] for row in queries["a"]))
            if mode != "none":
                self.assertEqual(len(queries["a"]), 2)
                self.assertEqual(len({row["id"] for row in queries["a"]}), 2)

    def test_text_retrieval_uses_best_caption_and_owners_by_id(self):
        texts = self.texts.copy()
        texts[4] = [1., 0., 0.]
        texts[5] = [0., 1., 0.]
        result = build_contexts([{"id": "query"}], self.bank, [[1., 0., 0.]], self.images,
                                texts, [self.bank[j]["id"] for j in self.owners], k=4)
        self.assertEqual(len({r["id"] for r in result["text"]["query"]}), 4)
        row = next(r for r in result["text"]["query"] if r["id"] == "c")
        self.assertTrue(row["caption"].startswith("Three horses"))

    def test_determinism_independent_query_order(self):
        queries = [{"id": "x"}, {"id": "y"}]
        first = self.build(queries, self.images[:2])
        second = self.build(list(reversed(queries)), self.images[:2][::-1])
        for mode in first:
            self.assertEqual(first[mode], second[mode])

    def test_exact_length_matching_and_copy_safety(self):
        original = self.build()
        result = match_context_lengths(original)
        for slot in range(2):
            lengths = [len(result[mode]["a"][slot]["caption"].split()) for mode in result if mode != "none"]
            self.assertEqual(len(set(lengths)), 1)
            self.assertTrue(all(result[mode]["a"][slot]["effective_words"] == lengths[0] for mode in result if mode != "none"))
        self.assertNotIn("length_matched", original["image"]["a"][0])
        self.assertEqual(result["none"]["a"], [])

    def test_conflict_is_annotated_synthetic_not_verified(self):
        result = self.build()
        item = result["conflict"]["a"][0]
        self.assertTrue(item["synthetic"])
        self.assertFalse(item["verified_false"])
        self.assertTrue(item["perturbation_applied"])
        self.assertNotEqual(item["caption"], result["image"]["a"][0]["caption"])

    def test_training_mixture_gate_and_no_self_training(self):
        contexts = self.build()
        self.assertEqual(training_context("a", contexts, "plain"), [])
        self.assertEqual(training_context("a", contexts, "noise", 42), training_context("a", contexts, "noise", 42))
        values = training_context("a", contexts, "retrieval")
        self.assertTrue(values)
        self.assertEqual(select_context(values, min_score=1.01), [])
        self.assertEqual(len(select_context(values, min_score=-1.)), 2)
        with self.assertRaises(ValueError):
            training_context("a", {"image": {"a": [{"id": "a"}]}}, "retrieval")

    def test_farthest_point_is_distinct_and_order_independent(self):
        ids = [r["id"] for r in self.bank]
        chosen = diversity_subset(ids, self.images, 3, seed=42)
        self.assertEqual(len(set(chosen)), 3)
        self.assertEqual(chosen, diversity_subset(ids[::-1], self.images[::-1], 3, seed=42))
        self.assertEqual(diversity_subset(ids, self.images, 0), [])
        self.assertEqual(coverage_subset(self.bank, 3), coverage_subset(self.bank[::-1], 3))

    def test_rejects_ambiguous_or_invalid_inputs(self):
        with self.assertRaises(ValueError):
            build_contexts([{"id": "q"}], self.bank, [[0., 0., 0.]], self.images, self.texts, self.owners)
        with self.assertRaises(ValueError):
            diversity_subset(["same", "same"], [[1, 0], [0, 1]], 1)
        with self.assertRaises(ValueError):
            self.build([{"id": "q"}, {"id": "q"}], self.images[:2])


if __name__ == "__main__":
    unittest.main()
