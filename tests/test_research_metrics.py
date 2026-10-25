import json
import math
import tempfile
import unittest
from pathlib import Path

from mmresearch.research_metrics import (
    FixedDFCider, answer_yes_no, binary_metrics, cider_d_scores, cider_scores,
    paired_effect, paired_seed_effect, per_sample_metrics, seed_summary,
)
from mmresearch.benchmarks import image_subset, load_pope, load_sugarcrepe


class CiderTests(unittest.TestCase):
    def setUp(self):
        self.df_refs = [["a small dog jumps high"], ["blue boats sail the ocean"]]

    def test_identical_and_disjoint(self):
        scores = cider_scores(["a small dog jumps high", "completely unrelated words here"],
                              [["a small dog jumps high"], ["a small dog jumps high"]], self.df_refs)
        self.assertAlmostEqual(scores[0], 10.0)
        self.assertEqual(scores[1], 0.0)

    def test_no_batch_fitted_df(self):
        single = cider_scores(["a small dog jumps"], [["a small dog jumps high"]], self.df_refs)[0]
        batch = cider_scores(["a small dog jumps", "invented test words do not fit idf"],
                             [["a small dog jumps high"], ["invented test words do not fit idf"]], self.df_refs)[0]
        self.assertAlmostEqual(single, batch)

    def test_document_is_image_group(self):
        scorer = FixedDFCider([["dog runs", "dog dog runs"], ["boat sails"]])
        self.assertEqual(scorer.document_frequency[("dog",)], 1)
        self.assertEqual(scorer.documents, 2)

    def test_references_are_averaged_not_best(self):
        a = cider_scores(["a small dog jumps high"], [["a small dog jumps high"]], self.df_refs)[0]
        b = cider_scores(["a small dog jumps high"], [["a small dog jumps high", "blue boats sail the ocean"]], self.df_refs)[0]
        self.assertAlmostEqual(b, a / 2)

    def test_empty_and_short_text(self):
        self.assertEqual(cider_scores([""], [["a small dog jumps high"]], self.df_refs), [0.0])
        self.assertTrue(math.isfinite(cider_scores(["dog"], [["dog"]], self.df_refs)[0]))

    def test_d_variant_penalizes_repetition_and_length(self):
        pred = "a small dog jumps high a small dog jumps high"
        reference = [["a small dog jumps high"]]
        self.assertLess(cider_d_scores([pred], reference, self.df_refs)[0],
                        cider_scores([pred], reference, self.df_refs)[0])

    def test_invalid_df_and_alignment(self):
        with self.assertRaises(ValueError):
            FixedDFCider([["only one image"]])
        with self.assertRaises(ValueError):
            cider_scores(["dog"], [], self.df_refs)


class DiagnosticsTests(unittest.TestCase):
    def test_context_overlap_is_content_only(self):
        score = per_sample_metrics("A red dog plays", ["A red dog plays with a ball"],
                                   [{"caption": "The red dog sleeps"}])
        self.assertAlmostEqual(score["context_copy_rate"], 2 / 3)
        self.assertAlmostEqual(score["reference_detail_recall"], 3 / 4)
        self.assertEqual(score["words"], 4)

    def test_no_context_is_zero_overlap(self):
        self.assertEqual(per_sample_metrics("a dog", ["a dog"])["context_copy_rate"], 0)

    def test_alignment_is_by_id(self):
        a = [{"id": "a", "token_f1": .1}, {"id": "b", "token_f1": .3}]
        b = [{"id": "b", "token_f1": .5}, {"id": "a", "token_f1": .3}]
        effect = paired_effect(a, b)
        self.assertAlmostEqual(effect["delta"], .2)
        self.assertAlmostEqual(effect["ci_low"], .2)
        self.assertEqual(effect["n_images"], 2)
        with self.assertRaises(ValueError):
            paired_effect(a, b + b[:1])

    def test_training_seed_uncertainty_separate(self):
        summary = seed_summary([{"seed": 1, "token_f1": .4}, {"seed": 2, "token_f1": .6}])
        self.assertAlmostEqual(summary["mean"], .5)
        self.assertAlmostEqual(summary["seed_sd"], math.sqrt(.02))
        self.assertEqual(summary["uncertainty_unit"], "training_seed")
        effect = paired_seed_effect({1: [{"id": "a", "token_f1": .1}]},
                                    {1: [{"id": "a", "token_f1": .4}]})
        self.assertAlmostEqual(effect["mean"], .3)
        self.assertIsNone(effect["seed_sd"])


class BinaryMetricsTests(unittest.TestCase):
    def test_strict_parser(self):
        for text in ("Yes", "YES.", "Answer: **Yes**.", "Yes, there is one."):
            self.assertEqual(answer_yes_no(text), "yes")
        for text in ("No.", "no, I cannot see it", "Answer: no"):
            self.assertEqual(answer_yes_no(text), "no")
        for text in ("There is no dog", "Maybe yes", "Yes/no", "No or yes", "", None, "Yes probably"):
            self.assertEqual(answer_yes_no(text), "invalid")

    def test_invalid_is_not_dropped(self):
        result = binary_metrics(["Yes.", "uncertain", "No.", "uncertain"], ["yes", "yes", "no", "no"], require_balanced=True)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["invalid_rate"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["invalid_negative"], 1)

    def test_label_quality(self):
        with self.assertRaises(ValueError):
            binary_metrics(["yes"], ["yes"])
        with self.assertRaises(ValueError):
            binary_metrics(["yes", "no"], ["yes", "unknown"])
        with self.assertRaises(ValueError):
            binary_metrics(["yes", "yes", "no"], ["yes", "yes", "no"], require_balanced=True)


class BenchmarkLoaderTests(unittest.TestCase):
    def test_pope_preserves_labels_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            (root / "image.jpg").touch()
            path = root / "pope.json"
            examples = [{"question_id": 1, "image": "image.jpg", "text": "Is there a dog?", "label": "yes"},
                        {"question_id": 2, "image": "image.jpg", "text": "Is there a cat?", "label": "no"}]
            path.write_text("\n".join(json.dumps(row) for row in examples), encoding="utf-8")
            rows, provenance = load_pope(path, root)
            self.assertEqual([r["label"] for r in rows], ["yes", "no"])
            self.assertTrue(provenance["balanced_labels"])
            self.assertEqual(len(provenance["annotation_sha256"]), 64)
            examples[1]["image"] = "../outside.jpg"
            path.write_text("\n".join(json.dumps(row) for row in examples), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pope(path, root, require_images=False)

    def test_sugarcrepe_pair_schema_and_missing_images(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / "replace_obj.json"
            path.write_text(json.dumps({"0": {"filename": "image.jpg", "caption": "A dog jumps.",
                                              "negative_caption": "A cat jumps."}}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_sugarcrepe(path, folder)
            rows, provenance = load_sugarcrepe(path, folder, require_images=False)
            self.assertEqual(rows[0]["id"], "replace_obj:0")
            self.assertEqual(provenance["missing_images"], 1)
            self.assertFalse(provenance["images_verified_present"])

    def test_subsampling_keeps_image_clusters(self):
        rows = [{"image_id": "a", "id": "1"}, {"image_id": "a", "id": "2"},
                {"image_id": "b", "id": "3"}, {"image_id": "b", "id": "4"}]
        subset = image_subset(rows, 1)
        self.assertEqual(len(subset), 2)
        self.assertEqual(len({r["image_id"] for r in subset}), 1)


if __name__ == "__main__":
    unittest.main()
