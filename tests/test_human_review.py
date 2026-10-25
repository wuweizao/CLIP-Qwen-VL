import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmresearch.common import ROOT, read_json, save_json, save_rows
from mmresearch.human_review import run, ingest_human_scores
from mmresearch.research_common import run_name


@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Requires Pillow for synthetic review images")
class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        temporary_root = ROOT / ".cache/tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="human-review-", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.out = self.root / "outputs"
        self.reports = self.root / "reports"
        (self.data / "images").mkdir(parents=True)
        for identifier in ("q1", "q2", "b1", "b2"):
            Image.new("RGB", (16, 12), "blue").save(self.data / "images" / (identifier + ".png"))
        self.queries = [{"id": i, "image": "data/images/" + i + ".png", "captions": ["A blue image."]} for i in ("q1", "q2")]
        self.bank = [{"id": i, "image": "data/images/" + i + ".png", "captions": ["An image."]} for i in ("b1", "b2")]
        self.cfg = {"seeds": [42], "core_train_images": 2}
        self.names = ["zero_shot"] + [run_name(regime, 42, "random", 2, "language", "epochs") for regime in ("plain", "noise")]
        for index, name in enumerate(self.names):
            save_rows(self.out / "evaluations" / name / "test_none_full/predictions.jsonl",
                      [{"id": row["id"], "prediction": "A blue image number %d." % index} for row in self.queries])
        save_json(self.out / "contexts/test_full.json", {"image": {
            row["id"]: [{"id": "b1", "caption": "An image."}] for row in self.queries}})
        self.patches = [patch("mmresearch.human_review.ROOT", self.root),
                        patch("mmresearch.human_review.paths", return_value=(self.data, self.out, self.reports)),
                        patch("mmresearch.human_review.load_split", side_effect=lambda cfg, split: self.queries if split == "test" else self.bank)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def csv_rows(self, name="blind_scores.csv"):
        with (self.out / "human_review" / name).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    def rewrite(self, rows, fields, name="blind_scores.csv"):
        path = self.out / "human_review" / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_materials_are_blind_and_all_labels_start_blank(self):
        status = run(self.cfg)
        directory = self.out / "human_review"
        page = (directory / "review.html").read_text(encoding="utf-8")
        csv_text = (directory / "blind_scores.csv").read_text(encoding="utf-8-sig")
        for name in self.names:
            self.assertNotIn(name, page)
            self.assertNotIn(name, csv_text)
        fields, rows = self.csv_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r["blind_id"] for r in rows}), 6)
        self.assertTrue(all(not r["factual_correctness"] and not r["detail_coverage"] for r in rows))
        self.assertEqual(status["caption_items_scored"], 0)
        self.assertFalse(status["human_scoring_completed"])
        self.assertFalse(status["automatic_fact_labels_created"])
        self.assertIn("data:image/jpeg;base64,", page)
        mapping = read_json(directory / "mapping.json")
        self.assertEqual({v["model"] for v in mapping["captions"].values()}, set(self.names))
        self.assertFalse((directory / "annotations.json").exists())

    def test_blank_template_is_not_a_human_result_and_partial_import_stays_partial(self):
        run(self.cfg)
        path = self.out / "human_review/blind_scores.csv"
        with self.assertRaises(ValueError):
            ingest_human_scores(self.cfg, path)
        fields, rows = self.csv_rows()
        rows[0].update(factual_correctness="2", detail_coverage="1", reviewer="annotator")
        status = ingest_human_scores(self.cfg, self.rewrite(rows, fields))
        self.assertEqual(status["caption_items_scored"], 1)
        self.assertEqual(status["caption_items_unscored"], 5)
        self.assertFalse(status["human_scoring_completed"])
        annotation = read_json(self.out / "human_review/annotations.json")["captions"][rows[0]["blind_id"]]
        self.assertEqual(annotation["label_source"], "human_provided")
        self.assertFalse(annotation["annotator_identity_authenticated"])
        run(self.cfg)
        self.assertEqual(self.csv_rows()[1][0]["factual_correctness"], "2")

    def test_bad_labels_and_unknown_ids_are_rejected(self):
        run(self.cfg)
        fields, rows = self.csv_rows()
        rows[0].update(factual_correctness="5", detail_coverage="1")
        with self.assertRaises(ValueError):
            ingest_human_scores(self.cfg, self.rewrite(rows, fields))
        rows[0].update(blind_id="unknown", factual_correctness="2")
        with self.assertRaises(ValueError):
            ingest_human_scores(self.cfg, self.rewrite(rows, fields))

    def test_relevance_labels_are_explicit_and_missing_conditions_stay_pending(self):
        (self.out / "evaluations" / self.names[-1] / "test_none_full/predictions.jsonl").unlink()
        status = run(self.cfg)
        self.assertEqual(status["package_status"], "partial_materials")
        fields, rows = self.csv_rows("relevance_scores.csv")
        self.assertTrue(all(not row["relevance"] for row in rows))
        rows[0]["relevance"] = "partial"
        status = ingest_human_scores(self.cfg, self.rewrite(rows, fields, "relevance_scores.csv"))
        self.assertEqual(status["package_status"], "partial_materials")
        self.assertEqual(status["relevance_items_scored"], 1)
        self.assertFalse(status["human_scoring_completed"])


if __name__ == "__main__":
    unittest.main()
