import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from mmresearch.research_data import assert_disjoint, instructions_for, near_duplicate_audit


class ResearchDataTests(unittest.TestCase):
    def test_reject_exact_overlap_and_duplicates(self):
        with self.assertRaises(ValueError):
            assert_disjoint({"train": [{"id": "a"}], "test": [{"id": "a"}]})
        with self.assertRaises(ValueError):
            assert_disjoint({"train": [{"id": "a"}, {"id": "a"}]})
        assert_disjoint({"train": [{"id": "a"}], "test": [{"id": "b"}]})

    def test_cross_split_dhash_is_a_review_flag(self):
        rows = {"train": [{"id": "a"}, {"id": "b"}], "test": [{"id": "c"}]}
        audit = near_duplicate_audit(rows, {"a": 0, "b": 255, "c": 3}, threshold=2)
        self.assertEqual(audit["flagged_pairs"], 1)
        self.assertEqual(audit["pairs"][0]["left_id"], "a")
        self.assertEqual(audit["pairs"][0]["hamming_distance"], 2)
        self.assertIn("No human verification", audit["interpretation"])

    def test_dhash_audit_reports_truncation(self):
        rows = {"train": [{"id": "a"}], "test": [{"id": "b"}, {"id": "c"}]}
        audit = near_duplicate_audit(rows, {"a": 0, "b": 0, "c": 0}, max_saved_pairs=1)
        self.assertEqual(audit["flagged_pairs"], 2)
        self.assertEqual(len(audit["pairs"]), 1)
        self.assertTrue(audit["pairs_truncated"])

    def test_instructions_preserve_human_sources_and_image_group(self):
        row = {"id": "a", "image": "data/images/a.png", "split": "train", "captions": ["one", "two", "three"]}
        values = instructions_for([row], 2)
        self.assertEqual([r["answer"] for r in values], ["one", "two"])
        self.assertEqual({r["image_id"] for r in values}, {"a"})
        self.assertTrue(all(r["answer_source"] == "public_human_caption" for r in values))


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Requires Pillow")
class PreparedResearchDataTests(unittest.TestCase):
    def test_prepare_excludes_exploration_and_is_immutable(self):
        from PIL import Image
        from mmresearch import research_data as module
        from mmresearch.common import read_rows, save_rows
        scratch = Path(__file__).resolve().parents[1] / ".cache/test_research_data"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            self.assertIn(scratch.resolve(), root.resolve().parents)
            (root / "data/images").mkdir(parents=True)
            original = Image.new("RGB", (16, 16), "red")
            identifier = module.image_id(original)
            original.save(root / ("data/images/%s.png" % identifier))
            save_rows(root / "data/test.jsonl", [{"id": identifier, "image": "data/images/%s.png" % identifier,
                      "captions": ["An explored image."], "split": "test"}])
            def item(color):
                return {"image": Image.new("RGB", (16, 16), color), "caption_0": "A %s picture." % color,
                        "caption_1": "The color is %s." % color}
            sources = {"train": [item("red"), item("blue"), item("green")],
                       "validation": [item("yellow")],
                       "test": [item("red"), item("white"), item("black")]}
            fake_api = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(
                dataset_info=lambda *args, **kwargs: types.SimpleNamespace(sha=module.REVISION)))
            cfg = {"train_images": 2, "validation_images": 1, "test_images": 2}
            with patch.object(module, "ROOT", root), patch.object(module, "_stream_split", side_effect=lambda cfg, split, info: sources[split]), patch.dict("sys.modules", {"huggingface_hub": fake_api}):
                source = module.prepare(cfg)
                confirmation = read_rows(root / "data/research_v2/test.jsonl")
                self.assertNotIn(identifier, {r["id"] for r in confirmation})
                self.assertEqual(source["counts"], {"train": 2, "validation": 1, "test": 2, "exploration": 1, "instructions": 4})
                self.assertEqual(source["exact_duplicate_rejections"], {"train": 1, "validation": 0, "test": 1})
                self.assertEqual(source, module.prepare(cfg))
                self.assertEqual(read_rows(root / "data/test.jsonl")[0]["split"], "test")
                with self.assertRaises(ValueError):
                    module.prepare({**cfg, "seed": 7})
                (root / "data/research_v2/test.jsonl").write_text("", encoding="utf-8")
                with self.assertRaises(ValueError):
                    module.prepare(cfg)

    def test_original_data_directory_cannot_be_destination(self):
        from mmresearch.research_data import _directory
        with self.assertRaises(ValueError):
            _directory({"data_dir": "data"})
        with self.assertRaises(ValueError):
            _directory({"data_dir": "../elsewhere"})


if __name__ == "__main__":
    unittest.main()
