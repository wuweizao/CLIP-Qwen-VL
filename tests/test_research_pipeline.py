"""Research v2 protocol checks; synthetic generation and CPU-only processing."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mmresearch.common import ROOT, read_rows, save_rows
from mmresearch.research_evaluate import Generator, tune_gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": i, "split": "validation"} for i in "abcd"]
        self.plain = [{"id": i, "token_f1": value} for i, value in zip("abcd", [.8, .8, .4, .4])]
        self.rag = [{"id": i, "token_f1": value} for i, value in zip("abcd", [.2, .3, .8, .9])]
        self.contexts = {i: [{"score": value}] for i, value in zip("abcd", [.1, .2, .8, .9])}
        self.cfg = {"gate_score_quantiles": [0, .5, 1]}

    def test_gate_needs_only_validation_ids_and_precomputed_scores(self):
        result = tune_gate(self.rows, self.plain, self.rag, self.contexts, self.cfg)
        self.assertEqual(result["selection_split"], "validation")
        self.assertAlmostEqual(result["threshold"], .5)
        self.assertAlmostEqual(result["validation_token_f1"], .825)
        self.assertAlmostEqual(result["use_fraction"], .5)
        # There are deliberately no caption/reference labels in the API input.
        self.assertTrue(all(set(row) == {"id", "split"} for row in self.rows))

    def test_gate_rejects_test_rows_duplicates_and_mispairing(self):
        invalid = [{**row, "split": "test"} for row in self.rows]
        with self.assertRaises(ValueError):
            tune_gate(invalid, self.plain, self.rag, self.contexts, self.cfg)
        with self.assertRaises(ValueError):
            tune_gate(self.rows + self.rows[:1], self.plain, self.rag, self.contexts, self.cfg)
        with self.assertRaises(ValueError):
            tune_gate(self.rows, self.plain, self.rag[:-1], self.contexts, self.cfg)

    def test_gate_tie_prefers_abstaining(self):
        result = tune_gate(self.rows, self.plain, self.plain, self.contexts, self.cfg)
        self.assertEqual(result["use_fraction"], 0)


TORCH_READY = importlib.util.find_spec("torch") is not None
PROCESSOR_READY = TORCH_READY and importlib.util.find_spec("transformers") is not None and Path(
    "data/models/Qwen2.5-VL-3B-Instruct/tokenizer.json").exists() and Path("data/train.jsonl").exists()


@unittest.skipUnless(PROCESSOR_READY, "Requires local Qwen processor and image data")
class ContextCollatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        cls.processor = AutoProcessor.from_pretrained("data/models/Qwen2.5-VL-3B-Instruct", max_pixels=100352)

    def test_context_and_image_tokens_never_become_target_labels(self):
        from mmresearch.research_train import ContextCollator
        from mmresearch.vl import messages
        row = read_rows("data/train.jsonl")[0]
        row = {**row, "answer": row["captions"][0], "context": [
            {"id": "different-image", "caption": "UNRELATEDTEXT purple helicopter beside an elephant."}]}
        batch = ContextCollator(self.processor)([row])
        labels = batch["labels"][0]
        target = self.processor.decode(labels[labels != -100], skip_special_tokens=True).strip()
        self.assertEqual(target, row["answer"])
        full = self.processor.decode(batch["input_ids"][0], skip_special_tokens=False)
        self.assertIn("UNRELATEDTEXT", full)
        self.assertNotIn("UNRELATEDTEXT", target)
        self.assertEqual(batch["_image_id"], row["id"])
        self.assertGreater(int((labels == -100).sum()), 0)
        with self.assertRaises(ValueError):
            ContextCollator(self.processor)([row, row])


@unittest.skipUnless(TORCH_READY, "Requires CPU PyTorch")
class GenerationProtocolTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        temporary_root = ROOT / ".cache" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline-", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.out = self.root / "outputs"
        self.out.mkdir()
        self.rows = [{"id": key, "image": key + ".png", "captions": [caption], "split": "test"}
                     for key, caption in zip("abc", ("a dog runs", "a horse walks", "a blue boat"))]
        np.savez(self.out / "index.npz", test=np.eye(3))
        self.cfg = {"inference_batch_size": 2, "max_new_tokens": 8}

        class Batch(dict):
            def to(self, device):
                return self

        class Model:
            device = "cpu"
            calls = 0
            def eval(inner):
                return inner
            def generate(inner, input_ids, **kwargs):
                inner.calls += 1
                return torch.cat([input_ids, torch.ones((len(input_ids), 1), dtype=torch.long)], dim=1)

        def fake_inputs(processor, rows, contexts=None, **kwargs):
            return Batch(input_ids=torch.ones((len(rows), 2), dtype=torch.long),
                         attention_mask=torch.ones((len(rows), 2), dtype=torch.long))

        self.model = Model()
        self.processor = SimpleNamespace(batch_decode=lambda values, **kw: ["a dog runs" for _ in values],
                                         tokenizer=SimpleNamespace(encode=lambda text, **kw: text.split()))
        self.patches = [
            patch("mmresearch.research_evaluate.paths", return_value=(self.root / "data", self.out, self.root / "reports")),
            patch("mmresearch.research_evaluate.fingerprint", return_value={"study": "synthetic"}),
            patch("mmresearch.research_evaluate.load_split", side_effect=lambda cfg, split: self.rows),
            patch("mmresearch.research_evaluate.load", return_value=(self.model, self.processor)),
            patch("mmresearch.research_evaluate.inputs", side_effect=fake_inputs),
            patch("torch.cuda.is_available", return_value=False),
        ]
        for item in self.patches:
            item.start()
        self.generator = Generator(self.cfg, "zero_shot")
        self.generator.clip = SimpleNamespace(
            device="cpu", model=SimpleNamespace(to=lambda device: None),
            encode=lambda captions: np.tile([1., 0., 0.], (len(captions), 1)))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_shuffle_preserves_targets_and_reports_actual_input_image(self):
        values = self.generator.generate(self.rows, "test", "none", {"none": {}}, shuffled=True)
        self.assertEqual([r["id"] for r in values], list("abc"))
        self.assertEqual([r["input_image_id"] for r in values], list("bca"))
        self.assertEqual([r["image"] for r in values], ["b.png", "c.png", "a.png"])
        self.assertEqual([r["references"] for r in values], [r["captions"] for r in self.rows])
        self.assertEqual(values[0]["clipscore"], 2.5)
        self.assertEqual(values[0]["input_clipscore"], 0.)

    def test_partial_resume_generates_only_missing_rows(self):
        values = self.generator.generate(self.rows, "test", "none", {"none": {}})
        directory = self.out / "evaluations/zero_shot/test_none_full"
        save_rows(directory / "predictions.jsonl", values[:2])
        (directory / "metrics.json").unlink()
        self.model.calls = 0
        completed = self.generator.generate(self.rows, "test", "none", {"none": {}})
        self.assertEqual(self.model.calls, 1)
        self.assertEqual([r["id"] for r in completed], list("abc"))
        self.model.calls = 0
        self.generator.generate(self.rows, "test", "none", {"none": {}})
        self.assertEqual(self.model.calls, 0)

    def test_changed_context_cannot_reuse_completed_run(self):
        self.generator.generate(self.rows, "test", "none", {"none": {}})
        with self.assertRaises(ValueError):
            self.generator.generate(self.rows, "test", "none", {"none": {}, "image": {"a": []}})


if __name__ == "__main__":
    unittest.main()
