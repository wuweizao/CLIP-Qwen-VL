"""Local integration checks using downloaded processor and real data, no GPU needed."""
import importlib.util
import unittest
from pathlib import Path

READY = Path("data/models/Qwen2.5-VL-3B-Instruct/tokenizer.json").exists() and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(READY, "Requires downloaded processor and local data")
class MultimodalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        cls.processor = AutoProcessor.from_pretrained("data/models/Qwen2.5-VL-3B-Instruct", max_pixels=100352)

    def test_assistant_only_targets(self):
        from mmresearch.common import read_rows
        from mmresearch.vl import SFTCollator
        row = read_rows("data/instructions.jsonl")[0]
        batch = SFTCollator(self.processor)([row])
        labels = batch["labels"][0]
        self.assertGreater(int((labels == -100).sum()), 0)
        text = self.processor.decode(labels[labels != -100], skip_special_tokens=True).strip()
        self.assertEqual(text, row["answer"])

    def test_split_disjoint(self):
        from mmresearch.common import read_rows
        sets = [set(r["id"] for r in read_rows("data/%s.jsonl" % s)) for s in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_rag_only_training_images(self):
        from mmresearch.common import read_rows, read_json
        train = {r["id"] for r in read_rows("data/train.jsonl")}
        context = read_json("outputs/test_contexts.json")
        test = {r["id"] for r in read_rows("data/test.jsonl")}
        self.assertEqual(set(context), test)
        for values in context.values():
            self.assertTrue(all(c["id"] in train and c["id"] not in test for c in values))
