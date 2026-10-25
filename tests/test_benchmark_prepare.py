import json
import unittest

from scripts.prepare_benchmarks import CATEGORIES, VARIANTS, parse_pope, select_pope, select_sugar, transfer_rows


class BenchmarkPreparationTests(unittest.TestCase):
    def sugar_source(self):
        return {str(i): {"filename": "%012d.jpg" % i, "caption": "A positive caption %d." % i,
                        "negative_caption": "An incorrect caption %d." % i} for i in range(40)}

    def pope_sources(self):
        output = {}
        for variant in VARIANTS:
            rows = []
            for image in range(4):
                for q in range(6):
                    rows.append({"question_id": image * 6 + q + 1,
                                 "image": "COCO_val2014_%012d.jpg" % image,
                                 "text": "Is there an object?", "label": "yes" if q % 2 == 0 else "no"})
            output[variant] = rows
        return output

    def test_sugar_is_stratified_and_seeded(self):
        source = self.sugar_source()
        rows = [row for category in CATEGORIES for row in select_sugar(source, category, 16, 7)]
        self.assertEqual(len(rows), 112)
        self.assertEqual(select_sugar(source, "add_att", 16, 7), select_sugar(dict(reversed(list(source.items()))), "add_att", 16, 7))
        self.assertTrue(all(r["positive"] != r["negative"] for r in rows))
        self.assertEqual(len({r["id"] for r in rows}), 112)

    def test_sugar_rejects_paths_and_missing_labels(self):
        with self.assertRaises(ValueError):
            select_sugar({"a": {"filename": "../image.jpg", "caption": "a", "negative_caption": "b"}}, "add_att", 1)
        with self.assertRaises(ValueError):
            select_sugar({"a": {"filename": "000000000001.jpg", "caption": "a", "negative_caption": "a"}}, "add_att", 1)

    def test_pope_retains_same_images_and_balance(self):
        source = self.pope_sources()
        rows = select_pope(source, questions=12, seed=7)
        self.assertEqual(len(rows), 36)
        images = []
        for variant in VARIANTS:
            group = [r for r in rows if r["variant"] == variant]
            self.assertEqual(sum(r["label"] == "yes" for r in group), 6)
            self.assertEqual(sum(r["label"] == "no" for r in group), 6)
            images.append({r["image_id"] for r in group})
        self.assertEqual(images[0], images[1])
        self.assertEqual(images[1], images[2])
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual(rows, select_pope(source, 12, 7))

    def test_pope_cannot_silently_rebalance_or_fabricate(self):
        with self.assertRaises(ValueError):
            select_pope(self.pope_sources(), questions=11)
        source = self.pope_sources()
        source["popular"] = []
        with self.assertRaises(ValueError):
            select_pope(source, questions=12)
        with self.assertRaises(ValueError):
            parse_pope(json.dumps({"question_id": 1, "image": "COCO_val2014_000000000001.jpg", "text": "Q?", "label": "maybe"}).encode())

    def test_transfer_is_unique_and_discloses_selection(self):
        rows = select_sugar(self.sugar_source(), "add_att", 2)
        transfer = transfer_rows(rows + rows)
        self.assertEqual(len(transfer), 2)
        self.assertTrue(all(len(r["captions"]) == 1 for r in transfer))
        self.assertTrue(all("not representative COCO" in r["selection_scope"] for r in transfer))
        self.assertTrue(all("not independently verified" in r["reference_source"] for r in transfer))


if __name__ == "__main__":
    unittest.main()
