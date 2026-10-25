import copy
import unittest

from mmresearch.research_aux_controls import match_banks


class BankSizeControlTests(unittest.TestCase):
    def test_matches_each_slot_and_preserves_source_identity(self):
        full = {"q": [{"id": "a", "caption": "A dog runs in the grass.", "score": .9, "source": "image"},
                      {"id": "b", "caption": "Two young children play.", "score": .8, "source": "image"}]}
        subset = {"q": [{"id": "c", "caption": "A cat sits.", "score": .7, "source": "image"},
                        {"id": "a", "caption": "Three small children play by a tree.", "score": .6, "source": "image"}]}
        original_full, original_subset = copy.deepcopy(full), copy.deepcopy(subset)
        a, b = match_banks(full, subset, ["q"])
        self.assertEqual(full, original_full)
        self.assertEqual(subset, original_subset)
        self.assertEqual([len(r["caption"].split()) for r in a["q"]], [3, 4])
        self.assertEqual([len(r["caption"].split()) for r in b["q"]], [3, 4])
        self.assertEqual([r["id"] for r in a["q"]], ["a", "b"])
        self.assertEqual([r["id"] for r in b["q"]], ["c", "a"])
        self.assertEqual(a["q"][0]["score"], .9)
        self.assertEqual(a["q"][0]["source"], "image")
        self.assertEqual(a["q"][0]["pre_cross_bank_caption"], full["q"][0]["caption"])
        self.assertTrue(a["q"][0]["cross_bank_length_matched"])
        self.assertEqual(a["q"][0]["effective_words"], 3)
        self.assertTrue(a["q"][0]["cross_bank_trimmed"])
        self.assertFalse(b["q"][0]["cross_bank_trimmed"])

    def test_requires_matched_nonempty_counts_and_valid_owners(self):
        good = {"q": [{"id": "a", "caption": "A caption."}]}
        for bad in ({}, {"q": []}, {"q": [{"id": "q", "caption": "A caption."}]},
                    {"q": [{"id": "a", "caption": ""}]},
                    {"q": [{"id": "a", "caption": "A."}, {"id": "b", "caption": "B."}]}):
            with self.assertRaises(ValueError):
                match_banks(good, bad, ["q"])
        with self.assertRaises(ValueError):
            match_banks(good, good, ["q", "q"])

    def test_extra_queries_are_not_added_or_used(self):
        full = {"q": [{"id": "a", "caption": "A dog."}], "other": [{"id": "b", "caption": "Other."}]}
        subset = {"q": [{"id": "c", "caption": "A cat."}]}
        a, b = match_banks(full, subset, ["q"])
        self.assertEqual(set(a), {"q"})
        self.assertEqual(set(b), {"q"})


if __name__ == "__main__":
    unittest.main()
