"""Pairing and abstention-statistic guards for the research report."""
import tempfile
import unittest
from pathlib import Path

from mmresearch.common import save_json, save_rows
from mmresearch.research_report import _effects, _read_runs, _seed_groups, _similarity_utility_data


def run_record(mode, ids, value=.5):
    return {"model": "plain_s42_random_n128_language_epochs", "regime": "plain", "seed": 42,
            "sampling": "random", "count": 128, "scope": "language", "budget": "epochs",
            "split": "test", "mode": mode, "bank": "full", "shuffled": False,
            "n": len(ids), "token_f1": value,
            "rows": [{"id": identifier, "token_f1": value} for identifier in ids]}


class DiagnosticPairingTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["image_%d" % i for i in range(256)]
        self.cfg = {"confirmation_images": 256, "diagnostic_images": 64}

    def test_exact_diagnostic_prefix_is_explicitly_labeled(self):
        records = [run_record("none", self.ids, .4), run_record("image", self.ids, .5),
                   run_record("text", self.ids[:64], .7)]
        effects = _effects(records, self.cfg)
        diagnostic = next(e for e in effects if e["effect"] == "plain:text-image")
        primary = next(e for e in effects if e["effect"] == "plain:image-none")
        self.assertEqual(diagnostic["n_images"], 64)
        self.assertAlmostEqual(diagnostic["mean_delta"], .2)
        self.assertEqual(diagnostic["cohort"]["analysis_subset"], "diagnostic_subset")
        self.assertEqual(primary["n_images"], 256)
        self.assertEqual(primary["cohort"]["analysis_subset"], "confirmation")

    def test_diagnostic_arbitrary_overlap_or_reordered_prefix_rejected(self):
        for ids in (self.ids[1:65], list(reversed(self.ids[:64])), self.ids[:63]):
            with self.assertRaises(ValueError):
                _effects([run_record("image", self.ids), run_record("conflict", ids)], self.cfg)

    def test_primary_pairing_never_silently_intersects(self):
        with self.assertRaises(ValueError):
            _effects([run_record("none", self.ids), run_record("image", self.ids[:64])], self.cfg)
        with self.assertRaises(ValueError):
            _effects([run_record("image", self.ids), run_record("text", self.ids[:64])])


class ConditionalCopyTests(unittest.TestCase):
    def test_conditional_copy_separates_use_rate_from_copy_when_used(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            out = Path(folder)
            directory = out / "evaluations/zero_shot/test_adaptive_full"
            directory.mkdir(parents=True)
            rows = [{"id": "a", "context": [{"caption": "a dog"}], "context_copy_rate": .8},
                    {"id": "b", "context": [], "context_copy_rate": 0.}]
            save_rows(directory / "predictions.jsonl", rows)
            save_json(directory / "identity.json", {"ids": ["a", "b"]})
            save_json(directory / "metrics.json", {"model": "zero_shot", "n": 2,
                "split": "test", "mode": "adaptive", "bank": "full", "shuffled": False,
                "context_copy_rate": .4, "input_tokens": 120., "context_tokens": 4.})
            run = _read_runs(out)[0]
            self.assertEqual(run["context_copy_rate"], .4)
            self.assertEqual(run["context_copy_rate_when_used"], .8)
            self.assertEqual(run["retrieval_used_images"], 1)
            self.assertEqual(run["input_tokens"], 120.)

    def test_no_retrieval_is_undefined_not_zero_conditional_copy(self):
        run = run_record("none", ["a"])
        run["context_copy_rate_when_used"] = None
        summary = _seed_groups([run])[0]
        self.assertIsNone(summary["context_copy_rate_when_used_mean"])
        self.assertIsNone(summary["context_copy_rate_when_used_seed_sd"])
        self.assertEqual(summary["context_copy_rate_when_used_n_defined_runs"], 0)


class SimilarityUtilityTests(unittest.TestCase):
    def test_fixed_seed_average_known_correlation_and_quartiles(self):
        ids = ["image_%d" % i for i in range(8)]
        cfg = {"regimes": ["plain", "retrieval", "noise"], "seeds": [42, 43],
               "core_train_images": 128, "confirmation_images": 8}
        runs = []
        for seed in cfg["seeds"]:
            baseline, retrieved = run_record("none", ids, .1), run_record("image", ids, .1)
            baseline["seed"] = retrieved["seed"] = seed
            for i, row in enumerate(retrieved["rows"]):
                row["context"] = [{"score": .1 * i}]
                row["token_f1"] += .02 * i + (.01 if seed == 43 else 0)
            runs.extend([baseline, retrieved])
        queries, quartiles, summaries = _similarity_utility_data(runs, cfg)
        self.assertEqual(len(queries), 8)
        self.assertEqual(len(quartiles), 4)
        self.assertEqual(sum(q["n_images"] for q in quartiles), 8)
        self.assertEqual(summaries[0]["n_fixed_training_runs"], 2)
        self.assertAlmostEqual(summaries[0]["pearson_r"], 1.)
        self.assertAlmostEqual(queries[0]["delta_token_f1"], .005)
        retrieved["rows"][0]["id"] = "wrong"
        with self.assertRaises(ValueError):
            _similarity_utility_data(runs, cfg)

    def test_no_claim_for_partial_or_constant_pairs(self):
        cfg = {"regimes": ["plain"], "seeds": [42], "core_train_images": 128, "confirmation_images": 4}
        ids = ["a", "b", "c", "d"]
        baseline, retrieved = run_record("none", ids), run_record("image", ids)
        for row in retrieved["rows"]:
            row["context"] = [{"score": .5}]
        _, _, summaries = _similarity_utility_data([baseline, retrieved], cfg)
        self.assertIsNone(summaries[0]["pearson_r"])
        self.assertEqual(_similarity_utility_data([baseline], cfg), ([], [], []))
        retrieved["rows"] = retrieved["rows"][:2]
        with self.assertRaises(ValueError):
            _similarity_utility_data([baseline, retrieved], cfg)


class FourArmEffectsTests(unittest.TestCase):
    def records(self, both_value):
        records = []
        for regime, mode, value in (("plain", "image", .4), ("plain", "adaptive", .45),
                                    ("noise", "image", .5), ("noise", "adaptive", both_value)):
            record = run_record(mode, ["a", "b"], value)
            record["regime"] = regime
            record["model"] = regime + "_s42_random_n128_language_epochs"
            for row in record["rows"]:
                row.update({metric: value for metric in ("cider_d_fixed_df", "clipscore", "words", "context_copy_rate")})
            records.append(record)
        return records

    def test_additive_and_synergistic_four_arm_contrasts(self):
        label = "interaction:(noise_adaptive-noise_image)-(plain_adaptive-plain_image)"
        for both_value, expected_interaction in ((.55, 0.), (.62, .07)):
            effects = _effects(self.records(both_value))
            noise_only = next(e for e in effects if e["effect"] == "noise_only:noise_image-plain_image" and e["metric"] == "token_f1")
            interaction = next(e for e in effects if e["effect"] == label and e["metric"] == "token_f1")
            self.assertAlmostEqual(noise_only["mean_delta"], .1)
            self.assertAlmostEqual(interaction["mean_delta"], expected_interaction)
            self.assertAlmostEqual(interaction["image_ci_low"], expected_interaction)
            self.assertEqual(interaction["n_images"], 2)
            self.assertEqual({e["metric"] for e in effects if e["effect"] == label},
                             {"token_f1", "cider_d_fixed_df", "clipscore", "words"})

    def test_incomplete_arm_is_pending_and_mismatched_ids_rejected(self):
        records = self.records(.62)
        effects = _effects(records[:3])
        self.assertTrue(any(e["effect"] == "noise_only:noise_image-plain_image" for e in effects))
        self.assertFalse(any(e["effect"].startswith("interaction:(noise_adaptive") for e in effects))
        records[2]["rows"][0]["id"] = "unmatched"
        with self.assertRaises(ValueError):
            _effects(records)


if __name__ == "__main__":
    unittest.main()
