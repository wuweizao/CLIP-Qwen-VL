"""CPU-only verification of published generated text against the pinned data."""
import argparse
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import ROOT, read_rows, read_json


def score(prediction, reference):
    p = re.findall(r"[a-z0-9]+", prediction.lower())
    r = re.findall(r"[a-z0-9]+", reference.lower())
    if not p or not r:
        return float(p == r)
    counts_p, counts_r = Counter(p), Counter(r)
    overlap = sum(min(n, counts_r.get(word, 0)) for word, n in counts_p.items())
    return 2.0 * overlap / (len(p) + len(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/research_v2/predictions_core.jsonl")
    parser.add_argument("--data", default="data/research_v2/test.jsonl")
    parser.add_argument("--config", default="configs/research.json")
    args = parser.parse_args()
    predictions = read_rows(ROOT / args.predictions)
    references = {r["id"]: r["captions"] for r in read_rows(ROOT / args.data)}
    cfg = read_json(ROOT / args.config)
    from mmresearch.research_common import run_name
    models = {"zero_shot"} | {run_name(regime, seed, count=cfg["core_train_images"])
                                for regime in cfg["regimes"] for seed in cfg["seeds"]}
    expected_conditions = {(model, mode) for model in models for mode in cfg["evaluation_modes"]}
    expected_images = list(references)[:cfg["confirmation_images"]]
    seen, values = set(), defaultdict(list)
    for row in predictions:
        identity = (row["model"], row["mode"], row["image_id"])
        if identity in seen:
            raise ValueError("Duplicate published prediction")
        seen.add(identity)
        calculated = max(score(row["prediction"], ref) for ref in references[row["image_id"]])
        if abs(calculated - row["token_f1"]) > 1e-12:
            raise ValueError("Token-F1 mismatch: " + repr(identity))
        values[(row["model"], row["mode"])].append(calculated)
    if set(values) != expected_conditions or any((model, mode, image_id) not in seen
            for model, mode in expected_conditions for image_id in expected_images):
        raise ValueError("Incomplete published core matrix")
    if len(seen) != len(expected_conditions) * len(expected_images):
        raise ValueError("Unexpected prediction outside core matrix")
    print("Verified %d published predictions in %d conditions using independent Counter-based token-F1; no torch/model load." % (len(predictions), len(values)))
