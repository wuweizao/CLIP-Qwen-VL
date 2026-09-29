"""Verify completed experimental artifacts and recompute all caption metrics."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmresearch.common import ROOT, config, read_json, read_rows, save_json, digest
from mmresearch.metrics import caption_scores


def main():
    cfg = config()
    expected = {r["id"] for r in read_rows(ROOT / "data/test.jsonl")}
    manifests = {s: digest(ROOT / "data" / (s + ".jsonl")) for s in ("train", "validation", "test")}
    verified = []
    for model in ["zero_shot"] + ["sft_%g" % f for f in cfg["fractions"]]:
        for prompt in cfg["prompts"]:
            for rag in (False, True):
                name = "%s_%s_%s_test" % (model, prompt, "rag" if rag else "plain")
                directory = ROOT / "outputs/evaluations" / name
                rows = read_rows(directory / "predictions.jsonl")
                if {r["id"] for r in rows} != expected or len(rows) != len(expected):
                    raise ValueError("Missing or duplicate predictions: " + name)
                if any(not r["prediction"].strip() for r in rows):
                    raise ValueError("Empty prediction: " + name)
                metrics = read_json(directory / "metrics.json")
                recalculated = caption_scores([r["prediction"] for r in rows], [r["references"] for r in rows])
                if any(abs(metrics[k] - v) > 1e-12 for k, v in recalculated.items()):
                    raise ValueError("Metric mismatch: " + name)
                if read_json(directory / "provenance.json")["manifests"] != manifests:
                    raise ValueError("Mismatched data provenance: " + name)
                verified.append(name)
    for fraction in cfg["fractions"]:
        directory = ROOT / "outputs" / ("sft_%g" % fraction)
        if not (directory / "adapter/adapter_model.safetensors").exists():
            raise ValueError("Missing adapter weights")
        history = read_json(directory / "history.json")
        losses = [r["eval_loss"] for r in history if "eval_loss" in r]
        if len(losses) != cfg["epochs"] or not all(math.isfinite(x) for x in losses):
            raise ValueError("Missing or invalid validation losses")
    save_json(ROOT / "reports/verification.json", {"status": "passed", "completed_evaluations": len(verified),
              "predictions_per_evaluation": len(expected), "recomputed_metrics": True,
              "adapter_count": len(cfg["fractions"]), "evaluations": verified})
    print("Verified %d evaluations, %d images each, %d adapters" % (len(verified), len(expected), len(cfg["fractions"])))


if __name__ == "__main__":
    main()
