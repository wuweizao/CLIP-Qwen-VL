"""Recompute primary metrics and audit paired intervention artifacts."""
import math
import importlib.metadata
import platform
import numpy as np
from .common import ROOT, read_json, read_rows, save_json, digest
from .research_common import paths, load_split, fingerprint, run_name
from .research_metrics import per_sample_metrics


def run(cfg):
    data, out, report = paths(cfg)
    expected_ids = {r["id"] for r in load_split(cfg, "test")[:cfg["confirmation_images"]]}
    train_ids = {r["id"] for r in load_split(cfg, "train")}
    if expected_ids & train_ids:
        raise ValueError("Train/test overlap")
    evaluated, predictions = [], 0
    for path in sorted((out / "evaluations").glob("*/*/metrics.json")):
        m = read_json(path)
        identity = read_json(path.parent / "identity.json")
        rows = read_rows(path.parent / "predictions.jsonl")
        if len(rows) != m["n"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Prediction completeness mismatch")
        if identity["data_manifests"] != fingerprint(cfg)["data_manifests"]:
            raise ValueError("Stale dataset provenance")
        if m["split"] == "test" and not {r["id"] for r in rows} <= expected_ids:
            raise ValueError("Unexpected confirmation IDs")
        for row in rows:
            if not row["prediction"].strip():
                raise ValueError("Empty prediction")
            if any(c["id"] not in train_ids or c["id"] == row["id"] for c in row["context"]):
                raise ValueError("Context source leakage")
            recalculated = per_sample_metrics(row["prediction"], row["references"], row["context"])
            for key, value in recalculated.items():
                if abs(row[key] - value) > 1e-10:
                    raise ValueError("Metric mismatch: " + key)
        for key in ("token_f1", "rouge_l", "words", "context_copy_rate", "reference_detail_recall", "clipscore"):
            if not math.isfinite(m[key]) or abs(m[key] - np.mean([r[key] for r in rows])) > 1e-10:
                raise ValueError("Aggregate mismatch: " + key)
        evaluated.append(str(path.parent.relative_to(out)))
        predictions += len(rows)
    missing = []
    for seed in cfg["seeds"]:
        for regime in cfg["regimes"]:
            model = run_name(regime, seed, count=cfg["core_train_images"])
            adapter = out / "models" / model / "adapter/adapter_model.safetensors"
            if not adapter.exists():
                missing.append(str(adapter.relative_to(out)))
            gate = read_json(out / "gates" / (model + ".json"))
            if gate["selection_split"] != "validation" or set(gate["validation_ids"]) & expected_ids:
                raise ValueError("Gate selected on confirmation data")
            for mode in cfg["evaluation_modes"]:
                path = out / "evaluations" / model / ("test_%s_full" % mode) / "metrics.json"
                if not path.exists():
                    missing.append(str(path.relative_to(out)))
                elif read_json(path)["n"] != len(expected_ids):
                    raise ValueError("Incomplete core condition")
    if missing:
        raise ValueError("Missing planned core artifacts: " + repr(missing))
    training_runs = []
    for path in sorted((out / "models").glob("*/training.json")):
        training = read_json(path)
        if not (path.parent / "adapter/adapter_model.safetensors").exists():
            raise ValueError("Missing saved adapter")
        history = read_json(path.parent / "history.json")
        losses = [r["eval_loss"] for r in history if "eval_loss" in r]
        if not losses or not all(math.isfinite(v) for v in losses):
            raise ValueError("Missing/invalid validation objective")
        gate_path = out / "gates" / (path.parent.name + ".json")
        if gate_path.exists() and set(training["checkpoint_validation_ids"]) & set(read_json(gate_path)["validation_ids"]):
            raise ValueError("Checkpoint selection overlaps retrieval gate validation")
        training_runs.append({"model": path.parent.name, **training, "validation_losses": losses,
                              "adapter_sha256": digest(path.parent / "adapter/adapter_model.safetensors")})
    save_json(report / "training_runs.json", training_runs)
    source = read_json(data / "source.json")
    for filename, expected_hash in source["manifest_sha256"].items():
        if digest(data / filename) != expected_hash:
            raise ValueError("Dataset manifest modified: " + filename)
    from PIL import Image
    from .research_data import image_id
    verified_images = 0
    for split in ("train", "validation", "test"):
        for row in load_split(cfg, split):
            with Image.open(ROOT / row["image"]) as image:
                if image_id(image) != row["id"]:
                    raise ValueError("Image pixels do not match manifest ID")
            verified_images += 1
    save_json(report / "data_source.json", source)
    save_json(report / "near_duplicate_audit.json", read_json(data / "near_duplicate_audit.json"))
    save_json(report / "benchmark_source.json", read_json(data / "benchmark_source.json"))
    save_json(report / "sampling_selections.json", read_json(out / "selections.json"))
    save_json(report / "gates.json", {path.stem: read_json(path) for path in sorted((out / "gates").glob("*.json"))})
    save_json(report / "reproducibility.json", {"config": cfg, "fingerprint": fingerprint(cfg),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "datasets", "numpy", "Pillow")},
        "model_sources": {k: read_json(ROOT / cfg[k] / "download_provenance.json") for k in ("clip_model", "vl_model")}})
    result = {"status": "passed", "completed_evaluations": len(evaluated),
              "completed_predictions": predictions, "core_conditions": len(cfg["seeds"]) * len(cfg["regimes"]) * len(cfg["evaluation_modes"]),
              "saved_adapters": len(training_runs), "image_pixel_hashes_verified": verified_images,
              "recomputed_lexical_metrics": True, "context_leakage_checks": True,
              "validation_only_gate_checks": True, "human_review": "pending unless separately imported",
              "evaluations": evaluated}
    save_json(report / "verification.json", result)
    from .research_findings import run as summarize_research_questions
    summarize_research_questions(cfg)
    print(result, flush=True)
