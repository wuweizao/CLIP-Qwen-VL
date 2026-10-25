"""Paired context interventions, validation-only gating, resumable generation."""
import gc
import hashlib
import json
import time
import numpy as np
from .common import ROOT, read_json, read_rows, save_json, save_rows, seed_all, digest
from .research_common import paths, load_split, fingerprint, run_name, context_path
from .vl import load, inputs
from .research_metrics import per_sample_metrics, cider_scores, cider_d_scores


def model_name(cfg, regime, seed, sampling="random", count=None, scope="language", budget="epochs"):
    return "zero_shot" if regime == "zero_shot" else run_name(regime, seed, sampling, count or cfg["core_train_images"], scope, budget)


def adapter_path(cfg, name):
    return None if name == "zero_shot" else str(paths(cfg)[1] / "models" / name / "adapter")


def tune_gate(rows, plain, retrieved, contexts, cfg):
    """Select an image-similarity threshold without reading confirmation references."""
    a = {r["id"]: r["token_f1"] for r in plain}
    b = {r["id"]: r["token_f1"] for r in retrieved}
    ids = [r["id"] for r in rows]
    if any(r.get("split") != "validation" for r in rows):
        raise ValueError("Gate selection is restricted to validation rows")
    if len(ids) != len(set(ids)) or len(plain) != len(ids) or len(retrieved) != len(ids):
        raise ValueError("Duplicate or missing validation records")
    if set(ids) != set(a) or set(ids) != set(b):
        raise ValueError("Validation pairing mismatch")
    scores = {i: float(contexts[i][0]["score"]) for i in ids}
    thresholds = sorted(set([-2., 2.] + [float(np.quantile(list(scores.values()), q)) for q in cfg["gate_score_quantiles"]]))
    candidates = [{"threshold": threshold,
                   "validation_token_f1": float(np.mean([b[i] if scores[i] >= threshold else a[i] for i in ids])),
                   "use_fraction": float(np.mean([scores[i] >= threshold for i in ids]))}
                  for threshold in thresholds]
    best = max(candidates, key=lambda c: (c["validation_token_f1"], -c["use_fraction"], c["threshold"]))
    return {**best, "selection_split": "validation", "selection_metric": "token_f1",
            "validation_ids": ids, "candidates": candidates,
            "interpretation": "Similarity threshold, not a calibrated correctness probability"}


class Generator:
    def __init__(self, cfg, name):
        self.cfg, self.name = cfg, name
        self.model, self.processor = load(cfg, adapter_path(cfg, name))
        self.model.eval()
        self.clip = None

    def generate(self, rows, split, mode, contexts=None, bank="full", shuffled=False, gate=None):
        import torch
        cfg = self.cfg
        _, out, _ = paths(cfg)
        tag = "%s_%s_%s%s" % (split, mode, bank, "_shuffled" if shuffled else "")
        directory = out / "evaluations" / self.name / tag
        identity = {**fingerprint(cfg), "model": self.name, "split": split, "mode": mode,
                    "bank": bank, "shuffled": shuffled, "ids": [r["id"] for r in rows],
                    "gate_threshold": gate["threshold"] if gate else None,
                    "context_sha256": hashlib.sha256(json.dumps(contexts or {}, sort_keys=True).encode()).hexdigest(),
                    "evaluation_code_sha256": digest(__file__),
                    "metric_code_sha256": digest(ROOT / "mmresearch/research_metrics.py"),
                    "adapter_sha256": digest(out / "models" / self.name / "adapter/adapter_model.safetensors") if self.name != "zero_shot" else None}
        if (directory / "metrics.json").exists():
            if read_json(directory / "identity.json") != identity:
                raise ValueError("Stale evaluation: " + str(directory))
            return read_rows(directory / "predictions.jsonl")
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "identity.json").exists() and read_json(directory / "identity.json") != identity:
            raise ValueError("Stale partial predictions")
        save_json(directory / "identity.json", identity)
        context_map = (contexts or {}).get(mode, {}) if mode != "adaptive" else (contexts or {}).get("image", {})
        if mode == "adaptive":
            if gate is None:
                raise ValueError("Adaptive retrieval requires a validation-selected gate")
            context_map = {i: values if values and values[0]["score"] >= gate["threshold"] else []
                           for i, values in context_map.items()}
        if mode == "none":
            context_map = {}
        actual_rows = [{**r, "_input_image_id": r["id"]} for r in rows]
        if shuffled:
            if len(rows) < 2:
                raise ValueError("Image shuffle requires at least two images")
            actual_rows = [{**r, "image": rows[(i + 1) % len(rows)]["image"],
                            "_input_image_id": rows[(i + 1) % len(rows)]["id"]} for i, r in enumerate(rows)]
        partial = directory / "predictions.jsonl"
        results = read_rows(partial) if partial.exists() else []
        if [r["id"] for r in results] != [r["id"] for r in rows[:len(results)]]:
            raise ValueError("Partial prediction order mismatch")
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for i in range(len(results), len(actual_rows), cfg["inference_batch_size"]):
                chunk = actual_rows[i:i + cfg["inference_batch_size"]]
                batch = inputs(self.processor, chunk, contexts=context_map).to(self.model.device)
                generated = self.model.generate(**batch, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
                strings = self.processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                input_lengths = batch["attention_mask"].sum(dim=1).tolist()
                for row, prediction, input_length in zip(chunk, strings, input_lengths):
                    context = context_map.get(row["id"], [])
                    if any(c["id"] == row["id"] for c in context):
                        raise ValueError("Evaluation self-caption leakage")
                    results.append({"id": row["id"], "input_image_id": row["_input_image_id"],
                        "image": row["image"], "prediction": prediction.strip(), "references": row["captions"],
                        "context": context, "context_words": sum(len(c["caption"].split()) for c in context),
                        "input_tokens": input_length,
                        "context_tokens": len(self.processor.tokenizer.encode("\n".join(c["caption"] for c in context), add_special_tokens=False)),
                        **per_sample_metrics(prediction.strip(), row["captions"], context)})
                save_rows(partial, results)
                if len(results) % 32 == 0 or len(results) == len(rows):
                    print("%s/%s: %d/%d" % (self.name, tag, len(results), len(rows)), flush=True)
        df_refs = [r["captions"] for r in load_split(cfg, "train")]
        cider = cider_scores([r["prediction"] for r in results], [r["references"] for r in results], df_refs)
        cider_d = cider_d_scores([r["prediction"] for r in results], [r["references"] for r in results], df_refs)
        for row, c, d in zip(results, cider, cider_d):
            row["cider_fixed_df"] = float(c)
            row["cider_d_fixed_df"] = float(d)
        if self.clip is None:
            from .retrieval import Retriever
            self.clip = Retriever(cfg)
        self.clip.model.to(self.clip.device)
        prediction_features = self.clip.encode([r["prediction"] for r in results])
        self.clip.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        index = np.load(out / "index.npz")
        original = load_split(cfg, split)
        positions = {r["id"]: i for i, r in enumerate(original)}
        image_features = index[split]
        for row, feature in zip(results, prediction_features):
            row["clipscore"] = float(2.5 * max(0., feature @ image_features[positions[row["id"]]]))
            row["input_clipscore"] = float(2.5 * max(0., feature @ image_features[positions[row["input_image_id"]]]))
        save_rows(partial, results)
        keys = ["token_f1", "rouge_l", "words", "context_copy_rate", "reference_detail_recall",
                "cider_fixed_df", "cider_d_fixed_df", "clipscore", "input_clipscore", "context_words"]
        keys += ["input_tokens", "context_tokens"]
        metrics = {k: float(np.mean([r[k] for r in results])) for k in keys}
        metrics.update({"model": self.name, "split": split, "mode": mode, "bank": bank,
                        "shuffled": shuffled, "n": len(results), "seconds_this_invocation": time.perf_counter() - start,
                        "retrieval_use_fraction": float(np.mean([bool(r["context"]) for r in results])),
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
                        "cider_protocol": "regex tokens + fixed training document frequency; not COCO leaderboard comparable",
                        "copy_protocol": "Lexical content overlap proxy, not a hallucination rate"})
        save_json(directory / "metrics.json", metrics)
        print("Completed evaluation:", metrics, flush=True)
        return results


def run_bundle(cfg, regime, seed, sampling="random", count=None, scope="language", budget="epochs", full=False):
    seed_all(seed)
    _, out, _ = paths(cfg)
    name = model_name(cfg, regime, seed, sampling, count, scope, budget)
    generator = Generator(cfg, name)
    validation = load_split(cfg, "validation")[:cfg["gate_validation_images"]]
    contexts = read_json(context_path(cfg, "validation"))
    a = generator.generate(validation, "validation", "none", contexts)
    b = generator.generate(validation, "validation", "image", contexts)
    gate = tune_gate(validation, a, b, contexts["image"], cfg)
    save_json(out / "gates" / (name + ".json"), {**gate, "identity": fingerprint(cfg), "model": name})
    tests = load_split(cfg, "test")[:cfg["confirmation_images"]]
    test_contexts = read_json(context_path(cfg, "test"))
    modes = cfg["evaluation_modes"] if full else ["none"]
    for mode in modes:
        generator.generate(tests, "test", mode, test_contexts, gate=gate)
    if full and seed == cfg["seeds"][0] and regime in cfg["regimes"]:
        diagnostics = tests[:cfg["diagnostic_images"]]
        for mode in cfg["diagnostic_modes"]:
            generator.generate(diagnostics, "test", mode, test_contexts)
        generator.generate(diagnostics, "test", "none", test_contexts, shuffled=True)
        generator.generate(diagnostics, "test", "image", test_contexts, shuffled=True)
        subset = read_json(context_path(cfg, "test", "subset"))
        generator.generate(diagnostics, "test", "image", subset, bank="subset")
    del generator
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
