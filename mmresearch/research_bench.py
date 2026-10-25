"""Pinned external-subset diagnostics for research v2, with resumable inference.

No benchmark labels are synthesized here. POPE tests yes/no object presence;
SugarCrepe tests supplied caption pairs; transfer captions use the benchmark's
positive descriptions and are not a representative COCO caption evaluation.
"""

import gc
import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, digest, read_json, read_rows, save_json, save_rows, seed_all
from .research_common import fingerprint, load_split, paths
from .research_evaluate import adapter_path, model_name
from .research_metrics import (answer_yes_no, binary_metrics, cider_d_scores,
                               cider_scores, per_sample_metrics)


def answer_choice(text):
    """Frozen strict leading A/B parser; invalid replies count as incorrect."""
    if not isinstance(text, str):
        return "invalid"
    cleaned = re.sub(r"^answer\s*:\s*", "", text.strip(), flags=re.IGNORECASE).lstrip("*_` ")
    match = re.match(r"^([AB])\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return "invalid"
    suffix = cleaned[match.end():].lstrip("*_` ")
    if not suffix or suffix.startswith((".", ",", "!", "\n", "\r", ":", ";")):
        return match.group(1).upper()
    return "invalid"


def balanced_options(rows, seed=20261025):
    """Deterministic A/B balancing within category, independent of input order."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("type", row.get("category", "all"))].append(row)
    assignment = {}
    for category, group in groups.items():
        ordered = sorted(group, key=lambda row: hashlib.sha256((str(seed) + ":" + row["id"]).encode()).hexdigest())
        for i, row in enumerate(ordered):
            positive_option = "A" if i % 2 == 0 else "B"
            assignment[row["id"]] = {**row, "positive_option": positive_option,
                "option_a": row["positive"] if positive_option == "A" else row["negative"],
                "option_b": row["negative"] if positive_option == "A" else row["positive"]}
    if len(assignment) != len(rows):
        raise ValueError("Duplicate SugarCrepe IDs")
    return [assignment[row["id"]] for row in rows]


def _mean(rows, keys):
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _validate_rows(rows, task):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicated %s benchmark rows" % task)
    for row in rows:
        if not (ROOT / row["image"]).is_file():
            raise FileNotFoundError("Benchmark image missing: " + row["image"])
        if row.get("label_source") and row["label_source"] != "official_benchmark_annotation":
            raise ValueError("Unsupported benchmark label provenance")
    if task == "pope":
        for variant in {row["variant"] for row in rows}:
            counts = Counter(row["label"] for row in rows if row["variant"] == variant)
            if set(counts) != {"yes", "no"} or counts["yes"] != counts["no"]:
                raise ValueError("POPE variant requires balanced explicit yes/no labels")
    elif task == "sugarcrepe":
        if any(not row["positive"].strip() or not row["negative"].strip() or row["positive"].strip() == row["negative"].strip() for row in rows):
            raise ValueError("Invalid SugarCrepe pair")
    elif task == "transfer" and any(not row.get("captions") for row in rows):
        raise ValueError("Transfer rows need explicit supplied reference captions")


def _identity(cfg, name, task, data, adapter):
    source = data / "benchmark_source.json"
    base_provenance = Path(cfg["vl_model"]) / "download_provenance.json"
    return {**fingerprint(cfg), "model": name, "task": task,
            "benchmark_manifest_sha256": digest(data / (task + ".jsonl")),
            "benchmark_source_sha256": digest(source),
            "adapter_sha256": digest(Path(adapter) / "adapter_model.safetensors") if adapter else None,
            "base_model_source_sha256": digest(base_provenance) if base_provenance.exists() else None,
            "protocol_version": 1, "benchmark_seed": cfg["benchmark_seed"],
            "max_new_tokens": cfg["max_new_tokens"] if task == "transfer" else 8}


def _existing(directory, task, identity, rows):
    path = directory / (task + "_metrics.json")
    if not path.exists():
        return None
    metrics = read_json(path)
    if metrics.get("identity") != identity:
        raise ValueError("Stale completed benchmark: " + str(path))
    predictions = read_rows(directory / (task + "_predictions.jsonl"))
    if [row["id"] for row in predictions] != [row["id"] for row in rows]:
        raise ValueError("Completed benchmark prediction ID mismatch")
    if metrics.get("n") != len(predictions):
        raise ValueError("Completed benchmark prediction count mismatch")
    return metrics


def _partial(directory, task, identity, rows):
    identity_path = directory / (task + ".identity")
    predictions_path = directory / (task + "_predictions.jsonl")
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Stale partial benchmark identity")
    if predictions_path.exists() and not identity_path.exists():
        raise ValueError("Partial benchmark predictions lack identity")
    save_json(identity_path, identity)
    predictions = read_rows(predictions_path) if predictions_path.exists() else []
    if [row["id"] for row in predictions] != [row["id"] for row in rows[:len(predictions)]]:
        raise ValueError("Partial benchmark order mismatch")
    return predictions


def _image(relative):
    from PIL import Image
    with Image.open(ROOT / relative) as opened:
        return opened.convert("RGB")


def _generate(model, processor, rows, questions, max_new_tokens):
    import torch
    chats = []
    for question in questions:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        chats.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    images = [_image(row["image"]) for row in rows]
    batch = processor(text=chats, images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    return [text.strip() for text in processor.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)]


def _sugar_clip(cfg, rows):
    """Release the CLIP GPU model before loading Qwen to fit a 16GB GPU."""
    import torch
    from .retrieval import Retriever
    retriever = Retriever(cfg)
    unique = {}
    for row in rows:
        unique.setdefault(row["image_id"], row["image"])
    ids = list(unique)
    image_features = retriever.encode([unique[i] for i in ids], images=True)
    image_map = dict(zip(ids, image_features))
    text_features = retriever.encode([caption for row in rows for caption in (row["positive"], row["negative"])])
    result = {}
    for i, row in enumerate(rows):
        image = image_map[row["image_id"]]
        positive = float(image @ text_features[2 * i])
        negative = float(image @ text_features[2 * i + 1])
        result[row["id"]] = {"clip_positive_similarity": positive, "clip_negative_similarity": negative,
                             "clip_margin": positive - negative, "clip_correct": positive > negative,
                             "clip_tie": positive == negative}
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _task_metrics(task, predictions, identity, seconds, peak):
    result = {"status": "completed", "task": task, "model": identity["model"],
              "n": len(predictions), "n_images": len({r["image_id"] for r in predictions}),
              "subset_only": True, "training_domain": "Flickr8k caption instructions",
              "identity": identity, "seconds_this_invocation": seconds, "peak_gpu_gib": peak}
    if task == "pope":
        variants = {}
        for variant in sorted({row["variant"] for row in predictions}):
            group = [row for row in predictions if row["variant"] == variant]
            variants[variant] = {**binary_metrics([r["answer"] for r in group], [r["label"] for r in group], require_balanced=True),
                                 "n_images": len({r["image_id"] for r in group})}
        result.update({**binary_metrics([r["answer"] for r in predictions], [r["label"] for r in predictions], require_balanced=True),
                       "variants": variants, "task_shift": "object-existence yes/no; not the caption training task",
                       "invalid_policy": "invalid outputs retained and counted as incorrect in accuracy",
                       "dependence": "same images and correlated questions across variants; not independent image observations"})
    elif task == "sugarcrepe":
        types = {}
        for category in sorted({row["category"] for row in predictions}):
            group = [row for row in predictions if row["category"] == category]
            types[category] = {"n": len(group), "accuracy": float(np.mean([r["correct"] for r in group])),
                               "clip_accuracy": float(np.mean([r["clip_correct"] for r in group])),
                               "invalid_rate": float(np.mean([r["parsed_answer"] == "invalid" for r in group])),
                               "positive_option_counts": dict(Counter(r["positive_option"] for r in group))}
        result.update({"accuracy": float(np.mean([r["correct"] for r in predictions])),
                       "clip_accuracy": float(np.mean([r["clip_correct"] for r in predictions])),
                       "clip_ties": sum(r["clip_tie"] for r in predictions),
                       "invalid_rate": float(np.mean([r["parsed_answer"] == "invalid" for r in predictions])),
                       "types": types, "parser": "strict_leading_A_B_v1",
                       "clip_tie_policy": "exact positive==negative ties count as incorrect",
                       "task_shift": "balanced caption-pair choice; not the caption training task"})
    else:
        keys = ("token_f1", "rouge_l", "words", "context_copy_rate", "reference_detail_recall", "cider_fixed_df", "cider_d_fixed_df")
        result.update({**_mean(predictions, keys),
                       "reference_source": "SugarCrepe official positive captions; authorship not independently verified",
                       "selection_scope": "benchmark-conditioned subset, not representative COCO or independent human evaluation",
                       "cider_protocol": "regex tokenization; fixed Flickr8k training-reference document frequency; not COCO leaderboard comparable",
                       "clipscore": None, "clipscore_status": "not_computed_to_avoid_a_second_live_GPU_model"})
    return result


def run(cfg, regime="zero_shot", seed=42):
    """Run complete external diagnostics for a base model or a core adapter.

    Resuming requires the same config/data fingerprints, annotation manifests,
    and adapter SHA256; completed summaries are never silently mixed.
    """
    import torch
    from .vl import load
    data, out, _ = paths(cfg)
    source = read_json(data / "benchmark_source.json")
    if source.get("status") != "complete":
        raise ValueError("Benchmark preparation is not complete")
    name = model_name(cfg, regime, seed)
    adapter = adapter_path(cfg, name)
    directory = out / "benchmarks" / name
    directory.mkdir(parents=True, exist_ok=True)
    tasks = {}
    identities = {}
    finished = {}
    for task in ("pope", "sugarcrepe", "transfer"):
        rows = read_rows(data / (task + ".jsonl"))
        _validate_rows(rows, task)
        if task == "sugarcrepe":
            rows = balanced_options(rows, cfg["benchmark_seed"])
        tasks[task] = rows
        identities[task] = _identity(cfg, name, task, data, adapter)
        prior = _existing(directory, task, identities[task], rows)
        if prior is not None:
            finished[task] = prior
    if len(finished) == len(tasks):
        print("Reusing completed external benchmarks:", name, flush=True)
        return finished
    clip_results = _sugar_clip(cfg, tasks["sugarcrepe"]) if "sugarcrepe" not in finished else None
    seed_all(seed)
    model, processor = load(cfg, adapter)
    model.eval()
    try:
        for task, rows in tasks.items():
            if task in finished:
                continue
            identity = identities[task]
            predictions = _partial(directory, task, identity, rows)
            start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for i in range(len(predictions), len(rows), cfg["inference_batch_size"]):
                chunk = rows[i:i + cfg["inference_batch_size"]]
                if task == "pope":
                    questions = [row["question"] + "\nAnswer yes or no first." for row in chunk]
                elif task == "sugarcrepe":
                    questions = ["Which caption best matches this image? Reply with A or B only.\nA: %s\nB: %s" % (row["option_a"], row["option_b"]) for row in chunk]
                else:
                    questions = ["Describe this image in one concise English sentence."] * len(chunk)
                answers = _generate(model, processor, chunk, questions, identity["max_new_tokens"])
                for row, answer in zip(chunk, answers):
                    if task == "pope":
                        entry = {**row, "answer": answer, "parsed_answer": answer_yes_no(answer)}
                    elif task == "sugarcrepe":
                        parsed = answer_choice(answer)
                        entry = {**row, "answer": answer, "parsed_answer": parsed,
                                 "correct": parsed == row["positive_option"], **clip_results[row["id"]]}
                    else:
                        entry = {**row, "prediction": answer, "references": row["captions"],
                                 **per_sample_metrics(answer, row["captions"], [])}
                    predictions.append(entry)
                save_rows(directory / (task + "_predictions.jsonl"), predictions)
                if len(predictions) % 32 == 0 or len(predictions) == len(rows):
                    print("bench %s/%s: %d/%d" % (name, task, len(predictions), len(rows)), flush=True)
            if task == "transfer":
                df_refs = [row["captions"] for row in load_split(cfg, "train")]
                c = cider_scores([row["prediction"] for row in predictions], [row["references"] for row in predictions], df_refs)
                d = cider_d_scores([row["prediction"] for row in predictions], [row["references"] for row in predictions], df_refs)
                for row, a, b in zip(predictions, c, d):
                    row["cider_fixed_df"], row["cider_d_fixed_df"] = a, b
                save_rows(directory / (task + "_predictions.jsonl"), predictions)
            if len(predictions) != len(rows):
                raise ValueError("Benchmark completion count mismatch")
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
            summary = _task_metrics(task, predictions, identity, time.perf_counter() - start, peak)
            save_json(directory / (task + "_metrics.json"), summary)
            finished[task] = summary
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finished
