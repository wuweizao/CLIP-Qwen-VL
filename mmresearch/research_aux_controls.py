"""Supplementary bank-size diagnostic with matched per-slot context word counts.

This module does not alter the main experiment plan, its contexts, or its
evaluation implementation. It compares retrieval at inference time for the same
already-trained model. Word matching does not imply equal BPE token counts,
caption semantics, or retrieval quality.
"""
import copy
import gc
import hashlib
import json

from .common import digest, read_json, read_rows, save_json, seed_all
from .research_common import context_path, fingerprint, load_split, paths
from .research_metrics import paired_effect


BANKS = ("full_length_matched", "subset_length_matched")
EFFECT_METRICS = ("token_f1", "cider_d_fixed_df", "clipscore", "words", "context_tokens")


def match_banks(fullmap, subsetmap, query_ids):
    """Deep-copy and trim both banks to the same word count in every slot.

    The query-ID sequence must be unique and fully present in each map. Context
    counts must be equal and nonzero for each query. Owner IDs may differ across
    the banks, as expected when the candidate bank changes; rank and provenance
    are retained. Every selected owner must differ from the query image ID.
    """
    query_ids = list(query_ids)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("Unique nonempty query IDs are required")
    full, subset = {}, {}
    for query_id in query_ids:
        if query_id not in fullmap or query_id not in subsetmap:
            raise ValueError("Both banks must contain every requested query ID")
        a, b = copy.deepcopy(fullmap[query_id]), copy.deepcopy(subsetmap[query_id])
        if not a or len(a) != len(b):
            raise ValueError("Both banks require the same nonzero context count")
        for values in (a, b):
            owners = [r["id"] for r in values]
            if len(owners) != len(set(owners)) or query_id in owners:
                raise ValueError("Duplicate context owner or current-image leakage")
            if any(not isinstance(r.get("caption"), str) or not r["caption"].strip() for r in values):
                raise ValueError("Every context needs a nonempty caption")
        for slot, (left, right) in enumerate(zip(a, b)):
            left_words, right_words = left["caption"].split(), right["caption"].split()
            target = min(len(left_words), len(right_words))
            for item, words in ((left, left_words), (right, right_words)):
                original = item["caption"]
                item["pre_cross_bank_caption"] = original
                item["pre_cross_bank_words"] = len(words)
                item["caption"] = " ".join(words[:target])
                item["cross_bank_length_matched"] = True
                item["cross_bank_matching_rule"] = "per-query per-ranked-slot shared minimum words; no padding"
                item["cross_bank_slot"] = slot
                item["cross_bank_target_words"] = target
                item["effective_words"] = target
                item["cross_bank_trimmed"] = len(words) > target
                if len(words) > target:
                    item["truncated"] = True
                    item["complete_sentence_boundary"] = False
        full[query_id], subset[query_id] = a, b
    return full, subset


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _names(cfg):
    from .research_evaluate import model_name
    seed = cfg["seeds"][0]
    return [(regime, model_name(cfg, regime, seed)) for regime in ("zero_shot", *cfg["regimes"])]


def _protocol(cfg, rows, full, subset):
    _, out, _ = paths(cfg)
    selections = read_json(out / "selections.json")["random"]
    train_ids = {r["id"] for r in load_split(cfg, "train")}
    subset_ids = set(selections)
    ids = [r["id"] for r in rows]
    if len(subset_ids) != len(selections) or not subset_ids <= train_ids:
        raise ValueError("Invalid training-only subset bank definition")
    for query_id in ids:
        if len(full[query_id]) != cfg["rag_k"] or len(subset[query_id]) != cfg["rag_k"]:
            raise ValueError("Bank-size control must retain configured K contexts")
        if any(c["id"] not in train_ids for c in full[query_id]) or any(c["id"] not in subset_ids for c in subset[query_id]):
            raise ValueError("Context owner is outside the declared training bank")
    return {"control": "supplementary_inference_bank_size_word_matched", "identity": fingerprint(cfg),
        "protocol_version": 1, "control_code_sha256": digest(__file__),
        "registration": "fixed before this supplementary generation; introduced after the main experiment plan",
        "status_scope": "supplementary diagnostic, not a new preregistered confirmatory claim",
        "split": "test", "query_ids": ids, "n": len(rows), "k": cfg["rag_k"],
        "seed": cfg["seeds"][0], "models": [name for _, name in _names(cfg)],
        "source_context_sha256": {"full": digest(context_path(cfg, "test")),
                                   "subset": digest(context_path(cfg, "test", "subset"))},
        "matched_context_sha256": {"full": _hash({"image": full}), "subset": _hash({"image": subset})},
        "bank_images": {"full": len(train_ids), "subset": len(subset_ids)},
        "matching": "Per query and rank slot: trim both captions to their shared minimum whitespace-word count. Keep original owner IDs, ranking and metadata. No padding or extra descriptions.",
        "slot_word_counts": {i: [len(c["caption"].split()) for c in full[i]] for i in ids},
        "context_owner_ids": {"full": {i: [c["id"] for c in full[i]] for i in ids},
                              "subset": {i: [c["id"] for c in subset[i]] for i in ids}},
        "held_fixed": ["trained adapter", "query images and references", "generation prompt and budget",
                       "retrieval algorithm", "K", "per-slot whitespace-word counts", "within-bank rank order"],
        "not_matched": ["BPE token count", "caption semantics", "retrieval similarity", "selected context owner identity"],
        "training_memory": "Retrieval-aware and noise adapters were trained with the full training memory. This control changes only inference memory and does not retrain adapters.",
        "limitations": ["Trimming can remove semantic details or end a sentence mid-phrase.",
                        "Only the first configured seed and the fixed diagnostic image subset are assessed.",
                        "Changing bank size changes retrieved content; results are conditional effects of these banks, not isolated effects of memory capacity in all settings."]}


def summarize(cfg):
    """Summarize completed pairs only; incomplete controls stay explicitly pending."""
    _, out, report = paths(cfg)
    protocol = read_json(out / "bank_size_control_protocol.json")
    if protocol["identity"] != fingerprint(cfg):
        raise ValueError("Stale bank-size control protocol")
    expected = protocol["query_ids"]
    effects, model_rows, pending = [], [], []
    for regime, name in _names(cfg):
        predictions = {}
        for bank in BANKS:
            directory = out / "evaluations" / name / ("test_image_" + bank)
            if not (directory / "metrics.json").exists():
                pending.append({"model": name, "bank": bank})
                continue
            metrics = read_json(directory / "metrics.json")
            identity = read_json(directory / "identity.json")
            rows = read_rows(directory / "predictions.jsonl")
            if [r["id"] for r in rows] != expected or metrics["n"] != len(expected):
                raise ValueError("Bank-size control prediction IDs or count mismatch")
            if identity["data_manifests"] != protocol["identity"]["data_manifests"] or identity["model"] != name or identity["bank"] != bank:
                raise ValueError("Bank-size control evaluation identity mismatch")
            source = "full" if bank == BANKS[0] else "subset"
            if identity["context_sha256"] != protocol["matched_context_sha256"][source]:
                raise ValueError("Matched context provenance mismatch")
            for row in rows:
                contexts = row["context"]
                if [c["id"] for c in contexts] != protocol["context_owner_ids"][source][row["id"]]:
                    raise ValueError("Control context owner/rank mismatch")
                if [len(c["caption"].split()) for c in contexts] != protocol["slot_word_counts"][row["id"]]:
                    raise ValueError("Control context word counts changed")
                if not all(c.get("cross_bank_length_matched") for c in contexts):
                    raise ValueError("Missing cross-bank length matching provenance")
            predictions[bank] = rows
            model_rows.append({"regime": regime, "model": name, "bank": bank, "n": len(rows),
                               **{key: metrics[key] for key in EFFECT_METRICS}})
        if len(predictions) != 2:
            continue
        # Explicit B-A: full bank minus subset bank, on identical image IDs.
        for metric in EFFECT_METRICS:
            effects.append({"model": name, "regime": regime,
                "effect": "full_length_matched - subset_length_matched",
                **paired_effect(predictions[BANKS[1]], predictions[BANKS[0]], metric=metric)})
    result = {"status": "completed" if not pending else "partial", "protocol": protocol,
              "completed_evaluations": len(model_rows), "expected_evaluations": len(_names(cfg)) * 2,
              "pending": pending, "run_metrics": model_rows, "effects": effects}
    save_json(report / "bank_size_effects.json", result)
    lines = ["# 补充诊断：检索库大小与上下文长度控制", "",
        "状态：%s；已完成评估 %d/%d。" % (result["status"], len(model_rows), result["expected_evaluations"]), "",
        "比较同一个已经训练完成的模型在 %d 与 %d 张训练图像检索库下的输出。固定 %d 张诊断图像、K=%d，以及每个排序位置的上下文词数；不对已有适配器重新训练。" %
        (protocol["bank_images"]["full"], protocol["bank_images"]["subset"], protocol["n"], protocol["k"]), "",
        "本控制在主实验方案之后提出，在本组推理前固定，属于补充诊断。检索训练与抗噪声模型原先使用完整训练检索库，因此这里考察推理时改变记忆库的条件效应。", "",
        "|模型|检索库|图像数|Token-F1|CIDEr-D变体|CLIPScore|输出词数|上下文 token数|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in model_rows:
        lines.append("|%s|%s|%d|%.4f|%.4f|%.4f|%.2f|%.2f|" % (row["model"], row["bank"], row["n"],
            row["token_f1"], row["cider_d_fixed_df"], row["clipscore"], row["words"], row["context_tokens"]))
    lines += ["", "差值方向为完整库减小库。下列 95% 区间仅按图像配对 bootstrap，不涵盖重新训练的不确定性。", "",
              "|模型|指标|差值|95% 图像区间|", "|---|---|---:|---|"]
    for effect in effects:
        lines.append("|%s|%s|%.4f|[%.4f, %.4f]|" % (effect["model"], effect["metric"], effect["delta"], effect["ci_low"], effect["ci_high"]))
    lines += ["", "词数匹配不等于 BPE token 数匹配；实际 token 数已记录。两库保留各自检索排序，选中的图像和描述内容仍可不同。截断可能损失语义细节，因此结果不能单独归因为抽象的记忆容量，也不能据单个种子宣称普遍因果规律。"]
    (report / "bank_size_effects.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run(cfg):
    """Run only the supplementary pairs, leaving the existing plan untouched."""
    from .research_evaluate import Generator
    data, out, report = paths(cfg)
    rows = load_split(cfg, "test")[:cfg["diagnostic_images"]]
    full, subset = match_banks(read_json(context_path(cfg, "test"))["image"],
                              read_json(context_path(cfg, "test", "subset"))["image"],
                              [r["id"] for r in rows])
    protocol = _protocol(cfg, rows, full, subset)
    protocol_path = out / "bank_size_control_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Refusing to replace a fixed supplementary control protocol")
    save_json(protocol_path, protocol)
    report.mkdir(parents=True, exist_ok=True)
    for regime, name in _names(cfg):
        seed_all(cfg["seeds"][0])
        generator = Generator(cfg, name)
        try:
            generator.generate(rows, "test", "image", {"image": full}, bank=BANKS[0])
            generator.generate(rows, "test", "image", {"image": subset}, bank=BANKS[1])
        finally:
            del generator
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summarize(cfg)
    return summarize(cfg)
