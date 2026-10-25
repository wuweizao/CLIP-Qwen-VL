"""Evidence-backed reports for research v2; safe to rerun during partial execution.

Only an evaluation with metrics.json AND a complete, unique prediction set is
counted. Image bootstrap uncertainty and variation across training seeds are
reported separately. No human judgments are inferred from automatic metrics.
"""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, read_rows, save_json
from .research_common import paths, run_name
from .research_metrics import paired_effect, seed_summary


METRICS = ("token_f1", "rouge_l", "cider_fixed_df", "cider_d_fixed_df", "clipscore",
           "input_clipscore", "words", "context_copy_rate", "reference_detail_recall",
           "context_words", "input_tokens", "context_tokens", "retrieval_use_fraction",
           "context_copy_rate_when_used")
COHORT = ("sampling", "count", "scope", "budget")
GROUP = ("regime",) + COHORT + ("split", "mode", "bank", "shuffled")


def _meta(out, model):
    if model == "zero_shot":
        return {"regime": "zero_shot", "seed": None, "sampling": "none", "count": 0,
                "scope": "none", "budget": "none"}
    training = out / "models" / model / "training.json"
    if training.exists():
        identity = read_json(training)["identity"]
        return {key: identity[key] for key in ("regime", "seed") + COHORT}
    match = re.fullmatch(r"(?P<regime>plain|retrieval|noise)_s(?P<seed>\d+)_(?P<sampling>.+?)_n(?P<count>\d+)_(?P<scope>language(?:_merger)?)_(?P<budget>epochs|matched_steps)", model)
    if not match:
        raise ValueError("Cannot identify training configuration: " + model)
    info = match.groupdict()
    info["seed"], info["count"] = int(info["seed"]), int(info["count"])
    return info


def _csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_runs(out):
    runs = []
    for path in sorted((out / "evaluations").glob("*/*/metrics.json")):
        metrics = read_json(path)
        predictions_path = path.parent / "predictions.jsonl"
        identity_path = path.parent / "identity.json"
        if not predictions_path.exists() or not identity_path.exists():
            raise ValueError("Completed metrics without prediction/identity artifacts: " + str(path))
        rows = read_rows(predictions_path)
        identity = read_json(identity_path)
        ids = [row["id"] for row in rows]
        if not rows or len(ids) != len(set(ids)) or len(rows) != metrics["n"]:
            raise ValueError("Incomplete or duplicated evaluation: " + str(path))
        if ids != identity["ids"]:
            raise ValueError("Prediction order/ID mismatch: " + str(path))
        if metrics["model"] != path.parent.parent.name:
            raise ValueError("Model identity mismatch: " + str(path))
        used = [row["context_copy_rate"] for row in rows if row.get("context")]
        metrics["context_copy_rate_when_used"] = float(np.mean(used)) if used else None
        metrics["retrieval_used_images"] = len(used)
        if any(not math.isfinite(float(metrics[key])) for key in METRICS if key in metrics and metrics[key] is not None):
            raise ValueError("Non-finite completed metric: " + str(path))
        runs.append({**metrics, **_meta(out, metrics["model"]), "rows": rows, "directory": str(path.parent)})
    return runs


def _seed_groups(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[tuple(run[key] for key in GROUP)].append(run)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(GROUP, key))
        seeds = [r["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate training seed within evaluation group")
        counts = {r["n"] for r in group}
        if len(counts) != 1:
            raise ValueError("Different evaluation sizes in seed group")
        expected_ids = {r["id"] for r in group[0]["rows"]}
        if any({r["id"] for r in run["rows"]} != expected_ids for run in group):
            raise ValueError("Different evaluation images across training seeds")
        record.update({"n_training_seeds": 0 if record["regime"] == "zero_shot" else len(group),
                       "n_runs": len(group), "n_images": next(iter(counts)), "seeds": seeds})
        for metric in METRICS:
            if not all(metric in run for run in group):
                continue
            valid = [run[metric] for run in group if run[metric] is not None]
            values = np.asarray(valid, dtype=float)
            record[metric + "_mean"] = float(values.mean()) if len(values) else None
            record[metric + "_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            record[metric + "_n_defined_runs"] = len(values)
        result.append(record)
    return result


def _deltas(a, b, metric):
    aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if not aa or len(aa) != len(a) or len(bb) != len(b) or set(aa) != set(bb):
        raise ValueError("Paired effects require identical unique nonempty image ID sets")
    return {i: float(bb[i][metric]) - float(aa[i][metric]) for i in sorted(aa)}


def _effect_summary(label, cohort, metric, seed_deltas):
    seeds = sorted(seed_deltas, key=str)
    image_ids = set(seed_deltas[seeds[0]])
    if any(set(seed_deltas[seed]) != image_ids for seed in seeds):
        raise ValueError("Effect aggregation requires the same test images across seeds")
    ids = sorted(image_ids)
    image_means = [{"id": i, metric: float(np.mean([seed_deltas[seed][i] for seed in seeds]))} for i in ids]
    zeros = [{"id": i, metric: 0.0} for i in ids]
    bootstrap = paired_effect(zeros, image_means, metric=metric)
    per_seed = [{"seed": seed, "delta": float(np.mean(list(seed_deltas[seed].values())))} for seed in seeds]
    stats = seed_summary(per_seed, metric="delta")
    return {"effect": label, "cohort": cohort, "metric": metric, "n_images": len(ids),
            "n_runs": len(seeds), "seeds": seeds, "mean_delta": stats["mean"], "seed_sd": stats["seed_sd"],
            "image_ci_low": bootstrap["ci_low"], "image_ci_high": bootstrap["ci_high"],
            "image_ci_interpretation": "evaluation-image bootstrap after averaging the available fixed training runs; excludes training-seed uncertainty",
            "per_seed": per_seed}


def _diagnostic_prefix(primary, diagnostic, cfg):
    """Project ONLY predeclared prefix diagnostics; never intersect arbitrary IDs."""
    if cfg is None:
        raise ValueError("Diagnostic pairing requires the predeclared experiment config")
    count = cfg["diagnostic_images"]
    if len(primary) != cfg["confirmation_images"] or len(diagnostic) != count:
        raise ValueError("Diagnostic size does not match the predeclared confirmation/prefix sizes")
    expected = [row["id"] for row in primary[:count]]
    actual = [row["id"] for row in diagnostic]
    if actual != expected or len(set(actual)) != count:
        raise ValueError("Diagnostic image IDs must be the exact predeclared ordered confirmation prefix")
    return primary[:count]


def _effects(runs, cfg=None):
    # Index trained models by a common training configuration and evaluation.
    lookup = {}
    for run in runs:
        key = (run["regime"], run["seed"]) + tuple(run[k] for k in COHORT) + (run["split"], run["mode"], run["bank"], run["shuffled"])
        if key in lookup:
            raise ValueError("Duplicate completed evaluation condition")
        lookup[key] = run
    pending = defaultdict(dict)
    for key, run in lookup.items():
        regime, seed, sampling, count, scope, budget, split, mode, bank, shuffled = key
        if split != "test" or shuffled:
            continue
        # Each effect has an explicit direction B-A.
        pairs = []
        if mode == "none":
            for other_mode in ("random", "image", "adaptive"):
                other_key = key[:7] + (other_mode, bank, False)
                if other_key in lookup:
                    pairs.append((regime + ":" + other_mode + "-none", run, lookup[other_key], False))
        if mode == "image":
            for other_mode in ("adaptive", "text", "rerank", "conflict", "partial"):
                other_key = key[:7] + (other_mode, bank, False)
                if other_key in lookup:
                    pairs.append((regime + ":" + other_mode + "-image", run, lookup[other_key], other_mode in ("text", "rerank", "conflict", "partial")))
        for label, a, b, diagnostic in pairs:
            baseline_rows = _diagnostic_prefix(a["rows"], b["rows"], cfg) if diagnostic else a["rows"]
            cohort = {"sampling": sampling, "count": count, "scope": scope, "budget": budget, "split": split, "bank": bank,
                      "analysis_subset": "diagnostic_subset" if diagnostic else "confirmation"}
            for metric in ("token_f1", "cider_d_fixed_df", "clipscore", "words", "context_copy_rate"):
                if metric not in a["rows"][0] or metric not in b["rows"][0]:
                    continue
                pending[(label, tuple(cohort.items()), metric)][seed] = _deltas(baseline_rows, b["rows"], metric)
        if regime != "plain" or mode != "image":
            continue
        cohort = {"sampling": sampling, "count": count, "scope": scope, "budget": budget, "split": split, "bank": bank,
                  "analysis_subset": "confirmation"}
        plain_none_key = ("plain", seed, sampling, count, scope, budget, split, "none", bank, False)
        retrieval_none_key = ("retrieval", seed, sampling, count, scope, budget, split, "none", bank, False)
        retrieval_image_key = ("retrieval", seed, sampling, count, scope, budget, split, "image", bank, False)
        noise_adaptive_key = ("noise", seed, sampling, count, scope, budget, split, "adaptive", bank, False)
        for metric in ("token_f1", "cider_d_fixed_df", "clipscore", "words"):
            if all(k in lookup for k in (plain_none_key, retrieval_none_key, retrieval_image_key)):
                plain_delta = _deltas(lookup[plain_none_key]["rows"], run["rows"], metric)
                retrieval_delta = _deltas(lookup[retrieval_none_key]["rows"], lookup[retrieval_image_key]["rows"], metric)
                if set(plain_delta) != set(retrieval_delta):
                    raise ValueError("Difference-in-differences requires the same image IDs")
                value = {i: retrieval_delta[i] - plain_delta[i] for i in plain_delta}
                pending[("interaction:(retrieval_image-retrieval_none)-(plain_image-plain_none)", tuple(cohort.items()), metric)][seed] = value
            if noise_adaptive_key in lookup:
                pending[("combined:noise_adaptive-plain_image", tuple(cohort.items()), metric)][seed] = _deltas(run["rows"], lookup[noise_adaptive_key]["rows"], metric)
    return [_effect_summary(label, dict(cohort), metric, values)
            for (label, cohort, metric), values in sorted(pending.items(), key=lambda item: str(item[0]))]


def _fmt(value, digits=4):
    return "—" if value is None else ("%." + str(digits) + "f") % value


def _similarity_utility_data(runs, cfg):
    """Descriptive query-level associations, after averaging fixed training runs.

    This function never selects gates or models and does not treat training
    seeds as independent copies of test images. Gallery Recall@K is a distinct
    task and is not an input to this analysis.
    """
    lookup = {}
    for run in runs:
        if (run["regime"] in cfg["regimes"] and run["sampling"] == "random"
                and run["count"] == cfg["core_train_images"] and run["scope"] == "language"
                and run["budget"] == "epochs" and run["split"] == "test"
                and run["bank"] == "full" and not run["shuffled"] and run["mode"] in ("none", "image")):
            key = (run["regime"], run["seed"], run["mode"])
            if key in lookup:
                raise ValueError("Duplicate similarity-utility condition")
            if len(run["rows"]) != cfg["confirmation_images"]:
                raise ValueError("Similarity-utility analysis requires the full predeclared confirmation set")
            lookup[key] = run
    queries, quartiles, summaries = [], [], []
    for regime in cfg["regimes"]:
        per_seed = {}
        for seed in cfg["seeds"]:
            pair = [(regime, seed, mode) for mode in ("none", "image")]
            if not all(key in lookup for key in pair):
                continue
            baseline, retrieved = [lookup[key]["rows"] for key in pair]
            delta = _deltas(baseline, retrieved, "token_f1")
            scores = {}
            for row in retrieved:
                if not row.get("context") or "score" not in row["context"][0]:
                    raise ValueError("Top-1 image context score missing")
                score = float(row["context"][0]["score"])
                if not math.isfinite(score):
                    raise ValueError("Non-finite top-1 context similarity")
                scores[row["id"]] = score
            per_seed[seed] = (scores, delta)
        if not per_seed:
            continue
        seeds = sorted(per_seed)
        ids = set(per_seed[seeds[0]][1])
        if any(set(per_seed[seed][1]) != ids for seed in seeds):
            raise ValueError("Similarity-utility aggregation requires matching images across seeds")
        regime_rows = []
        for identifier in sorted(ids):
            row = {"regime": regime, "id": identifier, "n_fixed_training_runs": len(seeds),
                   "top1_image_similarity": float(np.mean([per_seed[seed][0][identifier] for seed in seeds])),
                   "delta_token_f1": float(np.mean([per_seed[seed][1][identifier] for seed in seeds]))}
            regime_rows.append(row)
        x = np.asarray([row["top1_image_similarity"] for row in regime_rows])
        y = np.asarray([row["delta_token_f1"] for row in regime_rows])
        correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None
        summaries.append({"regime": regime, "n_images": len(ids), "n_fixed_training_runs": len(seeds),
                          "seeds": seeds, "pearson_r": correlation,
                          "interpretation": "descriptive association across test images; no p-value or causal interpretation; not used for gate/model selection"})
        ordered = sorted(regime_rows, key=lambda row: (row["top1_image_similarity"], row["id"]))
        for q, indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
            if not len(indices):
                continue
            group = [ordered[int(i)] for i in indices]
            quartiles.append({"regime": regime, "similarity_quartile": q, "n_images": len(group),
                "n_fixed_training_runs": len(seeds),
                "similarity_min": min(row["top1_image_similarity"] for row in group),
                "similarity_max": max(row["top1_image_similarity"] for row in group),
                "similarity_mean": float(np.mean([row["top1_image_similarity"] for row in group])),
                "delta_token_f1_mean": float(np.mean([row["delta_token_f1"] for row in group])),
                "positive_delta_fraction": float(np.mean([row["delta_token_f1"] > 0 for row in group]))})
        queries.extend(regime_rows)
    return queries, quartiles, summaries


def _similarity_utility_outputs(report, runs, cfg):
    queries, quartiles, summaries = _similarity_utility_data(runs, cfg)
    _csv(report / "similarity_utility_queries.csv", queries,
         ["regime", "id", "n_fixed_training_runs", "top1_image_similarity", "delta_token_f1"])
    _csv(report / "similarity_utility_quartiles.csv", quartiles,
         ["regime", "similarity_quartile", "n_images", "n_fixed_training_runs", "similarity_min",
          "similarity_max", "similarity_mean", "delta_token_f1_mean", "positive_delta_fraction"])
    save_json(report / "similarity_utility.json", {"summaries": summaries,
        "protocol": "Main random n128 language-epochs cohort only. Average available fixed-seed per-query differences, then summarize images. Equal-count quartiles break similarity ties by ID. Descriptive Pearson correlation; no p-value, causal claim, test-set threshold/model selection, or retrieval Recall claim."})
    if queries:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(cfg["regimes"]), figsize=(12, 3.8), sharey=True, squeeze=False)
        colors = {"plain": "#64748b", "retrieval": "#2563eb", "noise": "#0d9488"}
        for ax, regime in zip(axes[0], cfg["regimes"]):
            group = [row for row in queries if row["regime"] == regime]
            summary = next((row for row in summaries if row["regime"] == regime), None)
            if group:
                ax.scatter([row["top1_image_similarity"] for row in group],
                           [row["delta_token_f1"] for row in group], s=15, alpha=.55, color=colors[regime])
                ax.set_title("%s: n=%d, fixed runs=%d\nr=%s (descriptive)" % (
                    regime, len(group), summary["n_fixed_training_runs"], _fmt(summary["pearson_r"], 3)))
            else:
                ax.set_title(regime + ": pending paired runs")
            ax.axhline(0, color="#94a3b8", linewidth=.8)
            ax.set_xlabel("Top-1 image context similarity")
            ax.grid(alpha=.15)
        axes[0][0].set_ylabel("Token F1(image context) - Token F1(none)")
        fig.suptitle("Similarity and downstream utility; association only", fontsize=12)
        fig.tight_layout()
        fig.savefig(report / "similarity_utility.png", dpi=160)
        plt.close(fig)
    return quartiles, summaries, bool(queries)


def _benchmark_lines(out, report):
    completed = []
    for path in sorted((out / "benchmarks").glob("*/*_metrics.json")):
        payload = read_json(path)
        if payload.get("status") != "completed":
            continue
        task = payload.get("task")
        predictions_path = path.parent / (task + "_predictions.jsonl")
        rows = read_rows(predictions_path)
        if len(rows) != payload["n"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Incomplete external benchmark summary: " + str(path))
        completed.append(payload)
        save_json(report / "benchmark_summaries" / (path.parent.name + "_" + path.name), payload)
    if not completed:
        return ["尚无已完成的 POPE/SugarCrepe/跨来源描述结果；这些项目仍为 pending，不能据生成描述分数推断其表现。"]
    lines = ["以下均为固定的小规模外部诊断子集，不代表全量基准。完整来源、适配器哈希及解析协议保留在 `benchmark_summaries/`。",
             "", "### POPE 物体存在性", "",
             "|模型|负例变体|问题数|图像数|Accuracy|正类 F1|无效回答率|", "|---|---|---:|---:|---:|---:|---:|"]
    for item in completed:
        if item["task"] != "pope":
            continue
        for variant, values in sorted(item["variants"].items()):
            lines.append("|%s|%s|%s|%s|%s|%s|%s|" % (item["model"], variant, values["n"], values["n_images"],
                         _fmt(values["accuracy"]), _fmt(values["f1"]), _fmt(values["invalid_rate"])))
    lines += ["", "每个变体的 yes/no 标签平衡；三个变体共享图片且含多个相关问题，不能把问题总数当作独立图像样本量。无法解析的回答保留并计入 Accuracy 错误。", "",
              "### SugarCrepe 细粒度描述选择", "",
              "|模型|类型|样本数|Qwen Accuracy|CLIP Accuracy|Qwen 无效率|", "|---|---|---:|---:|---:|---:|"]
    for item in completed:
        if item["task"] != "sugarcrepe":
            continue
        for category, values in sorted(item["types"].items()):
            lines.append("|%s|%s|%s|%s|%s|%s|" % (item["model"], category, values["n"],
                         _fmt(values["accuracy"]), _fmt(values["clip_accuracy"]), _fmt(values["invalid_rate"])))
    lines += ["", "正例出现在 A/B 的次数按类型确定性平衡；Qwen 必须按固定首字母协议回答，其他输出计错。CLIP 按正负描述相似度直接比较，精确平分计错。", "",
              "### 跨来源图像描述诊断", "",
              "|模型|图像数|Token-F1|ROUGE-L|CIDEr-D变体|输出词数|", "|---|---:|---:|---:|---:|---:|"]
    for item in completed:
        if item["task"] == "transfer":
            lines.append("|%s|%s|%s|%s|%s|%s|" % (item["model"], item["n_images"], _fmt(item["token_f1"]),
                         _fmt(item["rouge_l"]), _fmt(item["cider_d_fixed_df"]), _fmt(item["words"])))
    lines += ["", "跨来源图像由 SugarCrepe 子集决定，参考为其官方正例描述，作者身份未独立核验；该集合受困难负例基准筛选条件影响，不是具有代表性的完整 COCO 描述测试集。该实验也不构成人工盲评。"]
    return lines


def _figures(cfg, report, runs, groups, out):
    if not runs:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"plain": "#64748b", "retrieval": "#2563eb", "noise": "#0d9488", "zero_shot": "#d97706"}
    core = [g for g in groups if g["split"] == "test" and g["bank"] == "full" and not g["shuffled"]
            and g["sampling"] == "random" and g["count"] == cfg["core_train_images"]
            and g["scope"] == "language" and g["budget"] == "epochs"
            and g["mode"] in cfg["evaluation_modes"]]
    created = []
    if core:
        modes = cfg["evaluation_modes"]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(modes))
        regimes = cfg["regimes"]
        width = .8 / len(regimes)
        for j, regime in enumerate(regimes):
            values, errors, positions = [], [], []
            for i, mode in enumerate(modes):
                group = next((g for g in core if g["regime"] == regime and g["mode"] == mode), None)
                if group:
                    values.append(group["token_f1_mean"])
                    errors.append(group["token_f1_seed_sd"] or 0.0)
                    positions.append(x[i] + (j - (len(regimes) - 1) / 2) * width)
            if positions:
                bars = ax.bar(positions, values, width, yerr=errors, capsize=4, color=colors[regime], label=regime)
                ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=8)
        ax.set(xticks=x, xticklabels=modes, ylim=(0, 1), ylabel="Token F1 (training-seed mean ± SD)",
               title="Confirmation set: available completed runs")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend()
        fig.tight_layout()
        name = "comparison_seed_mean_sd.png"
        fig.savefig(report / name, dpi=160)
        plt.close(fig)
        created.append(name)
    selected = [r for r in runs if r["split"] == "test" and not r["shuffled"] and r["bank"] == "full"
                and r["n"] == cfg["confirmation_images"]
                and (r["regime"] == "zero_shot" or
                     (r["sampling"] == "random" and r["count"] == cfg["core_train_images"]
                      and r["scope"] == "language" and r["budget"] == "epochs"))]
    if selected:
        shared_ids = {row["id"] for row in selected[0]["rows"]}
        if any({row["id"] for row in run["rows"]} != shared_ids for run in selected):
            raise ValueError("Length-quality plot requires the identical confirmation image set")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for regime in ("zero_shot", *cfg["regimes"]):
            values = [r for r in selected if r["regime"] == regime]
            if values:
                seeds = {r["seed"] for r in values}
                label = "zero-shot (fixed model)" if regime == "zero_shot" else "%s (%d training seed%s)" % (regime, len(seeds), "s" if len(seeds) != 1 else "")
                ax.scatter([r["words"] for r in values], [r["token_f1"] for r in values],
                           label=label, color=colors[regime], alpha=.75)
        ax.set(xlabel="Generated words (run mean)", ylabel="Token F1",
               title="Length versus lexical quality; no causal claim\nSame %d confirmation images; core configurations only" % cfg["confirmation_images"])
        ax.grid(alpha=.15)
        ax.legend()
        fig.tight_layout()
        name = "length_quality.png"
        fig.savefig(report / name, dpi=160)
        plt.close(fig)
        created.append(name)
    histories = []
    for path in sorted((out / "models").glob("*/history.json")):
        meta = _meta(out, path.parent.name)
        if meta["sampling"] == "random" and meta["count"] == cfg["core_train_images"] and meta["scope"] == "language" and meta["budget"] == "epochs":
            histories.append((path, meta))
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for path, meta in histories:
            history = read_json(path)
            label = "%s / seed %s" % (meta["regime"], meta["seed"])
            for axis, key in zip(axes, ("loss", "eval_loss")):
                points = [(r.get("step", r.get("epoch")), r[key]) for r in history if key in r]
                if points:
                    axis.plot(*zip(*points), color=colors[meta["regime"]], alpha=.6, marker="." if key == "eval_loss" else None, label=label)
                axis.set(xlabel="Optimizer step", ylabel="Loss", title="Training" if key == "loss" else "Validation (same no-context objective)")
                axis.grid(alpha=.15)
        axes[1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        name = "training_curves.png"
        fig.savefig(report / name, dpi=160)
        plt.close(fig)
        created.append(name)
    return created


def run(cfg):
    data, out, report = paths(cfg)
    report.mkdir(parents=True, exist_ok=True)
    runs = _read_runs(out)
    groups = _seed_groups(runs)
    effects = _effects(runs, cfg)
    utility_quartiles, utility_summaries, utility_figure = _similarity_utility_outputs(report, runs, cfg)
    columns = ["model", "regime", "seed", *COHORT, "split", "mode", "bank", "shuffled", "n", *METRICS,
               "retrieval_used_images", "seconds_this_invocation", "peak_gpu_gib"]
    _csv(report / "experiments.csv", runs, columns)
    group_columns = [*GROUP, "n_training_seeds", "n_runs", "n_images", "seeds"]
    group_columns += [metric + suffix for metric in METRICS for suffix in ("_mean", "_seed_sd", "_n_defined_runs")]
    _csv(report / "seed_summary.csv", groups, group_columns)
    save_json(report / "effects.json", effects)
    figures = _figures(cfg, report, runs, groups, out)
    if utility_figure:
        figures.append("similarity_utility.png")
    expected = []
    for regime in cfg["regimes"]:
        for seed in cfg["seeds"]:
            model = run_name(regime, seed, "random", cfg["core_train_images"], "language", "epochs")
            for mode in cfg["evaluation_modes"]:
                matched = [r for r in runs if r["model"] == model and r["split"] == "test" and r["mode"] == mode and r["bank"] == "full" and not r["shuffled"]]
                expected.append({"model": model, "mode": mode, "status": "completed" if matched else "pending"})
    completed_core = sum(row["status"] == "completed" for row in expected)
    source = read_json(data / "source.json") if (data / "source.json").exists() else None
    status = {"study_id": cfg.get("study_id"), "completed_evaluations": len(runs),
              "completed_core_conditions": completed_core, "expected_core_conditions": len(expected),
              "core_status": "completed" if completed_core == len(expected) else "partial",
              "human_evaluation": "pending_external_human_annotations_not_inferred",
              "completed_adapters": len(list((out / "models").glob("*/completed.json"))),
              "core_matrix": expected, "data_source": source}
    save_json(report / "progress.json", status)
    lines = ["# 多模态检索干扰研究：实际实验结果", "",
             "本报告只汇总已经写入完整预测与指标的运行。空白或 pending 不代表零分。", "",
             "- 主对照完成：%d/%d；已完成评估运行：%d；已保存 adapter：%d。" % (completed_core, len(expected), len(runs), status["completed_adapters"]),
             "- 人工相关性核验与盲评：pending。本报告没有把自动评分当作人工判断。",
             "- 预先固定的协议见 [protocol.md](protocol.md)；逐运行数据见 [experiments.csv](experiments.csv)。", "",
             "## 完成的确认集与预先固定诊断子集对比", "",
             "数值为可用训练种子的均值 ± 样本标准差；仅一个种子时不估计种子方差。Zero-shot 为固定基础模型，不计作独立训练种子。", "",
             "|训练方案|采样/图像数/预算|上下文|Bank|图像数|种子数|Token-F1|CIDEr-D变体|CLIPScore|输出词数|检索使用率|",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    confirmation_groups = [g for g in groups if g["split"] == "test" and not g["shuffled"]]
    for g in confirmation_groups:
        mean_sd = lambda metric: _fmt(g.get(metric + "_mean")) + (" ± " + _fmt(g[metric + "_seed_sd"]) if g.get(metric + "_seed_sd") is not None else "")
        lines.append("|%s|%s/%s/%s/%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|" % (
            g["regime"], g["sampling"], g["count"], g["scope"], g["budget"], g["mode"], g["bank"],
            g["n_images"], g["n_training_seeds"], mean_sd("token_f1"), mean_sd("cider_d_fixed_df"),
            mean_sd("clipscore"), mean_sd("words"), mean_sd("retrieval_use_fraction")))
    if not confirmation_groups:
        lines += ["", "尚无完整的确认集评估；不能据此解释方法优劣。"]
    lines += ["", "## 检索使用、复制代理与输入预算", "",
              "全体图像的复制代理在未使用检索时取零，因此自适应方法减少检索次数本身就可能降低其均值。下表同时给出仅在实际使用上下文的图像上计算的复制率；没有任何检索使用时该条件统计为未定义（—），不会填为零。条件均值仍是词汇代理，且不同策略选中的图像可能不同，不作事实正确性解释。", "",
              "|训练方案|上下文|Bank|图像数|检索使用率|全体复制代理|使用时复制代理|有定义运行数|输入 tokens|上下文 tokens|",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in confirmation_groups:
        lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|" % (
            g["regime"], g["mode"], g["bank"], g["n_images"], _fmt(g.get("retrieval_use_fraction_mean")),
            _fmt(g.get("context_copy_rate_mean")), _fmt(g.get("context_copy_rate_when_used_mean")),
            g.get("context_copy_rate_when_used_n_defined_runs", 0),
            _fmt(g.get("input_tokens_mean"), 1), _fmt(g.get("context_tokens_mean"), 1)))
    lines += ["", "## 配对效应与交互", "",
              "所有差值均按名称右侧减左侧的明确表达式计算，并要求图像 ID 完全一致。只有预先登记的 text/rerank/conflict/partial 诊断，才将主条件限定至完全相同、顺序一致的确认集前缀；不对任意缺失样本取交集。95% 图像 bootstrap 区间是在可用固定训练运行取平均后重采样图像，不能解释成训练随机种子的不确定性。种子 SD 单独列出；不作多重比较校正。", "",
              "|效应（表达式即差值方向）|配置/Bank|分析子集|图像数|种子数|Token-F1 Δ|种子 SD|95% 图像区间|",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for e in effects:
        if e["metric"] != "token_f1":
            continue
        c = e["cohort"]
        lines.append("|%s|%s/%s/%s/%s/%s|%s|%s|%s|%s|%s|[%s, %s]|" % (
            e["effect"], c["sampling"], c["count"], c["scope"], c["budget"], c["bank"],
            c.get("analysis_subset", "confirmation"), e["n_images"], e["n_runs"],
            _fmt(e["mean_delta"]), _fmt(e["seed_sd"]), _fmt(e["image_ci_low"]), _fmt(e["image_ci_high"])))
    lines += ["", "其他指标的效应和逐种子差值见 [effects.json](effects.json)。交互项 `(retrieval_image-retrieval_none)-(plain_image-plain_none)` 描述训练中加入检索后，推理检索效应发生的变化；它本身不能证明唯一的心理或神经机制。", "",
              "## 相似度与下游效用的描述性关联", "",
              "仅使用主 128 图像随机采样配置、完整确认集、full 检索库的 image 与 none 配对输出。每张图片先对可用固定训练种子的分数差取平均，再分析 top-1 图像上下文相似度与 Token-F1 变化的关联。相似度不是相关性真值或正确概率；Pearson r 仅为描述性统计，不计算显著性、不作因果解释，也不据此调整检索阈值或选模型。", "",
              "候选图库上的 Recall@K 衡量另一项图文匹配任务；本分析没有训练图像检索的人工相关性标签，不能把相似度与效用的关联改写成 Recall 提升导致生成提升。", "",
              "|训练方案|图像数|固定训练运行数|Pearson r（描述性）|", "|---|---:|---:|---:|"]
    for item in utility_summaries:
        lines.append("|%s|%s|%s|%s|" % (item["regime"], item["n_images"], item["n_fixed_training_runs"], _fmt(item["pearson_r"])))
    lines += ["", "|训练方案|相似度四分组|图像数|相似度均值|Token-F1 Δ 均值|正向 Δ 比例|",
              "|---|---:|---:|---:|---:|---:|"]
    for item in utility_quartiles:
        lines.append("|%s|%s|%s|%s|%s|%s|" % (item["regime"], item["similarity_quartile"], item["n_images"],
                     _fmt(item["similarity_mean"]), _fmt(item["delta_token_f1_mean"]), _fmt(item["positive_delta_fraction"])))
    lines += ["", "原始图像级摘要见 [similarity_utility_queries.csv](similarity_utility_queries.csv)，分组见 [similarity_utility_quartiles.csv](similarity_utility_quartiles.csv)。", "",
              "## 验证集选择的检索策略", "", "|模型|阈值|验证指标|验证使用率|选择集合|", "|---|---:|---:|---:|---|"]
    for path in sorted((out / "gates").glob("*.json")):
        gate = read_json(path)
        if gate.get("selection_split") != "validation":
            raise ValueError("Gate is not explicitly validation-selected: " + str(path))
        lines.append("|%s|%s|%s|%s|%s|" % (path.stem, _fmt(gate.get("threshold")), _fmt(gate.get("validation_token_f1")), _fmt(gate.get("use_fraction")), gate["selection_split"]))
    lines += ["", "## 外部基准与人工评估", ""]
    lines += _benchmark_lines(out, report)
    lines += ["", "人工核验相关上下文和人工盲评需要独立的人类标注。只有真实标注经完整性核验后才能报告事实正确性或人工偏好；当前报告不生成此类分数。", "",
              "## 图像打乱对照", "", "|模型|上下文|Token-F1|原图 CLIPScore|输入图 CLIPScore|", "|---|---|---:|---:|---:|"]
    for r in runs:
        if r["shuffled"] and r["split"] == "test":
            lines.append("|%s|%s|%s|%s|%s|" % (r["model"], r["mode"], _fmt(r["token_f1"]), _fmt(r.get("clipscore")), _fmt(r.get("input_clipscore"))))
    lines += ["", "打乱条件保持参考答案与检索上下文不变，因此应分别报告相对原图与实际输入图片的一致性；仅凭打乱后仍有非零词汇分数，不能证明模型完全没有使用图像。", "",
              "## 未完成的主对照", "", "|模型|上下文|状态|", "|---|---|---|"]
    for row in expected:
        if row["status"] != "completed":
            lines.append("|%s|%s|pending|" % (row["model"], row["mode"]))
    if completed_core == len(expected):
        lines.append("|主对照全部具备完整指标|—|completed|")
    lines += ["", "## 指标和解释边界", "",
              "- `cider_fixed_df` / `cider_d_fixed_df` 分别为 CIDEr_fixed_df_regex / CIDErD_fixed_df_regex：使用英文 regex 分词和固定训练参考 DF；与 PTB 分词、其他 DF 协议的 COCO 排行榜数值不直接可比。",
              "- CLIPScore 使用同一 CLIP 表征体系，可能偏向该检索器；它是辅助证据，不能独立证明事实正确性。",
              "- `context_copy_rate` 是输出内容词在上下文中出现的比例；`reference_detail_recall` 是最佳参考描述内容词召回。两者均为词汇代理，不是幻觉率或完整视觉细节覆盖。",
              "- 随机、部分相关和合成冲突是受控文本干预，不代表经过人工核验的真假标签。",
              "- 检索阈值是验证集选择的相似度决策规则，不是校准后的正确概率。",
              "- 多种子均值和 SD 不替代独立数据分布验证；数据比例、更新步数、采样策略、检索库大小与可训练参数量必须分别标记。",
              "- 本报告不因文件存在就宣称六个研究方向全部完成，未产生的外部基准和人工评估仍待完成。"]
    for name in figures:
        lines += ["", "![%s](%s)" % (name.removesuffix(".png"), name)]
    (report / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Research report: %d completed evaluations; core %d/%d" % (len(runs), completed_core, len(expected)), flush=True)
    return status
