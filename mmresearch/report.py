import csv
from .common import ROOT, read_json, read_rows, save_json
from .metrics import token_f1, paired_bootstrap


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    source = ROOT / "data/source.json"
    if source.exists():
        save_json(report / "data_manifest.json", {"source": read_json(source),
            "image_ids": {split: [r["id"] for r in read_rows(ROOT / "data" / (split + ".jsonl"))]
                          for split in ("train", "validation", "test")}})
    for path in (ROOT / "outputs").glob("sft_*/provenance.json"):
        save_json(report / (path.parent.name + "_provenance.json"), read_json(path))
    retrieval_provenance = ROOT / "outputs/retrieval_provenance.json"
    if retrieval_provenance.exists():
        save_json(report / "retrieval_provenance.json", read_json(retrieval_provenance))
    metrics = [read_json(p) for p in sorted((ROOT / "outputs/evaluations").glob("*/metrics.json"))]
    runs = {}
    for path in sorted((ROOT / "outputs/evaluations").glob("*/metrics.json")):
        m = read_json(path)
        runs[(m["model"], m["prompt"], m["rag"], m["split"])] = read_rows(path.parent / "predictions.jsonl")
    columns = ["model", "prompt", "rag", "split", "n", "token_f1", "rouge_l", "seconds"]
    with (report / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    lines = ["# 实验结果（仅包含实际完成的运行）", "", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["|" + "|".join(str(m.get(c, "")) for c in columns) + "|" for m in metrics]
    if not metrics:
        lines += ["", "尚无完成的 Qwen-VL 评估，不填入模拟指标。"]
    comparisons = []
    for key, rows in runs.items():
        baseline_key = ("zero_shot", key[1], key[2], key[3])
        if key[0] == "zero_shot" or baseline_key not in runs:
            continue
        baseline = {r["id"]: r for r in runs[baseline_key]}
        if set(baseline) != {r["id"] for r in rows}:
            raise ValueError("Cannot compare mismatched test images")
        a = [max(token_f1(baseline[r["id"]]["prediction"], ref) for ref in r["references"]) for r in rows]
        b = [max(token_f1(r["prediction"], ref) for ref in r["references"]) for r in rows]
        comparisons.append({"model": key[0], "prompt": key[1], "rag": key[2], "split": key[3], **paired_bootstrap(a, b)})
    if comparisons:
        save_json(report / "paired_comparisons.json", comparisons)
        lines += ["", "## 相对相同提示词/RAG设置的 zero-shot 差异", "", "图像级配对 bootstrap，2000 次，95% 百分位区间；不作多重比较校正，不代表跨训练种子的变异。", "", "|Model|Prompt|RAG|Token-F1 Δ|95% CI|", "|---|---|---|---|---|"]
        lines += ["|%s|%s|%s|%.4f|[%.4f, %.4f]|" % (c["model"], c["prompt"], c["rag"], c["delta"], c["ci_low"], c["ci_high"]) for c in comparisons]
    baseline_key = ("zero_shot", "concise", False, "test")
    tuned_key = ("sft_1", "concise", False, "test")
    if baseline_key in runs and tuned_key in runs:
        baseline = {r["id"]: r for r in runs[baseline_key]}
        examples = []
        for row in runs[tuned_key]:
            before = baseline[row["id"]]["prediction"]
            delta = max(token_f1(row["prediction"], ref) for ref in row["references"]) - max(token_f1(before, ref) for ref in row["references"])
            examples.append({"id": row["id"], "delta_token_f1": delta, "baseline": before,
                             "sft": row["prediction"], "references": row["references"]})
        examples.sort(key=lambda r: r["delta_token_f1"])
        save_json(report / "prediction_examples.json", examples[:3] + examples[-3:])
        lines += ["", "## 输出变化样例", "", "以下按 Token-F1 变化选取两端样本，仅解释词汇指标变化，不将参考答案中未出现的内容自动判为幻觉。"]
        for example in examples[:2] + examples[-2:]:
            lines += ["", "- 图像 `%s`（Δ %.4f）" % (example["id"][:12], example["delta_token_f1"]),
                      "  - zero-shot: " + example["baseline"], "  - SFT: " + example["sft"],
                      "  - 参考之一: " + example["references"][0]]
    retrieval = ROOT / "outputs/retrieval.json"
    if retrieval.exists():
        r = read_json(retrieval)
        lines += ["", "## CLIP retrieval", "", "```json", __import__("json").dumps(r, indent=2), "```"]
    for path in sorted((ROOT / "outputs").glob("sft_*/history.json")):
        history = read_json(path)
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in (("loss", "train"), ("eval_loss", "validation")):
            points = [(r["epoch"], r[key]) for r in history if key in r]
            if points:
                ax.plot(*zip(*points), marker="o", label=label)
        ax.set(xlabel="Epoch", ylabel="Cross entropy loss", title=path.parent.name)
        ax.legend()
        fig.tight_layout()
        name = path.parent.name + "_loss.png"
        fig.savefig(report / name, dpi=160)
        plt.close(fig)
        lines += ["", "![Training curve](%s)" % name]
    if metrics:
        fig, ax = plt.subplots(figsize=(10, max(3, len(metrics) * 0.4)))
        labels = ["%s / %s / %s" % (m["model"], m["prompt"], "RAG" if m["rag"] else "plain") for m in metrics]
        bars = ax.barh(labels, [m["token_f1"] for m in metrics],
                       color=[{"zero_shot": "#64748b", "sft_0.25": "#3b82f6", "sft_1": "#0d9488"}[m["model"]] for m in metrics])
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.15)
        ax.set(xlim=(0, 1), xlabel="Best-reference token F1", title="Caption experiments")
        fig.tight_layout()
        fig.savefig(report / "comparison.png", dpi=160)
        plt.close(fig)
        lines += ["", "![Comparison](comparison.png)"]
    (report / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
