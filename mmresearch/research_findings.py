"""Answer the study questions from completed artifacts without selecting models."""
import numpy as np
from .common import read_json, read_rows, save_json
from .research_common import paths, run_name
from .research_metrics import paired_effect


def _mean_sd(values):
    return {"mean": float(np.mean(values)), "training_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_training_runs": len(values)}


def run(cfg):
    _, out, report = paths(cfg)
    evaluation = out / "evaluations"
    summary, budgets, shuffled, adoption = [], [], [], []
    metrics = ("token_f1", "rouge_l", "cider_d_fixed_df", "clipscore", "words")
    baseline = read_json(evaluation / "zero_shot/test_none_full/metrics.json")
    for regime in cfg["regimes"]:
        for mode in cfg["evaluation_modes"]:
            runs = [read_json(evaluation / run_name(regime, seed, count=cfg["core_train_images"]) /
                              ("test_%s_full" % mode) / "metrics.json") for seed in cfg["seeds"]]
            summary.append({"regime": regime, "mode": mode,
                            **{metric: _mean_sd([r[metric] for r in runs]) for metric in metrics},
                            "retrieval_use_fraction": _mean_sd([r["retrieval_use_fraction"] for r in runs])})
        name = run_name(regime, cfg["seeds"][0], count=cfg["core_train_images"])
        for mode in ("none", "image"):
            normal = read_rows(evaluation / name / ("test_%s_full" % mode) / "predictions.jsonl")
            changed = read_rows(evaluation / name / ("test_%s_full_shuffled" % mode) / "predictions.jsonl")
            if [r["id"] for r in normal[:cfg["diagnostic_images"]]] != [r["id"] for r in changed]:
                raise ValueError("Shuffle comparison is not the fixed diagnostic prefix")
            shuffled.append({"regime": regime, "mode": mode, "training_seed": cfg["seeds"][0],
                "effect_direction": "shuffled-minus-correct-image",
                "token_f1": paired_effect(normal[:len(changed)], changed, "token_f1"),
                "target_clipscore": paired_effect(normal[:len(changed)], changed, "clipscore"),
                "actual_input_clipscore": paired_effect(normal[:len(changed)], changed, "input_clipscore")})
        from .research_perturbation import compare_adoption
        original = read_rows(evaluation / name / "test_image_full/predictions.jsonl")[:cfg["diagnostic_images"]]
        conflict = read_rows(evaluation / name / "test_conflict_full/predictions.jsonl")
        adoption.append({"regime": regime, "training_seed": cfg["seeds"][0], **compare_adoption(original, conflict)})
    for path in sorted((out / "models").glob("*/training.json")):
        training = read_json(path)
        identity = training["identity"]
        score_path = evaluation / path.parent.name / "test_none_full/metrics.json"
        if not score_path.exists():
            continue
        score = read_json(score_path)
        budgets.append({"model": path.parent.name, **{k: identity[k] for k in ("regime", "seed", "sampling", "count", "scope", "budget")},
            **{k: training[k] for k in ("optimizer_steps", "unique_supervised_images", "supervised_microbatches", "trainable_parameters")},
            **{k: score[k] for k in metrics}})
    effects = read_json(report / "effects.json")
    human = read_json(report / "human_review_status.json")
    data = {"baseline": {k: baseline[k] for k in metrics}, "main_conditions": summary,
            "paired_effects": effects, "shuffled_image_effects": shuffled, "synthetic_term_adoption": adoption, "training_budget_audit": budgets,
            "human_evaluation": human}
    save_json(report / "findings.json", data)
    lines = ["# 研究问题、结果与证据边界", "", "本页只汇总已完成的固定实验，不根据确认集挑选新的门控阈值或隐藏负结果。完整结果见 [results.md](results.md)，协议与修订见 [protocol.md](protocol.md)。", "",
             "## 1. 检索与训练方式的交互", "",
             "主实验使用相同的 %d 张确认图片、%d 个训练随机种子；均值 ± SD 表示训练种子的变化，不是置信区间。" % (cfg["confirmation_images"], len(cfg["seeds"])), "",
             "|训练方式|推理上下文|Token-F1 均值 ± SD|ROUGE-L|CLIPScore|平均词数|检索使用率|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        f = r["token_f1"]
        lines.append("|%s|%s|%.4f ± %.4f|%.4f|%.4f|%.2f|%.3f|" %
                     (r["regime"], r["mode"], f["mean"], f["training_seed_sd"], r["rouge_l"]["mean"],
                      r["clipscore"]["mean"], r["words"]["mean"], r["retrieval_use_fraction"]["mean"]))
    lines += ["", "固定 zero-shot 无检索基线：Token-F1 %.4f，ROUGE-L %.4f，CLIPScore %.4f，平均 %.2f 词。" %
              (baseline["token_f1"], baseline["rouge_l"], baseline["clipscore"], baseline["words"]), "",
              "不能只比较训练方案的绝对分数来解释格式适应：应同时考察各方案的 image−none 配对效应，以及两方案之间该效应的差。检索训练还可读取标注记忆库，因此交互证据不能唯一归因于格式变化。", "",
              "## 2. 自适应使用与抗噪声训练", "", "所有门控阈值仅使用前 64 张验证图片；checkpoint 损失选模使用另外 64 张验证图片。完整候选阈值与选择记录在 gates 对应的归档摘要中。", "",
              "|配对效应（名称中 A−B 的方向）|Token-F1 差|固定训练运行下的图像 bootstrap 95% 区间|训练种子差值 SD|", "|---|---:|---:|---:|"]
    selected = [e for e in effects if e["metric"] == "token_f1" and e["n_images"] == cfg["confirmation_images"]
                and e["n_runs"] == len(cfg["seeds"]) and e["cohort"].get("count") == cfg["core_train_images"]
                and e["cohort"].get("scope") == "language" and e["cohort"].get("budget") == "epochs"
                and e["cohort"].get("sampling") == "random" and e["cohort"].get("bank") == "full"]
    for e in selected:
        lines.append("|%s|%.4f|[%.4f, %.4f]|%.4f|" %
                     (e["effect"], e["mean_delta"], e["image_ci_low"], e["image_ci_high"], e["seed_sd"]))
    lines += ["", "上述图像区间没有覆盖重新训练的全部随机性，也没有作多重比较校正。自适应方法可能通过停止使用检索回到无检索基线；不能把它相对固定 Top-K 的改善当成外部信息必然有益。总体复制率与实际使用检索时的条件复制率必须分别查看。", "",
              "## 3. 是否依赖图像，以及是否看得更准", "",
              "|训练方式|上下文|打乱图片后的 Token-F1 差|95% 图像区间|", "|---|---|---:|---:|"]
    for r in shuffled:
        e = r["token_f1"]
        lines.append("|%s|%s|%.4f|[%.4f, %.4f]|" % (r["regime"], r["mode"], e["delta"], e["ci_low"], e["ci_high"]))
    lines += ["", "补充的合成词采用分析：仅统计长度匹配后仍暴露、且原上下文没有的替换词；该描述性分析不是最初的主指标。替换词没有被人工确认对图片为假，因此不能称为错误物体复制率或幻觉率。", "",
              "|训练方式|有效干预图像|正常上下文的采用率|合成上下文的采用率|新采用次数|", "|---|---:|---:|---:|---:|"]
    for r in adoption:
        if r["n_eligible_images"]:
            lines.append("|%s|%d|%.3f|%.3f|%d|" % (r["regime"], r["n_eligible_images"], r["normal_adoption_rate"], r["conflict_adoption_rate"], r["newly_adopts"]))
    lines += ["", "图片打乱只使用固定的前 64 张确认图像，并保留原参考答案和原检索上下文。它检查输出对图片的依赖，不能单独证明原始输出没有幻觉。目标图片 CLIPScore 和实际输入图片 CLIPScore 分开保存，避免把对错误输入的忠实描述误标成幻觉。", "",
              "Token-F1/CIDEr 的提高可能与输出风格和长度变化有关；CLIPScore 来自同一个检索编码器，只作辅助证据。POPE 检查官方标注的物体存在性问题，SugarCrepe 检查成对描述选择；它们均为固定子集且只使用第一个训练种子的 adapter，不代表全量或多种子外部基准。", "",
              "## 4. 数据、更新步数与可训练参数的实际预算", "",
              "|模型|候选/选中图像|实际监督图像|优化步数|可训练参数|无检索 Token-F1|", "|---|---:|---:|---:|---:|---:|"]
    for r in budgets:
        lines.append("|%s|%d|%d|%d|%d|%.4f|" % (r["model"], r["count"], r["unique_supervised_images"],
                     r["optimizer_steps"], r["trainable_parameters"], r["token_f1"]))
    lines += ["", "32/128 图像的固定轮数与固定更新步数比较分开呈现；1024 图像等步数组最多只会访问其中部分样本，实际监督数量以上表为准。语义覆盖只读 CLIP 特征与固定词表，未读取整个人工描述池进行选样。随机性分析固定了数据子集，不涵盖重新抽取子集的全部方差。连接模块扩展的参数更多，不是等容量的位置因果对照。", "",
              "## 5. 检索规模与组合语义", "",
              "512 图像/2560 描述图库的 Recall 与训练库上下文的生成效用是不同任务。相似度—效用分析是描述性关联，不能证明 Recall 改善导致生成改善。词数匹配的库规模补充实验见 [bank_size_effects.md](bank_size_effects.md)，SugarCrepe 分类型结果见完整报告。", "",
              "## 6. 人工评估状态与尚不能下的结论", "",
              "人工描述评分已完成 %d/%d；相关性核验已完成 %d/%d。" %
              (human["caption_items_scored"], human["caption_items_ready"], human["relevance_items_scored"], human["relevance_items_ready"]), "",
              "用户已选择先继续自动实验。没有人工标注时，不把词汇覆盖率、上下文复制率或 CLIPScore 当成人工事实准确率，也不报告人工核验的高相关上下文条件。材料和导入工具已经交付，可在后续补充。跨来源描述集仅由 SugarCrepe 子集决定，不能概括为完整 COCO 或所有领域的泛化性能。"]
    (report / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Research question summary written", flush=True)
