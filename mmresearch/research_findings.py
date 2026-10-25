"""Answer the study questions from completed artifacts without selecting models."""
import re
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
    for mode in ("none", "image"):
        shifted_path = evaluation / "zero_shot" / ("test_%s_full_shuffled" % mode) / "predictions.jsonl"
        if shifted_path.exists():
            normal = read_rows(evaluation / "zero_shot" / ("test_%s_full" % mode) / "predictions.jsonl")[:cfg["diagnostic_images"]]
            changed = read_rows(shifted_path)
            if [r["id"] for r in normal] != [r["id"] for r in changed]:
                raise ValueError("Baseline shuffle does not match the fixed diagnostic prefix")
            shuffled.append({"regime": "zero_shot", "mode": mode, "training_seed": None,
                "effect_direction": "shuffled-minus-correct-image",
                "token_f1": paired_effect(normal, changed, "token_f1"),
                "target_clipscore": paired_effect(normal, changed, "clipscore"),
                "actual_input_clipscore": paired_effect(normal, changed, "input_clipscore")})
    for path in sorted((out / "models").glob("*/training.json")):
        training = read_json(path)
        identity = training["identity"]
        score_path = evaluation / path.parent.name / "test_none_full/metrics.json"
        if not score_path.exists():
            continue
        score = read_json(score_path)
        selected_step = re.search(r"checkpoint-(\d+)$", str(training.get("best_checkpoint", "")))
        budgets.append({"model": path.parent.name, **{k: identity[k] for k in ("regime", "seed", "sampling", "count", "scope", "budget")},
            **{k: training[k] for k in ("optimizer_steps", "unique_supervised_images", "supervised_microbatches", "trainable_parameters")},
            "selected_checkpoint_step": int(selected_step.group(1)) if selected_step else None,
            **{k: score[k] for k in metrics}})
    sampling_summary = []
    for strategy in ("random", "diversity", "coverage"):
        selected = [r for r in budgets if r["regime"] == "plain" and r["sampling"] == strategy
                    and r["count"] == cfg["core_train_images"] and r["scope"] == "language" and r["budget"] == "epochs"]
        if {r["seed"] for r in selected} != set(cfg["seeds"]):
            raise ValueError("Sampling comparison is missing planned training seeds")
        if len({r["optimizer_steps"] for r in selected}) != 1 or any(r["unique_supervised_images"] != cfg["core_train_images"] for r in selected):
            raise ValueError("Sampling comparison does not have matched supervision budgets")
        sampling_summary.append({"strategy": strategy, "n_images": cfg["core_train_images"],
            "optimizer_steps": selected[0]["optimizer_steps"], **{key: _mean_sd([r[key] for r in selected]) for key in metrics}})
    budget_summary = []
    for count, budget in ((32, "epochs"), (32, "matched_steps"),
                          (cfg["core_train_images"], "epochs"), (cfg["train_images"], "single_epoch"),
                          (cfg["train_images"], "matched_steps")):
        selected = [r for r in budgets if r["regime"] == "plain" and r["sampling"] == "random"
                    and r["scope"] == "language" and r["count"] == count and r["budget"] == budget]
        if selected:
            budget_summary.append({"candidate_images": count, "budget": budget, "seeds": [r["seed"] for r in selected],
                "actual_supervised_images": sorted({r["unique_supervised_images"] for r in selected}),
                "executed_steps": sorted({r["optimizer_steps"] for r in selected}),
                "selected_checkpoint_steps": sorted({r["selected_checkpoint_step"] for r in selected}),
                **{key: _mean_sd([r[key] for r in selected]) for key in metrics}})
    effects = read_json(report / "effects.json")
    human = read_json(report / "human_review_status.json")
    external = []
    for regime in ("zero_shot", *cfg["regimes"]):
        name = "zero_shot" if regime == "zero_shot" else run_name(regime, cfg["seeds"][0], count=cfg["core_train_images"])
        metric_paths = {task: out / "benchmarks" / name / (task + "_metrics.json") for task in ("pope", "sugarcrepe", "transfer")}
        if all(path.exists() for path in metric_paths.values()):
            results = {task: read_json(path) for task, path in metric_paths.items()}
            external.append({"regime": regime, "model": name, "seed": None if regime == "zero_shot" else cfg["seeds"][0],
                "pope_accuracy": results["pope"]["accuracy"], "pope_f1": results["pope"]["f1"],
                "pope_invalid_rate": results["pope"]["invalid_rate"],
                "sugarcrepe_accuracy": results["sugarcrepe"]["accuracy"],
                "sugarcrepe_invalid_rate": results["sugarcrepe"]["invalid_rate"],
                "transfer_token_f1": results["transfer"]["token_f1"]})
    data = {"baseline": {k: baseline[k] for k in metrics}, "main_conditions": summary,
            "paired_effects": effects, "shuffled_image_effects": shuffled, "synthetic_term_adoption": adoption, "training_budget_audit": budgets,
            "sampling_strategy_comparison": sampling_summary, "data_step_budget_comparison": budget_summary,
            "external_diagnostics": external,
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
    by_condition = {(r["regime"], r["mode"]): r for r in summary}
    plain_none = by_condition[("plain", "none")]
    plain_image = by_condition[("plain", "image")]
    plain_adaptive = by_condition[("plain", "adaptive")]
    lines += ["", "普通 SFT 的无检索 Token-F1 为 %.4f，固定图像检索为 %.4f，平均差为 %.2f 个百分点；自适应相对固定检索恢复 %.2f 个百分点，自适应减无检索的差为 %+.2f 个百分点。这里衡量的是词汇重合，不能据此直接判断视觉事实质量。" %
              (plain_none["token_f1"]["mean"], plain_image["token_f1"]["mean"],
               100 * (plain_image["token_f1"]["mean"] - plain_none["token_f1"]["mean"]),
               100 * (plain_adaptive["token_f1"]["mean"] - plain_image["token_f1"]["mean"]),
               100 * (plain_adaptive["token_f1"]["mean"] - plain_none["token_f1"]["mean"]))]
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
    selected_by_name = {e["effect"]: e for e in selected}
    component_names = ("plain:adaptive-image", "noise_only:noise_image-plain_image", "combined:noise_adaptive-plain_image")
    if all(name in selected_by_name for name in component_names):
        component = [selected_by_name[name]["mean_delta"] for name in component_names]
        lines += ["", "相对普通 SFT＋固定检索，仅自适应、仅抗噪声训练、二者结合的平均差依次为 %+.2f、%+.2f、%+.2f 个百分点。结合组的观察均值没有超过单独自适应组，因此不能把两个组件写成已证实的叠加收益。" % tuple(100 * v for v in component)]
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
              "外部诊断均为无检索推理；POPE 的 360 问共享 20 张图片，SugarCrepe 为 112 对描述，跨来源描述为 108 张图片。下表只列三项任务均已完成的模型。", "",
              "|模型|POPE Accuracy|POPE 正类 F1|SugarCrepe Accuracy|跨来源 Token-F1|", "|---|---:|---:|---:|---:|"]
    for r in external:
        lines.append("|%s|%.4f|%.4f|%.4f|%.4f|" % (r["regime"], r["pope_accuracy"], r["pope_f1"],
                     r["sugarcrepe_accuracy"], r["transfer_token_f1"]))
    lines += ["", "同一外部子集上的模型差异可作描述性比较，不能据单个种子和少量图片宣称通用视觉能力提升。跨来源集合多数图片只有 1 条参考，而 Flickr 确认集通常有 5 条；最佳参考 F1 对参考数量敏感，因此两数据集的绝对分数差不解释为跨域退化幅度。无效回答率、每个 POPE 变体及 SugarCrepe 类型的明细见完整报告。", "",
              "## 4. 数据、更新步数与可训练参数的实际预算", "",
              "相同监督图像数、最大更新步数和生成设置下的选样比较（均为普通 SFT、无检索推理，使用同一验证选模规则；所选 checkpoint 步数见明细）：", "",
              "|选样策略|监督图像|优化步数|Token-F1 均值 ± SD|CLIPScore 均值|平均词数|", "|---|---:|---:|---:|---:|---:|"]
    for r in sampling_summary:
        lines.append("|%s|%d|%d|%.4f ± %.4f|%.4f|%.2f|" %
                     (r["strategy"], r["n_images"], r["optimizer_steps"], r["token_f1"]["mean"],
                      r["token_f1"]["training_seed_sd"], r["clipscore"]["mean"], r["words"]["mean"]))
    lines += ["", "随机选样的图像数与更新预算比较；`epochs` 为两轮，`single_epoch` 为一轮，`matched_steps` 为固定 32 次更新。仅一个种子的千张扩展不估计训练种子 SD。", "",
              "|候选图像|预算规则|种子数|全程实际监督图像|执行步数|所选 checkpoint 步数|Token-F1 均值 ± SD|",
              "|---|---|---:|---|---|---|---:|"]
    for r in budget_summary:
        score = r["token_f1"]
        display = "%.4f" % score["mean"]
        if score["training_seed_sd"] is not None:
            display += " ± %.4f" % score["training_seed_sd"]
        lines.append("|%d|%s|%d|%s|%s|%s|%s|" % (r["candidate_images"], r["budget"], len(r["seeds"]),
            ", ".join(map(str, r["actual_supervised_images"])), ", ".join(map(str, r["executed_steps"])),
            ", ".join(map(str, r["selected_checkpoint_steps"])), display))
    lines += ["", "相同轮数的组仍含更新量差异；等步数组控制最大更新次数，但不等于相同 token 数、重复访问次数或 FLOPs。此处不根据测试分数选择最佳数据比例。", "",
              "各训练运行的实际访问与参数预算：", "",
              "|模型|配置训练图像数|实际监督图像|执行优化步数|所选 checkpoint 步数|可训练参数|无检索 Token-F1|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in budgets:
        lines.append("|%s|%d|%d|%d|%s|%d|%.4f|" % (r["model"], r["count"], r["unique_supervised_images"],
                     r["optimizer_steps"], r["selected_checkpoint_step"], r["trainable_parameters"], r["token_f1"]))
    lines += ["", "32/128 图像的固定轮数与固定更新步数比较分开呈现；1024 图像等步数组最多只会访问其中部分样本，实际监督数量以上表为准。表中的实际监督图像与微批次数记录整个训练过程的消耗；若选择了更早的 checkpoint，不将全程访问量当作该 checkpoint 已见的数量。语义覆盖使用 CLIP 特征与固定词表，不使用人工描述内容作为选样依据。随机性分析固定了数据子集，不涵盖重新抽取子集的全部方差。`language_merger` 是语言 LoRA 加视觉 merger 全参数训练，通过 PEFT 的 modules_to_save 保存 merger，并非对 merger 再加同秩 LoRA；参数更多，不是等容量的位置因果对照。", "",
              "## 5. 检索规模与组合语义", "",
              "512 图像/2560 描述图库的 Recall 与训练库上下文的生成效用是不同任务。相似度—效用分析是描述性关联，不能证明 Recall 改善导致生成改善。图像检索取首条描述，描述检索/重排可访问每图全部描述，候选描述数不同；结果不能单独归因为算法改进。词数匹配的库规模补充实验见 [bank_size_effects.md](bank_size_effects.md)，SugarCrepe 分类型结果见完整报告。", "",
              "## 6. 人工评估状态与尚不能下的结论", "",
              "人工描述评分已完成 %d/%d；相关性核验已完成 %d/%d。" %
              (human["caption_items_scored"], human["caption_items_ready"], human["relevance_items_scored"], human["relevance_items_ready"]), "",
              "用户已选择先继续自动实验。没有人工标注时，不把词汇覆盖率、上下文复制率或 CLIPScore 当成人工事实准确率，也不报告人工核验的高相关上下文条件。材料和导入工具已经交付，可在后续补充。跨来源描述集仅由 SugarCrepe 子集决定，不能概括为完整 COCO 或所有领域的泛化性能。"]
    (report / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Research question summary written", flush=True)
