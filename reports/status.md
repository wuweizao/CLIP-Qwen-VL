# 项目完成记录

2026-09-29，本机 RTX 5070 Ti 16GB、Python 3.10.21、PyTorch 2.11.0+cu128 实际运行完成。

|阶段|交付与验证|
|---|---|
|CLIP 检索|Flickr8k 128/32/64 图像划分；双向检索；Recall@1/5/10；文本/图像查询入口|
|Qwen-VL baseline|256 条真实人工描述构成的 instruction；4 组 zero-shot 推理；Token-F1、ROUGE-L|
|LoRA SFT|25%/100% 两组训练；各两轮；仅训练 3,686,400 个参数；两个 adapter 与训练/验证曲线|
|消融|共 12 组 × 64 图像 = 768 条预测；两个 Prompt；两种训练比例；有/无 CLIP 检索增强|
|工程化|README、requirements、环境快照、推理/检索/完整运行脚本、CI、实验 CSV/图表、简历总结、本地 Git 仓库|

## 主要结果

- CLIP：image→text Recall@1 = 93.75%；text→image Recall@1 = 83.44%。仅针对 64 图像/320 描述候选池。
- 主对比（concise、无 RAG）：zero-shot Token-F1 0.4877 → SFT25% 0.5602 → SFT100% 0.5711。
- SFT100% 相对 zero-shot 提升 8.34 个百分点，图像级配对 bootstrap 95% 区间为 [5.17, 11.59] 个百分点。
- 同一主对比的 ROUGE-L：0.4628 → 0.5402 → 0.5566。
- RAG 不稳定：zero-shot concise Token-F1 从 0.4877 升到 0.4965，但 SFT100% 从 0.5711 降至 0.5522。
- SFT100% 两轮验证损失分别为 1.5670、1.5437；SFT25% 为 2.0748、2.0182。

## 已通过检查

- 9 项单元/集成检查，包含多正例 Recall、文本指标、配对区间、嵌套采样、跨集合去重、RAG 来源、assistant-only 标签。
- 12 组完整预测、每组 64 个唯一测试 ID，重新计算指标与文件数值一致。
- 两组 adapter 权重存在，每组均有两轮有限的验证损失。
- 文本→图像、图像→文本、单图 adapter 推理入口实际运行成功。
- 权重 SHA256、数据版本、数据清单哈希、依赖版本均记录。

## 交付位置

- `outputs/sft_0.25/adapter/`、`outputs/sft_1/adapter/`：LoRA adapter。
- `outputs/evaluations/`：逐样本预测、评估参数、指标与来源记录。
- `reports/results.md`、`reports/experiments.csv`：完整结果。
- `reports/*_loss.png`、`reports/comparison.png`：真实训练和比较图。
- `reports/verification.json`：完整性核验结果。

数据、基础模型和 adapter 保留本地，已被 Git 忽略；GitHub 目标仓库为 [wuweizao/CLIP-Qwen-VL](https://github.com/wuweizao/CLIP-Qwen-VL)，上传内容包含代码、配置、文档与实验报告。

## 结论边界

本研究是单随机种子的小规模工程实验，词汇指标的改善不能直接解释为通用视觉能力提高。置信区间不涵盖重新训练的随机性；数据比例组并非等更新步数；未评估原始预训练数据污染、CIDEr/SPICE 或人工幻觉率。检索库固定使用全训练集，因此 25%+RAG 组仍可访问完整训练集的检索描述。
