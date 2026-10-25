# 多模态检索干扰研究：实际实验结果

本报告只汇总已经写入完整预测与指标的运行。空白或 pending 不代表零分。

- 主对照完成：20/36；已完成评估运行：57；已保存 adapter：5。
- 人工相关性核验与盲评：pending。本报告没有把自动评分当作人工判断。
- 预先固定的协议见 [protocol.md](protocol.md)；逐运行数据见 [experiments.csv](experiments.csv)。

## 完成的确认集与预先固定诊断子集对比

数值为可用训练种子的均值 ± 样本标准差；仅一个种子时不估计种子方差。Zero-shot 为固定基础模型，不计作独立训练种子。

|训练方案|采样/图像数/预算|上下文|Bank|图像数|种子数|Token-F1|CIDEr-D变体|CLIPScore|输出词数|检索使用率|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|noise|random/128/language/epochs|adaptive|full|256|1|0.5745|0.9851|0.8020|9.8984|0.3047|
|noise|random/128/language/epochs|conflict|full|64|1|0.5484|0.8938|0.8096|10.6250|1.0000|
|noise|random/128/language/epochs|image|full|256|1|0.5636|0.9666|0.8014|9.8516|1.0000|
|noise|random/128/language/epochs|image|subset|64|1|0.5112|0.8061|0.8057|10.8750|1.0000|
|noise|random/128/language/epochs|none|full|256|1|0.5768|0.9827|0.8019|9.8945|0.0000|
|noise|random/128/language/epochs|partial|full|64|1|0.5361|0.8469|0.7969|10.0781|1.0000|
|noise|random/128/language/epochs|random|full|256|1|0.5641|0.9219|0.7922|9.2461|1.0000|
|noise|random/128/language/epochs|rerank|full|64|1|0.5212|0.8244|0.8084|10.5781|1.0000|
|noise|random/128/language/epochs|text|full|64|1|0.5310|0.8334|0.8130|10.4844|1.0000|
|plain|random/128/language/epochs|adaptive|full|256|2|0.5731 ± 0.0089|0.9757 ± 0.0360|0.8036 ± 0.0001|10.0020 ± 0.1188|0.5352 ± 0.3259|
|plain|random/128/language/epochs|conflict|full|64|1|0.5362|0.8557|0.8146|10.7500|1.0000|
|plain|random/128/language/epochs|image|full|256|2|0.5606 ± 0.0026|0.9431 ± 0.0037|0.8028 ± 0.0002|9.9590 ± 0.0635|1.0000 ± 0.0000|
|plain|random/128/language/epochs|image|subset|64|1|0.5112|0.7760|0.8033|10.6406|1.0000|
|plain|random/128/language/epochs|none|full|256|2|0.5764 ± 0.0028|0.9818 ± 0.0188|0.8013 ± 0.0012|9.9551 ± 0.2624|0.0000 ± 0.0000|
|plain|random/128/language/epochs|partial|full|64|1|0.5309|0.8868|0.8021|10.5156|1.0000|
|plain|random/128/language/epochs|random|full|256|2|0.5618 ± 0.0031|0.9630 ± 0.0033|0.7989 ± 0.0007|9.7129 ± 0.0249|1.0000 ± 0.0000|
|plain|random/128/language/epochs|rerank|full|64|1|0.5223|0.7913|0.8120|10.6406|1.0000|
|plain|random/128/language/epochs|text|full|64|1|0.5320|0.8348|0.8132|10.4688|1.0000|
|retrieval|random/128/language/epochs|adaptive|full|256|2|0.5775 ± 0.0030|0.9956 ± 0.0093|0.8047 ± 0.0010|10.1738 ± 0.0691|0.3047 ± 0.0000|
|retrieval|random/128/language/epochs|conflict|full|64|1|0.5447|0.8740|0.8015|10.3281|1.0000|
|retrieval|random/128/language/epochs|image|full|256|2|0.5636 ± 0.0002|0.9690 ± 0.0087|0.7997 ± 0.0020|9.8223 ± 0.2293|1.0000 ± 0.0000|
|retrieval|random/128/language/epochs|image|subset|64|1|0.5093|0.7764|0.8008|10.0781|1.0000|
|retrieval|random/128/language/epochs|none|full|256|2|0.5795 ± 0.0012|1.0076 ± 0.0115|0.8043 ± 0.0001|10.2910 ± 0.0193|0.0000 ± 0.0000|
|retrieval|random/128/language/epochs|partial|full|64|1|0.5384|0.8860|0.7989|9.8594|1.0000|
|retrieval|random/128/language/epochs|random|full|256|2|0.5626 ± 0.0011|0.9403 ± 0.0054|0.7916 ± 0.0015|9.2520 ± 0.2514|1.0000 ± 0.0000|
|retrieval|random/128/language/epochs|rerank|full|64|1|0.5428|0.8418|0.8070|10.2188|1.0000|
|retrieval|random/128/language/epochs|text|full|64|1|0.5364|0.8620|0.8172|10.3594|1.0000|
|zero_shot|none/0/none/none|adaptive|full|256|0|0.5150|0.6691|0.8436|15.3320|0.7656|
|zero_shot|none/0/none/none|image|full|256|0|0.5110|0.6775|0.8429|15.1836|1.0000|
|zero_shot|none/0/none/none|none|full|256|0|0.5174|0.6842|0.8436|15.6797|0.0000|
|zero_shot|none/0/none/none|random|full|256|0|0.5074|0.6617|0.8401|15.6797|1.0000|

## 检索使用、复制代理与输入预算

全体图像的复制代理在未使用检索时取零，因此自适应方法减少检索次数本身就可能降低其均值。下表同时给出仅在实际使用上下文的图像上计算的复制率；没有任何检索使用时该条件统计为未定义（—），不会填为零。条件均值仍是词汇代理，且不同策略选中的图像可能不同，不作事实正确性解释。

|训练方案|上下文|Bank|图像数|检索使用率|全体复制代理|使用时复制代理|有定义运行数|输入 tokens|上下文 tokens|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|noise|adaptive|full|256|0.3047|0.1218|0.3999|1|159.2|5.7|
|noise|conflict|full|64|1.0000|0.1891|0.1891|1|187.6|19.6|
|noise|image|full|256|1.0000|0.2515|0.2515|1|187.5|19.4|
|noise|image|subset|64|1.0000|0.2006|0.2006|1|187.3|19.3|
|noise|none|full|256|0.0000|0.0000|—|0|147.1|0.0|
|noise|partial|full|64|1.0000|0.2063|0.2063|1|187.4|19.5|
|noise|random|full|256|1.0000|0.0600|0.0600|1|187.5|19.4|
|noise|rerank|full|64|1.0000|0.3026|0.3026|1|187.3|19.3|
|noise|text|full|64|1.0000|0.3200|0.3200|1|187.6|19.6|
|plain|adaptive|full|256|0.5352|0.1688|0.3389|2|168.5|10.2|
|plain|conflict|full|64|1.0000|0.1847|0.1847|1|187.6|19.6|
|plain|image|full|256|1.0000|0.2461|0.2461|2|187.5|19.4|
|plain|image|subset|64|1.0000|0.2039|0.2039|1|187.3|19.3|
|plain|none|full|256|0.0000|0.0000|—|0|147.1|0.0|
|plain|partial|full|64|1.0000|0.1899|0.1899|1|187.4|19.5|
|plain|random|full|256|1.0000|0.0564|0.0564|2|187.5|19.4|
|plain|rerank|full|64|1.0000|0.2885|0.2885|1|187.3|19.3|
|plain|text|full|64|1.0000|0.3063|0.3063|1|187.6|19.6|
|retrieval|adaptive|full|256|0.3047|0.1222|0.4011|2|159.2|5.7|
|retrieval|conflict|full|64|1.0000|0.1943|0.1943|1|187.6|19.6|
|retrieval|image|full|256|1.0000|0.2518|0.2518|2|187.5|19.4|
|retrieval|image|subset|64|1.0000|0.2103|0.2103|1|187.3|19.3|
|retrieval|none|full|256|0.0000|0.0000|—|0|147.1|0.0|
|retrieval|partial|full|64|1.0000|0.1904|0.1904|1|187.4|19.5|
|retrieval|random|full|256|1.0000|0.0614|0.0614|2|187.5|19.4|
|retrieval|rerank|full|64|1.0000|0.3222|0.3222|1|187.3|19.3|
|retrieval|text|full|64|1.0000|0.3211|0.3211|1|187.6|19.6|
|zero_shot|adaptive|full|256|0.7656|0.1616|0.2110|1|177.8|14.6|
|zero_shot|image|full|256|1.0000|0.1921|0.1921|1|187.5|19.4|
|zero_shot|none|full|256|0.0000|0.0000|—|0|147.1|0.0|
|zero_shot|random|full|256|1.0000|0.0437|0.0437|1|187.5|19.4|

## 配对效应与交互

所有差值均按名称右侧减左侧的明确表达式计算，并要求图像 ID 完全一致。只有预先登记的 text/rerank/conflict/partial 诊断，才将主条件限定至完全相同、顺序一致的确认集前缀；不对任意缺失样本取交集。95% 图像 bootstrap 区间是在可用固定训练运行取平均后重采样图像，不能解释成训练随机种子的不确定性。种子 SD 单独列出；不作多重比较校正。

|效应（表达式即差值方向）|配置/Bank|分析子集|图像数|种子数|Token-F1 Δ|种子 SD|95% 图像区间|
|---|---|---|---:|---:|---:|---:|---|
|combined:noise_adaptive-plain_image|random/128/language/epochs/full|confirmation|256|1|0.0158|—|[0.0023, 0.0300]|
|interaction:(retrieval_image-retrieval_none)-(plain_image-plain_none)|random/128/language/epochs/full|confirmation|256|2|-0.0001|0.0008|[-0.0120, 0.0124]|
|noise:adaptive-image|random/128/language/epochs/full|confirmation|256|1|0.0109|—|[-0.0033, 0.0252]|
|noise:adaptive-none|random/128/language/epochs/full|confirmation|256|1|-0.0023|—|[-0.0122, 0.0075]|
|noise:conflict-image|random/128/language/epochs/full|diagnostic_subset|64|1|0.0063|—|[-0.0085, 0.0220]|
|noise:image-none|random/128/language/epochs/full|confirmation|256|1|-0.0132|—|[-0.0302, 0.0032]|
|noise:partial-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0060|—|[-0.0330, 0.0187]|
|noise:random-none|random/128/language/epochs/full|confirmation|256|1|-0.0127|—|[-0.0288, 0.0041]|
|noise:rerank-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0209|—|[-0.0587, 0.0148]|
|noise:text-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0111|—|[-0.0475, 0.0281]|
|plain:adaptive-image|random/128/language/epochs/full|confirmation|256|2|0.0125|0.0063|[0.0020, 0.0241]|
|plain:adaptive-none|random/128/language/epochs/full|confirmation|256|2|-0.0033|0.0061|[-0.0141, 0.0071]|
|plain:conflict-image|random/128/language/epochs/full|diagnostic_subset|64|1|0.0089|—|[-0.0023, 0.0208]|
|plain:image-none|random/128/language/epochs/full|confirmation|256|2|-0.0158|0.0002|[-0.0326, 0.0007]|
|plain:partial-image|random/128/language/epochs/full|diagnostic_subset|64|1|0.0035|—|[-0.0128, 0.0185]|
|plain:random-none|random/128/language/epochs/full|confirmation|256|2|-0.0146|0.0058|[-0.0298, 0.0007]|
|plain:rerank-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0050|—|[-0.0417, 0.0271]|
|plain:text-image|random/128/language/epochs/full|diagnostic_subset|64|1|0.0047|—|[-0.0270, 0.0389]|
|retrieval:adaptive-image|random/128/language/epochs/full|confirmation|256|2|0.0139|0.0028|[0.0009, 0.0273]|
|retrieval:adaptive-none|random/128/language/epochs/full|confirmation|256|2|-0.0020|0.0018|[-0.0092, 0.0060]|
|retrieval:conflict-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0030|—|[-0.0157, 0.0078]|
|retrieval:image-none|random/128/language/epochs/full|confirmation|256|2|-0.0159|0.0010|[-0.0309, -0.0008]|
|retrieval:partial-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0093|—|[-0.0334, 0.0140]|
|retrieval:random-none|random/128/language/epochs/full|confirmation|256|2|-0.0169|0.0001|[-0.0322, -0.0022]|
|retrieval:rerank-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0049|—|[-0.0417, 0.0323]|
|retrieval:text-image|random/128/language/epochs/full|diagnostic_subset|64|1|-0.0113|—|[-0.0464, 0.0236]|
|zero_shot:adaptive-image|none/0/none/none/full|confirmation|256|1|0.0040|—|[-0.0041, 0.0127]|
|zero_shot:adaptive-none|none/0/none/none/full|confirmation|256|1|-0.0024|—|[-0.0156, 0.0098]|
|zero_shot:image-none|none/0/none/none/full|confirmation|256|1|-0.0064|—|[-0.0218, 0.0084]|
|zero_shot:random-none|none/0/none/none/full|confirmation|256|1|-0.0100|—|[-0.0252, 0.0043]|

其他指标的效应和逐种子差值见 [effects.json](effects.json)。交互项 `(retrieval_image-retrieval_none)-(plain_image-plain_none)` 描述训练中加入检索后，推理检索效应发生的变化；它本身不能证明唯一的心理或神经机制。

## 相似度与下游效用的描述性关联

仅使用主 128 图像随机采样配置、完整确认集、full 检索库的 image 与 none 配对输出。每张图片先对可用固定训练种子的分数差取平均，再分析 top-1 图像上下文相似度与 Token-F1 变化的关联。相似度不是相关性真值或正确概率；Pearson r 仅为描述性统计，不计算显著性、不作因果解释，也不据此调整检索阈值或选模型。

候选图库上的 Recall@K 衡量另一项图文匹配任务；本分析没有训练图像检索的人工相关性标签，不能把相似度与效用的关联改写成 Recall 提升导致生成提升。

|训练方案|图像数|固定训练运行数|Pearson r（描述性）|
|---|---:|---:|---:|
|plain|256|2|0.0847|
|retrieval|256|2|0.0122|
|noise|256|1|0.0019|

|训练方案|相似度四分组|图像数|相似度均值|Token-F1 Δ 均值|正向 Δ 比例|
|---|---:|---:|---:|---:|---:|
|plain|1|64|0.6959|-0.0323|0.3438|
|plain|2|64|0.7728|-0.0130|0.4062|
|plain|3|64|0.8248|-0.0194|0.3281|
|plain|4|64|0.8816|0.0016|0.3906|
|retrieval|1|64|0.6959|-0.0206|0.3750|
|retrieval|2|64|0.7728|-0.0102|0.2969|
|retrieval|3|64|0.8248|-0.0228|0.2969|
|retrieval|4|64|0.8816|-0.0098|0.3750|
|noise|1|64|0.6959|-0.0209|0.3750|
|noise|2|64|0.7728|-0.0096|0.3750|
|noise|3|64|0.8248|-0.0167|0.2656|
|noise|4|64|0.8816|-0.0058|0.2656|

原始图像级摘要见 [similarity_utility_queries.csv](similarity_utility_queries.csv)，分组见 [similarity_utility_quartiles.csv](similarity_utility_quartiles.csv)。

## 验证集选择的检索策略

|模型|阈值|验证指标|验证使用率|选择集合|
|---|---:|---:|---:|---|
|noise_s42_random_n128_language_epochs|0.8409|0.5859|0.2500|validation|
|plain_s42_random_n128_language_epochs|0.7421|0.5868|0.7500|validation|
|plain_s43_random_n128_language_epochs|0.8409|0.5837|0.2500|validation|
|retrieval_s42_random_n128_language_epochs|0.8409|0.5819|0.2500|validation|
|retrieval_s43_random_n128_language_epochs|0.8409|0.5766|0.2500|validation|
|zero_shot|0.7421|0.4963|0.7500|validation|

## 外部基准与人工评估

尚无已完成的 POPE/SugarCrepe/跨来源描述结果；这些项目仍为 pending，不能据生成描述分数推断其表现。

人工核验相关上下文和人工盲评需要独立的人类标注。只有真实标注经完整性核验后才能报告事实正确性或人工偏好；当前报告不生成此类分数。

## 图像打乱对照

|模型|上下文|Token-F1|原图 CLIPScore|输入图 CLIPScore|
|---|---|---:|---:|---:|
|noise_s42_random_n128_language_epochs|image|0.2673|0.4358|0.7972|
|noise_s42_random_n128_language_epochs|none|0.2633|0.4413|0.8020|
|plain_s42_random_n128_language_epochs|image|0.2685|0.4292|0.7965|
|plain_s42_random_n128_language_epochs|none|0.2643|0.4425|0.8005|
|retrieval_s42_random_n128_language_epochs|image|0.2750|0.4392|0.7928|
|retrieval_s42_random_n128_language_epochs|none|0.2707|0.4373|0.8079|

打乱条件保持参考答案与检索上下文不变，因此应分别报告相对原图与实际输入图片的一致性；仅凭打乱后仍有非零词汇分数，不能证明模型完全没有使用图像。

## 未完成的主对照

|模型|上下文|状态|
|---|---|---|
|plain_s44_random_n128_language_epochs|none|pending|
|plain_s44_random_n128_language_epochs|random|pending|
|plain_s44_random_n128_language_epochs|image|pending|
|plain_s44_random_n128_language_epochs|adaptive|pending|
|retrieval_s44_random_n128_language_epochs|none|pending|
|retrieval_s44_random_n128_language_epochs|random|pending|
|retrieval_s44_random_n128_language_epochs|image|pending|
|retrieval_s44_random_n128_language_epochs|adaptive|pending|
|noise_s43_random_n128_language_epochs|none|pending|
|noise_s43_random_n128_language_epochs|random|pending|
|noise_s43_random_n128_language_epochs|image|pending|
|noise_s43_random_n128_language_epochs|adaptive|pending|
|noise_s44_random_n128_language_epochs|none|pending|
|noise_s44_random_n128_language_epochs|random|pending|
|noise_s44_random_n128_language_epochs|image|pending|
|noise_s44_random_n128_language_epochs|adaptive|pending|

## 指标和解释边界

- `cider_fixed_df` / `cider_d_fixed_df` 分别为 CIDEr_fixed_df_regex / CIDErD_fixed_df_regex：使用英文 regex 分词和固定训练参考 DF；与 PTB 分词、其他 DF 协议的 COCO 排行榜数值不直接可比。
- CLIPScore 使用同一 CLIP 表征体系，可能偏向该检索器；它是辅助证据，不能独立证明事实正确性。
- `context_copy_rate` 是输出内容词在上下文中出现的比例；`reference_detail_recall` 是最佳参考描述内容词召回。两者均为词汇代理，不是幻觉率或完整视觉细节覆盖。
- 随机、部分相关和合成冲突是受控文本干预，不代表经过人工核验的真假标签。
- 检索阈值是验证集选择的相似度决策规则，不是校准后的正确概率。
- 多种子均值和 SD 不替代独立数据分布验证；数据比例、更新步数、采样策略、检索库大小与可训练参数量必须分别标记。
- 本报告不因文件存在就宣称六个研究方向全部完成，未产生的外部基准和人工评估仍待完成。

![comparison_seed_mean_sd](comparison_seed_mean_sd.png)

![length_quality](length_quality.png)

![training_curves](training_curves.png)

![similarity_utility](similarity_utility.png)
