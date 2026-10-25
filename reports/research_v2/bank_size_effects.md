# 补充诊断：检索库大小与上下文长度控制

状态：completed；已完成评估 8/8。

比较同一个已经训练完成的模型在 1024 与 128 张训练图像检索库下的输出。固定 64 张诊断图像、K=2，以及每个排序位置的上下文词数；不对已有适配器重新训练。

本控制在主实验方案之后提出，在本组推理前固定，属于补充诊断。检索训练与抗噪声模型原先使用完整训练检索库，因此这里考察推理时改变记忆库的条件效应。

|模型|检索库|图像数|Token-F1|CIDEr-D变体|CLIPScore|输出词数|上下文 token数|
|---|---|---:|---:|---:|---:|---:|---:|
|zero_shot|full_length_matched|64|0.4777|0.5393|0.8475|16.61|17.69|
|zero_shot|subset_length_matched|64|0.4939|0.5722|0.8432|16.33|16.92|
|plain_s42_random_n128_language_epochs|full_length_matched|64|0.5339|0.8689|0.8145|10.67|17.69|
|plain_s42_random_n128_language_epochs|subset_length_matched|64|0.5254|0.8324|0.8021|10.48|16.92|
|retrieval_s42_random_n128_language_epochs|full_length_matched|64|0.5420|0.8707|0.8013|9.83|17.69|
|retrieval_s42_random_n128_language_epochs|subset_length_matched|64|0.5257|0.7939|0.8003|9.92|16.92|
|noise_s42_random_n128_language_epochs|full_length_matched|64|0.5434|0.9201|0.8091|10.33|17.69|
|noise_s42_random_n128_language_epochs|subset_length_matched|64|0.5273|0.8347|0.8022|10.48|16.92|

差值方向为完整库减小库。下列 95% 区间仅按图像配对 bootstrap，不涵盖重新训练的不确定性。

|模型|指标|差值|95% 图像区间|
|---|---|---:|---|
|zero_shot|token_f1|-0.0162|[-0.0367, 0.0048]|
|zero_shot|cider_d_fixed_df|-0.0329|[-0.1470, 0.0849]|
|zero_shot|clipscore|0.0043|[-0.0076, 0.0158]|
|zero_shot|words|0.2812|[-1.0320, 1.6250]|
|zero_shot|context_tokens|0.7656|[0.3906, 1.2031]|
|plain_s42_random_n128_language_epochs|token_f1|0.0085|[-0.0151, 0.0361]|
|plain_s42_random_n128_language_epochs|cider_d_fixed_df|0.0366|[-0.0507, 0.1364]|
|plain_s42_random_n128_language_epochs|clipscore|0.0123|[0.0004, 0.0239]|
|plain_s42_random_n128_language_epochs|words|0.1875|[-0.4688, 0.8281]|
|plain_s42_random_n128_language_epochs|context_tokens|0.7656|[0.3906, 1.2031]|
|retrieval_s42_random_n128_language_epochs|token_f1|0.0163|[-0.0094, 0.0430]|
|retrieval_s42_random_n128_language_epochs|cider_d_fixed_df|0.0768|[-0.0298, 0.1849]|
|retrieval_s42_random_n128_language_epochs|clipscore|0.0009|[-0.0126, 0.0138]|
|retrieval_s42_random_n128_language_epochs|words|-0.0938|[-0.9066, 0.6562]|
|retrieval_s42_random_n128_language_epochs|context_tokens|0.7656|[0.3906, 1.2031]|
|noise_s42_random_n128_language_epochs|token_f1|0.0160|[-0.0066, 0.0386]|
|noise_s42_random_n128_language_epochs|cider_d_fixed_df|0.0855|[-0.0272, 0.2065]|
|noise_s42_random_n128_language_epochs|clipscore|0.0069|[-0.0060, 0.0197]|
|noise_s42_random_n128_language_epochs|words|-0.1562|[-0.9844, 0.6719]|
|noise_s42_random_n128_language_epochs|context_tokens|0.7656|[0.3906, 1.2031]|

词数匹配不等于 BPE token 数匹配；实际 token 数已记录。两库保留各自检索排序，选中的图像和描述内容仍可不同。截断可能损失语义细节，因此结果不能单独归因为抽象的记忆容量，也不能据单个种子宣称普遍因果规律。
