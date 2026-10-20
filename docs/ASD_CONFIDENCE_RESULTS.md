# ASD 置信度小规模开发检查

仅4道已观察开发题；不是独立泛化验证。alpha=.5、价格2冻结，固定价格1/3为对照；原ASD预算仍按未加权regret扣除。有效选择价格在1到3之间，固定对照不等于扫描所有中间价格。概率按原始logits的T=1归一化，仅作启发式特征，生成仍为greedy，没有进行正确率校准。

已审计机制28、完整自然计时56、128前缀计时56、分支16次。全部质量/截断保留；计时输出与机制一致，旧四组保留之前未受短上限影响的前缀。

## math500

| 方法 | 完整秒（两重复） | token（两重复） | TPS | 每100token轮 | 正确 | 截断 | regret | 相对Utility时延 | 等128前缀秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 41.271497 | 4420 | 107.096 | 11.946 | 3/3 | 0 | 0.000 | +16.45% | 10.985624 |
| ASD | 41.246308 | 4662 | 113.028 | 11.197 | 3/3 | 0 | 23.434 | +16.38% | 9.327110 |
| ASD-Run | 34.775637 | 4262 | 122.557 | 10.277 | 3/3 | 0 | 6.310 | -1.87% | 8.432863 |
| ASD-Utility | 35.440130 | 4376 | 123.476 | 10.192 | 3/3 | 0 | 20.375 | +0.00% | 8.726646 |
| Utility-Price1 | 38.691360 | 4572 | 118.166 | 10.674 | 3/3 | 0 | 23.347 | +9.17% | 8.314391 |
| Utility-Price3 | 34.899002 | 4204 | 120.462 | 10.466 | 3/3 | 0 | 14.674 | -1.53% | 9.637994 |
| Confidence-Utility | 34.105988 | 4010 | 117.575 | 10.673 | 3/3 | 0 | 23.030 | -3.76% | 9.361886 |

Confidence/Utility分解：长度比0.916362，轮密度比1.047229，每轮成本比1.002828；三者乘积为时延比0.962355。各项来自不同轨迹，不能当成独立因果贡献。
等128token集合：`math500:1, math500:4, math500:7`；均遵守EOS，128上限可能改变最后一块决策，裁短块前的前缀已审计。

## gsm8k

| 方法 | 完整秒（两重复） | token（两重复） | TPS | 每100token轮 | 正确 | 截断 | regret | 相对Utility时延 | 等128前缀秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFlash | 5.165532 | 320 | 61.949 | 21.250 | 1/1 | 0 | 0.000 | -22.91% | 3.860950 |
| ASD | 5.366239 | 458 | 85.348 | 15.284 | 1/1 | 0 | 7.936 | -19.92% | 2.864903 |
| ASD-Run | 4.198692 | 308 | 73.356 | 17.532 | 1/1 | 0 | 1.577 | -37.34% | 3.151964 |
| ASD-Utility | 6.700870 | 480 | 71.632 | 18.333 | 0/1 | 0 | 6.373 | +0.00% | 3.450365 |
| Utility-Price1 | 6.705945 | 522 | 77.841 | 16.858 | 1/1 | 0 | 6.949 | +0.08% | 3.451458 |
| Utility-Price3 | 4.343506 | 316 | 72.752 | 17.722 | 1/1 | 0 | 2.881 | -35.18% | 3.157554 |
| Confidence-Utility | 6.725093 | 522 | 77.620 | 16.858 | 1/1 | 0 | 5.553 | +0.36% | 3.463667 |

Confidence/Utility分解：长度比1.087500，轮密度比0.919540，每轮成本比1.003615；三者乘积为时延比1.003615。各项来自不同轨迹，不能当成独立因果贡献。
等128token集合：`gsm8k:21`；均遵守EOS，128上限可能改变最后一块决策，裁短块前的前缀已审计。

## 逐题与固定价格对照

| 请求 | 方法 | 秒（两重复） | token（单次） | 验证轮（单次） | 正确 | 与置信度组完整token相同 |
|---|---|---:|---:|---:|---|---|
| math500:1 | DFlash | 10.631472 | 722 | 68 | True | False |
| math500:1 | ASD | 11.332565 | 806 | 72 | True | False |
| math500:1 | ASD-Run | 9.352640 | 707 | 59 | True | False |
| math500:1 | ASD-Utility | 10.602599 | 794 | 67 | True | False |
| math500:1 | Utility-Price1 | 12.482591 | 864 | 79 | True | False |
| math500:1 | Utility-Price3 | 9.642653 | 703 | 61 | True | False |
| math500:1 | Confidence-Utility | 9.521803 | 664 | 60 | True | True |
| math500:4 | DFlash | 15.542460 | 653 | 96 | True | False |
| math500:4 | ASD | 17.964084 | 700 | 111 | True | False |
| math500:4 | ASD-Run | 12.687630 | 586 | 77 | True | False |
| math500:4 | ASD-Utility | 13.016513 | 601 | 79 | True | False |
| math500:4 | Utility-Price1 | 13.607841 | 588 | 83 | True | False |
| math500:4 | Utility-Price3 | 12.973818 | 575 | 79 | True | False |
| math500:4 | Confidence-Utility | 13.636791 | 576 | 83 | True | True |
| math500:7 | DFlash | 15.097565 | 835 | 100 | True | False |
| math500:7 | ASD | 11.949660 | 825 | 78 | True | False |
| math500:7 | ASD-Run | 12.735367 | 838 | 83 | True | False |
| math500:7 | ASD-Utility | 11.821017 | 793 | 77 | True | False |
| math500:7 | Utility-Price1 | 12.600928 | 834 | 82 | True | False |
| math500:7 | Utility-Price3 | 12.282531 | 824 | 80 | True | False |
| math500:7 | Confidence-Utility | 10.947394 | 765 | 71 | True | True |
| gsm8k:21 | DFlash | 5.165532 | 160 | 34 | True | False |
| gsm8k:21 | ASD | 5.366239 | 229 | 35 | True | False |
| gsm8k:21 | ASD-Run | 4.198692 | 154 | 27 | True | False |
| gsm8k:21 | ASD-Utility | 6.700870 | 240 | 44 | False | False |
| gsm8k:21 | Utility-Price1 | 6.705945 | 261 | 44 | True | False |
| gsm8k:21 | Utility-Price3 | 4.343506 | 158 | 28 | True | False |
| gsm8k:21 | Confidence-Utility | 6.725093 | 261 | 44 | True | True |

## 相同前缀的单次例外干预

严格续写；每对只改变一个强制token，缓存重新计算。仅单例外后果，不等于近似策略完整rollout。matched表示源状态的regret差<=.25、run差<=2、pair_mass差>=.1；unmatched不能作为控制这些因素后的证据。

| 请求/位置 | matched | pair_mass | regret | run | 同状态核查 | proposal正确 | recovery正确 | proposal/recovery总token | proposal/recovery后续轮 |
|---|---|---:|---:|---:|---|---|---|---|---|
| math500:1/142 | True | 0.5344 | 0.2669 | 3 | True | True | True | 790/746 | 66/64 |
| math500:1/459 | True | 1.0000 | 0.3927 | 1 | True | True | True | 809/873 | 31/33 |
| math500:4/20 | False | 0.3197 | 1.7226 | 2 | True | True | True | 692/680 | 115/126 |
| math500:4/45 | False | 0.9979 | 2.6110 | 2 | True | True | True | 683/692 | 123/110 |
| math500:7/130 | True | 0.5025 | 0.5862 | 3 | True | True | True | 737/783 | 59/62 |
| math500:7/204 | True | 0.9975 | 0.4870 | 1 | True | True | True | 794/758 | 52/49 |
| gsm8k:21/17 | False | 0.8813 | 0.8823 | 2 | True | True | True | 229/232 | 43/42 |
| gsm8k:21/67 | False | 1.0000 | 1.4159 | 1 | True | True | True | 236/238 | 30/32 |

## 相同源状态的离线选择

原ASD、Utility及Confidence-Utility三个源的每块独立借用各自实际账本；没有推进替代策略状态，不能当作全请求节省。全部proposal/EOS已记录，因此可精确裁剪候选EOS。这是离线补充诊断，干预选样始终仅来自原ASD。

| 请求 | 源轨迹 | 选择器 | 与Utility相同块 | 多接块 | 少接块 |
|---|---|---|---:|---:|---:|
| math500:1 | ASD | DFlash | 64 | 0 | 8 |
| math500:1 | ASD | ASD | 68 | 4 | 0 |
| math500:1 | ASD | ASD-Run | 70 | 0 | 2 |
| math500:1 | ASD | ASD-Utility | 72 | 0 | 0 |
| math500:1 | ASD | Utility-Price1 | 70 | 2 | 0 |
| math500:1 | ASD | Utility-Price3 | 70 | 0 | 2 |
| math500:1 | ASD | Confidence-Utility | 71 | 1 | 0 |
| math500:1 | ASD-Utility | DFlash | 56 | 0 | 11 |
| math500:1 | ASD-Utility | ASD | 61 | 6 | 0 |
| math500:1 | ASD-Utility | ASD-Run | 61 | 0 | 6 |
| math500:1 | ASD-Utility | ASD-Utility | 67 | 0 | 0 |
| math500:1 | ASD-Utility | Utility-Price1 | 63 | 4 | 0 |
| math500:1 | ASD-Utility | Utility-Price3 | 65 | 0 | 2 |
| math500:1 | ASD-Utility | Confidence-Utility | 64 | 3 | 0 |
| math500:1 | Confidence-Utility | DFlash | 50 | 0 | 10 |
| math500:1 | Confidence-Utility | ASD | 52 | 8 | 0 |
| math500:1 | Confidence-Utility | ASD-Run | 57 | 0 | 3 |
| math500:1 | Confidence-Utility | ASD-Utility | 60 | 0 | 0 |
| math500:1 | Confidence-Utility | Utility-Price1 | 56 | 4 | 0 |
| math500:1 | Confidence-Utility | Utility-Price3 | 60 | 0 | 0 |
| math500:1 | Confidence-Utility | Confidence-Utility | 58 | 2 | 0 |
| math500:4 | ASD | DFlash | 108 | 0 | 3 |
| math500:4 | ASD | ASD | 107 | 4 | 0 |
| math500:4 | ASD | ASD-Run | 111 | 0 | 0 |
| math500:4 | ASD | ASD-Utility | 111 | 0 | 0 |
| math500:4 | ASD | Utility-Price1 | 109 | 2 | 0 |
| math500:4 | ASD | Utility-Price3 | 111 | 0 | 0 |
| math500:4 | ASD | Confidence-Utility | 110 | 1 | 0 |
| math500:4 | ASD-Utility | DFlash | 71 | 0 | 8 |
| math500:4 | ASD-Utility | ASD | 76 | 3 | 0 |
| math500:4 | ASD-Utility | ASD-Run | 72 | 0 | 7 |
| math500:4 | ASD-Utility | ASD-Utility | 79 | 0 | 0 |
| math500:4 | ASD-Utility | Utility-Price1 | 77 | 2 | 0 |
| math500:4 | ASD-Utility | Utility-Price3 | 76 | 0 | 3 |
| math500:4 | ASD-Utility | Confidence-Utility | 77 | 2 | 0 |
| math500:4 | Confidence-Utility | DFlash | 76 | 0 | 7 |
| math500:4 | Confidence-Utility | ASD | 78 | 5 | 0 |
| math500:4 | Confidence-Utility | ASD-Run | 80 | 0 | 3 |
| math500:4 | Confidence-Utility | ASD-Utility | 83 | 0 | 0 |
| math500:4 | Confidence-Utility | Utility-Price1 | 79 | 4 | 0 |
| math500:4 | Confidence-Utility | Utility-Price3 | 82 | 0 | 1 |
| math500:4 | Confidence-Utility | Confidence-Utility | 80 | 3 | 0 |
| math500:7 | ASD | DFlash | 67 | 0 | 11 |
| math500:7 | ASD | ASD | 75 | 3 | 0 |
| math500:7 | ASD | ASD-Run | 75 | 0 | 3 |
| math500:7 | ASD | ASD-Utility | 78 | 0 | 0 |
| math500:7 | ASD | Utility-Price1 | 77 | 1 | 0 |
| math500:7 | ASD | Utility-Price3 | 76 | 0 | 2 |
| math500:7 | ASD | Confidence-Utility | 78 | 0 | 0 |
| math500:7 | ASD-Utility | DFlash | 65 | 0 | 12 |
| math500:7 | ASD-Utility | ASD | 67 | 10 | 0 |
| math500:7 | ASD-Utility | ASD-Run | 74 | 0 | 3 |
| math500:7 | ASD-Utility | ASD-Utility | 77 | 0 | 0 |
| math500:7 | ASD-Utility | Utility-Price1 | 74 | 3 | 0 |
| math500:7 | ASD-Utility | Utility-Price3 | 76 | 0 | 1 |
| math500:7 | ASD-Utility | Confidence-Utility | 75 | 2 | 0 |
| math500:7 | Confidence-Utility | DFlash | 66 | 0 | 5 |
| math500:7 | Confidence-Utility | ASD | 67 | 4 | 0 |
| math500:7 | Confidence-Utility | ASD-Run | 69 | 0 | 2 |
| math500:7 | Confidence-Utility | ASD-Utility | 71 | 0 | 0 |
| math500:7 | Confidence-Utility | Utility-Price1 | 68 | 3 | 0 |
| math500:7 | Confidence-Utility | Utility-Price3 | 70 | 0 | 1 |
| math500:7 | Confidence-Utility | Confidence-Utility | 69 | 2 | 0 |
| gsm8k:21 | ASD | DFlash | 30 | 0 | 5 |
| gsm8k:21 | ASD | ASD | 33 | 2 | 0 |
| gsm8k:21 | ASD | ASD-Run | 33 | 0 | 2 |
| gsm8k:21 | ASD | ASD-Utility | 35 | 0 | 0 |
| gsm8k:21 | ASD | Utility-Price1 | 35 | 0 | 0 |
| gsm8k:21 | ASD | Utility-Price3 | 33 | 0 | 2 |
| gsm8k:21 | ASD | Confidence-Utility | 35 | 0 | 0 |
| gsm8k:21 | ASD-Utility | DFlash | 37 | 0 | 7 |
| gsm8k:21 | ASD-Utility | ASD | 38 | 6 | 0 |
| gsm8k:21 | ASD-Utility | ASD-Run | 40 | 0 | 4 |
| gsm8k:21 | ASD-Utility | ASD-Utility | 44 | 0 | 0 |
| gsm8k:21 | ASD-Utility | Utility-Price1 | 43 | 1 | 0 |
| gsm8k:21 | ASD-Utility | Utility-Price3 | 41 | 0 | 3 |
| gsm8k:21 | ASD-Utility | Confidence-Utility | 43 | 1 | 0 |
| gsm8k:21 | Confidence-Utility | DFlash | 38 | 0 | 6 |
| gsm8k:21 | Confidence-Utility | ASD | 36 | 8 | 0 |
| gsm8k:21 | Confidence-Utility | ASD-Run | 42 | 0 | 2 |
| gsm8k:21 | Confidence-Utility | ASD-Utility | 44 | 0 | 0 |
| gsm8k:21 | Confidence-Utility | Utility-Price1 | 41 | 3 | 0 |
| gsm8k:21 | Confidence-Utility | Utility-Price3 | 42 | 0 | 2 |
| gsm8k:21 | Confidence-Utility | Confidence-Utility | 42 | 2 | 0 |

## 本轮判断

三道MATH的完整自然时延变化-3.76%，输出长度变化-8.36%，自然TPS变化-4.78%，每token轮密度变化+4.72%；等128token前缀时延变化+7.28%。总体没有新增吞吐优势。
这不抹去局部正例：math500:7的置信度组71轮/765token，Utility为77轮/793token，轮密度确有改善；但math500:4变成83轮/576token，对照79轮/601token，方向相反。均为旧开发题，尚不构成场景判别器。
gsm8k:21置信度组修复了旧Utility错答，但固定价格1和3也答对。置信度组相对价格3时延变化+54.83%；没有置信度独有的纠错或加速证据。
八个单例外状态的16分支全部答对，无有害/无害质量标签差异，不能计算或宣称风险预测能力；这项干预未识别完整Utility轨迹的致错例外。只有math500:1/7满足预定regret/run配对，另两题为unmatched。
具体指标局限：选中的原ASD状态math500:4/45，Target首选概率约0.9296，proposal约0.0683，二者和约0.9979，regret约2.611。pair_mass很高，但Target对两者的偏好明显不相同。以集中度一律降低价格可能削弱原regret惩罚；这说明该启发式定义有缺陷，并不证明这个源状态是置信度组退化的唯一原因。
阶段决定：保留采集/干预/校验工具，停止扩大这条简单pair_mass动态定价规则；不在四题上反向调权重或继续扫alpha。尚不能据此否定所有置信度方法。若后续继续，应先在已知错误的Utility轨迹定位有害例外，再检验新增信号能否在控制regret和后缀长度后区分它们。

## 解释边界

概率集中度是启发式，不是答案正确率；首选和proposal概率接近仍可对应相反数字/结论。原始regret、运行开销、答案质量和长度必须共同判断。单个开发正例、局部更长接受或更短回答均不能证明稳定吞吐增量。
本轮不训练新风险模型、不根据结果扫描alpha。固定价格对照若能取得同等/更好行为，置信度独有价值仍不成立；不扩大该启发式到新题，除非机制和含全部开销的对照同时支持。

冻结执行commit `e7ec17b5f3b832b91f1c68cceb74395ac7607f85`；runtime `4097fa46857e96b785fd4a9d2b88ef94dfd9dd0d700d4fe42d83ac537d1c934f`。原发布证据与执行源码前后SHA不变。
重建：`python -m scripts.write_asd_confidence_report`。原始公开题输出、概率、EOS、命令与测试在results/asd-confidence；权重不入库。
