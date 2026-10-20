# ASD 置信度小检查（运行前冻结）

开发诊断，不作新题泛化或校准正确率声明。候选来自对话中提出的启发式：相同regret下，Target概率集中在首选和proposal之间的分歧更值得尝试。该方向可能错误，不能把高熵或低熵直接解释成低语义风险。

## 样本和控制

- 仅旧开发题：math500:1、math500:4、math500:7、gsm8k:21。包含旧正例、Run退化例、Utility真实错答案例；均已观察，不新增筛题。
- 共享原prompt、Qwen3-8B/DFlash版本、FP32/eager、greedy、enable_thinking=false、空历史。自然EOS完整上限2048；短前缀上限128，遵守EOS。
- 七组：严格DFlash、原ASD(B8/g.25/M2)、Run、Utility价格2、Utility价格1、Utility价格3、Confidence-Utility。置信度组alpha=.5、价格2，原ASD可行域不改，w=1+.5*(1-2*(p_top+p_proposal))，仅例外计价。原始regret账本照常扣除，不按w打折预算。
- 每题七组各1机制、2自然计时、2前缀计时，共140生成；两个GPU各独立进程，同题同卡交错。加载/预热/插桩不计入速度，所有概率计算计入置信度组的正常在线耗时。128前缀只测起始区，不声称完全相同文本工作量。
- 所有质量/截断结果保留，不按表现提前结束或扫描alpha/价格。原组对照旧三题补全及gsm21预算结果，允许仅在旧短上限裁短块起点之后分歧。

## 同前缀干预

只从原ASD机制轨迹选状态，不看分支结果。候选是每块首个被实际接受、非EOS的mismatch，位置在输出前512token内，之前的块内token均精确匹配。保存proposal全部ID、Target首选ID、概率/排名、Draft概率、未接受proposal EOS位置、源账本和原始regret。

每题至多选两状态：优先找regret差<=.25、连续匹配run差<=2、pair_mass差>=.1的配对，最大化pair_mass差，平局按出现位置；无合格配对则选pair_mass最低和最高的两个不同状态并明确标记unmatched，不用这种配对作控制regret的证据。候选不足如实保留。

每状态两分支，在完全相同prompt和生成前缀上，分别强制一个proposal或Target recovery token，之后使用严格DFlash续写到EOS或总2048token上限，重算各分支缓存/特征，不跨分支复用旧logits或已评分suffix。最多16分支，重prefill与墙钟单列，不当作线上加速；严格续写仅估计单次例外的局部后果，不等于整个近似策略的效果。

先对源相同前缀做独立Target前向核查argmax和regret（绝对容差1e-3）；超差标记，不伪造同状态因果结论。质量按现有评分，不改解析器/答案。精确匹配风险与任务质量风险分别报告。

## 判定

主要看置信度组相对Utility及固定价格1/3是否有独立信号：每token验证轮、完整答案质量、等128token吞吐、真实在线总耗时。更短输出单列，不当作验证效率改善。相同源状态的离线选择、样本配对、分支因果检查与完整on-policy计时分列；4题/最多8状态不提供质量保证或显著性结论。

若仅复制固定价格对照、质量退化、概率计算抵消收益，或pair_mass无法区分分支后果，则停止扩大该启发式，保留负结果；不由本轮拟合新风险模型。

## 文献和运行入口

- ASD: https://arxiv.org/html/2608.03447v3 （regret是局部偏好代价，不是正确率）
- DISCO: https://arxiv.org/html/2405.04304v5 （Draft置信度预测验证机会）
- MARS: https://arxiv.org/html/2601.15498 （Target margin近似验证）
- Calibration: https://aclanthology.org/2021.tacl-1.57/ （概率不自动校准）
- DFlash: https://arxiv.org/html/2602.06036 （并行Draft置信度不能直接当AR链概率）

独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-confidence`，环境`../.venv-dflash/bin/python`。入口：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_confidence_probe --config configs/runtime-dflash.json --task-ids math500:1 math500:4 --output results/raw/asd-confidence/math-a
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ../.venv-dflash/bin/python -u -m scripts.run_asd_confidence_probe --config configs/runtime-dflash.json --task-ids math500:7 gsm8k:21 --output results/raw/asd-confidence/mixed-b
```
