# ASD 实际后缀的新题与收益归因

2026-10-20，所有新输出生成之前冻结。依据ASD的连续可行前缀、请求账本与实际前缀后缀语义：
https://arxiv.org/html/2608.03447v3 。不宣称无损、论文复现或算法首创。

- 新题固定为GSM8K test源码位置24..31（8题），MATH-500位置8..23（16题），保持来源顺序，不按难度、长度、答案或结果筛选；来源commit/SHA、许可与旧切片一致。
- 五组：严格DFlash、原ASD、ASD-Run、ASD-Utility、ASD-Conservative。原四组配置保持：B8/g0.25/M2，Run使用到下一失配的实际后缀长度，Utility价格2。
- Conservative只把原ASD的固定gate改为0.0625，不使用后缀匹配形状。用已发表旧24题的1115个Run非终止状态，从固定网格0.01/0.02/0.04/0.0625/0.1/0.125/0.1875/0.25选择总regret最接近Run者，平手按例外数差异、较小gate。借用同状态的Run剩余账本，仅作开发校准，不把离线重选当真实rollout或性能。旧状态Run47.585608、对照50.556339 regret，例外107/141；新题实际消耗不保证匹配，不在新题重调。
- 同一Qwen3-8B+DFlash、FP32/eager、greedy、batch1、非思考、四CPU线程。各题同卡五组交错；MATH 8..15 GPU0、16..23 GPU1，GSM 24..27 GPU0、28..31 GPU1。每个请求从原prompt及零账本开始。
- 自然EOS上限2048：每题五组各1机制+2无逐token插桩计时，共120机制+240自然计时。随后每题五组各2次128上限前缀计时，共240；总600正式生成，预热另计。前缀仍遵守EOS，不强制越过EOS。质量失败、截断或零激活均保留，不中止整个套件。
- 严格保持原prompt、Target输入和评分器；所有在线推理与验证成本计入，加载/预热/机制插桩另列。MATH主指标是保守boxed规范化精确匹配，不是符号等价；保存全部未过评分文本，单列人工等价审计，不更改原rows或主评分。

归因检查预先固定：

1. 新题是否保留自然时延、自然TPS及任务质量；逐题列出退化、答案变化和截断。
2. 恒等分解 `T = N × (rounds/N) × (T/rounds)`，报告输出长度、每token轮数和每轮成本的比率。此分解描述已实现路径，不等于三个独立因果效应。
3. 短前缀检查：报告全部128上限运行；另外只在五组全部达到128token的共同集合比较等token工作量。保留全组，不挑某一方法的成功样本。审计新短上限裁短验证块之前的输出与2048轨迹一致；最后裁短块可改变gate/bonus。该前缀实验只描述回答开头，不替代完整答案结果。
4. 对比Run与固定保守ASD：若保守组即可重现收益，不能归因为实际后缀信息的独特价值。匹配开发regret不保证新题regret或质量匹配，实测差异全部报告。
5. 同候选块、同剩余账本的离线比较：分别在五组实际非终止状态上重算Run、原ASD与Conservative的局部接受前缀，统计选择一致/各自多接、低收益支付及regret/局部增量。排除终止块以免未保存的proposal EOS混淆；分轨迹报告，不把跨分叉块相减当节约的Target前向。严格不使用重算前缀去提交未经相应前缀评分的token。

先运行：

```powershell
.venv\Scripts\python.exe -m scripts.prepare_asd_attribution --gsm-source artifacts/asd-attribution-gsm-source.jsonl --math-source artifacts/asd-budget-math500-source.jsonl
.venv\Scripts\python.exe -m pytest -q tests/test_asd.py tests/test_asd_budget.py tests/test_asd_completion.py tests/test_asd_attribution.py --basetemp .test-tmp-asd-attribution
```

每个服务器分片：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --data data/research/asd-attribution/records.json --policies configs/asd-attribution-policies.json --benchmark math500 --task-ids math500:8 math500:9 math500:10 math500:11 math500:12 math500:13 math500:14 math500:15 --max-new-tokens 2048 --prefix-tokens 128 --protocol docs/ASD_ATTRIBUTION_PROTOCOL.md --output results/raw/asd-attribution/math-a
```

另一卡运行math-b（16..23），各卡随后gsm-a（24..27）/gsm-b（28..31）；精确命令与退出码由run-pipeline.py保存。
冻结源码在独立`asd-attribution`目录，执行前后核验Git原始字节清单及旧发布证据；实验不会修改infra冻结分支或默认CLI/服务。
