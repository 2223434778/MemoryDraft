# 前缀风险拼接续测：解码一致性与任务质量分别报告

2026-10-19。用户明确要求继续测试，认为原停止条件过严。本修订在任何风险机制/计时结果之前冻结，仅调整评测执行条件；不调整算法、模型、候选、权重、阈值、长度或Target输入。首次停止的协议、12个AR和报告全部保留原字节。

## 固定设置

沿用原6项目/12请求、五组Suffix-R / Guided-Edit / Splice-Edit / Risk-NoConfidence / Risk-Splice，以及原56状态/9项目训练系数。0.5风险边界规则、32token上限、top8/4次查询、精确suffix优先、冷却和严格Target greedy验证均不变。使用已修复FP32均匀熵边界的交付代码。

所有12个Target答案已观察；本轮称为项目隔离的开发续测，不再称新holdout。旧训练项目与这些项目仍无交集，但不能因此把答案已知的本轮当独立确认。没有按当前答案调整参数或写回历史。

复用原`results/prefix-risk/mechanism/fresh-ar.jsonl.gz`的全部12个AR参考，不重新生成、不删失败例。核对压缩文件SHA、原provenance、输入/校准SHA、模型revision/精度/attention/线程/模板/预算，再用当前tokenizer核对每个prompt身份与token数。原AR构造52.955987s是历史成本，本轮不重复支付、不混入在线延迟。

## 修订的执行条件

- 硬停止：任何加速输出token/prompt/合法快照不匹配相应AR；内存隔离、KV/计数审计、来源核验不通过；执行异常或无效计时。这些保证不放宽。
- Target任务合同错误不再终止解码器评测。vector-ingest:0的正文遗漏单列，五组仍必须逐token复现同一AR；不能声称加速器修复任务输出或该错误输出可投入实际编辑流程。
- 完整运行60次机制检查。即使Risk没有改选候选，也继续120次未插桩交错计时，以测量实际开销和不触发情况。仅实际风险改变、轮数变化、端到端成本才能解释增量；相同路径小计时差只视为成本/噪声观察。
- 原3%/1%/项目数等继续投入门槛保留为描述性指标，不作为收集120次计时的前置条件，不以单个任务失败或小收益提前停止。两重复与原种子顺序不变，不反复增加重复挑有利结果。

## 报告要求

主结果包含全部12请求；主场景8请求/4项目和控制4请求/2项目分别报告，同时单列合同正确11请求与错误1请求，避免把生成错误短答案的速度当可用任务收益。报告每请求/项目时间、严格一致数、机制/计时路径一致数、Draft前向/Target轮数、风险评估及改选、候选同状态接受差异。所有softmax/熵/CPU-GPU同步/索引/检索/策略/KV/前向/解码成本计入在线请求延迟；加载/预热和原AR成本单列。

原结果与新的续测目录分开：服务器新`/media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk-continuation`，输出`results/raw/prefix-risk-continuation/{mechanism,timing}`；发布到`results/prefix-risk/continuation`，不覆盖原provenance、报告或publication。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --continue-evaluation --stage mechanism --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk-continuation/mechanism
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_prefix_risk_probe --continue-evaluation --stage timing --config configs/runtime-ot-benefit.json --mechanism results/raw/prefix-risk-continuation/mechanism --output results/raw/prefix-risk-continuation/timing
```
