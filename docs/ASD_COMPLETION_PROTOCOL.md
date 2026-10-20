# ASD 截断数学题的完整解答复查

2026-10-20。接上一轮预算实验，用户授权补全此前3道MATH截断题。
预先固定math500:1、math500:4、math500:7；这是已观察题的诊断，不是新题泛化测试。
保留DFlash、原ASD、ASD-Run、ASD-Utility四组，B8/g0.25/M2、Utility价格2不变。
输出上限统一2048，避免1024仍截断导致再次运行；模型、权重、FP32/eager、greedy、非思考、题目、提示及评分器不变。
若2048仍截断，也保留原样报告，不把未完成解答当成完整质量证据。

每题4机制+8自然EOS计时（两次交错重复），共12机制+24计时=36生成；另计预热。
math500:1/4在物理GPU0，math500:7在物理GPU1；同题四组同卡交错，全部在线成本计入。
新进程从原始prompt重新生成，请求账本从零起始；不拼接截断文本或重置中途ASD预算。

逐方法核对旧512token输出与新输出的公共前缀。旧上限会裁短最后的验证块，
使ASD剩余长度分母变化，因此要求旧首次裁短块之前的输出相同，另外记录完整512前缀是否相同。
前缀核对只作离线审计，不向模型输入旧输出、当前答案或测试标签。
任务输入、评分器和runtime package SHA必须与旧实验一致。

报告完整答案正确数/截断、自然EOS时延、token数、自然TPS、验证轮、regret、例外和主动缩短。
逐题列出结果，不丢弃质量失败或退化；保留原512结果，并列解释截断结论与完整解答结论。
自然输出长度变化不能冒充相同工作量加速；没有忽略EOS的强制续写阶段。
3题只是补全诊断，不做显著性或质量非劣性保证，不在本批调整门槛。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark math500 --task-ids math500:1 math500:4 --max-new-tokens 2048 --previous results/asd-budget --protocol docs/ASD_COMPLETION_PROTOCOL.md --output results/raw/asd-completion/part-a
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark math500 --task-ids math500:7 --max-new-tokens 2048 --previous results/asd-budget --protocol docs/ASD_COMPLETION_PROTOCOL.md --output results/raw/asd-completion/part-b
```
