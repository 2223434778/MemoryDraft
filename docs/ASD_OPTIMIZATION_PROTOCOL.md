# ASD 剩余优化空间排查（执行前冻结）

用户授权尝试剩余优化方向。本轮按有限实现与实测收益决定投入，不能穷尽所有算法；不重启已停止的置信度价格/未来二选一预测调参。独立服务器目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-optimization`，不改变已冻结infra引用或旧结果。

## 依据和边界

- ASD采用原B8/g.25/M2；Utility价格2，不改Target prompt、EOS、评分或请求账本规则。零预算仍严格。
- AdaFlash https://arxiv.org/html/2607.19223v3 使用部署分布蒸馏及自适应验证长度；DScale https://arxiv.org/html/2609.37532v1 面向高并发组织验证计算；BlockPilot https://arxiv.org/html/2606.31315v1 按请求选块长。本轮只借鉴思想、独立薄适配，不复制上游代码，不声称动态长度首创，也不称为这些系统的复现。
- 本运行时batch=1，不能用论文并发8..32收益预估本机收益。FP32/eager是原严格profile；其他profile只通过实验loader启用，不放宽生产配置限制。
- 全部是已观察开发任务，不作新题泛化或显著性结论。所有错误/截断/EOS保留，最终答案正确不保证推理全文正确。

## 阶段A：执行成本和长度

固定math500:1/4/7、gsm8k:21，2048自然EOS每臂1次、遵守EOS的128前缀每臂2次，组内按冻结种子交错。自然计时仅1次是可行性数据，不发布稳定百分比结论；配对128重复检查确定性。全部在线请求成本由原engine.generate计入。

FP32/eager有11臂：D-old/D-fast、U-old/U-fast、ASD-fast、D-verify7/U-verify7、D-draft8/U-draft8、U-recent/U-budget。D为严格；U为Utility；ASD为原最长可行前缀。fast只增量扫描本轮新增token查EOS，必须逐ID/账本与old一致。draft8指含anchor的8token块（7proposal），不超过原训练块16；verify7保留完整Draft16，只在Target调用前截到7proposal，不能退还已付Draft成本。

动态长度仅取7或原15proposal；前4轮全长，之后前4轮平均推进<=4取7，否则15。budget臂额外在已花regret>=6时取7。选择只依赖过去推进与当前账本，不读取当前Target验证、未来标签/答案，不训练预测器。截短后按实际验证块计算原ASD可行域，原始regret实际扣费、不恢复已花预算。

另测FP32/SDPA和BF16/SDPA，每个profile均D-fast/ASD-fast/U-fast三臂。每题每profile另跑严格AR128，记录实际首分歧，不把跨shape浮点差异掩盖或当作无损。SDPA实际Target/Draft配置及dtype记录在provenance，基础RuntimeConfig保持原合同。每profile每worker额外一个128token profiler，仅作带插桩成本定位，排除主计时。

主检查共4题*(11+3+3)*3=204次生成，另12个AR128、6个profiler及短warmup（单列）。两卡独立进程；同题同profile始终同卡。

## 阶段B：冻结迁移规则

迁移题固定math500:8/9、gsm8k:24/25，均既有题。始终执行FP32/eager的D-fast/ASD-fast/U-fast。

- 执行profile：在A中D-fast等128前缀总时间比原profile低至少5%、自然正确数不低于原profile，才有资格迁移；只迁移合格者中128总时间最少的profile，平局保持原profile。不能将相同自然正确数当质量非劣保证。
- 长度：各候选分别比较同profile D-fast或U-fast；A中等128总时间低至少5%、自然正确数不低且自然截断数不高，才迁移该候选。没有合格者就停止该长度策略，不换阈值。若双方未达到128，则本轮该候选不按短输出作速度通过。
- 所有选中臂在B自然1次/128两次，保留基线，不在B重选超参。选择器实现与协议在GPU前冻结。

## 阶段C：Draft前缀适应诊断

复用上一轮4题16冻结源状态，每状态proposal/recovery两个forced token，共32条严格128续写。相同公共前缀、独立重新prefill、记录第一Draft块接受、后续轮和实际token。不是完整Utility动作比较，不是在线计时，不沿用改变token前的suffix logits。

只有proposal分支第一块平均严格接受比recovery低>=20%，且双方达到128的共同集合中proposal的每token轮数高>=10%，才值得追加小训练可行性阶段；两项均需共同集>=8状态/3题。否则停止“ASD前缀适配”训练投入。即使通过，也只是必要性诊断：两个前缀本身可有不同难度，不能证明训练必能修复；需另立等训练成本的普通蒸馏对照。

## 执行和审计

冻结Git与完整原始blob manifest，部署前后逐文件SHA。旧证据受保护；新结果位于`results/raw/asd-optimization`，发布为`results/asd-optimization`，JSONL无损gzip。禁止上传权重、大型profiler trace、私有记录。新增API必须有真实tiny Target/KV、流式/EOS及错误参数测试。

服务器准确入口：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_optimization_pipeline`。pipeline按GPU0 math1/math4、GPU1 math7/gsm21依次启动三个profile；B分为GPU0 math8/gsm24、GPU1 math9/gsm25。
