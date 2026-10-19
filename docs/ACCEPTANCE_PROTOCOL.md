# 接受长度校准诊断（执行前冻结）

日期：2026-10-19。目标是修复旧收益门控的接受长度预测，不改变 Target 验证，不恢复完整 OT 对齐矩阵方案。默认 suffix 和独立 infra 版本不变。

## 有界采集

复用 `splice_probe.inputs` 的 16 个已观察任务，运行 Splice-Edit、Splice-OT、Splice-TailOT 各一次，共 48 次。使用 `configs/runtime-ot-benefit.json` 的 FP32/eager、四线程、greedy 配置。每次严格核对原 AR token、prompt、用户/时间快照；其中旧 matrix 八请求的 24 条路径还核对原计数。

在每次实际 Draft 前向后，读取现有 logits；不额外调用 Target。记录每位置 top1 概率、归一化熵、top1/top2 概率差，以及候选对应 token 的 Draft 概率。只保留小型特征，不保存完整词表 logits。只读 shadow shortlist 可越过线上四次查询上限，必须标注为离线候选，不能算线上实际提议。去重身份为请求、位置、完整 Draft、候选 token/边界/位置，重复路径不增加样本。

原 14 个固定状态必须匹配完整 Draft，并采集原候选的置信度，以免用不同候选替换困难样本。所有接受标签只在生成通过 strict 检查后，用已有 AR 答案事后计算；答案不进入检索库或特征。新增状态来自同批已观察项目，是开发诊断，不是独立 holdout。

## 预测与消融

估计生存曲线 `S(k)=P(A>=k)`，预计接受长度为 `sum(S[:L])`。每次拟合排除当前整个项目；至少保留三个其他项目。训练只用新采集的普通 shadow 候选，原 14 帧的额外候选不另加训练权重。

- Draft 头：截断位置/15、前缀最小和平均 top1 概率、最大归一化熵、最小概率差。
- 历史头：只使用 Draft 已通过拼接边界的训练候选，估计条件接续生存；特征为历史长度/32、边界/15、候选 cheap 分数、与 Draft 的 LCP/15、是否当前 prompt、Draft 可观察区间比例、该区间历史 token 最小和平均概率。
- 拼接前共享 Draft 生存；拼接后为 `S_D(boundary) * S_history(k | boundary passed)`。两头曲线用累计最小值保证单调。完整接受的块右删失，不虚构下一 token 拒绝标签。
- 项目/状态/候选/截断位置分层等权。固定正则 logistic：截距 penalty 0.0001、其余 0.01，L-BFGS-B maxiter 300、ftol 1e-12、gtol 1e-8。不根据本轮结果调参。
- no-confidence 消融把所有概率/熵/概率差特征置零，其他训练规则相同。

## 判定

主要比较原 14 状态、原策略已选动作及原长度：旧邻居预测、条件模型无置信度、条件模型有置信度。报告 MAE、偏差、逐位置 Brier；旧生存曲线由旧邻居接受长度的加权经验分布恢复，并检查积分等于原预测。额外报告全部去重状态/候选指标、Edit/TailOT 同状态排序和 payment/shipment 失败例。

预先设定开发通过条件：14 状态完整覆盖；新模型 MAE 相对旧模型至少降低 20%；Brier 优于旧模型；MAE 和 Brier 均优于 no-confidence 消融；按项目等权、固定种子 20261019、5000 次项目 bootstrap 的旧减新 MAE 改善 95% 区间下界大于零。样本小，bootstrap 只是开发不确定性描述，不能消除跨折训练依赖或代替独立测试。

条件不满足就报告局部改善/失败，不上线新路由、不换阈值重跑。不宣称吞吐改善：置信度计算、shadow 检索、离线校准有成本，本轮插桩时间不用于速度比较。即使通过，也只证明接受预测在该开发集改善，须独立数据及全成本路由测试后才能替换运行策略。

## 命令

服务器工作目录为新独立 `/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-acceptance`，模型软链接指向父目录合法缓存。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.collect_acceptance_probe --config configs/runtime-ot-benefit.json --output results/raw/acceptance/collection
OMP_NUM_THREADS=4 PYTHONPATH=. ../.venv-dflash/bin/python -m scripts.evaluate_acceptance_probe --input results/raw/acceptance/collection --output results/raw/acceptance/evaluation
```

设计参考：[SVIP, arXiv:2411.18462v2](https://arxiv.org/abs/2411.18462v2) 使用 Draft 不确定性支持自适应长度。这里借鉴低成本风险信号，不把熵直接当正确概率，也不声称复现 SVIP 或其结论自动适用于 DFlash。
