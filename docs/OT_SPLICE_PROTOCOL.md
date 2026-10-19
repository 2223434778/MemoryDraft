# Draft内部锚点拼接：小规模开发实验

2026-10-19。此轮只使用已经观察过Target答案的任务，不是新holdout确认。

- 复用原诊断8请求，以及上一轮Guided验证的edited-continuation 8请求，共16。输入、历史、AR参考不变；全部原结果目录逐文件SHA保护。参考答案只在生成结束后评分，不进入检索器。
- 六组：Suffix-R、Guided-Edit、Guided-OT、Splice-Edit、Splice-OT、Splice-TailOT。原方法不改。精确suffix优先，未命中后仍支付一次正常DFlash前向；检索和拼接复用现有post-draft hook，验证器不改。
- 拼接在15个Draft候选内枚举所有4-token内部锚点，最多12窗口、每窗口64个后缀项；记录/特殊token边界内，锚点前至少8历史token。保留Draft到锚点结束的前缀，再接锚点之后的历史原文，合计最多32token，需长于原Draft块。
- 三个Splice组先按同一Edit分数去重相同候选，保留最高cheap分、短Draft前缀、早历史位置的表示；再同样保留top8。排序平分同样优先短Draft前缀、早历史位置，阈值均0.65。最多4次查询，包括无候选。该cheap筛选可能删除OT本可选中的候选，是固定预算下的限制。
- Splice-Edit使用原Edit分数；Splice-OT使用原词面Sinkhorn分数；Splice-TailOT使用完全相同的传输计划，将整体行贡献和末8个query位置的单位质量贡献各加权一半。没有新增语义编码器、单调约束或训练；不把该启发分数解释为接受概率，也不声称复现OPW。
- 明确统计保留Draft前缀接受量、历史尾部接受量，以及超出原Draft长度的接受量。后者仍不是相对其他方法的净增量；净效果看配对总轮和端到端时间。Draft前缀拒绝不能映射成历史端点失败。未确认候选仍全部由固定Target逐token验证。
- 机制阶段每组每请求一次，共96次，独立包装器保存调用/轮次；不保存插桩总时延作为性能证据。若某一个Splice组至少在2请求上同时少于两个Guided基线的总轮，且各请求至少一次内部拼接的事后连续接受长度高于同状态原Draft，则允许进入计时。
- 若门槛失败，停止该机制的GPU扩展并报告。若通过，固定同一package/runner/任务，16请求×6方法×2重复，共192次独立未插桩计时，顺序预定随机交错。不能依据结果调参后重测同批宣称确认。
- 全部token/prompt/snapshot严格核对既有AR，首个不一致即停止。沿用Qwen3-8B、原DFlash、FP32/eager、4线程、batch1、原输出预算。计时包含快照、分词、索引、GPU→CPU候选同步、全部候选搜索/评分、Draft、Target验证、KV维护及解码；模型加载/预热单列。
- 即使计时有收益，此轮也只说明已观察任务上的开发信号。OT增量必须相对同池同拼接的Splice-Edit单列。适用场景和新项目验证需后续独立固定；不以本批结果证明新颖性或生产泛化。

机制参考：[RASD](https://aclanthology.org/2025.findings-acl.320/)的模型/检索候选融合；[OPW](https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Order-Preserving_Wasserstein_Distance_CVPR_2017_paper.html)仅作为后续序列顺序约束的相关工作，本轮未实现该算法。未复制新增上游代码。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_splice_probe --stage mechanism --config configs/runtime-dflash.json --output artifacts/mechanism-v1
# 仅机制门槛通过才执行：
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_splice_probe --stage timing --config configs/runtime-dflash.json --mechanism artifacts/mechanism-v1 --output artifacts/timing-v1
```
