# ASD 原论文任务上的小成本组合检查

2026-10-20。用户指出此前一直沿用旧编辑任务，要求尝试 ASD 使用的任务。
选 GSM8K test 前8条、MBPP 官方测试 task_id 11–18，按来源顺序固定，不按模型输出筛选。
来源/固定提交/许可在 `data/research/asd-bench/records.json`，保留官方 MIT/Apache-2.0 文本。

模型继续使用现成 Qwen3-8B + DFlash、FP32/eager、greedy、非思考模式；同批 Target prompt 始终相同。
这是同 benchmark 的组合实验，未复现原论文 DSpark、Qwen3-14B、原始完整提示或数据切分。
ASD参数改用官方stable：B=8、g=0.25、M=2；不按本批结果扫阈值。

四方法：DFlash、DFlash-ASD、Splice-Edit、Splice-Edit-ASD。
每题使用不同 user_id 与空历史，检索资源只有当前输入文本（MBPP含公开测试）。
参考解法、答案标签、当前生成及其他题目的生成不入索引。
因此本轮首先判断ASD在数学/代码生成上能否工作；不据此宣称长期记忆价值。

GSM8K零样本指令要求逐步解答，并用 `#### <answer>` 结束，另支持boxed数字。
MBPP输入题目和原公开3测试，输出实现；不提供当前题目的参考代码。
GSM8K对最终数值评分；MBPP在限制import/动态调用的全新限时子进程执行公开测试。
该MBPP评分是公开测试通过，不代表隐藏测试或一般安全sandbox。
所有评分在生成计时之外执行；质量错误/输出截断不阻断整轮。

每题四方法各1机制（strict使用B=0），保存紧凑regret、例外、路线；DFlash strict作为本批参考，不额外重跑AR。
每方法2次自然EOS计时（最多512token），再2次固定128token（忽略EOS，仅吞吐）。
共64机制+128自然+128固定，共320次生成；所有预定样本完成。
自然EOS输出必须与对应机制一致；严格拼接必须与DFlash strict相同。
零预算、预算累计、真实近似KV路径由已有测试覆盖。
分别报告两benchmark的正确数、截断、长度、回合、例外及全部在线成本。
小样本质量差异只报告计数，不声称非劣性；不混合不同指标凑总准确率。

运行（独立服务器 `asd-bench`）：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_bench_probe --config configs/runtime-ot-benefit.json --output results/raw/asd-bench
```
