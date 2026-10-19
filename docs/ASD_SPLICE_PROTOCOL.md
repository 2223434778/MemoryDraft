# ASD 与历史拼接：直接开发实验

2026-10-19。用户要求以合理结合逻辑直接实施，不再增加机会门槛前置阶段。

假设：DFlash 短前缀定位旧接续；ASD 在已有 Target 分数上接受少量低 logit 差距的 token，可能释放长历史尾部。
这是一条近似解码实验路径，允许输出改变，不计入既有严格无损结果。默认 runtime/CLI/服务仍使用严格验证。

来源：Approximate Speculative Decoding v3，https://arxiv.org/abs/2608.03447v3。
官方代码 https://github.com/Kissmetothemoon/ASD ，核验提交 `9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6`，Apache-2.0。
本次按论文公式独立实现 GPU tensor 选择，不复制第三方源码、不引入新模型或依赖。

ASD 每请求预算 B=2，局部门槛 g=0.25，每块最多2例外；为小成本开发配置，不声称复现论文 Qwen3-14B/DSpark 的 B=8 实验。
原规则价值为剩余候选长度；唯一新变体取 `1 + 下一个不匹配前连续匹配数`，用已计算的 Target argmax，不额外前向。
每轮必须提交连续前缀；预算跨 R/D 路由累计。EOS 之后未交付的例外不计费。
零预算按 token identity 截断，包括不同 token 的并列最大 logits。

使用旧8请求：payment-runbook:0/1、shipment-contract:0/1、inventory-query:0/1、edge-router:1、short-settings:1。
覆盖已知拼接、引导收益及短控制，全部已观察，5项目，不作泛化确认；不按本轮结果删样本。
旧 prompt、合法时间快照、参考答案和配置保持；当前输出不写回历史。

9方法：DFlash / Suffix / Guided-Edit / Splice-Edit 各自 strict 与 ASD，加 Splice-ASD-Run。
不叠加 OT 或旧风险预测器。拼接仍用原候选池、阈值、最长32；DFlash通常15候选。
原版 ASD 的剩余长度门槛随 K 改变，因此原版组合收益不能自动归因于历史质量；Run 是一项控制该现象的消融。

每方法/请求先1次机制运行，strict 对照实际使用 B=0 并与旧 AR 比较；再2次自然 EOS 交错计时，输出须与机制相同。
最后2次固定64 token计时，明确忽略EOS，仅衡量等工作量吞吐；尤其短控制继续生成的结果不能作任务答案或质量证据。
合计72机制+144自然EOS+144固定工作量。所有已规划方法完成，不设速度/任务质量停止门槛；输入、缓存和实现错误须修复。
全部在线检索、特征、同步和前向计入；机制计时与正式时延分开。
近似组分别报告 AR 一致率、任务合同、输出长度、近似 token 数、累计 regret、验证轮及自然EOS时延。
同预算不意味着同质量。旧配置任务中的数值/标识符错误可能被放宽，任务检查必须保留。

运行：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_splice_probe --config configs/runtime-ot-benefit.json --output results/raw/asd-splice
```

服务器独立目录 `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-splice`；沿用现有 Qwen3-8B/DFlash、FP32/eager、greedy、A6000。
旧 infra、OT、风险结果保留；记录实际运行源码和数据哈希。
