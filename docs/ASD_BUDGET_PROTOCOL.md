# 按实际前缀收益分配 ASD 预算

2026-10-20。用户授权验证第一方向。只做线性候选上的收益选择，不启用检索、训练、OT或动态树。

旧 GSM8K 0..7 是已观察开发数据：48个近似接受块中29个仅多接受1–2候选token，消耗约63.6% regret。
这些同状态局部量不能换算成完整请求净加速。本轮新样本在生成前固定：GSM8K test位置8..23（16题），
MATH-500官方HF文件位置0..7（8题），不筛题、不将参考解法或当前生成写入历史。
HF MATH卡未单独声明许可，卡片指定的原始openai/prm800k仓库MIT notice随子集保存；保留来源SHA/commit。

四组均同一Qwen3-8B/DFlash、FP32/eager、greedy、非思考、相同Target输入、自然EOS、最多512token。
严格DFlash是本批参考，不额外跑AR。近似组B=8/g=0.25/M=2，请求预算始终累积、不按块重置。

- DFlash：计时走原严格验证；机制用B=0记录。
- ASD：原剩余块长价值，最长可行前缀。
- ASD-Run：价值为当前例外加后续连续精确匹配长度，门槛不变；这是更严格的局部规则。
- ASD-Utility：保留ASD原可行集合，选择最大化“额外提交token数 − 2 × regret”的连续前缀。
  比较基线是同状态严格前缀；包含recovery/bonus，遇proposal EOS截断收益，禁止计算EOS后尾部。
  同效用先低regret，再长前缀；只对实际选中的前缀计费。价格2在本批前冻结，不在新结果上扫描。

Utility使用已评分当前块，不预测当前历史尾部长度。它不保证未来有更好机会，也不保证质量。
这一轮是固定价格的最小检验；先不加入依赖未知剩余输出长度的预算节奏控制。

每题4机制+8计时（2次交错重复），共96机制、192自然EOS计时，总288次。
同一题的所有方法在同一GPU交错运行；GSM8K用物理GPU0，MATH-500用物理GPU1并行。
不跨benchmark比较GPU绝对速度，也不合并两类质量指标。插桩时间和模型加载不计正式时延，
所有在线选前缀、同步、起草、验证、缓存、token化和文本解码计入。
机制和计时输出必须相同，预算与输入身份必须一致；质量失败、零激活或截断不停止整轮。

GSM8K用既有最终数值评分；MATH用平衡括号提取最后boxed、去空白/left/right/text/角度记号等后精确匹配。
不使用宽泛的字符串末尾猜数字，也不声称一般符号等价。保存原文和解析结果，对所有不匹配单列复查，
不得静默修标签。参考解法必须全部通过评分器检查。

报告每任务族的质量/截断、自然EOS耗时、token数、自然输出TPS、轮数、regret、例外、
主动缩短块数、局部额外接受量与预算使用位置；长度改变时不将时延变化全归为推理效率。
不做忽略EOS的固定工作量阶段。两次重复的小样本不声称统计显著或质量非劣性。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark gsm8k --output results/raw/asd-budget/gsm8k
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark math500 --output results/raw/asd-budget/math500
```
