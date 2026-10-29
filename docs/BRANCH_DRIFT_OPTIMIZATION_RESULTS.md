# 按需JS与门控节点分配：七任务开发检查

固定开发诊断子集，单次配对计时。负数更快；自然回答与共同128前缀分别报告。
AR参考用于逐token保真和既定质量检查；开放任务没有新增偏好评分。

完成924计时、66匹配输入；Lazy/eager的132对token、停止、逐轮路径和预算完全一致。

|任务|方法|自然 vs ASD|自然 vs Lazy|自然 vs 严格树|AR相同|正确/可评分|截断|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|ASD|+0.00%|-6.76%|-0.22%|0/8|8/8|0|
|gsm8k|BudgetTree|+3.23%|-6.44%|+0.00%|8/8|8/8|0|
|gsm8k|DriftTree-ASD|+10.37%|+0.09%|+7.06%|4/8|8/8|0|
|gsm8k|LazyDrift-ASD|+10.27%|+0.00%|+6.97%|4/8|8/8|0|
|gsm8k|GateTree-ASD|+6.86%|-2.98%|+3.71%|6/8|8/8|0|
|gsm8k|TokenCap-ASD|+10.23%|-0.04%|+6.93%|5/8|8/8|0|
|gsm8k|GateCap-ASD|+6.91%|-2.93%|+3.76%|6/8|8/8|0|
|math500|ASD|+0.00%|-3.61%|+30.45%|0/12|8/12|0|
|math500|BudgetTree|-5.55%|-14.54%|+0.00%|12/12|8/12|0|
|math500|DriftTree-ASD|+11.16%|+0.14%|+37.48%|3/12|8/12|1|
|math500|LazyDrift-ASD|+11.01%|+0.00%|+37.30%|3/12|8/12|1|
|math500|GateTree-ASD|-4.49%|-13.11%|+1.64%|5/12|8/12|1|
|math500|TokenCap-ASD|+1.96%|-8.79%|+6.71%|4/12|7/12|0|
|math500|GateCap-ASD|-4.95%|-14.00%|+0.61%|7/12|8/12|1|
|humaneval|ASD|+0.00%|+3.24%|+8.03%|5/8|7/8|0|
|humaneval|BudgetTree|-5.17%|-4.56%|+0.00%|8/8|7/8|0|
|humaneval|DriftTree-ASD|-0.31%|+0.07%|+5.11%|7/8|6/8|0|
|humaneval|LazyDrift-ASD|-0.36%|+0.00%|+5.04%|7/8|6/8|0|
|humaneval|GateTree-ASD|-4.96%|-3.41%|+1.10%|8/8|7/8|0|
|humaneval|TokenCap-ASD|-1.55%|-1.11%|+3.81%|8/8|7/8|0|
|humaneval|GateCap-ASD|-4.82%|-3.27%|+1.25%|8/8|7/8|0|
|mbpp|ASD|+0.00%|+15.00%|+7.36%|2/8|5/8|0|
|mbpp|BudgetTree|-5.24%|+8.91%|+0.00%|8/8|4/8|0|
|mbpp|DriftTree-ASD|-10.83%|+0.20%|-4.70%|7/8|5/8|0|
|mbpp|LazyDrift-ASD|-10.98%|+0.00%|-4.88%|7/8|5/8|0|
|mbpp|GateTree-ASD|-1.04%|+13.26%|+4.50%|8/8|4/8|0|
|mbpp|TokenCap-ASD|-11.05%|-0.06%|-4.94%|7/8|5/8|0|
|mbpp|GateCap-ASD|-0.94%|+13.39%|+4.60%|8/8|4/8|0|
|mmlu|ASD|+0.00%|+5.13%|+4.27%|6/14|1/14|0|
|mmlu|BudgetTree|-0.78%|+1.47%|+0.00%|14/14|5/14|0|
|mmlu|DriftTree-ASD|-2.46%|+0.01%|-1.09%|11/14|5/14|0|
|mmlu|LazyDrift-ASD|-2.49%|+0.00%|-1.09%|11/14|5/14|0|
|mmlu|GateTree-ASD|-2.00%|-0.05%|-1.35%|10/14|5/14|0|
|mmlu|TokenCap-ASD|-2.02%|+0.42%|-0.69%|12/14|5/14|0|
|mmlu|GateCap-ASD|-2.08%|-0.09%|-1.36%|11/14|5/14|0|
|mtbench|ASD|+0.00%|+10.98%|+13.97%|0/8|—|0|
|mtbench|BudgetTree|-11.14%|-2.46%|+0.00%|8/8|—|0|
|mtbench|DriftTree-ASD|-8.78%|+0.12%|+2.84%|2/8|—|0|
|mtbench|LazyDrift-ASD|-8.89%|+0.00%|+2.72%|2/8|—|0|
|mtbench|GateTree-ASD|-8.70%|+0.28%|+2.85%|3/8|—|0|
|mtbench|TokenCap-ASD|-9.00%|-0.13%|+2.59%|2/8|—|0|
|mtbench|GateCap-ASD|-11.08%|-1.92%|+0.58%|4/8|—|0|
|alpaca|ASD|+0.00%|+4.63%|+6.90%|1/8|—|0|
|alpaca|BudgetTree|-4.40%|-2.02%|+0.00%|8/8|—|0|
|alpaca|DriftTree-ASD|-2.33%|+0.10%|+2.25%|5/8|—|0|
|alpaca|LazyDrift-ASD|-2.42%|+0.00%|+2.15%|5/8|—|0|
|alpaca|GateTree-ASD|-0.74%|+1.99%|+4.13%|5/8|—|0|
|alpaca|TokenCap-ASD|-1.36%|+0.83%|+2.91%|7/8|—|0|
|alpaca|GateCap-ASD|-1.45%|+1.33%|+3.47%|6/8|—|0|

## 相同生成量

|任务|方法|共同128 vs ASD|有效数|共同128 vs Lazy|有效数|
|---|---|---:|---:|---:|---:|
|gsm8k|LazyDrift-ASD|+3.07%|8|+0.00%|8|
|gsm8k|GateTree-ASD|-0.26%|8|-3.19%|8|
|gsm8k|TokenCap-ASD|+3.05%|8|-0.02%|8|
|gsm8k|GateCap-ASD|-0.33%|8|-3.26%|8|
|math500|LazyDrift-ASD|+7.45%|12|+0.00%|12|
|math500|GateTree-ASD|+3.32%|12|-3.50%|12|
|math500|TokenCap-ASD|+8.47%|12|+0.88%|12|
|math500|GateCap-ASD|+3.94%|12|-2.88%|12|
|humaneval|LazyDrift-ASD|—|0|—|0|
|humaneval|GateTree-ASD|—|0|—|0|
|humaneval|TokenCap-ASD|—|0|—|0|
|humaneval|GateCap-ASD|—|0|—|0|
|mbpp|LazyDrift-ASD|—|0|—|0|
|mbpp|GateTree-ASD|—|0|—|0|
|mbpp|TokenCap-ASD|—|0|—|0|
|mbpp|GateCap-ASD|—|0|—|0|
|mmlu|LazyDrift-ASD|-0.63%|4|+0.00%|4|
|mmlu|GateTree-ASD|-1.72%|4|-1.10%|4|
|mmlu|TokenCap-ASD|+1.24%|4|+1.83%|4|
|mmlu|GateCap-ASD|-1.73%|4|-1.11%|4|
|mtbench|LazyDrift-ASD|+2.77%|6|+0.00%|6|
|mtbench|GateTree-ASD|+1.28%|6|-1.10%|6|
|mtbench|TokenCap-ASD|+2.77%|6|+0.00%|6|
|mtbench|GateCap-ASD|+1.17%|6|-1.19%|6|
|alpaca|LazyDrift-ASD|-2.67%|5|+0.00%|7|
|alpaca|GateTree-ASD|-1.21%|5|+1.32%|7|
|alpaca|TokenCap-ASD|-2.38%|5|+0.26%|7|
|alpaca|GateCap-ASD|-2.21%|5|+0.68%|7|

## 检查量与在线成本

下表自然回答汇总。gate和JS为CPU取值边界的wall摘要，JS包含在gate中，二者不能相加。构树入口wall可能包含尚未完成的Draft GPU工作；不能全归因于CPU构树，未插入额外同步单独剖析。未记录项用—，全部在线工作均在总请求wall内。

|方法|AR相同|近似token|JS边|JS批|跳过边|gate秒|JS秒|平均节点|平均树深|构树入口wall秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ASD|14|355|0|0|0|0.000|0.000|0.00|0.00|—|
|BudgetTree|66|0|0|0|0|0.000|0.000|24.88|8.29|—|
|DriftTree-ASD|39|60|5848|2250|0|1.257|0.707|28.59|8.29|25.873|
|LazyDrift-ASD|39|60|737|683|5111|0.861|0.211|28.59|8.29|25.861|
|GateTree-ASD|45|44|479|448|2452|0.685|0.138|16.79|8.64|24.290|
|TokenCap-ASD|45|44|476|451|2424|0.742|0.139|28.40|7.88|23.731|
|GateCap-ASD|50|27|298|287|1293|0.637|0.090|16.75|8.60|24.333|

## 诊断题与新增失败

|输入|方法|秒|token|正确|AR相同|停止|
|---|---|---:|---:|---|---|---|
|math500:372|DriftTree-ASD|33.525|4096|False|False|length|
|math500:372|LazyDrift-ASD|33.502|4096|False|False|length|
|math500:372|GateCap-ASD|7.619|824|True|True|eos|
|math500:372|ASD|30.061|3275|False|False|eos|
|math500:372|TokenCap-ASD|8.001|824|True|False|eos|
|math500:372|GateTree-ASD|7.622|824|True|True|eos|
|math500:372|BudgetTree|7.662|824|True|True|eos|
|math500:101|GateCap-ASD|31.549|4096|False|False|length|
|math500:101|DriftTree-ASD|37.233|3539|True|False|eos|
|math500:101|GateTree-ASD|29.141|4096|False|False|length|
|math500:101|TokenCap-ASD|39.066|3460|False|False|eos|
|math500:101|ASD|18.994|2072|True|False|eos|
|math500:101|LazyDrift-ASD|37.187|3539|True|False|eos|
|math500:101|BudgetTree|30.983|3316|False|True|eos|
|humaneval:76|TokenCap-ASD|0.810|59|True|True|eos|
|humaneval:76|GateCap-ASD|0.779|59|True|True|eos|
|humaneval:76|ASD|0.825|59|True|True|eos|
|humaneval:76|DriftTree-ASD|0.884|58|False|False|eos|
|humaneval:76|BudgetTree|0.799|59|True|True|eos|
|humaneval:76|LazyDrift-ASD|0.885|58|False|False|eos|
|humaneval:76|GateTree-ASD|0.777|59|True|True|eos|

## 复现与证据

规则、固定参数与任务选择见[协议](BRANCH_DRIFT_OPTIMIZATION_PROTOCOL.md)。
源码身份、实际命令与恢复结果见[公开证据](../results/branch-drift-optimization/)。
`python -m scripts.report_branch_drift_optimization --input results/branch-drift-optimization`可从压缩发布数据重建。
