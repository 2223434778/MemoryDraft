# 保真门控固定的树宽度实测

66匹配输入、792新计时，单次配对开发检查；负数更快。全部错误和截断保留。

|任务|方法|自然 vs ASD|自然 vs 惩罚1|自然 vs 严格树|AR相同|正确/可评分|截断|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|ASD|+0.00%|-3.89%|-0.23%|0/8|8/8|0|
|gsm8k|BudgetTree|+3.24%|-3.46%|+0.00%|8/8|8/8|0|
|gsm8k|GateCap-ASD|+6.81%|+0.00%|+3.67%|6/8|8/8|0|
|gsm8k|WidthHalf-ASD|+8.76%|+1.15%|+4.71%|4/8|8/8|0|
|gsm8k|WidthZero-ASD|+6.27%|-0.87%|+2.63%|5/8|8/8|0|
|gsm8k|WidthFeedback-ASD|+5.38%|-1.44%|+2.09%|5/8|8/8|0|
|math500|ASD|+0.00%|+29.95%|+30.50%|0/12|8/12|0|
|math500|BudgetTree|-5.58%|-0.61%|+0.00%|12/12|8/12|0|
|math500|GateCap-ASD|-4.93%|+0.00%|+0.65%|7/12|8/12|1|
|math500|WidthHalf-ASD|-7.69%|-1.39%|-0.82%|6/12|9/12|0|
|math500|WidthZero-ASD|-6.60%|-1.18%|-0.59%|5/12|9/12|0|
|math500|WidthFeedback-ASD|-10.83%|-3.80%|-3.24%|7/12|9/12|0|
|humaneval|ASD|+0.00%|+6.01%|+8.16%|5/8|7/8|0|
|humaneval|BudgetTree|-5.25%|-0.94%|+0.00%|8/8|7/8|0|
|humaneval|GateCap-ASD|-4.79%|+0.00%|+1.39%|8/8|7/8|0|
|humaneval|WidthHalf-ASD|-4.98%|-0.46%|+0.67%|8/8|7/8|0|
|humaneval|WidthZero-ASD|-4.68%|-0.36%|+0.58%|8/8|7/8|0|
|humaneval|WidthFeedback-ASD|-4.12%|+0.73%|+2.15%|8/8|7/8|0|
|mbpp|ASD|+0.00%|+3.63%|+7.48%|2/8|5/8|0|
|mbpp|BudgetTree|-5.33%|-3.59%|+0.00%|8/8|4/8|0|
|mbpp|GateCap-ASD|-1.33%|+0.00%|+4.28%|8/8|4/8|0|
|mbpp|WidthHalf-ASD|-1.59%|+0.03%|+4.19%|8/8|4/8|0|
|mbpp|WidthZero-ASD|-4.78%|-3.01%|+0.60%|8/8|4/8|0|
|mbpp|WidthFeedback-ASD|-3.58%|-1.90%|+2.11%|8/8|4/8|0|
|mmlu|ASD|+0.00%|+5.68%|+4.33%|6/14|1/14|0|
|mmlu|BudgetTree|-0.85%|+1.36%|+0.00%|14/14|5/14|0|
|mmlu|GateCap-ASD|-1.82%|+0.00%|-1.03%|11/14|5/14|0|
|mmlu|WidthHalf-ASD|-2.09%|-0.10%|-1.24%|12/14|5/14|0|
|mmlu|WidthZero-ASD|-3.06%|-0.46%|-1.67%|12/14|5/14|0|
|mmlu|WidthFeedback-ASD|-3.84%|-1.35%|-2.53%|11/14|5/14|0|
|mtbench|ASD|+0.00%|+13.60%|+13.95%|0/8|—|0|
|mtbench|BudgetTree|-11.13%|+0.18%|+0.00%|8/8|—|0|
|mtbench|GateCap-ASD|-11.03%|+0.00%|+0.62%|4/8|—|0|
|mtbench|WidthHalf-ASD|-14.01%|-3.12%|-2.65%|3/8|—|0|
|mtbench|WidthZero-ASD|-13.52%|-1.97%|-2.02%|3/8|—|0|
|mtbench|WidthFeedback-ASD|-12.70%|-1.52%|-1.59%|3/8|—|0|
|alpaca|ASD|+0.00%|+3.03%|+6.89%|1/8|—|0|
|alpaca|BudgetTree|-4.38%|-3.24%|+0.00%|8/8|—|0|
|alpaca|GateCap-ASD|-1.47%|+0.00%|+3.44%|6/8|—|0|
|alpaca|WidthHalf-ASD|-1.15%|+0.13%|+3.54%|6/8|—|0|
|alpaca|WidthZero-ASD|-2.42%|-1.39%|+1.88%|7/8|—|0|
|alpaca|WidthFeedback-ASD|-3.25%|-1.98%|+1.35%|7/8|—|0|

## 共同128/EOS

|任务|方法|vs ASD|有效数|vs 惩罚1|有效数|
|---|---|---:|---:|---:|---:|
|gsm8k|GateCap-ASD|-0.26%|8|+0.00%|8|
|gsm8k|WidthHalf-ASD|+3.66%|8|+4.97%|8|
|gsm8k|WidthZero-ASD|-1.56%|8|-0.40%|8|
|gsm8k|WidthFeedback-ASD|-2.65%|8|-2.30%|8|
|math500|GateCap-ASD|+3.91%|12|+0.00%|12|
|math500|WidthHalf-ASD|+4.69%|12|+0.76%|12|
|math500|WidthZero-ASD|+6.11%|12|+2.02%|12|
|math500|WidthFeedback-ASD|+4.88%|12|+0.94%|12|
|humaneval|GateCap-ASD|—|0|—|0|
|humaneval|WidthHalf-ASD|—|0|—|0|
|humaneval|WidthZero-ASD|—|0|—|0|
|humaneval|WidthFeedback-ASD|—|0|—|0|
|mbpp|GateCap-ASD|—|0|—|0|
|mbpp|WidthHalf-ASD|—|0|—|0|
|mbpp|WidthZero-ASD|—|0|—|0|
|mbpp|WidthFeedback-ASD|—|0|—|0|
|mmlu|GateCap-ASD|-1.82%|4|+0.00%|4|
|mmlu|WidthHalf-ASD|-3.79%|4|-1.93%|4|
|mmlu|WidthZero-ASD|-2.09%|4|-0.21%|4|
|mmlu|WidthFeedback-ASD|-1.87%|4|-0.08%|4|
|mtbench|GateCap-ASD|+1.21%|6|+0.00%|6|
|mtbench|WidthHalf-ASD|-0.04%|6|-1.23%|6|
|mtbench|WidthZero-ASD|-0.90%|6|-2.09%|6|
|mtbench|WidthFeedback-ASD|-0.97%|6|-2.00%|6|
|alpaca|GateCap-ASD|-2.28%|5|+0.00%|7|
|alpaca|WidthHalf-ASD|-4.20%|5|-2.06%|7|
|alpaca|WidthZero-ASD|-2.35%|5|-0.95%|7|
|alpaca|WidthFeedback-ASD|-2.76%|5|-0.61%|7|

## 自然请求计算与宽度

不同方法可能处于不同生成轨迹。节点/深度和轮数为请求汇总，不是同状态因果分项；所有反馈开销计入wall。

|任务|方法|总秒|输出token|验证轮|平均节点|平均最大深度|系数0/.5/1轮数|
|---|---|---:|---:|---:|---:|---:|---|
|gsm8k|ASD|24.170|2412|328|—|—|0/0/0|
|gsm8k|BudgetTree|25.016|2586|315|28.26|9.68|0/0/0|
|gsm8k|GateCap-ASD|25.977|2586|337|19.30|9.68|0/0/337|
|gsm8k|WidthHalf-ASD|26.382|2581|337|23.27|9.52|0/337/0|
|gsm8k|WidthZero-ASD|25.711|2563|322|28.52|9.44|322/0/0|
|gsm8k|WidthFeedback-ASD|25.561|2593|323|25.44|9.67|167/119/37|
|math500|ASD|139.564|15313|1740|—|—|0/0/0|
|math500|BudgetTree|111.439|12810|1332|24.14|9.74|0/0/0|
|math500|GateCap-ASD|111.942|13592|1369|17.20|10.29|0/0/1369|
|math500|WidthHalf-ASD|108.691|12281|1329|19.36|9.54|0/1329/0|
|math500|WidthZero-ASD|109.186|12596|1300|24.34|9.79|1300/0/0|
|math500|WidthFeedback-ASD|99.504|11681|1216|21.76|9.84|523/443/250|
|humaneval|ASD|6.111|660|70|—|—|0/0/0|
|humaneval|BudgetTree|5.691|613|60|26.85|11.50|0/0/0|
|humaneval|GateCap-ASD|5.755|613|62|19.56|11.74|0/0/62|
|humaneval|WidthHalf-ASD|5.726|613|61|22.61|11.61|0/61/0|
|humaneval|WidthZero-ASD|5.723|613|60|26.85|11.50|60/0/0|
|humaneval|WidthFeedback-ASD|5.801|613|62|22.55|11.81|18/18/26|
|mbpp|ASD|6.565|338|83|—|—|0/0/0|
|mbpp|BudgetTree|6.777|363|81|25.91|6.65|0/0/0|
|mbpp|GateCap-ASD|7.006|363|86|17.31|6.87|0/0/86|
|mbpp|WidthHalf-ASD|6.925|363|84|20.01|6.81|0/84/0|
|mbpp|WidthZero-ASD|6.816|363|81|25.91|6.65|81/0/0|
|mbpp|WidthFeedback-ASD|6.811|363|82|21.59|6.83|30/21/31|
|mmlu|ASD|24.136|2211|310|—|—|0/0/0|
|mmlu|BudgetTree|24.516|2419|293|26.56|9.26|0/0/0|
|mmlu|GateCap-ASD|23.568|2284|290|17.72|9.34|0/0/290|
|mmlu|WidthHalf-ASD|23.583|2294|286|20.95|9.22|0/286/0|
|mmlu|WidthZero-ASD|22.789|2174|270|26.45|9.09|270/0/0|
|mmlu|WidthFeedback-ASD|22.607|2174|271|22.92|9.18|109/102/60|
|mtbench|ASD|55.953|3639|719|—|—|0/0/0|
|mtbench|BudgetTree|50.935|3589|609|23.73|6.55|0/0/0|
|mtbench|GateCap-ASD|51.172|3538|635|14.93|6.79|0/0/635|
|mtbench|WidthHalf-ASD|48.512|3504|593|18.17|6.82|0/593/0|
|mtbench|WidthZero-ASD|48.500|3589|577|24.14|6.82|577/0/0|
|mtbench|WidthFeedback-ASD|50.366|3545|609|19.88|6.70|253/240/116|
|alpaca|ASD|39.892|1843|561|—|—|0/0/0|
|alpaca|BudgetTree|38.332|2011|506|24.73|5.03|0/0/0|
|alpaca|GateCap-ASD|39.536|2060|536|15.23|5.24|0/0/536|
|alpaca|WidthHalf-ASD|40.419|2104|542|18.05|5.16|0/542/0|
|alpaca|WidthZero-ASD|40.033|2087|525|24.44|5.06|525/0/0|
|alpaca|WidthFeedback-ASD|38.041|2009|508|19.38|5.18|159/198/151|

## 诊断题

|输入|方法|秒|token|正确|AR相同|停止|
|---|---|---:|---:|---|---|---|
|gsm8k:750|WidthHalf-ASD|5.078|383|True|False|eos|
|gsm8k:750|WidthZero-ASD|4.808|365|True|False|eos|
|gsm8k:750|GateCap-ASD|4.568|395|True|True|eos|
|gsm8k:750|WidthFeedback-ASD|4.421|395|True|True|eos|
|gsm8k:750|BudgetTree|4.482|395|True|True|eos|
|gsm8k:750|ASD|3.160|231|True|False|eos|
|math500:372|GateCap-ASD|7.610|824|True|True|eos|
|math500:372|WidthHalf-ASD|7.440|824|True|True|eos|
|math500:372|WidthFeedback-ASD|7.625|824|True|True|eos|
|math500:372|ASD|30.064|3275|False|False|eos|
|math500:372|WidthZero-ASD|7.700|824|True|True|eos|
|math500:372|BudgetTree|7.655|824|True|True|eos|
|math500:101|GateCap-ASD|31.550|4096|False|False|length|
|math500:101|WidthZero-ASD|28.723|3115|True|False|eos|
|math500:101|WidthFeedback-ASD|19.228|2194|True|False|eos|
|math500:101|ASD|18.993|2072|True|False|eos|
|math500:101|WidthHalf-ASD|25.707|2668|True|False|eos|
|math500:101|BudgetTree|30.974|3316|False|True|eos|
|humaneval:76|ASD|0.824|59|True|True|eos|
|humaneval:76|WidthFeedback-ASD|0.790|59|True|True|eos|
|humaneval:76|GateCap-ASD|0.777|59|True|True|eos|
|humaneval:76|BudgetTree|0.792|59|True|True|eos|
|humaneval:76|WidthHalf-ASD|0.794|59|True|True|eos|
|humaneval:76|WidthZero-ASD|0.797|59|True|True|eos|

规则见[协议](GATE_WIDTH_PROTOCOL.md)，[原始公开摘要](../results/gate-width/)包含输入身份、预算、反馈历史重放和源码清单。
开放任务本轮无新增偏好评分。AR一致率不等同正确率或全序列分布距离。
`python -m scripts.report_gate_width --input results/gate-width`可从压缩输入重建。
