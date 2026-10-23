# ASD论文七任务：五方法扩样本比较

241个未使用源任务，MT-Bench展开为249个匹配输入请求；4980计时+249 AR，另40次自有历史第二轮。
自然EOS各2次、128/EOS各1次、固定256/EOS忽略各1次。全部在线成本计入wall，成本表/权重/ASD B8/g.25/m2冻结。
DFlash底座、A6000/FP32/RMSNorm；不是论文DSpark/L20和原始完整提示/样本协议的精确复现。

|任务|方法|自然相对ASD|自然相对第一阶段组合|质量正确/已评分|截断|等128相对ASD（N）|固定256相对ASD|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|DFlash|+12.87%|+26.84%|63/64|0|+11.71%（64）|+9.72%|
|gsm8k|ASD|+0.00%|+12.99%|63/64|0|+0.00%（64）|+0.00%|
|gsm8k|BudgetTree|-0.38%|+11.86%|63/64|0|-2.58%（64）|-2.96%|
|gsm8k|BudgetTree-ASD|-9.86%|+0.00%|63/64|0|-11.64%（63）|-10.05%|
|gsm8k|JointTree-ASD|-6.74%|+3.93%|63/64|0|-11.49%（64）|-10.01%|
|math500|DFlash|+6.23%|+26.67%|26/32|0|+15.41%（32）|+13.23%|
|math500|ASD|+0.00%|+41.38%|26/32|3|+0.00%（32）|+0.00%|
|math500|BudgetTree|-0.24%|+18.95%|26/32|0|+6.75%（32）|+5.70%|
|math500|BudgetTree-ASD|-11.42%|+0.00%|28/32|0|-4.70%（32）|-3.57%|
|math500|JointTree-ASD|-7.73%|+18.98%|23/32|2|-4.11%（32）|-4.19%|
|humaneval|DFlash|+5.14%|+14.11%|27/32|0|+9.02%（6）|+40.99%|
|humaneval|ASD|+0.00%|+8.68%|28/32|0|+0.00%（6）|+0.00%|
|humaneval|BudgetTree|-1.62%|+6.31%|27/32|0|-0.21%（6）|+1.54%|
|humaneval|BudgetTree-ASD|-6.34%|+0.00%|27/32|0|-6.10%（5）|-20.17%|
|humaneval|JointTree-ASD|-4.82%|+1.73%|26/32|0|-3.48%（4）|-11.30%|
|mbpp|DFlash|+1.17%|+16.20%|12/32|0|-16.49%（2）|+35.63%|
|mbpp|ASD|+0.00%|+19.43%|14/32|0|+0.00%（3）|+0.00%|
|mbpp|BudgetTree|-15.75%|-3.89%|12/32|0|-25.68%（2）|+13.39%|
|mbpp|BudgetTree-ASD|+71.85%|+0.00%|12/32|1|-13.85%（1）|-3.31%|
|mbpp|JointTree-ASD|+100.87%|+5.47%|13/32|1|-12.48%（2）|-6.52%|
|mmlu|DFlash|+24.83%|-2.54%|32/57|0|+22.88%（4）|+7.66%|
|mmlu|ASD|+0.00%|-3.15%|25/57|0|+0.00%（4）|+0.00%|
|mmlu|BudgetTree|+21.57%|-3.09%|32/57|0|+7.40%（4）|+0.06%|
|mmlu|BudgetTree-ASD|+27.25%|+0.00%|22/57|0|-10.97%（4）|-5.60%|
|mmlu|JointTree-ASD|+27.90%|+0.36%|22/57|0|-14.44%（4）|-4.15%|
|mtbench|DFlash|+11.60%|+27.38%|另评|0|+16.42%（14）|+9.95%|
|mtbench|ASD|+0.00%|+14.46%|另评|0|+0.00%（14）|+0.00%|
|mtbench|BudgetTree|-2.37%|+11.17%|另评|0|+0.96%（14）|-4.91%|
|mtbench|BudgetTree-ASD|-10.49%|+0.00%|另评|0|-10.55%（14）|-14.77%|
|mtbench|JointTree-ASD|-9.35%|+1.55%|另评|0|-12.05%（14）|-13.94%|
|alpaca|DFlash|+10.04%|+33.38%|另评|0|+10.48%（11）|+3.06%|
|alpaca|ASD|+0.00%|+24.32%|另评|0|+0.00%（11）|+0.00%|
|alpaca|BudgetTree|-9.01%|+9.89%|另评|0|-8.03%（11）|-12.73%|
|alpaca|BudgetTree-ASD|-12.92%|+0.00%|另评|0|-14.62%（11）|-15.47%|
|alpaca|JointTree-ASD|-13.94%|+3.15%|另评|0|-15.70%（11）|-16.37%|

负数为时延降低；速度表用逐请求比值的题等权均值；另在JSON保存总token/总wall TPS、长度与请求区间。
MT-Bench主速度表第二轮共享严格AR的首轮回答；下表的质量审查使用各方法接续自身首轮回答。
代码标准测试不进提示，MBPP可用challenge另记；数学保守主评分、有限标量等价和待复核单列。

|固定256计算吞吐|方法|token/s|相对ASD TPS变化|
|---|---|---:|---:|
|gsm8k|DFlash|78.156|-8.32%|
|gsm8k|ASD|85.250|+0.00%|
|gsm8k|BudgetTree|88.700|+4.05%|
|gsm8k|BudgetTree-ASD|95.721|+12.28%|
|gsm8k|JointTree-ASD|95.660|+12.21%|
|math500|DFlash|87.888|-11.83%|
|math500|ASD|99.683|+0.00%|
|math500|BudgetTree|95.307|-4.39%|
|math500|BudgetTree-ASD|104.989|+5.32%|
|math500|JointTree-ASD|105.721|+6.06%|
|humaneval|DFlash|38.938|-15.86%|
|humaneval|ASD|46.279|+0.00%|
|humaneval|BudgetTree|58.556|+26.53%|
|humaneval|BudgetTree-ASD|73.795|+59.46%|
|humaneval|JointTree-ASD|65.952|+42.51%|
|mbpp|DFlash|84.129|-25.65%|
|mbpp|ASD|113.158|+0.00%|
|mbpp|BudgetTree|101.291|-10.49%|
|mbpp|BudgetTree-ASD|119.816|+5.88%|
|mbpp|JointTree-ASD|123.832|+9.43%|
|mmlu|DFlash|60.994|-5.71%|
|mmlu|ASD|64.688|+0.00%|
|mmlu|BudgetTree|66.119|+2.21%|
|mmlu|BudgetTree-ASD|70.128|+8.41%|
|mmlu|JointTree-ASD|68.843|+6.42%|
|mtbench|DFlash|42.803|-8.19%|
|mtbench|ASD|46.619|+0.00%|
|mtbench|BudgetTree|50.693|+8.74%|
|mtbench|BudgetTree-ASD|56.214|+20.58%|
|mtbench|JointTree-ASD|55.132|+18.26%|
|alpaca|DFlash|43.415|-2.64%|
|alpaca|ASD|44.594|+0.00%|
|alpaca|BudgetTree|52.098|+16.83%|
|alpaca|BudgetTree-ASD|53.094|+19.06%|
|alpaca|JointTree-ASD|53.862|+20.78%|

上表TPS变化正数更快。固定256包含EOS后续写的计算，不能直接当自然请求质量或响应收益。

|开放任务|候选相对ASD|本地偏好（含平局）|已评/任务|相同输出|换序一致|
|---|---|---:|---:|---:|---:|
|mtbench|DFlash|50.00%|8/8|0|0|
|mtbench|BudgetTree|50.00%|8/8|0|0|
|mtbench|BudgetTree-ASD|54.17%|6/8|0|0|
|mtbench|JointTree-ASD|50.00%|8/8|0|0|
|alpaca|DFlash|50.00%|16/16|2|2|
|alpaca|BudgetTree|50.00%|16/16|2|2|
|alpaca|BudgetTree-ASD|51.56%|16/16|1|1|
|alpaca|JointTree-ASD|50.00%|16/16|1|1|

本地盲评使用同一Qwen3-8B Target、A/B双向换序和独立rubric；属于有偏好的辅助诊断，
不是独立GPT-4裁判、官方MT-Bench分数或AlpacaEval胜率。输入超过4096 token不截断裁判材料，标为未评。
开放任务不伪造准确率；模型加载、预热、冷编译和离线裁判成本另列。
