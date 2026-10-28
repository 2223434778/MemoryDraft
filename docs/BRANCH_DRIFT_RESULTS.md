# 分支漂移门控：七任务实测结果

六方法同窗自然EOS比较；另五方法测128/EOS前缀。负数表示时延减少。配对比值按题平均；单次计时。各列均直接配对，不能将两个平均比值相除来推导方法间的比值。
冻结旧AR输出仅供质量和轨迹参考，不用于本轮AR时延加速比。MT第二轮共享冻结AR首轮历史。

|任务|输入数|Drift vs 原树ASD|vs ASD|vs 严格树|vs Coverage|正确：原树ASD → Drift|AR逐token相同：原树ASD → Drift|
|---|---:|---:|---:|---:|---:|---|---|
|gsm8k|16|+10.53%|+2.56%|+3.99%|+5.74%|15 → 16 / 16|0 → 7 / 16|
|math500|12|+13.41%|+11.10%|+37.46%|+59.21%|10 → 8 / 12|0 → 3 / 12|
|humaneval|16|+1.01%|-4.24%|+3.27%|+1.53%|13 → 12 / 16|11 → 15 / 16|
|mbpp|16|+4.21%|-11.36%|-0.64%|+6.33%|9 → 9 / 16|8 → 14 / 16|
|mmlu|57|-2.44%|-1.56%|-0.21%|+0.99%|17 → 28 / 57|37 → 54 / 57|
|mtbench|16|-2.50%|-15.84%|+0.77%|-4.11%|开放质量另列|3 → 6 / 16|
|alpaca|16|+17.83%|-3.04%|+1.93%|+10.81%|开放质量另列|1 → 7 / 16|

## 等128-token前缀

仅比较双方实际生成128token的请求；自然EOS短回答不强制续写。各对照有效题数可以不同。

|任务|Drift vs 原树ASD|有效题数|vs ASD|有效题数|vs 严格树|有效题数|
|---|---:|---:|---:|---:|---:|---:|
|gsm8k|+13.20%|16|-1.52%|16|+4.17%|16|
|math500|+5.65%|12|+7.62%|12|+2.13%|12|
|humaneval|+4.45%|1|+7.86%|1|+4.44%|1|
|mbpp|-3.87%|1|-12.57%|1|-3.78%|1|
|mmlu|+12.34%|6|+7.57%|6|+2.34%|6|
|mtbench|+12.24%|10|+1.66%|10|+1.41%|10|
|alpaca|+13.43%|12|-3.95%|12|+1.11%|14|

## 自然质量、长度与轮数

数学沿用既定评分与有限标量等价补充；代码使用原提供测试，MMLU采用仅字母输出合同。开放任务不计正确率。

|任务|方法|正确/可评分|输出token|验证轮|近似token|与AR相同|上限截断|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|AR冻结参考|16/16|4715|历史参考，不作同窗计时|0|16|0|
|gsm8k|DFlash|16/16|4715|729|0|16|0|
|gsm8k|ASD|16/16|4427|650|105|0|0|
|gsm8k|BudgetTree|16/16|4715|597|0|16|0|
|gsm8k|BudgetTree-ASD|15/16|4556|562|107|0|0|
|gsm8k|CoverageTree-ASD|16/16|4611|569|77|0|0|
|gsm8k|DriftTree-ASD|16/16|4720|606|12|7|0|
|math500|AR冻结参考|8/12|12810|历史参考，不作同窗计时|0|12|0|
|math500|DFlash|8/12|12810|1486|0|12|0|
|math500|ASD|8/12|15313|1740|141|0|0|
|math500|BudgetTree|8/12|12810|1332|0|12|0|
|math500|BudgetTree-ASD|10/12|14963|1503|136|0|1|
|math500|CoverageTree-ASD|7/12|15267|1496|116|0|1|
|math500|DriftTree-ASD|8/12|16470|1672|32|3|1|
|humaneval|AR冻结参考|13/16|1213|历史参考，不作同窗计时|0|16|0|
|humaneval|DFlash|13/16|1213|139|0|16|0|
|humaneval|ASD|13/16|1379|148|10|9|0|
|humaneval|BudgetTree|13/16|1213|122|0|16|0|
|humaneval|BudgetTree-ASD|13/16|1274|130|8|11|0|
|humaneval|CoverageTree-ASD|13/16|1213|120|5|13|0|
|humaneval|DriftTree-ASD|12/16|1212|122|1|15|0|
|mbpp|AR冻结参考|8/16|770|历史参考，不作同窗计时|0|16|0|
|mbpp|DFlash|8/16|770|177|0|16|0|
|mbpp|ASD|9/16|745|156|14|10|0|
|mbpp|BudgetTree|8/16|770|138|0|16|0|
|mbpp|BudgetTree-ASD|9/16|819|121|12|8|0|
|mbpp|CoverageTree-ASD|9/16|843|118|10|8|0|
|mbpp|DriftTree-ASD|9/16|795|124|2|14|0|
|mmlu|AR冻结参考|28/57|3241|历史参考，不作同窗计时|0|57|0|
|mmlu|DFlash|28/57|3241|489|0|57|0|
|mmlu|ASD|16/57|3297|474|73|32|0|
|mmlu|BudgetTree|28/57|3241|429|0|57|0|
|mmlu|BudgetTree-ASD|17/57|3352|415|64|37|0|
|mmlu|CoverageTree-ASD|28/57|2988|381|42|51|0|
|mmlu|DriftTree-ASD|28/57|2913|401|6|54|0|
|mtbench|AR冻结参考|—|5008|历史参考，不作同窗计时|0|16|0|
|mtbench|DFlash|—|5008|1152|0|16|0|
|mtbench|ASD|—|5287|1174|105|2|0|
|mtbench|BudgetTree|—|5008|916|0|16|0|
|mtbench|BudgetTree-ASD|—|5394|958|95|3|0|
|mtbench|CoverageTree-ASD|—|5184|965|75|2|0|
|mtbench|DriftTree-ASD|—|4929|883|16|6|0|
|alpaca|AR冻结参考|—|5661|历史参考，不作同窗计时|0|16|0|
|alpaca|DFlash|—|5661|1917|0|16|0|
|alpaca|ASD|—|5394|1699|107|1|0|
|alpaca|BudgetTree|—|5661|1479|0|16|0|
|alpaca|BudgetTree-ASD|—|4989|1297|116|1|0|
|alpaca|CoverageTree-ASD|—|5145|1334|96|1|0|
|alpaca|DriftTree-ASD|—|5731|1480|21|7|0|

## 分支门控覆盖

下表统计局部通过原ASD门槛的树边，包含最终未被提交的旁支；不能当作实际输出token的比例。
Coverage与Drift使用相同构树和停止/真实兄弟覆盖要求，后续生成状态可因JS门控而分化。

|任务|方法|局部候选边|缺失兄弟|停止拦截|比较数|JS拒绝|平均JS|选中例外|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|gsm8k|CoverageTree-ASD|964|357|9|598|0|0.7448|77|
|gsm8k|DriftTree-ASD|1148|382|11|755|609|0.7491|12|
|math500|CoverageTree-ASD|1392|377|0|1015|0|0.8061|116|
|math500|DriftTree-ASD|4303|1124|3|3176|2811|0.8152|32|
|humaneval|CoverageTree-ASD|62|31|15|16|0|0.5067|5|
|humaneval|DriftTree-ASD|63|31|15|17|9|0.5357|1|
|mbpp|CoverageTree-ASD|104|47|3|54|0|0.8770|10|
|mbpp|DriftTree-ASD|130|63|6|61|49|0.8034|2|
|mmlu|CoverageTree-ASD|595|203|39|353|0|0.8086|42|
|mmlu|DriftTree-ASD|847|273|40|534|465|0.8196|6|
|mtbench|CoverageTree-ASD|1070|424|4|642|0|0.8101|75|
|mtbench|DriftTree-ASD|2368|837|10|1521|1317|0.8104|16|
|alpaca|CoverageTree-ASD|1302|620|3|679|0|0.7968|96|
|alpaca|DriftTree-ASD|3805|1657|6|2142|1814|0.7883|21|

## 开放任务独立奖励偏好

固定Skywork-Reward-V2-Qwen3-8B，仅为独立偏好代理，非官方MT-Bench/AlpacaEval，也非正确率。相同文本按平局。

|任务|方法|对照|胜/平/负|偏好率|
|---|---|---|---:|---:|
|mtbench|CoverageTree-ASD|ASD|11/2/3|75.00%|
|mtbench|CoverageTree-ASD|BudgetTree|8/3/5|59.38%|
|mtbench|CoverageTree-ASD|BudgetTree-ASD|9/2/5|62.50%|
|mtbench|DriftTree-ASD|ASD|10/2/4|68.75%|
|mtbench|DriftTree-ASD|BudgetTree|3/8/5|43.75%|
|mtbench|DriftTree-ASD|BudgetTree-ASD|8/3/5|59.38%|
|mtbench|DriftTree-ASD|CoverageTree-ASD|5/2/9|37.50%|
|alpaca|CoverageTree-ASD|ASD|7/2/7|50.00%|
|alpaca|CoverageTree-ASD|BudgetTree|5/1/10|34.38%|
|alpaca|CoverageTree-ASD|BudgetTree-ASD|9/1/6|59.38%|
|alpaca|DriftTree-ASD|ASD|11/2/3|75.00%|
|alpaca|DriftTree-ASD|BudgetTree|3/9/4|46.88%|
|alpaca|DriftTree-ASD|BudgetTree-ASD|10/1/5|65.62%|
|alpaca|DriftTree-ASD|CoverageTree-ASD|10/1/5|65.62%|

## 执行与复核

共1639条新计时，149条冻结AR参考。GPU加载/预热与离线奖励评分单列；在线构树、JS、传输、验证、KV全部计入wall。
严格方法与冻结AR身份差异0条；严格树与同窗DFlash自然差异0题。详见audit.json，不将未诊断差异自动称数值误差。
原数据以前已用于研究，本轮不称新保留集。完整回答和前缀指标分别报告；相同AR token数不是语义质量指标。
原始小型公开输出、账本与轮级摘要见[results/branch-drift](../results/branch-drift/)，无完整logits/KV/权重。
复核命令：`python -m scripts.write_branch_drift_report --input results/raw/branch-drift`。实现与选择规则见[协议](BRANCH_DRIFT_PROTOCOL.md)。
