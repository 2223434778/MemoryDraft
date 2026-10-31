# 新场景扩展：实测结果

182固定新请求、三组546自然生成；Qwen3-8B/DFlash、单A6000、FP32/eager/RMSNorm。
文献基线另见[引用表](EXPANDED_LITERATURE_BASELINES.md)，不同配置不相除排名。

|任务|方法|配对耗时vs DFlash|配对耗时vs ASD|正确/可评分 或 指标|截断|DFlash一致|实际tokens/s|平均首token秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ifeval|DFlash|+0.00%|+2.24%|52/64|1|64/64|30.97|0.088|
|ifeval|ASD|+2.69%|+0.00%|53/64|2|4/64|31.48|0.088|
|ifeval|BufferedKV-ASD|-15.33%|-13.86%|52/64|3|55/64|39.41|0.088|
|livecodebench|DFlash|+0.00%|-5.60%|8/24|0|24/24|74.26|0.429|
|livecodebench|ASD|+46.94%|+0.00%|9/24|2|1/24|87.69|0.432|
|livecodebench|BufferedKV-ASD|-7.34%|-13.63%|8/24|0|23/24|81.29|0.429|
|canitedit|DFlash|+0.00%|+3.91%|17/31|0|32/32|117.41|0.554|
|canitedit|ASD|-2.65%|+0.00%|15/31|0|16/32|118.03|0.554|
|canitedit|BufferedKV-ASD|-3.22%|+0.60%|17/31|0|31/32|121.82|0.553|
|longbench-qasper|DFlash|+0.00%|+0.76%|43.14|0|8/8|3.09|3.973|
|longbench-qasper|ASD|-0.68%|+0.00%|45.38|0|4/8|3.30|3.977|
|longbench-qasper|BufferedKV-ASD|-1.59%|-0.86%|43.14|0|8/8|3.14|3.969|
|longbench-multifieldqa_en|DFlash|+0.00%|+2.06%|55.60|0|8/8|4.30|3.660|
|longbench-multifieldqa_en|ASD|-1.90%|+0.00%|56.00|0|6/8|3.77|3.658|
|longbench-multifieldqa_en|BufferedKV-ASD|-2.46%|-0.47%|55.60|0|8/8|4.40|3.657|
|longbench-multi_news|DFlash|+0.00%|+8.86%|26.46|4|8/8|24.83|1.925|
|longbench-multi_news|ASD|-6.88%|+0.00%|26.41|3|0/8|26.12|1.929|
|longbench-multi_news|BufferedKV-ASD|-20.43%|-13.30%|26.46|4|7/8|31.35|1.921|
|longbench-gov_report|DFlash|+0.00%|+4.04%|34.87|5|8/8|21.12|4.211|
|longbench-gov_report|ASD|-3.69%|+0.00%|35.48|5|0/8|21.59|4.202|
|longbench-gov_report|BufferedKV-ASD|-19.48%|-16.24%|34.61|5|7/8|26.24|4.199|
|aime25|DFlash|+0.00%|+17.76%|5/30|11|30/30|94.27|0.216|
|aime25|ASD|-2.61%|+0.00%|7/30|11|0/30|99.93|0.216|
|aime25|BufferedKV-ASD|-8.78%|+4.60%|5/30|10|7/30|102.84|0.216|

IFEval主表为严格整请求通过数；QA为F1×100，摘要为ROUGE-L×100，不是正确率。CanItEdit按统一canonical可运行掩码评分；AIME按最终boxed整数等价评分。
时延含prefill与全部在线工作，输出变短也影响完整请求时间；配对均值与混合总量比不是同一统计。DFlash一致性不代替任务质量，也不冒充本轮AR验证。

## IFEval分项

|方法|严格请求|宽松请求|严格指令|宽松指令|
|---|---:|---:|---:|---:|
|DFlash|52/64|52/64|85/97|85/97|
|ASD|53/64|56/64|86/97|89/97|
|BufferedKV-ASD|52/64|53/64|85/97|86/97|

## 混合请求总量

|方法|秒|token|实际tokens/s|DFlash一致|
|---|---:|---:|---:|---:|
|DFlash|2266.395|131275|57.92|182/182|
|ASD|2245.920|135321|60.25|31/182|
|BufferedKV-ASD|1972.739|131201|66.51|146/182|

## 评分与范围

IFEval官方strict/loose；LiveCodeBench v6-only按难度固定8/8/8，官方全测试2s/test；CanItEdit descriptive32，原生成评分及修订评分同时保留。
编辑规范答案可运行31/32，未评分['canitedit:78_llm_inference']。评分修订3修复对象方法、科学依赖及源码检查环境；统一重评全部编辑输出，不修改token、时间或测试，旧评分保留。
LongBench仅四个子集各8，3840-token首尾截取，保留原始长度/截取标记；不称完整长文榜单。QA128/摘要512输出上限，其他代码/指令2048、AIME4096，截断均计入结果。
未做新参数扫描，三组同题配对，不能把单次子集点估计外推为所有请求稳定保证。原始长文/完整测试载荷不另行发布，数据版本/题目ID/SHA及获取/评分代码可复核。

## 执行记录

生成2026-10-31T04:01:06Z至2026-10-31T05:51:31Z，加载20.54s、一次合成校准7.30s另列。
协议：[EXPANDED_SUITE_PROTOCOL.md](EXPANDED_SUITE_PROTOCOL.md)。[公开结果](../results/expanded-suite/)。
`python -m scripts.report_expanded_suite --input results/expanded-suite` 可从公开压缩行重新汇总。报告/独立评分为GPU执行后追加，不冒充冻结推理源码。
