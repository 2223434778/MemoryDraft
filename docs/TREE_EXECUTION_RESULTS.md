# 树验证执行优化：直接对比结果

同一高保真接受策略、七任务66匹配输入、三组198次自然4096/EOS计时。各一次，负数表示更快。

|任务|方法|配对耗时 vs 原版|配对耗时 vs 仅去重|AR完全一致|正确/可评分|截断|
|---|---|---:|---:|---:|---:|---:|
|gsm8k|MinGainFeedback-ASD|+0.00%|+0.24%|7/8|8/8|0|
|gsm8k|ReuseScores-ASD|-0.24%|+0.00%|7/8|8/8|0|
|gsm8k|BufferedKV-ASD|-0.45%|-0.20%|7/8|8/8|0|
|math500|MinGainFeedback-ASD|+0.00%|+0.25%|9/12|9/12|0|
|math500|ReuseScores-ASD|-0.25%|+0.00%|9/12|9/12|0|
|math500|BufferedKV-ASD|-1.02%|-0.77%|9/12|9/12|0|
|humaneval|MinGainFeedback-ASD|+0.00%|-0.02%|8/8|7/8|0|
|humaneval|ReuseScores-ASD|+0.02%|+0.00%|8/8|7/8|0|
|humaneval|BufferedKV-ASD|-0.29%|-0.31%|8/8|7/8|0|
|mbpp|MinGainFeedback-ASD|+0.00%|-0.02%|8/8|4/8|0|
|mbpp|ReuseScores-ASD|+0.02%|+0.00%|8/8|4/8|0|
|mbpp|BufferedKV-ASD|-0.57%|-0.59%|8/8|4/8|0|
|mmlu|MinGainFeedback-ASD|+0.00%|+0.06%|13/14|5/14|0|
|mmlu|ReuseScores-ASD|-0.06%|+0.00%|13/14|5/14|0|
|mmlu|BufferedKV-ASD|-0.10%|-0.05%|13/14|5/14|0|
|mtbench|MinGainFeedback-ASD|+0.00%|+0.19%|4/8|—|0|
|mtbench|ReuseScores-ASD|-0.19%|+0.00%|4/8|—|0|
|mtbench|BufferedKV-ASD|-0.85%|-0.66%|4/8|—|0|
|alpaca|MinGainFeedback-ASD|+0.00%|+0.27%|7/8|—|0|
|alpaca|ReuseScores-ASD|-0.27%|+0.00%|7/8|—|0|
|alpaca|BufferedKV-ASD|-0.43%|-0.17%|7/8|—|0|

## 总量及一致性

|方法|总秒|总wall vs 原版|AR完全一致|输出token|验证轮|近似token|峰allocated GiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|MinGainFeedback-ASD|255.793|+0.00%|56/66|23605|3151|13|35.49|
|ReuseScores-ASD|255.221|-0.22%|56/66|23605|3151|13|35.49|
|BufferedKV-ASD|253.159|-1.03%|56/66|23605|3151|13|35.86|

132个优化/原版配对：输出差异0，逐轮路径/接受/反馈/原始regret轨迹差异0，累计账本差异0。
总wall比与逐题配对均值为不同统计，不相除推导其他对照。分配、写入、扩容、选择和释放开销计在线；模型加载/预热单列。
缓存组KV更新131567次、分配5079次；扩容时历史复制9365438464 bytes。这不是与旧动态缓存实测流量的直接对照，不据此推断带宽瓶颈占比。
准确度以输出保持为本轮要求；出现输出差异的选项不作为保持输出的优化接纳。差异详情保留在audit.json。

## 执行记录

实际GPU源码commit `ff50208b7c8912aae1cb118c015b006873f57729`，521文件运行前后SHA核验。实际ZIP SHA `8e9256d92e98e2d91deecccc783884b9e55fa8a18c3277f571f8a7deb06a8132`。
生成2026-10-30T09:56:16Z至2026-10-30T10:09:01Z，加载20.64秒，峰allocated 38509075456 bytes。
目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/tree-execution-v1`，GPU1，原Qwen3-8B/DFlash、FP32/eager/RMSNorm、成本表。
命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_tree_execution`。
服务恢复退出码：{'memorydraft-chat-large-model.service': 0, 'memorydraft-chat-large-adapter.service': 0}。本机76 passed/2 skipped/14.71s；服务器78 passed/8.72s。
旧目录与线上默认保持；报告为生成后的离线后处理，不冒充冻结GPU源码。

开发输入非新保留集；MMLU仍为字母格式合同，开放聊天未新增偏好评分。
协议：[TREE_EXECUTION_PROTOCOL.md](TREE_EXECUTION_PROTOCOL.md)。[公开数据](../results/tree-execution/)。
复现：`python -m scripts.report_tree_execution --input results/tree-execution`。
