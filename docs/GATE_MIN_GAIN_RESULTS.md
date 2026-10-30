# 最小增量接受：直接对比结果

七任务66匹配输入，三组198次自然4096/EOS计时。沿用既有开发输入，各一次，负数表示更快。
只增加“近似路径至少领先同树严格路径2个token”的提交条件；预算、KV、反馈均以实际提交路径更新。

|任务|方法|配对耗时 vs ASD|配对耗时 vs 原反馈|AR完全一致|正确/可评分|截断|
|---|---|---:|---:|---:|---:|---:|
|gsm8k|ASD|+0.00%|-2.20%|0/8|8/8|0|
|gsm8k|WidthFeedback-ASD|+5.46%|+0.00%|5/8|8/8|0|
|gsm8k|MinGainFeedback-ASD|+4.34%|-1.03%|7/8|8/8|0|
|math500|ASD|+0.00%|+33.68%|0/12|8/12|0|
|math500|WidthFeedback-ASD|-10.79%|+0.00%|7/12|9/12|0|
|math500|MinGainFeedback-ASD|-8.04%|+2.71%|9/12|9/12|0|
|humaneval|ASD|+0.00%|+5.03%|5/8|7/8|0|
|humaneval|WidthFeedback-ASD|-3.94%|+0.00%|8/8|7/8|0|
|humaneval|MinGainFeedback-ASD|-4.20%|-0.29%|8/8|7/8|0|
|mbpp|ASD|+0.00%|+5.60%|2/8|5/8|0|
|mbpp|WidthFeedback-ASD|-3.56%|+0.00%|8/8|4/8|0|
|mbpp|MinGainFeedback-ASD|-3.48%|+0.14%|8/8|4/8|0|
|mmlu|ASD|+0.00%|+6.76%|6/14|1/14|0|
|mmlu|WidthFeedback-ASD|-3.77%|+0.00%|11/14|5/14|0|
|mmlu|MinGainFeedback-ASD|-1.93%|+1.52%|13/14|5/14|0|
|mtbench|ASD|+0.00%|+15.66%|0/8|—|0|
|mtbench|WidthFeedback-ASD|-12.70%|+0.00%|3/8|—|0|
|mtbench|MinGainFeedback-ASD|-12.45%|+0.18%|4/8|—|0|
|alpaca|ASD|+0.00%|+5.44%|1/8|—|0|
|alpaca|WidthFeedback-ASD|-3.28%|+0.00%|7/8|—|0|
|alpaca|MinGainFeedback-ASD|-2.84%|+0.42%|7/8|—|0|

## 混合请求总量

|方法|总秒|AR完全一致|输出token|验证轮|近似token|回退轮|本轮放弃推进token|
|---|---:|---:|---:|---:|---:|---:|---:|
|ASD|295.916|14/66|26416|3811|355|0|0|
|WidthFeedback-ASD|248.324|49/66|22978|3071|35|0|0|
|MinGainFeedback-ASD|255.476|56/66|23605|3151|13|20|20|

新方案总wall相对原反馈+2.88%，相对ASD-13.67%。总时间比区别于逐题配对均值，包含生成路径/长度变化。
预定速度代价≤3%：True；AR完全一致数增加：True。

## 正确集合变化

|任务|新方案较原反馈新增正确|新方案较原反馈失去正确|
|---|---|---|
|gsm8k|—|—|
|math500|—|—|
|humaneval|—|—|
|mbpp|—|—|
|mmlu|—|—|
|mtbench|—|—|
|alpaca|—|—|

## 既有关键题

|输入|方法|秒|token|正确|AR完全一致|停止|
|---|---|---:|---:|---|---|---|
|gsm8k:750|WidthFeedback-ASD|4.379|395|True|True|eos|
|gsm8k:750|MinGainFeedback-ASD|4.378|395|True|True|eos|
|gsm8k:750|ASD|3.122|231|True|False|eos|
|math500:372|WidthFeedback-ASD|7.625|824|True|True|eos|
|math500:372|MinGainFeedback-ASD|7.623|824|True|True|eos|
|math500:372|ASD|30.028|3275|False|False|eos|
|math500:101|MinGainFeedback-ASD|25.303|2716|True|False|eos|
|math500:101|WidthFeedback-ASD|19.206|2194|True|False|eos|
|math500:101|ASD|18.970|2072|True|False|eos|

与上一轮比较的132个原基线输出/输入身份，差异0；旧时延未参与本轮配对。

## 执行记录

实际GPU源码commit `d7618fd99a2767478dcaed0a760f4826f92685aa`，515文件运行前后SHA核验。实际ZIP SHA `8f478f3e199e68527939c10e8a1d9f5e80b42c81084e7d8a0a996c7ee0b089f2`。
生成时间2026-10-30T08:24:08Z至2026-10-30T08:37:28Z，加载20.89秒单列，峰allocated 38339675136 bytes。
目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/gate-min-gain-v1`，GPU1，原Qwen3-8B/DFlash、FP32/eager/RMSNorm、成本表。
命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_gate_min_gain`。
服务恢复退出码：{'memorydraft-chat-large-model.service': 0, 'memorydraft-chat-large-adapter.service': 0}。本机63 passed/14.41s；服务器63 passed/5.20s。
原反馈版与新方案均保留，未变更线上默认。报告脚本为生成后的离线后处理，不冒充冻结GPU源码。

MMLU沿用字母格式合同；开放聊天未新增偏好评分。AR一致不等同正确率，单次开发计时不作普遍保证。
协议：[GATE_MIN_GAIN_PROTOCOL.md](GATE_MIN_GAIN_PROTOCOL.md)。[公开数据](../results/gate-min-gain/)。
复现：`python -m scripts.report_gate_min_gain --input results/gate-min-gain`。
