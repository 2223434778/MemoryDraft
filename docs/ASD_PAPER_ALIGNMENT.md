# 与ASD论文及公开协议的对应关系

2026-10-24复核。[arXiv记录](https://arxiv.org/abs/2608.03447)当前仍为2026-08-30的v3；本轮依据[第4节](https://arxiv.org/html/2608.03447v3#S4)及已核验仓库快照`9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6`。

|项目|本轮对应设置与范围|
|---|---|
|任务覆盖|使用论文列出的GSM8K、MATH-500、HumanEval、MBPP、MMLU、MT-Bench、Alpaca七个任务名；冻结514新源任务/530匹配请求，完整源版本、SHA、许可、切分在data/research/asd-paper-extension/manifest.json|
|模型与底座|Qwen3-8B/DFlash、native16、A6000、FP32/RMSNorm；论文主表为Qwen3-14B/DSpark和L20，另有其他配对。共同默认预算B8/g.25/M2不意味着跨模型质量或绝对TPS可直接换算|
|固定吞吐|每请求DFlash→ASD→DFlash、严格树→树ASD→严格树，完成量均256；分别使用相邻严格wall均值。论文每条件四次，本轮固定三连各一次，自然各两次|
|自然行为|4096上限、自然EOS；完整请求时延、长度、截断、数学/代码/选择题评分单列。固定吞吐含EOS后计算，另作停止诊断|
|接受长度|实际输出Draft token/轮数，裁去未发出的尾部，排除Target root/recovery/bonus；验证节点与调用量并列，树与线性不混用提议节点分母|
|单参数控制|同一预定切片同时评价ASD与树ASD的B4/B12/g.125/g.5/M1，B0/B8复用主轨迹；不称为论文全部控制、无界设置或跨家族复现|
|漂移|上游B-C-B设置3%速度资格检查；本轮所有漂移请求均保留、完整报告，不以有利性能筛题|
|开放质量|80真实开放单元×4组，固定独立训练奖励权重；另有事前题干性质检查。奖励偏好及性质通过数保留各自含义，不命名为官方MT-Bench分数、AlpacaEval胜率或全回答准确率|

公开[README](https://github.com/Kissmetothemoon/ASD/blob/9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6/README.md)说明数据、答案和生成输出不随代码仓库提供；[评测文档](https://github.com/Kissmetothemoon/ASD/blob/9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6/docs/EVALUATION.md)要求使用外部runner与dataset-root。已阅读的这些公开材料不足以确认每个任务的完整原始数据版本、提示切分和开放质量裁判定义。因此本轮明确公开自己的这些选择，不根据同名任务猜测细节。

具体地，本项目Alpaca为固定`stanford_alpaca/alpaca_data.json`中的研究子集，MMLU为固定公开test源、MT-Bench为固定FastChat问题；不将Stanford Alpaca改称AlpacaEval。HumanEval/MBPP使用原标准测试、单次greedy完成，未声称完整EvalPlus。跨模型文献数字仍单列为原文报告值；新增方法的增量使用本机同环境对照。

上游发布检查中的速度目标和质量阈值是其可配置协议默认值，不是算法保证，也没有替换本项目已冻结的评估与用户接受标准。报告保留实际局部收益、质量取舍和失败场景，不将所有任务均胜出设为贡献成立的必要条件。

调用量字段以验证轮数为定义，不包含初始Target prefill；完整请求wall仍包含prefill及全部在线成本。共同128输出仅控制生成量，不要求不同近似路径的中间状态完全相同，不把它当同状态因果干预。

核验文件Git blob：README `a999325c5109b5a61c0f019bc227d452a7415fc1`；EVALUATION `a81f6bf12146e73c164f0a6ab0876c6b64a6cc5f`；[适配文档](https://github.com/Kissmetothemoon/ASD/blob/9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6/docs/DEEPSPEC_DSPARK.md) `5925b34e374c28e5fe2325a4dec29faeead1f4e3`。本次只读复核，没有导入新的上游推理代码、改变库版本或修改正在生成的实验。
