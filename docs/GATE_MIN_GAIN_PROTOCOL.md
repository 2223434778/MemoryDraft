# 最小增量接受：直接对比

用户只要求实现第一项，不新增下一步首选一致门、训练或复杂验证实验。

以反馈调宽为底座，先按现有门控选择最长可行路径。若含近似例外且比同一已验证树的严格路径少领先2个token，则提交严格路径和该叶子的Target恢复token；否则使用原选择。预算、例外计数、KV和宽度反馈均以最终提交路径更新。新增选择不调用Target，也不改变prompt、EOS、JS .1、raw regret ln4、B8/.25/cap2、树节点上限32。

三组：ASD、WidthFeedback-ASD、MinGainFeedback-ASD。沿用上一轮七任务62源/66匹配输入，含MT的共享冻结AR首轮上下文；只跑自然4096/EOS，共198新计时。每输入随机方法顺序、各一次，沿用原评分与AR冻结输出，不使用旧时延作同窗基线。不扫阈值，不加前缀、反事实或多个种子实验。包含所有失败、错误与截断；开放任务只报告时延/AR身份，无新增偏好评分。

报告每任务配对时延、总wall、AR完全一致数、正确数、验证轮、输出长度、回退次数。目标是相对原反馈版总wall增加不超过3%，同时提高AR一致性；这是判断标准，不是结果保证。MATH101可能损失现有有益路径，必须保留实际结果。更高AR一致性不等同更高正确率。

本机：`.venv\Scripts\python.exe -m pytest -q tests/test_gate_min_gain.py tests/test_gate_width.py tests/test_budget_tree.py tests/test_branch_drift.py --basetemp artifacts/pytest-gate-min-gain`。

GPU：复用隔离launcher，以`--study gate-min-gain`运行`scripts.run_gate_min_gain`，GPU1、原venv/模型/成本表；释放空闲27B后finally恢复，GPU0聊天持续运行。源码清单、执行元数据、压缩公开结果与简短报告随结果提交。
