# 树验证执行优化：直接对比

保持MinGainFeedback-ASD的权重、FP32/eager/RMSNorm、prompt、EOS、反馈调宽、JS .1、raw regret ln4、B8/.25/cap2、最小增量2与节点上限32。只优化实现，不调接受参数。

- MinGainFeedback-ASD：原高保真路径，DynamicCache与原分数提取。
- ReuseScores-ASD：同一Target前向的最大值、ID、候选分数和token列表只提取一次，严格选择/门控/反馈/流式过程共享摘要。
- BufferedKV-ASD：在分数复用基础上，Target和Draft各使用请求隔离的BufferedDynamicCache；256token块、几何扩容，返回有效长度view，继续使用原生crop和原树后缀整理。只支持单请求full-attention，不接收公共前缀cache回调。未使用容量不进入attention，不更改浮点精度或算子后端。

七任务62源/66匹配输入，三组各一次自然4096/EOS，共198计时；随机方法顺序，沿用冻结AR输入与评分。只做关键测试和直接完整回答对比，不扩大性质实验/反事实/阈值或种子扫描。

验收优先看优化组与同轮原高保真组的token、文本、停止和实际逐轮路径/接受长度/regret账本是否一致，然后报告每任务配对时延及混合总wall。出现输出差异则该选项不作为本次“保持输出”的优化接纳，即使最终答案恰好相同。全部失败/慢题保留；不从输出改变获得省时。新增分配/写入/扩容复制、分数提取和缓存释放均计在线成本。

本机关键检查：`.venv\Scripts\python.exe -m pytest -q tests/test_buffered_cache.py tests/test_tree_execution.py tests/test_gate_min_gain.py tests/test_branch_drift_optimization.py tests/test_dflash_reuse.py --basetemp artifacts/pytest-tree-execution --junitxml=artifacts/tree-execution-tests.xml --tb=short`。

GPU沿用隔离launcher `--study tree-execution`，实际执行`scripts.run_tree_execution`。GPU1空闲27B停用/恢复受finally保护，GPU0聊天继续运行。旧研究目录和线上默认不覆盖；本次不改为整树静态补齐attention，也不启用新的CUDA Graph或低精度。
