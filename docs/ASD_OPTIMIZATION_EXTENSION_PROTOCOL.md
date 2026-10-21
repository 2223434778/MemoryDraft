# ASD 优化排查追加归因（追加GPU执行前冻结）

这是看到阶段A结果后的追加，不冒充原预注册实验：FP32/eager各长度臂均未通过5%等128速度门槛；FP32/SDPA近似持平；BF16/SDPA的D-fast等128总秒9.178左右，对照14.876左右，出现执行组合收益。新增两个待核实因素：是否主要来自精度，以及长度在BF16下是否有不同作用。原协议、结果与执行源码不修改。

独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-optimization-extension`，冻结新Git/blob清单；等待原完整pipeline结束才开始，避免同GPU混跑。

- 任务仍为4旧开发题math500:1/4/7、gsm8k:21。BF16/SDPA重跑D-fast/ASD-fast/U-fast和全部6个原长度臂；BF16/eager只有3个核心臂。自然EOS每题每臂1次、128前缀2次，共144主生成；另8个AR128，短warmup单列。不再跑profiler或Draft分支。
- 不更改7/15验证长度、Draft8、近期4轮/均值4及已花预算6等阈值；不引入置信度或新训练。新的独立核心对照避免用上轮不同运行时的秒数为本轮长度臂定价。
- BF16各长度候选在原相同门槛下比较同轮BF16的匹配D/U核心臂：完整等128且总时间降低>=5%，自然正确数不低，截断数不高才继续。若通过，仅迁移至固定另外4旧题math500:8/9、gsm8k:24/25，保留同轮3核心对照；全不通过则停止，不扫参数。
- BF16/eager作为精度与attention归因控制，只解释开发样本机制，不自动晋升生产profile或将质量相同解释为非劣保证。所有实际AR/D数值分歧保留，不称为逐token无损。
- 不改变阶段A的profile/长度筛选或迁移结论；追加数据单列，不与原重复拼成新样本数或用重新运行挑最快。报告工具的新增矩阵身份核查只是离线加强，原summarize/draft_summary/select_transfer选择函数保持原冻结逻辑。

准确入口：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_optimization_extension`。结果`results/raw/asd-optimization-extension`，发布至`results/asd-optimization/extension`。
