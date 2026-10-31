# 新增八任务：补齐AR、严格成本树、BASTION

2026-10-31。用户授权优先补齐已有文献数据，并实测关键缺口。本轮复用已冻结的182请求：IFEval64、LCB v6分难度24、CanItEdit32、LongBench四子集各8、完整AIME25共30，不重新选题或扫参。

- 五组AR、DFlash、BudgetTree、官方BASTION、当前BufferedKV-ASD，每格一次自然EOS生成，预定910次。DFlash与当前方案提供同批锚点，旧ASD三组和九组数据保留，不跨窗口将旧计时充当本批实测。
- Qwen3-8B、原DFlash-b16、单A6000/GPU1、FP32/eager/RMSNorm、四线程、greedy，模型/模板/接受规则不变。输入token哈希、用户快照和各题输出上限与上一批DFlash逐项核验，182题全部一致才运行。
- 同GPU/相同后端复用公开冻结合成校准：成本树使用expanded-suite的calibration.json；BASTION使用external-trees的独立roofline校准及profile常数，不用测试答案或新输出调参数。两套校准原始探测和来源SHA保留。BASTION官方commit/许可/默认32..8192预算沿用既有核验，不改上游构树/验证/提交。
- 全请求时延计入tokenization、prefill、起草、构树、验证、提交与decode。加载、预热、旧一次校准及离线评分单列；BASTION内部统计同步保留并计时。AR没有草稿验证轮，实际输出长度与wall分别报告。
- 原提示、LongBench3840首尾截取、19/32截取标记和输出上限保持：QA128/摘要512、指令/代码编辑2048、AIME4096。编辑v3评分源码/同一规范31/32掩码复用；不重新变更测试，代码运行限制和CPU评分沿用。所有算法统一评分，无环境的1题不算模型错答但保留速度。
- 运行中记录全部输出、停止、峰显存、验证推进及树节点；结束逐题检查三个严格方法与本窗口AR身份，差异保留。DFlash和当前方案也与上一批已公开输出比对；评分正确不等于逐token保真，开放/摘要质量只按既定指标解释。
- A6000隔离expanded-baselines-v1目录；确认27B空闲后仅释放GPU1，finally恢复原27B及adapter，GPU0服务不改。旧main/infra冻结引用不动，原长文/完整测试载荷不上传GitHub，只发布紧凑生成记录、指标和来源。

GPU命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_expanded_baselines`。启动器执行专项检查后运行，并负责服务恢复。

本机快速检查：`.venv\Scripts\python.exe -m pytest -q tests/test_expanded_baselines.py tests/test_external_trees.py tests/test_expanded_suite.py tests/test_tree_execution.py --basetemp artifacts/pytest-expanded-baselines --junitxml=artifacts/expanded-baselines-tests.xml --tb=short`。
