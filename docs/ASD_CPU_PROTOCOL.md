# 短块ASD判定精简：CPU摘要方案（执行前冻结）

承接ASD_EXECUTION_FINDINGS.md的下一步。不改Target、Draft、输入、接受条件或默认运行时；只比较相同ASD策略的执行实现。显式研究模块CPUASDVerifier，原ASDVerifier完整保留。

## 实现边界

FP32、候选K≤15、词表小于2^24时，GPU计算每位置top值/ID及候选值，打包top、candidate、top-ID、proposal-ID共4K个FP32数值，一次传到CPU。ID在此范围可精确表示，传输没有完整词表logits。CPU以FP64完成原剩余长度/Run、cap、gate、请求预算、真实候选EOS、Utility及并列选择规则，更新同一账本。Target posterior的原argmax及所有KV/停止路径不改，本轮不额外合并它以免同时改变多个执行因素。

对top/candidate分数做共同二进制格点检查：把绝对系数总和限制在2^53以内，使regret差值、前缀和及charge无论求和顺序都在FP64中精确表示。极端跨度、非FP32、K>15、过大词表或自定义utility hook回退原Tensor实现；非有限top/candidate报错，不花预算。原已有1e-9最终预算防御仍保留，不增加新的接受/账本容差。不把数学约束或有限样本相同夸大为任意GPU/全部输入已验证。

## 固定任务和对照

仍用六类开发集各前两题，共12个已观察题，位置奇偶继续在相同A6000，validation/reserve不生成答案。原Qwen3-8B/DFlash b16 revision、FP32/eager、greedy、非思考、batch1、空历史，ASD/Run/Utility B8/g.25/M2/price2不变。

原生DFlash一组，加三策略各Tensor/CPU两组，共7组。遵守EOS最多128，每题每组5次，共**420次正式计时**。每repeat随机三策略组次序及每对Tensor/CPU先后，保证同策略两实现相邻；DFlash随机插在组间。种子`asd-cpu:<id>:<repeat>`。两worker各210计时。每worker7个相同warmup请求、上限128/EOS另计。

计时沿用engine.generate同步边界，包含tokenization、Draft/Target、摘要归约/复制、CPU精确性检查/判定、KV和停止检查。策略对象构造另测policy_setup_seconds，加到主latency_seconds；保留原engine_latency_seconds，两实现相同处理。模型加载/warmup及落盘/评分/影子审计不计。没有预先缓存候选或测试答案。

所有计时输出、输入SHA/snapshot、轮次和完整账本必须逐项与上一阶段16块128原轨迹相同；变更立即保留现场并停止，不能挑成功子集。CPU fast/fallback次数均保留并计入成本。原生DFlash仍与旧严格轨迹一致，不以新ASD冒充严格D。

## 独立实际状态检查

每题计时结束后，三原策略各生成一次，总**36次插桩**。每个实际块同时调用原Tensor验证器和持久CPU影子，逐项精确比较decision及summary；保留完整候选/Target top IDs、mismatch、regret、values、真实EOS信息，另用既有独立标量审计。影子支出不改变主预算。仅复制第0/4/12轮到独立回放，预计重现上一轮77状态，早结束不补轮。

每状态Tensor/CPU各5次warmup和30次计时，前置CUDA同步在计时外，调用及末尾同步计入；每次恢复实际spent_before并核对原决策。输入驻留GPU，不上传完整logits或大trace。回放收益与线上请求收益分开报告。没有新的自然4096答案评分，也不能从128一致声称所有自然长度已经逐项验证。

## 统计和取舍

主比较为每策略CPU/Tensor：每题先对五个相邻pair的log时延比取平均，题级几何比再等权合并。预定同时列六类各2题点估计和12题整体；5000次按类别各2题重采样的分层bootstrap，固定种子`asd-cpu-bootstrap:<method>`。重复/状态不是独立题，不计算每类n=2的可靠区间。保留逐对时延和回退次数，以检查约1%量级收益是否超过噪声。不用本轮挑价格/预算/任务或宣布算法新增量。

若一致且线上有小幅收益，保存为后续研究可选执行优化；不立即改默认infra。若只有微基准改善或线上区间含持平，诚实保留结果，不扩大同类验证器扫参，转向主要模型执行优化。设备账本/新模型/训练/精度切换都不混入此轮。

## 执行与发布

本地先`.venv\Scripts\python.exe -m pytest -q tests/test_asd_cpu.py tests/test_asd_cpu_probe.py tests/test_asd.py --basetemp .test-tmp-asd-cpu --junitxml=artifacts/asd-cpu-tests-local.xml`，随后适当完整回归。提交源码/协议/报表后导出Git blob/SHA，部署独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-cpu`，只读链接现有模型。运行前后全部源码和旧结果SHA核验，infra冻结引用保持。

服务器入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_cpu_pipeline`；子命令`CUDA_VISIBLE_DEVICES=0/1 ... -m scripts.run_asd_cpu_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-cpu/part-0或part-1`。小型结果与测试/报告推GitHub，失败现场不覆盖。
