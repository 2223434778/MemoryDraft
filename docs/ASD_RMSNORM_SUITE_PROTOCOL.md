# RMSNorm完整开发集自然EOS评估（执行前冻结）

承接12题RMSNorm-only约2.7%时延收益，扩到同一公开/合成数据池的48开发题，六类各8；不消费validation48/reserve96模型结果，不修改数据/评分或按输出筛题。先核查完整自然答案质量和数值轨迹，再谈是否升级执行选项。默认infra保持原样。

## 固定执行

Qwen3-8B/DFlash b16原revision、FP32 highest/TF32关闭、eager attention、greedy非思考、batch1空历史；沿用原RegionalRMSNorm，仅绑定原Qwen3RMSNorm.forward，包含Target/Draft全部norm，MLP仍eager。不改变输入、KV或接受契约。四方法原生DFlash/ASD/Run/Utility，近似组三者两profile均CPUASDVerifier；B8/g.25/M2/Utility价格2不变。eager与rmsnorm-inductor作为不同数值profile。

每类原顺序的偶/奇位置保持原GPU0/1分配，每worker24题。已有12题与新增profile评估36题均是development；不能把36题叫独立验证。本轮不再做norm同输入微回放或额外128前缀计时，避免重复前轮诊断。跨profile输出不同时，主结果保留，通过长度/验证密度/摊销成本描述，不能只用自然时延声称纯计算收益。

## 次数、顺序与质量

每worker两个profile各AR和四方法warmup32，合计20生成，单列wall、图数及编译计时。

对每题：

1. 两profile各AR一次，作为完整输出/质量及严格D对照，合计96 AR生成；AR单次时间不做速度统计。
2. 四方法×两profile×3相邻配对自然计时，每题24、总1152次。**首重复同时承担质量核查，已经计入1152，不能再加算384次生成或删去其线上编译成本。** 每profile的后两次须完全重现首轮token/文本/轮次/预算/停止原因。
3. 单对的两次生成都完成后才评分，评分在engine时钟外。同题/相同文本只评分一次；重复复用结果。共有480个方法/profile完整输出质量条目（含AR，非额外生成）；合计主要1248生成，另20短warmup。

策略及每对profile顺序用random.Random("rmsnorm-suite:<id>:<repeat>")固定；AR使用repeat=-1，策略列表仅AR。每对保持相邻。每轮请求新KV，不用上一次测试答案补历史，不强制EOS后续写，统一自然上限4096。全部错答/截断/数值分歧保留，不因质量下降而过滤样本。

eager首轮逐token/输入/轮次/账本/质量重现此前完整48题原生16块自然记录。compiled输入必须相同，输出变化如实保留。两个profile内分别比较DFlash与AR；跨profile AR也独立比较。严格D的数值分歧不是ASD例外放宽，更不能隐去。所有接受账本检查原预算/每轮cap，CPU fast/fallback调用数必须覆盖每轮。

沿用原评分：数学规范答案主评分，MATH另列有限标量等价与待复核；代码使用提供测试并单列不支持操作的未评估；SQuAD EM/F1；JSON严格类型/字段/顺序合同。canonical CPU预检仅这48题，先通过才加载GPU。评分器/参考答案不进入Target输入。

## 成本与资源

在线时延=engine同步wall（含编码/生成/解码）+策略构造+profile绑定/恢复。正式出现新编译完整计时并单列；不得减去首次调用后声称全部steady。加载、评分和落盘不计推理时延，明确列边界。

每worker仍最多8次backend编译、backend累计300秒、allocated峰值46GiB、独立空Inductor/Triton cache合计4GiB，关闭max-autotune/CUDA Graph，编译线程2。warmup后、每题后检查cache，图预算覆盖全程。根据旧48题实际自然时间，预计GPU0约1813秒、GPU1约1696秒，加编译/评分；全worker预算5400秒，执行前固定，不因结果不佳增预算。超时只终止本轮新建会话，不碰其他进程。两卡空闲才启动，不安装新依赖/下载权重。

outer_compile_inner包含backend，二者不相加；warmup_wall还含推理。报告末态图数与warmup图数、正式新编译、cache及显存。若估算摊销，只以本轮固定混合请求作条件描述，不能把前轮warmup/preflight成本当本轮纯编译。

## 冻结统计

每策略每题先合并3对log(编译/eager)比，再题等权；主结果全部48题。整体5000次类别内有放回重采样（各8题），种子rmsnorm-suite:overall:<method>，区间取排序索引125/4874；六类各8题用既有paired函数，种子rmsnorm-suite:family:<benchmark>:<method>。重复不算独立题。

预先登记旧12与新增profile评估36的分层结果，分别各类2/6题，种子rmsnorm-suite:old12:<method>与rmsnorm-suite:new36:<method>。两组都不是独立验证，禁止事后挑好看的子集。相同token子集仅列描述性几何比/数量，不替代全量主结果。

每题列首个token分歧、输出长度/EOS、轮数、账本变化及两profile质量；时延比=长度比×调用密度比×摊销成本比。摊销成本包含全部在线，不是纯Target内核。完整质量逐类列主正确、secondary、截断、未评估、待复核及SQuAD F1，逐题提升/下降均保留。小型开发集相同分数不等于统计质量非劣或通用无损。

## 复现与审计

源码、协议、统计及测试一起提交，导出Git原始blob/SHA，在独立/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-suite执行。入口：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_suite_pipeline`。worker `-m scripts.run_asd_rmsnorm_suite_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-rmsnorm-suite/part-0或part-1`。

本地先`.venv\Scripts\python.exe -m pytest -q tests/test_asd_rmsnorm_suite.py tests/test_asd_rmsnorm.py tests/test_asd_compile.py`，再完整回归；服务器相关测试通过才启动两worker。执行前后源码SHA、旧结果SHA、历史发布原blob及infra引用核验；旧12题RMSNorm统计可以内存重算，不改旧结果。小型生成/计时/评分和报告发布，排除模型、缓存、完整hidden/logits及凭据。重建`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_suite_report`。
