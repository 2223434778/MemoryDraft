# Target/Draft MLP区域编译（执行前冻结）

承接CPU判定小检查。只改原Qwen3MLP.forward的执行方式，复用现有PyTorch 2.5.1 Inductor，不改attention、KV、Target prompt、精度、块长或接受规则，也不重写MLP算式。默认infra不变。

## 固定编译范围与资源

以named_modules发现原Target和Draft中的Qwen3MLP，按class.forward共享编译入口，临时绑定、退出恢复，无权重复制。`torch.compile(fullgraph=True,dynamic=True)`，Inductor backend；关闭max-autotune和CUDA Graph，不提高FP32 matmul precision（保持highest/TF32关闭）。backend callback记录图操作、输入规格、状态和耗时；不允许静默用eager伪装成功。

模型仍Qwen3-8B/DFlash b16原revision、FP32、eager attention、greedy、batch1、非思考、空历史。原生DFlash、ASD、Run、Utility四策略；近似组三者均沿用本项目CPUASDVerifier，编译/未编译两组一致。B8/g.25/M2/Utility价格2不变。

每worker最多8次backend图编译、累计backend编译300秒、进程总1800秒；峰值allocated显存不超过46GiB，预检结束独立Inductor/Triton缓存总大小不超过4GiB。两GPU空闲才启动，各使用已授权目录独立缓存，编译线程2；不安装新包或下载模型。超预算保留现场失败，不提高预算掩盖重编译。超时只终止本轮创建的worker会话，不碰其他进程。

## 数据与顺序

仍六类各前两道已观察开发题12个，按每类第一/第二题分别到原GPU0/1；validation/reserve不生成模型答案。EOS前128，不强制短回答续写。

1. 每worker两个profile各AR及四策略在同一简单请求warmup32，共10次，合计20，保存wall/编译事件，不进入正式速度。
2. 每题两profile×AR及四策略各一次完整前128预检，共120生成。所有eager输入/轨迹/账本重现上阶段CPU基线，AR重现完整48题阶段的AR128。compiled保持输入及接受契约，输出差异原样记录，不能因近似账本末位变化或token差异删题。预检覆盖本批形状，编译准备wall和内部计时单列。
3. 预检全部完成后四策略×两profile×每题3次，共**288正式计时**。固定种子`asd-compile:<id>:<repeat>`随机策略及每对profile先后，保持相邻配对；每profile重复须完全重现自己的预检token/轮次/账本，否则停止而非合并。新编译事件即使出现也计入线上时延并标记，不能减去后称steady。engine同步时延之外加回策略构造及profile切换/恢复成本，单列三个字段；加载/预热/落盘不计。
4. 所有计时后12次独立eager DFlash诊断，hook只捕获第一层Target前两次MLP输入（prefill/首个验证块）及第一层Draft首次输入，共36实际状态。移除hook后在同一个clone输入上比较原/编译MLP输出：逐位相等、最大绝对误差、RMS和相对误差（分母floor1e-6）如实报告；不以allclose掩盖差异。每状态两profile各5次warmup、20次交错模块计时，前置同步在外、调用及后置同步在内，记录输入shape/stride，不发布完整hidden tensor。

主要生成合计420次（120预检+288计时+12诊断），另20warmup。原生DFlash在两个profile内分别与AR逐token比较，跨profile AR也比较。数值变化不是放宽接受条件，但不能因此宣称无损替换。128未覆盖完整长数学答案，本轮不做新的自然质量非劣声明。

## 分析

每策略编译/eager时延比：每题先合并3个相邻pair的log比，再对题等权；6类各2题点估计，整体12题5000次类别内重采样，复用CPU阶段estimate，种子`asd-cpu-bootstrap:compile:<method>`。重复和MLP状态不是独立请求。本批是开发子集，不做最终赢家选择。

逐题同时列token/轮次/账本一致性、首次token分歧、长度比、调用密度比、每调用摊销成本比和峰值显存。成本恒等式用于描述，摊销成本包含全部online，并非纯Target内核因果归属。不同输出长度/轨迹时，不能只拿时延比称计算加速；另列同token路径的描述性子集，保留所有主结果。

编译callback秒仅为backend子阶段。PyTorch版本内`compilation_time_metrics['_compile.compile_inner']`若存在，单列outer编译时间，不能与backend再相加；原始计时表也保留。cold_preparation_wall还包含预热和预检推理，不能全算纯编译。编译成本如需摊销，只作当前请求混合的条件估算，原始一次性成本不隐藏。

同输入MLP回放与在线请求分别报告；编译MLP仅能减少这一范围的算子/访存/调度，不等于整个推理引擎已经编译，也不预设大幅提速。失败不否定RMSNorm、attention、静态KV、CUDA Graph或其他精度后端等尚未测方向。

## 验证与部署

先`.venv\Scripts\python.exe -m pytest -q tests/test_regional_compile.py tests/test_asd_compile.py --basetemp .test-tmp-asd-compile --junitxml=artifacts/asd-compile-tests-local.xml`；本机Dynamo/eager仅测图捕获与恢复，不当Inductor性能。相关/完整回归后提交源码、协议及统计，导出Git blob/SHA到独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-compile`。服务器入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_compile_pipeline`；worker `CUDA_VISIBLE_DEVICES=0/1 TORCHINDUCTOR_COMPILE_THREADS=2 ... -m scripts.run_asd_compile_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-compile/part-0或part-1`，独立cache见provenance/启动日志。

执行前后源码、旧结果SHA、infra引用核验；保留失败，不上传缓存、模型或完整hidden/logits。小型记录/测试/报告发布GitHub。

区域编译依据：[PyTorch 2.5官方说明](https://pytorch.org/blog/pytorch2-5/)、[官方区域编译实例](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)。实际API以[PyTorch v2.5.1编译入口](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/__init__.py)和[Inductor backend](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/backends/inductor.py)及服务器安装源码核验，未照搬其他模型的速度数字。
