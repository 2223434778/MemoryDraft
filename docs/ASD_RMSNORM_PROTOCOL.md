# Target/Draft RMSNorm区域编译（执行前冻结）

承接MLP-only的微小收益，独立检查原始Qwen3RMSNorm.forward的Inductor融合。只改这一种模块的执行，MLP仍eager；不改原norm算式、attention、KV、FP32精度、highest/TF32关闭、原生16块、Target输入或接受规则。沿用原模型revision、greedy非思考空历史、CPUASDVerifier；AR、DFlash、ASD、Run、Utility各自契约不变。默认infra不变。

## 范围与预算

源码已核验服务器transformers4.57.3的Qwen3RMSNorm及DFlash原始模块：隐藏状态norm与Q/K head norm均属于该类；Draft还包含hidden_norm。运行时按类型发现全部模块并保存名称、类型、weight_shape、eps，不猜测总数量。按class.forward共享torch.compile(fullgraph=True,dynamic=True)，临时绑定实例forward并恢复，不复制权重，不启用max-autotune/CUDA Graph。编译FP32 reduction可能改变末位数值，作为单独数值profile记录。

每worker最多8次backend图编译、累计backend300秒、总1800秒；allocated峰值46GiB、准备完成cache合计4GiB。两个GPU启动前需空闲；独立空Inductor/Triton cache，编译线程2，不安装新包/下载权重。超预算失败保留，不提高阈值掩盖重编译；超时仅终止本轮worker会话。

## 数据、计时与诊断

六类各前两道已观察dev题，共12个，按每类第一/第二题分别GPU0/1。EOS前最多128，不强制续写，不生成validation/reserve或新的自然4096答案。

1. 每worker两profile（eager/rmsnorm-inductor）各AR和四策略warmup32，合计20次，单列成本。
2. 每题两profile×AR及四策略一次预检，共120生成。eager逐token/轮次/账本重现CPU阶段及旧AR。compiled输入必须相同；token、长度、EOS、轮次、账本变化全部保留，不能因不一致删题。两profile各自DFlash与AR比较。
3. 每题四策略×两profile×3次，共288相邻配对计时。复用种子asd-compile:<id>:<repeat>的策略与profile位置排序，RMSNorm标签替换MLP标签；每profile重复精确重现自身预检。在线总成本=engine同步时延+策略构造+绑定/恢复成本；正式新编译不得扣除，单列事件。
4. 正式计时之后12次独立eager DFlash诊断。每owner选择第一处input_layernorm、post_attention_layernorm、q_norm、k_norm、最终norm；Draft另选hidden_norm。Target每处前两次调用（prefill/首验证），Draft每处首次，共每题16状态、合计192。缺角色/调用即失败，不以重复层数当独立请求。hook移除后同一clone输入比较原forward与编译forward，保存shape、原始/回放stride、逐位相等、max_abs、RMS、max_relative（floor1e-6）；不上传hidden。各状态每profile5warmup+20交错计时，前置同步在外、调用与后置同步在内，复用mlp-replay:<repeat>顺序种子。clone的stride差异单列，不能冒充完整在线流水。

主要生成420（120+288+12），另20warmup。两profile数值差异不等于接受规则放宽，但不能据此宣称全输入无损。

## 分析与冻结

每题三pair先取log平均，各题等权。6类各2题点估计；整体12题5000次类别内bootstrap，复用estimate，种子asd-cpu-bootstrap:rmsnorm:<method>。不把重复/192状态当192独立请求。全量主结果保留输出变化；另外说明同token路径。报告长度比×调用密度比×每调用摊销成本比=时延比，摊销成本是全部在线而非纯Target内核因果量。

单独报告准备wall（含预热/预检推理）、backend秒、outer_compile_inner秒（与backend嵌套不相加）、cache、显存与重编译。模块回放用于检查是否确实减少小算子成本，不能直接外推完整请求比例。本批是开发小检查，不宣布通用赢家或自然答案质量非劣。

源码、脚本、协议、测试、统计一起提交并导出Git blob SHA；独立服务器目录/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm。入口：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_pipeline`。worker入口`-m scripts.run_asd_rmsnorm_probe --config configs/runtime-dflash.json --shard 0/1 --output results/raw/asd-rmsnorm/part-0或part-1`。

本地先`.venv\Scripts\python.exe -m pytest -q tests/test_regional_compile.py tests/test_asd_compile.py tests/test_asd_rmsnorm.py`，通过再完整回归；本机Dynamo/eager测试不当CUDA性能。服务器相关测试通过才启两个worker。重建报告：`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_report`。执行前后源码SHA、旧结果SHA和infra引用核验，小型trace/日志/报告发布，排除权重、缓存、完整hidden/logits与凭据。

本轮仅复用已核验的PyTorch2.5.1/Inductor与transformers4.57.3安装源码API，无新外部依赖；区域编译背景见ASD_COMPILE_PROTOCOL.md。若此范围仍无实用收益，停止对该单算子扫参，下一项需考虑更大执行边界或独立精度/质量评测，不从负结果否定未测方向。
