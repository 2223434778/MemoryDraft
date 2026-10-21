# ASD 剩余优化空间排查

全部为已观察任务的探索性排查；自然计时每题每臂1次，128前缀2次。错误/截断全部保留，不推断显著性或质量非劣。

严格profile与近似策略分开；改变dtype/attention不是ASD独有收益。生产默认仍FP32/eager、严格验证。

追加归因在看过首轮BF16正信号后单独冻结，不冒充原预注册，也不将重复旧题合并成独立样本。

## 主要判断

- development：BF16/SDPA严格D的等128秒较FP32/eager变化-38.30%；同BF16下基础ASD较严格D变化-11.73%。这些是固定前128token的探索性数据，不能推断完整长请求或质量非劣。
- transfer：BF16/SDPA严格D的等128秒较FP32/eager变化-41.19%；同BF16下基础ASD较严格D变化-21.11%。这些是固定前128token的探索性数据，不能推断完整长请求或质量非劣。
- FP32下全部6个长度变体未通过原速度门槛；不进入参数扫描。增量EOS逐ID/账本保持，但没有可确认收益，保留显式选项。
- BF16追加归因D-fast：SDPA/eager等128秒比0.9414；不同attention仍需单列数值轨迹，不把该比值当ASD独有增量。
- BF16追加归因ASD-fast：SDPA/eager等128秒比1.0518；不同attention仍需单列数值轨迹，不把该比值当ASD独有增量。
- BF16追加归因U-fast：SDPA/eager等128秒比0.9166；不同attention仍需单列数值轨迹，不把该比值当ASD独有增量。
- BF16全部6个长度复验均未通过门槛：短验证约慢15%–19%，短Draft约慢25%–29%，两种动态长度约慢1%；停止这些变体，不重新选阈值。逐臂比值见extension/length-selection.json。
- BF16/eager自身也明显快于原FP32/eager，支持精度变化是执行收益的重要来源；SDPA相对BF16/eager的作用随轨迹变化，不能解释为统一内核加速百分比。追加数据用于归因，未用于重新挑选迁移配置。
- 主开发/迁移BF16-SDPA严格D与同配置AR在6/8题的128前缀一致；math500:7位置118、gsm8k:25位置9出现数值路径分歧。该组合不能作为逐token无损配置晋升默认，当前生产合同保持不变。
- 本轮决定：保留基础ASD/简单Utility研究选项与执行配置证据，停止长度/预算调度变体及ASD前缀适配训练投入。没有穷尽其他模型、大并发引擎或全部训练方案，不能据此宣称ASD已无全局优化空间。

## development

|profile/臂|自然秒|token|正确|截断|128秒（两次）|128轮|等128完整|短验证次数|
|---|---:|---:|---:|---:|---:|---:|---|---:|
|bf16-sdpa/ASD-fast|14.612777|2478|4/4|0|8.101505|160|True|0|
|bf16-sdpa/D-fast|13.726213|2251|4/4|0|9.178037|184|True|0|
|bf16-sdpa/U-fast|13.060507|2372|4/4|0|7.873263|154|True|0|
|fp32-eager/ASD-fast|23.255963|2560|4/4|0|12.195242|144|True|0|
|fp32-eager/D-draft8|31.757918|2370|4/4|0|18.056905|232|True|0|
|fp32-eager/D-fast|23.175488|2370|4/4|0|14.876269|182|True|0|
|fp32-eager/D-old|23.185850|2370|4/4|0|14.821671|182|True|0|
|fp32-eager/D-verify7|30.741501|2370|4/4|0|16.985959|216|True|204|
|fp32-eager/U-budget|24.047422|2489|4/4|0|12.315708|146|True|8|
|fp32-eager/U-draft8|29.356411|2338|4/4|0|16.374390|206|True|0|
|fp32-eager/U-fast|21.090410|2428|3/4|0|12.192388|144|True|0|
|fp32-eager/U-old|21.083577|2428|3/4|0|12.169827|144|True|0|
|fp32-eager/U-recent|21.084590|2443|4/4|0|12.336246|146|True|8|
|fp32-eager/U-verify7|28.397662|2400|4/4|0|14.415605|178|True|170|
|fp32-sdpa/ASD-fast|23.719924|2560|4/4|0|12.289290|144|True|0|
|fp32-sdpa/D-fast|23.530884|2370|4/4|0|14.932691|182|True|0|
|fp32-sdpa/U-fast|21.291641|2428|3/4|0|12.239218|144|True|0|

## transfer

|profile/臂|自然秒|token|正确|截断|128秒（两次）|128轮|等128完整|短验证次数|
|---|---:|---:|---:|---:|---:|---:|---|---:|
|bf16-sdpa/ASD-fast|16.769576|2081|3/4|0|7.269562|146|True|0|
|bf16-sdpa/D-fast|20.760194|2582|3/4|0|9.215361|186|True|0|
|bf16-sdpa/U-fast|14.327283|2070|4/4|0|7.573390|150|True|0|
|fp32-eager/ASD-fast|31.030619|2473|3/4|0|12.740023|158|True|0|
|fp32-eager/D-fast|30.840897|2935|3/4|1|15.669007|200|True|0|
|fp32-eager/U-fast|35.221428|2962|3/4|1|12.762225|158|True|0|

## extension-attribution

|profile/臂|自然秒|token|正确|截断|128秒（两次）|128轮|等128完整|短验证次数|
|---|---:|---:|---:|---:|---:|---:|---|---:|
|bf16-eager/ASD-fast|14.578273|2424|4/4|0|7.633023|140|True|0|
|bf16-eager/D-fast|15.075632|2337|4/4|0|9.636287|182|True|0|
|bf16-eager/U-fast|14.213724|2410|4/4|0|8.531475|158|True|0|
|bf16-sdpa/ASD-fast|14.484387|2478|4/4|0|8.028093|160|True|0|
|bf16-sdpa/D-draft8|19.742863|2393|4/4|0|11.682070|240|True|0|
|bf16-sdpa/D-fast|13.590586|2251|4/4|0|9.071516|184|True|0|
|bf16-sdpa/D-verify7|19.223140|2393|4/4|0|10.782509|222|True|210|
|bf16-sdpa/U-budget|14.584967|2419|4/4|0|7.913838|156|True|28|
|bf16-sdpa/U-draft8|18.388630|2361|4/4|0|9.740483|196|True|0|
|bf16-sdpa/U-fast|12.965228|2372|4/4|0|7.820026|154|True|0|
|bf16-sdpa/U-recent|13.198782|2371|4/4|0|7.894194|156|True|14|
|bf16-sdpa/U-verify7|18.279818|2417|4/4|0|9.007244|180|True|170|

## 数值兼容性

AR和严格DFlash在同一profile的实际128前缀比较；一题可能检查多个严格长度臂。不一致保留，不能声称逐ID无损。

|运行分片|一致检查/总检查|首分歧（零起始）|
|---|---|---|
|development-bf16-sdpa-0|2/2|无|
|development-bf16-sdpa-1|1/2|math500:7/D-fast:118|
|development-fp32-eager-0|8/8|无|
|development-fp32-eager-1|8/8|无|
|development-fp32-sdpa-0|2/2|无|
|development-fp32-sdpa-1|2/2|无|
|transfer-bf16-sdpa-0|2/2|无|
|transfer-bf16-sdpa-1|1/2|gsm8k:25/D-fast:9|
|transfer-fp32-eager-0|2/2|无|
|transfer-fp32-eager-1|2/2|无|
|extension/attribution-bf16-eager-0|1/2|math500:4/D-fast:63|
|extension/attribution-bf16-eager-1|2/2|无|
|extension/attribution-bf16-sdpa-0|6/6|无|
|extension/attribution-bf16-sdpa-1|3/6|math500:7/D-fast:118, gsm8k:21/D-verify7:98, gsm8k:21/D-draft8:98|

## Draft前缀适应诊断

16对单token分支，其中双方达128的15对。第一块严格接受总数proposal/recovery={'proposal': 92, 'recovery': 88}，后续128总轮={'proposal': 268, 'recovery': 281}。
第一块接受比=1.0454545454545454，每token轮比=0.9537366548042705；冻结训练必要性门槛通过=False。

分支分别重prefill、之后严格续写；两个实际前缀可有不同难度。这是部署分布问题的必要性检查，不是训练效果或在线速度证据。

## 范围

验证截短不退还完整Draft计算；缩短Draft块需真实重跑，不能用长Draft事后裁剪冒充。长度调度只读过去推进与账本。动态截短后门控作用于实际块，因此自然轨迹可改变，必须分列质量与长度。

profiler独立带插桩，排除主计时。迁移选择由运行前冻结规则重建；仅当开发有至少5%等128优势且质量/截断通过才追加对应臂，没有事后扫描阈值。

审计主生成420次，另32条Draft分支、28条AR128和6次独立profiler；短warmup不计入主结果。8条原FP32完整D/Utility基线逐ID/输入/账本重现，旧结果原始SHA核验见postrun-audit.json。

重建：`.venv\Scripts\python.exe -m scripts.write_asd_optimization_report`。协议`docs/ASD_OPTIMIZATION_PROTOCOL.md`及`docs/ASD_OPTIMIZATION_EXTENSION_PROTOCOL.md`，原始证据`results/asd-optimization/`。
