# 跨任务基线第一阶段协议（GPU运行前冻结）

本轮目的是补足数据和评测基础，不新增接受规则，不用小切片对整个优化方向作最终否决。研究路线与待做项见ASD_RESEARCH_ROADMAP.md。前次所有证据及infra冻结引用保留。

## 数据与评分

`data/research/asd-suite/manifest.json`固定来源提交、原始哈希、抽样、分组及分片文件哈希。六类各32个，共192；本阶段仅48 development（每类8）。validation48与reserve96不生成模型答案、不供未来训练。输入不能包含隐藏的参考答案、HumanEval测试或canonical_solution；MBPP给定公开测试属于任务定义，按旧prompt保留。

- GSM8K排除旧0..31后按源ID固定哈希抽样，最终数值评分。
- MATH-500排除旧0..23后按学科轮转、组内固定哈希抽样；开发覆盖7学科。主指标normalized boxed exact，额外预定有限标量等价（整数、分数、有限小数以及单变量等号右侧）；其他主评分失败单列待复核，不事后添加任意符号规则。
- MBPP官方test ID19..510，排除旧11..18；HumanEval保守排除0..15。固定哈希各取32，代码先按参考实现检查全部192对应评分。程序以新子进程、空环境、时间/内存限制和import/builtin限制运行；这是算法任务评测器，不宣称通用安全沙箱。支持cmath等普通算法模块，canonical被评分器挡住时修评分器并保留失败，而非删题。
- SQuAD1.1开发源每文章只取一个哈希优先问题，32篇文章，split按文章隔离。输入提供原文段落，输出要求最短支持答案；按标准英文归一化、所有参考答案取最大EM/F1，不裁掉模型多余解释以虚增质量。
- JSON为新种子的事实提取合成控制，非生产流量/长期记忆。输入给自然语言事实和固定schema，隐藏expected不进入prompt；核验完整类型、字段、列表顺序，拒绝重复key/非有限常量。

数据原始下载GitHub API曾限流，改用只读git ls-remote固定提交后获取原始blob。MBPP f82046ba5aabbbb427dbfd38a254d26bff08b533；HumanEval b9453e8f1865df0a0057a1383767caa400acfbcf；SQuAD eee5fdbf62f8613a7812b03419e6b29617b74fd1。许可证与各源哈希在数据manifest/ATTRIBUTION。人工核对旧Spec-Bench代码任务未发现本池HumanEval相同函数签名。

## 方法与执行

固定原Qwen3-8B/DFlash版本、FP32/eager、batch1、greedy、非思考、空历史，不改Target prompt、EOS或接受规则。五组AR、严格DFlash、原ASD、ASD-Run、ASD-Utility，后四组配置沿用POLICIES：B8/g.25/M2，Run为exact-run，Utility价格2。原生产运行时与实验分离。

每请求每方法自然EOS上限4096，重复3次；遵守EOS的前128上限重复2次。每请求/阶段/重复的五组用固定种子随机交错；同请求始终同物理GPU，两个独立worker按每类源选择位置奇偶分配。每次新请求KV/预算重建。所有方法包括AR均做完整自然生成，AR不是仅128控制；真实代价全部由engine.generate计时，评分/结果落盘在计时外。

- 主生成48*5*(3+2)=**1200**。两卡各600，不把重复算成新题。
- 每类最先2个development请求，后四组另做一次带事件记录的完整机制回放，共**48**，输出/账本必须与未插桩自然运行一致；插桩秒数排除速度结果。短warmup单列。
- 自然重复必须token、账本和轮次完全一致；前缀重复同样核查。上限裁短最后块可能改变ASD的remaining gate，只在该块前核查自然/前缀一致；实际边界和分歧完整记录，不能用解释掩盖更早变化。
- 严格D与同profile AR的完整自然/128逐ID比较，记录首分歧和质量。出现浮点路径分歧不删题、不放宽接受条件，不宣称逐ID无损。若是运行时状态错误或非确定性，则保存现场并修机制后单独重跑，禁止合并失败/成功原始数据。

## 分析与边界

按任务类别分别报告：自然请求中位耗时（每题3次）、IQR/重复变化、最终质量、自然EOS/上限截断、输出长度、TPS、Target调用密度、regret/例外。AR调用数按输出token数，D/ASD按验证轮+prefill，不能拿AR的空trace当0次调用。

前128速度主要集合要求同一比较的两方法均完整达到128；真实提前EOS全部保留并单列。SQuAD可能没有足够等128样本，此时不报告该类别固定工作量速度。自然质量失败/截断不从主要请求汇总删除；可另列共同正确完整集合为诊断，并明确它是事后条件集合。

比较AR→D、D→ASD、ASD→Run/Utility；速度按题配对比值，5000次固定种子的请求bootstrap，重复先合并，不跨题重采样重复。八题/类别区间仍可能很宽；列出逐题正负方向，不能用未显著直接判断无效。输出长度与调用密度分开，不把路径缩短自动当每token效率。

质量按每题一次计算并核对重复文本，不将重复当更多正确样本；代码参考/评分器故障与模型错答分开；MATH待复核项与原主分数同时保存。不会在本批新开发结果上改变预算/价格、过滤坏题或扫描长度；后续方法根据基线机会另立冻结协议，独立验证后再判断适用场景。

## 复现

本地准备：`.venv\Scripts\python.exe -m scripts.prepare_asd_suite --sources artifacts/asd-evaluation-sources --output data/research/asd-suite`。输入原始GSM/MATH缓存和固定版本新源需按manifest哈希准备，脚本不在线取数据、不覆盖旧目录。

服务器独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-suite-stage1`。入口：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_suite_pipeline
```

pipeline冻结完整源码manifest，两个子进程分别CUDA_VISIBLE_DEVICES=0/1，`--shard 0/1 --benchmarks gsm8k math500 mbpp humaneval squad json --config configs/runtime-dflash.json --output results/raw/asd-suite-stage1/part-0或part-1`。执行前后SHA一致，旧证据受保护。只发布许可明确的选定公开记录、小型压缩trace及审计，模型权重/大型profiling trace不上传。
