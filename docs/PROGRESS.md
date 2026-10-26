# 2026-10-26 — 按用户决定停止DSpark，继续DFlash主线

- 用户决定放弃DSpark后续研究，继续DFlash。后续研究主线为DFlash + 成本树 + ASD；停止DSpark优化、扩样和训练投入，已完成的正负结果、源码、权重目录与实验分支保留，不删除或重写旧记录。
- 从已发布f56561438a68f7362ffc8cec61f409b248c5f4f0创建`codex/dflash-mainline`，原`codex/dspark-tree-opt`、`codex/dspark-tree`、`codex/asd-splice`及面试版引用不变。仅更新README和本进度文件，不修改推理实现、配置、Target或接受规则；当前CLI/服务默认严格验证，研究ASD显式启用。
- 检查`memorydraft/runtime/config.py`及configs/runtime入口：DSpark仍只存在于独立实验路径，无需切换默认模型。命令`git switch -c codex/dflash-mainline`；文档检查`git diff --check`；发布校验`.venv\Scripts\python.exe -m results.dspark-tree-opt.helpers.verify_publication --git`、`-m results.dspark-tree.helpers.verify_publication --git`、`-m results.asd-paper-extension.helpers.verify_publication --git`。本次没有新GPU运行或新性能结论；历史结果路径保持`results/dspark-tree*`及`results/asd-paper-extension`。

# 2026-10-26 — DSpark等价优化与复测完成

- 发布提交前的覆盖检查发现manifest误收本地__pycache__字节码（Git正常忽略），已仅移除该非结果条目；21原ZIP成员和全部实验/报告字节不变。修正后逐一核对索引/最终Git blobs再推送，保留提交修正记录。

- 请求冷LRU64行Markov偏置缓存、保守上界停搜已实现，逐行FP32 GEMV及原ASD/Target/权重/块长不变。独立分支codex/dspark-tree-opt，封存源692ce0f72afcb2a706171fdd9d8a312e4b4deddc已推送；2586 canonical Git文件/sourceSHA6467d3045a97049d993a3a5f0124d0de4b736f51492b32bb90550d8a5fe79f68，源ZIP64266365B/SHA26d0b3eaf24f41cd71336f314d739427834904ab235432428f9be743eeb5cbf0，服务端执行前后逐文件核验一致。
- 08:19:31 UTC在新目录/media/a6000/D/ZhaoBinHe/MemoryDraft/dspark-tree-opt以owned timeout PID73480启动scripts.run_dspark_tree_opt_pipeline。两卡官方骨干对照通过、164 passed/2 skipped；七族14源任务49真实GPU状态、四变体224消融行全部精确选树一致，构树耗时下降17.87%–67.88%。此前本机专项32、相关回归125 passed/2 skipped、后续参考覆盖/报告9 passed；旧扩展AR缺messages在部署前离线发现，保留来源、重建消息并核对实际prompt SHA/快照，未补造旧记录。
- 正式Alpaca16/HumanEval4/MT4对话，24源/28匹配输入，四组448主计时完整完成；112次优化/原DSpark配对生成的完整选树摘要、token/text/停止/ASD账本相同。自然时延较原DSpark分别省2.55%/4.26%/3.05%，固定TPS增2.74%/4.34%/3.26%。Alpaca较本轮DFlash组合省16.88%、固定TPS增12.42%，原DSpark在本轮已省14.68%，不把扩样差异归为工程增量。HumanEval/MT仍较DFlash慢44.53%/16.40%；Human四组均4/4，开放质量未另重评。自然Human轮次DFlash43/DSpark65不变，说明短块瓶颈仍在。
- 正式新编译图0，GPU峰39.284GiB、LRU上限38895616B；两卡结束后19MiB/0%。结果ZIP1155114B/SHA1297a8c074f597379ddb7a80d2d91cd17358c501367a57e9987409bafb0eaf15于08:37:42 UTC收尾校验下载。21原ZIP成员完整保留，发布results/dspark-tree-opt；本机逐字节重建analysis/report一致，审计记录输入参考来源（旧DSpark16/扩展12）、缓存计数及全部成本。仅复用旧AR输入/共同历史，不引用旧时延作新基线。
- 命令：服务器OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_dspark_tree_opt_pipeline；本机.venv\Scripts\python.exe -m results.dspark-tree-opt.helpers.audit以及-m results.dspark-tree-opt.helpers.verify_publication --git。结论docs/DSPARK_TREE_OPT_FINDINGS.md，自动表docs/DSPARK_TREE_OPT_RESULTS.md，协议DSPARK_TREE_OPT_PROTOCOL.md，原始行/消融/源码清单/测试/receipt均在独立结果目录。原DSpark、DFlash、面试引用保留，旧2576 Git对象单独保护；不宣称全新保留集或全部场景胜出。

# 项目进度

## 2026-10-26 — DSpark 八组比较完成，保留原 DFlash 主方案

- 独立实现DSpark官方权重加载、逐父token条件分数的best-first成本树和树上ASD，原DFlash解码/构树/配置未改。七族各4个已观察源任务，28源任务/32匹配输入、1024主计时、32自然AR全部完成；两worker各16输入/512计时。自然4096/EOS两次、共同128/EOS一次、固定256忽略EOS一次，正反顺序配对，无结果筛除或主参数变更。MT使用统一AR首轮历史。
- DSpark组合较DFlash组合的自然配对时延变化：GSM+16.61%、MATH+44.95%、HumanEval+50.90%、MBPP+22.81%、MMLU−1.48%、MT+20.14%、Alpaca−2.63%。五族正确数分别同为3/4、2/4、4/4、3/4、3/4；MATH正确集合变化：156由30度正确变60错误，327由4096截断变正常结束/15正确。开放回答未新裁判，不宣称质量保持。原生DSpark7与DFlash15及训练检查点差别明确列出。
- 保留新底座的场景收益：Alpaca中DSpark树+ASD较线性ASD自然省10.45%、固定TPS增20.76%、共同128两题省14.55%；接受长度2.632→3.917、验证152→120，输出552→590。相对原组合Alpaca共同128两题省11.78%、固定TPS增10.71%，自然省2.63%。当前继续保留DFlash组合为主，DSpark自由生成候选独立保存。
- 机制差别：HumanEval原组合接受10.233→DSpark6.262、轮43→65；MATH10.431→6.348、633→910，DSpark原生深度上限7。离线校准辅助构树DFlash约0.58ms、DSpark约9.3ms，骨干均约8.3ms；不把校准当在线因果profile。Human固定TPS增14.47%但自然慢50.90%，约53%固定token在首次EOS ID之后，指标分别报告。
- 源62e801067dcb990ea9ae9a18ed1c09f02e0d04ab/2531 canonical Git blobs前后全部一致；2026-10-26 14:25:24启动、15:00:13确认完成，PID1464已退出，两卡19MiB/0%。原始ZIP1600197B/SHA256cbf14e2dbaa69513c9c60203c535f6ac70a4b941301dc14e26c78efe4d985dd7，CRC和27原始成员核验通过。原始数据results/dspark-tree，auxiliary保存启动/完成/硬件片段，模型和全logits/KV未发布。
- 本机113 passed/2 skipped与补充20 passed、服务器151 passed/2 skipped分别保留；两卡官方源码context4/17骨干差0、Markov greedy一致，128自然严格对照均与AR相同、质量重复无变化、正式新图[0,0]、峰allocated40.128GiB。本机重建analysis.json/report.md逐字节相同，SHA分别eff6a14c3a385734ca2b966680083c5bc0ef568a744d036d829dbf825b0db4ee/85cf83d5f764db4fb4292414cb852edc59089550492c3657c26b66632ae45e94。
- 派生完整表命令`.venv\Scripts\python.exe -m results.dspark-tree.helpers.finalize_results`；新增DSPARK_TREE_FINDINGS/RESULTS和README入口。1851旧Git对象、原codex/asd-splice=8e7a80b、旧174发布文件和面试版三个引用核验通过；新结果以publication.json封存，字节检查入口`.venv\Scripts\python.exe -m results.dspark-tree.helpers.verify_publication --git`。发布分支codex/dspark-tree；后续若研究长窗口或条件投影批处理，独立登记，不改本轮结果。

## 2026-10-26 — DSpark 成本树独立接入与配对比较

- 为用户要求新建`codex/dspark-tree`，已验证DFlash分支`codex/asd-splice`保持`8e7a80b5b1af3b09886659e3549691be3ce6493e`；DFlash原解码/构树/配置/结果未修改，面试版三个引用保持原值。
- 锁定DeepSpec代码005e03b81cec38b7da6399833d609ee89a2587f2/MIT与官方DSpark-8B-block7检查点03326e5043815da1f81b109078b2889737c26017，4742170330B/SHA5c922d1f96c8d18dbdf005c40ea99878d9133d70eaa3be4bd1077ee5f2afc1e5本机下载核验通过，向A6000新模型目录传输中；模型无单独许可card元数据的事实保留，不上传权重。原始上游骨干/Markov文件重建到canonical Git blobs，待GPU与适配器逐状态核对。
- 新增DSpark骨干加载/Markov条件分数、父token感知成本树、独立解码薄适配与8组runner/report；沿用既有TreeCosts、树mask/ASD/选中路径KV提交。置信度裁剪关闭，原生DSpark7候选与旧DFlash15候选分别保留。评测计划在新输出前固定于DSPARK_TREE_PROTOCOL：已观察七族各4源任务/32匹配输入，1024主计时与32 AR，质量/长度/截断单列，不冒充新保留集。
- `.venv\Scripts\python.exe -m pytest -q tests/test_dspark.py tests/test_dspark_report.py tests/test_budget_tree.py tests/test_draft_tree.py tests/test_dflash_reuse.py --basetemp artifacts/pytest-dspark-ready --junitxml=artifacts/dspark-ready-tests.xml`：113 passed/2 skipped/15.95s。首版检查1项断言的排序方向错误、其余67通过，修正测试预期排序后DSpark15项及完整专项通过；失败XML保留。不把单测写作GPU性能结论，服务器实验待源封存、传输核验及骨干对照后启动。
- 源已固定62e801067dcb990ea9ae9a18ed1c09f02e0d04ab并推送，2531 canonical Git blobs/sourceSHA32c490f7b372c8cfad92c82d76c619eccde79a470f3a67b99a9419cf2cffe1d5；源码ZIP62547991B/SHA5726d3cddc04d1f6a561d95b572f23a9285b24f23cf64a23c935e4338f342de6，服务器逐文件核验通过。首次校验命令抄录SHA有误，断言在建root前停止，重新核对与更正后才部署，未启动错误源。官方权重全部SCP完成，独立目录models/dspark-qwen3-8b-block7。
- 服务器151 passed/2 skipped/10.95s；两卡的官方源码对照均通过，真实权重context4/17的骨干最大差0、Markov greedy token相同，Target共享embed/lm_head逐块相等。2026-10-26 14:25:24（UTC06:25:24）在新root`/media/a6000/D/ZhaoBinHe/MemoryDraft/dspark-tree`启动owned timeout PID1464/12000s，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_dspark_tree_pipeline`，每worker10800s；LOAD29.26/34.60s、两卡各4预热图。14:31四个GSM源任务八组完成，进入MATH，尚无严格AR差异；继续完整矩阵，不以中途结果调参。原1851配置/结果/核心Git对象SHA映射保留，174旧发布文件核验再次通过。

## 2026-10-25 — 树 + ASD 架构通俗说明

- 新增`docs/ASD_TREE_ARCHITECTURE.md`，从线性候选中断示例解释成本树与ASD的配合，覆盖DFlash单次分布复用、路径质量/成本选树、祖先attention、树上预算选路和KV/特征提交，并以已发布七任务结果解释接受长度增加与验证轮次减少带来的时延收益。明确BudgetTree-ASD与严格BudgetTree、Joint的对应关系，示例数字与实测数字分开。
- README顶部加入架构入口；核对依据为`memorydraft/backends/{budget_tree,draft_tree,dflash_reuse,asd}.py`、`scripts/run_asd_tree.py`及冻结结果报告。11个本地链接、4个实现文件的代码标识、3行接受长度/轮次/时延实测数据、UTF8/代码块检查和`git diff --check`通过；`.venv\Scripts\python.exe -m results.asd-paper-extension.helpers.verify_publication --git`再次核验174份结果文件、2份报告哈希及面试版引用通过。仅修改3份说明文档，不更改推理实现和原实验载荷，不重跑GPU；发布到`codex/asd-splice`。

## 2026-10-25 — 七任务独立扩样、预算控制和开放审查完成

- 两worker均complete257源任务/4770主计时/320控制/32自身历史第二轮；全514源任务、530匹配输入、9540主计时、640控制、530自然AR、64真实自身历史第二轮完成。172完整源任务复用原行字节，342源任务在隔离续跑会话完整执行；两会话32预热及原失败/数值诊断另列。GSM/MATH/HumanEval/MBPP/MMLU/MT/Alpaca全部保留，无按结果筛题或新seed。
- 正式完成观察05:51:48.895 UTC；2341源SHA前后一致，旧载荷不变。收尾观察器等两卡19MiB/0%后运行，两项实际前向分歧（323严格树/DFlash vs AR）均复现原轨迹、argmax成立，最大logits差1.52587890625e-5；未见其他身份差异。原正式ZIP17460709B/SHAcabf64683855b2c4741e78477044f72bb1688de8d84297bcd4b97083b101ec18保留；追加元数据/诊断后ZIP17500418B/SHA7040f141ad02b1264b9d21b810f8842cd2529acff692f00fbcfaa035622e5483，于05:54:04 UTC写receipt，5个≤4MiB块全部下载/校验。未上传模型/KV/logits/features。
- 严格树对DFlash530自然输出、三阶段1590对照token/text/停止/输入完全相同；自然时延GSM/MATH/Human/MBPP/MT/Alpaca分别省10.25%/6.19%/2.64%/14.87%/13.92%/17.15%，MMLU近乎持平。组合较ASD新GSM自然省7.72%/共同128省9.63%（N127），正确114→117/128；MBPP省10.21%、31→32/64，Alpaca省17.89%/共同128省18.36%（N41）。MT匹配单轮省5.51%、按对话合并省10.12%、自身历史双轮省12.93%；各自输入/区间单列。新MATH慢5.50%、44→45/64（主42→43）；Human省3.26%但53→52；MMLU39→37/114，四组逐题首选项相同78/114。质量取舍不删改，不将固定TPS+48.42%的HumanEval当自然提速幅度。
- 同一旧Skywork Reward V2 Qwen3-8B权重在3090/BF16/SDPA完成320行/223去重对话，8控制全过、重复差0、未评/裁剪0。监督源6d/2330文件前后一致，生成来源6d+1d39明确区分；05:56:41至05:58:27记录，子进程wall93.611s（源码前后核验另计）、权重校验13.957s、加载10.519s、任务前向38.171s，峰14.578GiB。组合MT7胜2平7负/50%，Alpaca26胜7平31负/46.09%；偏好不冒充准确率/官方评测。全部原始评分保留。
- 34条事前题干规则、136方法单元，100语义单元及10自动失败匿名审阅，26自动通过保留；110判定SHA d4ff8dd6cb717b08b21b829c8e40865013c0bb60f211a7028d1684dd4d8d069c，于06:08:08 UTC锁定，先Git8c48406ca7f6473ae87369de60bc23dda8922482冻结再揭示方法/查看RM逐题分数。ASD与组合34条逐项判定相同，MT13/16、Alpaca15/18；严格组12/16、17/18。秘书题长循环、漏Europe/She-her/None-LCS、漏删姓名和重复结论均逐条留证，性质通过不等于整答正确。实际消息/答案SHA及原自动判定136项独立绑定通过。
- 六个ASD/组合质量变化代码题以12个小反例/正确例核对：HE91把Is计作I、99负半数舍入错、103修复取整；MBPP190内部点误计、333修复return合同、74补长度检查。28个冻结评分变化case、8个MMLU合同变化、5个MATH得失及全部截断/输出SHA保留，不重写原测试成绩。7000首重复/控制发出位置轨迹、1117旧载荷、2341新/2330旧Git blobs、完整复用/资源独立审计通过；两卡峰42.799/36.392GiB、cache421054B、4图/正式新图0，相邻漂移>3%为2/1060，全保留。
- 统计JSON/MD在本机冻结1d源重建逐字节相同，SHA分别1ad0bd0e077a0c808317c1df0b04797850635da37696043b089519427b85285a/0fcb6241c0c9c2680dabc2db2f4960188071f15316d7555ad23fc294fc5d7407。发布工具首次因后缀白名单漏.jsonl停止，不完整副本保留artifacts/asd-extension-publication-attempt-1；补允许元数据类型后120原载荷SHA绑定发布成功，未重跑/改变实验。UTF8显示入口修正仅影响读取。结果results/asd-paper-extension，结论ASD_PAPER_EXTENSION_FINDINGS，完整表ASD_PAPER_EXTENSION_RESULTS；README/路线/论文对应同步。结果提交b06d5defcd27e226de405fa35b3dca73dac4f35d已通过`git push origin codex/asd-splice`推送；`.venv\Scripts\python.exe -m results.asd-paper-extension.helpers.verify_publication --git`核验120原载荷、52派生载荷、174份Git文件字节、2份报告哈希、匿名评审冻结及面试版引用全部通过。漏索引的STRICT_DIAG_LAUNCH.json来自artifacts忽略规则，检查后仅该文件强制加入；无结果文件遗漏，不启动新GPU实验。

## 2026-10-25 — 独立扩样续跑恢复

- 用户恢复额度后继续原任务，01:53:36 UTC服务器两卡19MiB/0%，新continuation root尚不存在，未重复启动旧失败会话。最后元数据补丁补齐身份比较右侧方法名及自身历史输入分歧记录；`python -m pytest -q tests/test_asd_extension_resume.py --basetemp artifacts/pytest-extension-resume-final --junitxml=artifacts/paper-extension-resume-final-local.xml`，17 passed/17.43s。原69专项与完整1055+7skip结果分别保留，不把新17项冒称全套回归。
- 续跑源1d39c56e5a81c896077258433b476af1a934ce6f/2341 canonical Git blobs，source SHAfa31110fd6487fbb98de3bbe9e2e3b5705142053d0f206d97721d83d91fdabd5；完整ZIP38136357B/SHA6cbbc5e015bfb898720dbb6c489ab3db80ea288e630923c7676b22793ec59f49，实际由原6d源+17文件delta276925B/SHAcbccda7c8170c653d95354802708cffda49ed864d97691940ac1b43f975dd5f2重建，部署后2341 SHA一致。本机canonical新副本与原2330源/真实失败载荷的完整组复用预检均通过。
- 新root`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-paper-extension-continuation`，owned timeout PID3143281于01:55:36.915545 UTC启动，outer22800/worker21600秒；入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_extension_continuation_pipeline`。原失败root/ZIP不改，两个新worker核验并复制83/89完整题原行字节；服务器181 passed/30.701s、0失败/错误后已开始剩余342题。暂未运行开放评分或更改其题干标准，不将已启动写作已完成。
- 03:49:18 UTC两worker各完成219源任务，合438/514；7956主计时、560控制、16自身历史MT第二轮。GSM/MATH/HumanEval/MBPP/MMLU已全部采集，MT继续，Alpaca尚未开始；不根据中途结果改参。03:03身份只读检查仍只有323的两个strict方法/同index602，未见其他差异，图数4、记录峰约36.1GiB。02:17硬件XML快照曾扁平混入最大/默认频率，原记录保留并于02:18:05显式current-clock查询纠正：89/84°C、SM1485/1665MHz、mem7600MHz、power272.37/298.99W（300W）；仅该时点，不冒称全程频率，未改硬件设置。
- 03:04:28 UTC另启只读等待/收尾owned timeout PID3191294（27000秒），`PYTHONPATH=. ../.venv-dflash/bin/python -u artifacts/finish_extension_server.py`，helper SHA61859e3455be5bce81aeeaf2b99ea3336f939389ac265168fd6b01460510c5da。等待主正式ZIP关闭且两GPU空闲后才进行逐个身份分歧的实际前向诊断，最终另写带auxiliary/diagnostic元数据的final ZIP，原正式ZIP不覆盖。当前仍waiting_for_formal_pipeline；额外观察器/打包器均在artifacts，不修改2341冻结源或计时代码。

## 2026-10-24 — 独立扩样中断与严格输出诊断

- 一题诊断已完成：AR重放与原参考一致，两近似组与原部分结果一致；严格树与DFlash均1125 tokens/index602同处分歧于AR1318 tokens。严格树/AR实际top2间隔7.62939453125e-6/1.71661376953125e-5，logits最大差1.52587890625e-5，严格树重复稳定且输出等于实际前向argmax；同前缀整段前向选AR词。支持数值形状差异解释，不将未来所有差异自动归此。源2330SHA再次核对。诊断ZIP82262B/SHA9ebabeb5e92e432217d6f7d6ed3b4043289d4ebad26a5752dc7c870c5e4a8930、失败辅助ZIP2982B/SHA2e26bfcf2f215edfd7decc67074820682bc9af1dac6cbad90e8dc94c00421093均下载/校验。
- 新增独立continuation入口与ASD_EXTENSION_CONTINUATION_PROTOCOL：原默认跨AR严格身份检查不变，仅续跑将跨形状差异逐项留证，不再声称全量bit-identical；接受规则/推理核心/输入/预算不改，同方法重复/strict前后/资源/质量仍硬检。新root不覆盖旧失败，完整83/89题原行字节复用，未完323整题重跑；本机对真实归档解压后核对3096主行/172参考/160控制均可完整复用，保留原2条323部分行。第一次直接读gzip归档目录被拒绝为缺失矩阵，解压后通过，未忽略缺项。
- 专项命令`python -m pytest -q tests/test_asd_extension_resume.py tests/test_asd_paper_extension.py tests/test_asd_paper.py tests/test_asd_paper_metrics.py tests/test_asd_tree_report.py --basetemp artifacts/pytest-extension-resume --junitxml=artifacts/paper-extension-resume-tests-local.xml`，69 passed/39.90s。随后额度上限使下一工具调用在执行前停止；最终小型元数据补丁及复验恢复后继续，未启动续跑GPU。

- 原冻结扩样在14:11:29 UTC因`math500:323`的严格BudgetTree与AR token不一致停止。原顺序BudgetTree-ASD/ASD/BudgetTree/DFlash，当时DFlash尚未执行；不能把此次分歧归给DFlash。失败行在断言前未落盘，原始错误输出没有保存，将以独立重放诊断，不假称恢复原失败行。
- 两worker分别完成83/89源任务（合172，GSM128和MATH44），1494/1602主计时、各80控制；另保留323的两条近似输出。worker0完成当前题后按peer-failed退出，原旧数据SHA均未变。原失败ZIP6470691B/SHA8bc43b76d55124012b364986d72d823dc8b064a8ae8ff1324c6d4ed72e39f691保留，14:15:46 UTC独立复核全部2330源SHA相同。其余领域尚未生成，本轮未完成，不将部分结果当完整七领域验证。
- 14:27:39.816130 UTC在物理GPU1启动一题正确性重放，owned timeout PID2727063/1200秒，使用原6d89480源/同配置、独立编译缓存。命令`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. TORCHINDUCTOR_CACHE_DIR=artifacts/strict-diag-inductor TRITON_CACHE_DIR=artifacts/strict-diag-triton ../.venv-dflash/bin/python -u artifacts/diagnose_extension_strict.py`；观察器SHA6b004de62526f9a04b406e888b5d82aa5162d4501fe5da31efd828723e8b4cc1。比较AR重放、四组、失败严格组重复及首次分歧同前缀的实际Target argmax/top2间隔。仅留小型派生数值，不发布全logits/KV；诊断计时不用于加速结论。尚未确定数值形状差异或实现问题，不放宽接受规则、不跳过失败题。

## 2026-10-24 — 七任务独立扩样与预算匹配消融准备

- 用户要求跨领域补齐未做实验，参考ASD v3第4节。冻结514新源任务/530匹配输入：GSM128/MATH64/HumanEval64/MBPP64/MMLU114/MT16双轮/Alpaca64，选中manifest SHA7b2934c9b76599529bbe9ca6dbfd1c2a9453edcfa784b3c2ce06ac48f37c7054。1117旧载荷SHA、476保守已用/保留ID排除，HumanEval尚可用84中选择64，57学科/8对话类别均覆盖；没有输出过滤或新种子。
- 四主组DFlash/ASD/BudgetTree/BudgetTree-ASD，继续原模型/提示/FP32/RMSNorm/native16/B8/g.25/M2和按卡冻结成本表。每请求自然2、prefix1、两个fixed256 strict-candidate-strict三连，共9540主计时+530自然AR+64自身历史MT第二轮。各族前4源任务预登记共32请求，5单参数控制×2方法×自然/固定，共640新增生成；B0/B8复用主轨迹。所有漂移保留，不筛题；非论文四重复/DSpark精确复现。
- 新增extension模块/准备/runner/pipeline/report与测试，现有tree execute只增加显式可选ASDConfig，默认行为保持；旧paper统计只增加可选arms/repeats。奖励runner增加study入口，既有Skywork权重本机审80单元×4=320行，不新增下载/训练/付费API，代理不冒充官方质量指标。协议ASD_PAPER_EXTENSION_PROTOCOL.md。
- 本机专项首次51 passed/1 error/33.91s：错误为pytest系统临时目录不可创建，发生在旧checkpoint fixture setup，非生成/评分失败；XML保留artifacts/paper-extension-tests-local.xml。改用工作区独立basetemp并补显式配置透传检查后复验；编译检查及跟踪文件diff检查已通过。远端两A6000均19MiB/0%，D盘401GiB可用。当前尚无本轮模型结果，冻结后部署独立asd-paper-extension root。
- 专项复验`python -m pytest -q tests/test_asd_paper_extension.py tests/test_asd_paper.py tests/test_asd_paper_metrics.py tests/test_asd_tree_report.py --basetemp artifacts/pytest-paper-extension-v2 --junitxml=artifacts/paper-extension-tests-v2-local.xml`，52 passed/28.42s。首临时目录错误及成功XML均保留。新增数据最大原始输入2771字符；后续自然历史仍按模型原8192输入上限，溢出留证不筛题。完整回归与固定源部署继续。
- 实际生成源6d89480c52e0d89714778668c25ef7027a9f0ac8，2330 canonical Git blobs/source SHAd39de7b926d6d49ca3750374ddfd52b012e12ee9e92bdfaa13623a41d3b81029。完整源ZIP38110237B/SHA5e4dbe6b575aa9c8ff456afa11bcc51002b3cfbd21a4a2f79b121d8fc75daff0上传超时，远端部分11567104B保留；改4a33ae5源+79文件差量1205416B/SHA56f6c366cef6aafa477c2ff9691bc57f126131a124824bbcb5f32cbc30826426，重建后2330 SHA全部相同。本机artifacts/asd-paper-extension-local另存同源副本；服务器独立asd-paper-extension root，owned outer PID2625729于2026-10-24T11:54:02.919422Z启动，outer22800/worker21600秒，启动两卡19MiB/0%。服务器专项164 passed/27.55s后两个worker已开始正式生成，参数/数据未修改，暂无汇总结论。
- 本机完整首轮在约85%处停滞，faulthandler明确位于旧HTTP/SSE测试的Windows asyncio socketpair/socket.accept；只核对并终止owned3108/13104，保留首日志及incomplete/noXML。相同源码在允许本机socket的环境重跑`python -m pytest -vv -o faulthandler_timeout=60 --basetemp artifacts/pytest-paper-extension-full-v2 --junitxml=artifacts/paper-extension-full-v2-local.xml`，1055 passed/7 skipped/162.49s，一条既有Starlette警告。没有为此更改运行/评分代码；首环境阻塞与新GPU实验分开。
- 运行期间仅查看新开放题的题干字段，12:36:25 UTC另冻结34项客观/明确性质（9自动/25匿名agent辅助内容检查，26源任务/32轮次/四组136单元），配置SHA7684f4063874ad1f86fbbe651365927277bc26d056dfde4f717a8b001e21993e。12:38:06 UTC服务器确认仅GSM38/39完成、当前GSM、无rollout文件，尚无本轮开放回答。6个来源上下文/歧义标签事前声明，全部保留主结果。附加协议ASD_EXTENSION_CONTRACT_PROTOCOL.md；不是GPU开始前的原始预登记、不改prompt/样本/预算/主评分，不冒称人工专家或官方裁判。初准备以文件入口运行导致模块未找到/缺配置，附加10通过1失败XML保留；模块入口重建后12 passed/4.10s。
- 题干规则及checker以7cf0daf事前提交并推送；13:10:27 UTC两卡均完成64个GSM源任务、各1152主计时/40控制，正运行MATH，未改配置。只读复核arXiv当前v3和ASD上游9426ce7的README/EVALUATION/DEEPSPEC文档；数据、答案、原始输出未随上游仓库提供，外部runner协议不足以确认所有原始切分/开放裁判。对应差异与精确blob记录于ASD_PAPER_ALIGNMENT.md，明确本项目Stanford Alpaca子集不称AlpacaEval；未据此改变数据或增加新GPU重复。
- 13:55:12 UTC只读硬件快照显示两卡软件温度限频Active、硬件大幅温控Not Active，89/84°C、SM1500/1260MHz、memory7600MHz、power272.76/299.11W（limit300W）。未更改风扇/功率/频率，保存服务器root/HARDWARE_SNAPSHOTS.jsonl。13:56:07对已完成fixed三连的严格前后作只读漂移检查：GPU0 158对中1对>3%（max1.046331/min.997881），GPU1 167对中0对（max1.004230/min.997474）；共325对全部保留。BASELINE_DRIFT_SNAPSHOTS.jsonl另存，最终须随归档发布；是中途快照，不冒称全程温度或最终漂移统计。当前仍运行MATH，无新实现错误。

## 2026-10-24 — 七任务剩余指标全部补测完成

- 原249匹配输入全部完成DFlash-before/AR/DFlash-after固定256且忽略EOS，共747正式生成+4预热；两个worker375/372，各125/124请求。三个新输出均逐token匹配旧fixed DFlash，输入/快照/精度保持；前后漂移范围.98286–1.00836，跨会话.97392–1.00784，>3%均0/249，无漂移筛题。没有重复旧五组自然/前缀实验或按新结果调参。
- 实际GPU源4a33ae58c8c35d5f0219ed3366a5150d70d02274/2259 canonical Git blobs，source SHAfbde4b6505e814dd8b861eddf66514ae597c5e4300a45b86ac1e26917b137165。完整源ZIP37165176B/SHA3d507f7f1bc0005eceafda149d5a8ee019c12d4763ccba5679c8950cf3ab53f8，实际delta64文件9157839B/SHAf44edeb53bea695557cf351b0382e4bacfd0ff55675b159dc1c8cfded068578c；新root asd-paper-metrics/PID2352078于04:20:49 UTC启动，执行验证文件末改05:05:41 UTC、完成审计05:06:24 UTC；前后与归档复核2259源SHA不变，两GPU最终19MiB/0%。
- 新AR GSM20.382TPS；当前严格DFlash/AR同窗直接3.845×。组合/AR跨会话4.696×、按新旧DFlash逐请求漂移校准估计4.709×；MATH对应5.168×/5.175×。旧五组时间不重写，主表明确直接实测、跨会话比与估计的范围，不将文献AR与本机时延相除。旧GSM完整-9.86%/等128-11.64%/63同正确集合、MATH完整-11.42%/正确26→28等结论保持。统一tau_selected/tau_emitted，GSM实际Draft接受长5.351→6.792、MATH6.566→7.665；独立枚举3735旧首重复轨迹每个输出位置的Draft/Target来源，全覆盖无重叠，bonus和节点成本分开。
- 另用独立训练Skywork Reward V2 Qwen3-8B在本机3090/BF16/SDPA离线评24开放单元×5组，共120行/87去重对话，全部可评、无裁剪。8控制全过、反序复算误差0；组合MT5胜0平3负/62.5%，Alpaca3胜2平11负/25%；Joint分别3/1/4及6/1/9，仍不升级。原192第一token位置裁判失败保留。9任务单元/45方法单元/65属性检查14未满足：ASD方向错、其他组代码次轮仍排序、所有组手机名未大写等，均保持原评分。RM会将严格排序方案排在线性ASD之上，大负分的bi-annual回答仍都每年两次；偏好不伪装准确率。7大差值/客观对照个案及输出SHA单列。
- 模型实际**14**文件/15152796883B，准备记录13为计数笔误，固定manifest与运行校验始终完整14。成功分块下载1435.318秒，首直连无进度的owned进程停止/日志保留、该首失败阶段未完整计秒。评分实际Python3.13.5/Torch2.6.0+cu124/Transformers4.52.2，校验13.365秒/加载11.284秒/任务前向18.801秒/全进程57.888秒、峰14.332GiB；04:42:31–04:43:29 UTC，2259本地冻结源前后SHA相同。不同于生成服务器4.57.3；两种用途与成本分开，权重仅本地artifacts。
- 生成每卡加载20.891/19.952秒、冷backend3.424/3.429秒包含预热，每卡4图/正式新图0、cache414726B，峰35.094/36.055GiB。本轮严格控制无需构树成本表；所有在线生成成本计wall，离线评价不混入速度。
- 本地最终相关28 passed/11.85s、服务器28 passed；完整首次在82%后停滞，核对owned18156/24788的命令和停滞CPU计数后终止，记录incomplete/noXML。带-vv/-o faulthandler_timeout=60及完整日志重跑1040 passed/7 skipped/167.98s，仅既有Starlette警告；未因此更改生成代码，首次原因未定。首fixture浅拷贝的3通过/1失败XML与后续27通过也保留。
- 原结果ZIP613909B/SHA59d2f5dc2abd0cfc43b25a4fce992ccf24890ee575adb65c26452888ea22d636保持服务器，追加deployment/completion/pipeline后614833B/SHA523e688cc02d13f50a7de2f1ae44cec2b609b09fdc778b7db277b64f845adfe5下载。32原始/本地检查载荷逐文件SHA，统计JSON/Markdown本地确定性重建逐字节相同，2259 Git blobs、747/249覆盖、精度/预算/资源/旧结果及infra三引用独立检查通过。报告ASD_PAPER_METRICS_FINDINGS/RESULTS，公开results/asd-paper-metrics；文献值单列ASD_LITERATURE_REFERENCE，不冒充本机实测。
- 精确命令见ASD_PAPER_METRICS_PROTOCOL；服务器`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_metrics_pipeline`。本地`python -m scripts.write_asd_paper_metrics_report --input results/asd-paper-metrics`及`python -m results.asd-paper-metrics.inspect_results`/`audit_open_contracts`/`audit_reward_differences`分别运行。README/路线表同步；接纳数学组合并保留所有正负结果，未启动新配对/训练/种子。

## 2026-10-24 — 七任务指标补全准备

- 用户授权继续未完成项，其他方法允许引用文献，不重复全五组实验。新增asd_paper_metrics、固定AR runner/pipeline、Skywork下载/pointwise评分、补充报告与测试。原241源任务/249主请求不变，计划747次DFlash-before/AR/DFlash-after固定256；原五组TPS保留，跨会话AR倍率和锚点校准估计明确区分，不当新候选配对实测。
- 独立训练奖励模型Skywork/Skywork-Reward-V2-Qwen3-8B固定6f19fdefb933293d4898bdb59a96f7223d998659，Apache-2.0，13文件15152796883B；模型卡/逐文件Git blob及LFS SHA在configs/asd-reward-model.json。本机RTX3090 BF16/SDPA离线评分，8独立好坏控制至少7通过才评24单元×5方法；无外部推理API，权重只在artifacts/models。本机下载中，服务器两个GPU空闲，401GiB可用。远端HF超时/mirror403，改本地已配置代理下载；一次SSH短暂超时后恢复。
- 首专项出现测试fixture浅拷贝使对照和变异行共用token list，修正为deepcopy；失败XML保留paper-metrics-tests-local.xml（3通过/1失败），未因此更改生成或验证逻辑。完整检查与固定源码部署继续，当前暂无新GPU性能或奖励评分结论。冻结协议ASD_PAPER_METRICS_PROTOCOL.md。
- 修正后相关27 passed/11.81s（paper-metrics-tests-v2-local.xml）；增加全七族统计集成检查后再跑最终专项。直连大权重请求无进度，核对owned进程4688/17008后终止；HTTP Range64KiB检查2.81秒成功，改为16MiB分块/每文件4线程，保留首下载日志与零字节part。原模型版本/哈希不变。

## 2026-10-24 — ASD七任务五方法扩样本完成

- 241新源任务/249匹配输入，五组4980正式计时+249 AR、40自身历史MT第二轮、18预热/零检查全部完成；192本地裁判记录另列。两个worker2500/2480，数学里程碑各960、parameter_changes=0；全部严格AR/token/text/停止、输入/快照、重复轨迹、实际预算≤8检查通过。源码60645db/2199 canonical Git blobs前后及最终复核SHA一致，未重复启动GPU实验。
- A6000隔离root asd-paper于2026-10-23 15:44:40 UTC启动，pipeline末日志18:27:09、最终归档审计18:54:32 UTC；不将审计时间当精确结束时刻。原ZIP8773881B/SHAdc34db532d6e752b4fc8e69b8a76d975059f7038882dccca1bfabc4d5f2155d2保留；追加deployment/completion/pipeline日志后的ZIP8774872B/SHAcc6a55bcbe59a333b985854780b35055f396464c38daf0ddb9f7d711d78a3bd5下载本机。31原始归档成员逐字节核验，补充统计不修改冻结主评分或载荷。
- GSM64题组合较ASD完整省9.86%（请求比95%CI .8714–.9308）、共同128省11.64%（N63），各组正确集合保持63/64；逐题输出长度比+1.91%、调用密度-18.02%。共同8064token验证1343→1069轮，固定256 TPS85.250→95.721/+12.28%，真实EOS后的强制计算另列。MATH32题组合完整省11.42%、等128省4.70%，正确26→28、ASD3截断→组合0；新增答对180/421/425、失去485，完整收益含避免循环。共同正确且EOS25题仅省2.17%，后验诊断保留。
- Joint较第一组合GSM自然慢3.93%、共同128慢.16%；MATH慢18.98%、正确28→23，另2截断；七任务直接自然配对未显示进一步收益。接纳第一组合的数学收益，Joint保留消融，不为复现旧8题的小增量在新题调参。HumanEval组合快6.34%但28→27，真实大小写处理错误保留。MBPP一例163由ASD28token/.844秒变组合4096/23.722秒，主时延慢71.85%；不删失败题。214只差浮点运算关联顺序/一个ULP，506是组合数代替排列数，不能把两者都称语义错误。
- MMLU主“只答字母”正确25→22，十处跨组质量变化均是正确字母后多写解释；五组逐题首选项均相同，补充提取均38/57，主评分不改。开放MT/Alpaca有速度信号，但176实际judge前向170选A/6选C、88对不同文本换序一致0；该裁判无效，约50%偏好不作为质量保持证据。4超长未评、12相同文本自动平局全部保留。真实自身历史MT速度另表，不混入共享AR首轮的匹配输入比较。
- 本机最终完整1028 passed/7 skipped/145.75s、矩阵扩充专项16 passed/9.19s；服务器171 passed/10.63s，相关及前次完整XML保留。Windows复算analysis.json/report.md与服务器逐字节相同（report-reconstruction.json）。独立审计1049旧数据/结果SHA、241无旧ID重叠、2199 Git blobs、33101联合构树块/26947缩短、完整矩阵和资源通过；infra三引用保持27e3619。9个MATH复核无新增等价答案，5个代码题25组原测试及20反例重新检查一致，完整24个变化/待复核案例保存quality-audit.json。
- 模型加载20.119/20.152秒、冷backend3.464/3.364秒包含相应预热wall；18预热/零检查29.467秒。每卡4图/正式新图0、cache411424B，生成峰35.94/36.15GiB；裁判加载19.836秒、前向248.732秒、峰38.86GiB，最终两卡19MiB/0%。成本表独立读取耗时漏记明确列缺项，在线构树/选择/验证仍计入wall。没有新训练/seed/阈值扫描、权重/KV/logits/features或私有历史发布。
- 结论docs/ASD_PAPER_FINDINGS.md、主表ASD_PAPER_RESULTS.md；公开results/asd-paper，协议ASD_PAPER_PROTOCOL.md。本地执行`python artifacts/publish_paper_results.py`后，`python -m scripts.write_asd_paper_report --input results/asd-paper`及`python -m results.asd-paper.{inspect_results,analyze_mechanism,review_quality,audit_quality,describe_results}`（分别运行模块，非字面花括号）。README/路线表同步，新题现均已观察；保留第一组合作为数学研究主线，未启动后续GPU。

## 2026-10-23 — 七任务扩样本执行启动

- 固定源码60645db3b6322e41362b9b875d5c88818c2c6c0f/2199 canonical Git blobs，source SHA4b29bbcead5ac60a8206f86b189a0f7fc4dfa7973b91b5ecb2ddd1066b883a66；完整本机ZIP28248779B/SHA24258dd3273e8b589c37a19020326a5620d6ebc40cd67a1df3d6dd8af7f1d2f5。实际以旧asd-joint-tree原源目录+77文件delta1840356B/SHA2cd1ffaff262ef5ca7dc9fbf56571c10fbd001e3f7bee434fe63c8a6b12ab55b，逐文件重建全部2199 SHA。
- 新root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-paper`、owned timeout PID1891785，于15:44:40 UTC启动，outer19000/worker14400/本地judge3600秒。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_paper_pipeline`。严格5组、七任务fixed256、自然EOS/128与自身历史MT分开，参数/样本/评委规则不再修改。
- 上游Alpaca/FastChat README原始字节包含尾空格，冻结时原样保留，不修改来源文本；后续.gitattributes为这两类原始README豁免whitespace检查。此为来源文本格式，不是算法/数据选择变化。当前尚无正式比较结果。
- 服务器171 passed/10.63s，随后正式生成启动；本地源外的复算/发布审计脚本准备中。计时元数据复核发现冻结worker未独立记录成本表读取耗时，最终需明确单列为缺项，不估造数字、不为此重跑主性能；表本身仍按既有SHA冻结，全部在线构树/选择/验证成本已计入wall。

## 2026-10-23 — ASD论文七任务五组扩样本（准备）

- 用户授权直接扩大数学并补齐论文任务。新增asd_paper数据/评分/统计、prepare/run/pipeline/judge/report及测试；复用原树/ASD/引擎，执行函数仅新增可选request_factory/fixed，旧默认保持。
- 固定241新源任务/249主请求：GSM64/MATH32/Human32/MBPP32/MMLU57/MT8双轮/Alpaca16。排除1049旧数据/结果文件中235题号及早期范围；数据manifest SHA7fcbccb1b830bcb7cfa7e8f7fe8705e73e0575466447ea2d8d952140a05d5ec6。新公共来源固定Git/HF commit及SHA/许可，Alpaca数据CC BY-NC 4.0仅研究；不进入面试版。
- 五组全保留，自然2×、128/EOS1×、全七任务fixed256/EOS忽略1×，4980计时+249 AR，另40自身历史MT第二轮；各卡先数学后扩展，成本表、权重、B8/g.25/m2冻结，无按结果筛法/改参。MT主比较共享AR首轮历史；真实双轮质量单列。开放本地192盲评换序只辅助，不冒充官方GPT-4/MTBench/AlpacaEval。
- 首相关79 passed/14.44秒，artifacts/paper-related-local.xml；217可评分参考本机预检全部通过，artifacts/paper-canonical-preflight-local.json。两A6000均19MiB/0%，D盘402GiB可用；新GPU生成尚未启动，完整回归与冻结源部署将继续。协议ASD_PAPER_PROTOCOL.md含准确命令与资源预算。
- 模型前接口复核：MBPP题干不总含函数名，32题新增只含标准函数签名的明确契约，参考函数体/tests仍隐藏，ID选择保持。最终数据manifest SHA56ccef799adfffe2caf0c6f5bd807484bfbc831563a2b90f80e041536078eca4；前一个7fcb快照未生成任何新模型结果。新增接口提取测试，最终相关检查继续。
- 接口后相关120 passed/2 skipped/16.70s；完整1028 passed/7 skipped/145.76s（前一完整1027 passed/7 skipped/142.29s也保留）。固定256覆盖全部七任务的矩阵调整后专项16 passed/9.19s，XML均在artifacts/paper-*local.xml。1049旧文件字节保持/241选择无旧ID重合/compileall/diff-check通过，准备提交canonical源；暂无新GPU结果。

## 2026-10-23 — 第二阶段剩余regret联合构树完成

- 六类各8已观察开发题，六组1440正式计时+48同题AR全部完成；每worker720，pilot24/720后续跑余24。原四组960次token/text/轮次/账本/停止/输入/快照逐项复现第一阶段；冻结各物理卡成本表，无新校准或调参，全部strict AR/input/repeat/实际B<=8检查通过。
- Joint较第一阶段组合：GSM完整省1.17%、共同128省2.48%（N7），答案均8/8；同896token验证126→固定主干123→Joint122轮，动态相对固定前缀仅另省0.62%。相对线性ASD，Joint完整省9.10%。接纳GSM小增量；MATH前缀省5.36%但完整慢4.97%、正确6/8保持；MBPP+0.36%、HumanEval+5.04%、SQuAD+0.02%、JSON-.06%，其余不升级。
- 固定完整主干MATH359错误长枚举至4096截断，正确6→5/8；Joint同题1291token最终6、正常EOS，避免该消融退化但仍比第一阶段711token更长。MATH448新Joint给5（17*5%83=2）、旧/固定给14（余72），仍错；Human134沿用旧错代码。质量个案JSON/输出SHA单列，不改冻结评分。
- 两新组8963轮（含重复）快照/主干/实际收费核对，Joint缩短2627轮；自然20/48请求触发，Joint/固定37/48完整输出相同。预算启发式不是语义正确概率；前缀减轮与完整长度效应分开，不把固定消融的大幅退化规避全算吞吐提升。
- 实际源31c0a74a1bbf5addb327597515908b879b108f83/2129 canonical blobs，source SHAfe066ff5fa5a49fb5c904d88343b9062ff840a93e1ce9eb312eadb54799e09dc；新root asd-joint-tree、owned timeout PID1851082、14:40:38→15:06:54 UTC启动/末审计。运行前后/末检查2129 SHA一致，两旧root未改。完整ZIP26651076B/SHA2ff2c65ac2b09af5ab720937a55c8963f7babc7beb201394b529ca0c14ec9b9b，实际314文件delta4143372B/SHAc98e685089f9c60ecd7d49e23f903321ecb10e63ad307647217a60a13b883364。
- 原pipeline ZIP1380224B/SHAa69a83204bfa004383da8a770d73677a346b721ddb64c583a08a98341b341bbd；追加部署/末审计/顶层log的final ZIP1381195B/SHAde372a1436001c9eff799b3faaed42f6a700ddd875ccfd764ddb54a836e569b4，28原始载荷逐字节发布。加载19.684/20.103s、只读表.000395/.002538s、20预热生成另计；峰allocated35.95/35.16GiB，每卡4图/正式0新图、cache413494B，末两GPU19MiB/0%。
- 本机完整1012 passed/7 skipped/145.86s，相关135 passed/2 skipped/16.28s，服务器166 passed/13.86s。统计JSON/MD跨平台独立重建逐字节一致；raw载荷/canonical Git源/960旧轨迹/预算/资源/infra三引用独立核查通过。初相关JSON键测试失败及后续通过XML全保留，未混入正式性能。
- 结论ASD_JOINT_TREE_FINDINGS.md、完整表ASD_JOINT_TREE_RESULTS.md、证据results/asd-joint-tree。复算`python -m scripts.write_asd_joint_tree_report`、`python -m results.asd-joint-tree.inspect_results`、`python -m results.asd-joint-tree.analyze_mechanism`、`python -m results.asd-joint-tree.review_quality`。保留Joint为GSM候选，其他任务保持第一阶段方案，默认服务和面试引用不变；未启动下一轮扫参。

## 2026-10-23 — 第二阶段联合构树执行启动

- 冻结源31c0a74a1bbf5addb327597515908b879b108f83，2129 canonical Git blobs，source SHAfe066ff5fa5a49fb5c904d88343b9062ff840a93e1ce9eb312eadb54799e09dc。完整ZIP26651076B/SHA2ff2c65ac2b09af5ab720937a55c8963f7babc7beb201394b529ca0c14ec9b9b；实际existing engineering ZIP+314文件delta4143372B/SHAc98e685089f9c60ecd7d49e23f903321ecb10e63ad307647217a60a13b883364重建，全部逐SHA一致。
- 完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/joint-full-tmp --junitxml=artifacts/joint-full-local.xml -x --tb=short`：1012 passed/7 skipped/145.86秒，一个既有starlette警告。服务器166 passed/13.86秒；模型加载后已开始正式六组生成，尚未比较性能。
- 新root `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-joint-tree`，timeout PID1851082，于14:40:38 UTC启动，outer7600/worker7200秒。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_joint_tree_pipeline`，manifest ASD_JOINT_TREE_SOURCE_MANIFEST.json。使用原两A6000和本地已有模型，无新下载/训练，之前root和证据不修改。
- pilot24/720计时+24 AR已通过coverage/strict AR/input/repeats/budget屏障，两worker自动继续余24；原四组实际执行行同时与第一阶段逐轮复核。这里只检查进度和预算分配触发，不根据中间速度/质量改参或删题。

## 2026-10-23 — 剩余regret联合构树（第二阶段准备）

- 用户要求直接进行第二步。新增opt-in固定主干/剩余预算主干，预算只读快照、实际选路扣费不变；节点总量沿用第一阶段冻结成本表，EOS叶子/32上限/原生16保持。规则和资源尺度详见ASD_JOINT_TREE_PROTOCOL.md，不以Draft置信度冒充错误概率。
- 六类各8已观察开发题，原四组加SpineTree-ASD/JointTree-ASD，共1440计时+48 AR，原四组960次轨迹与第一阶段逐项核对。自然EOS与共同完整128、质量和长度分别报告；固定主干为动态分配必要消融，不按新结果扫参。
- 首核心检查115 passed/2 skipped/11.95秒，artifacts/joint-first-local.xml。正在补相关/完整检查，尚无第二阶段GPU结果。服务器新隔离root asd-joint-tree，冻结canonical Git blobs再部署；旧结果与infra引用不动。
- 相关检查135 passed/2 skipped/16.28秒，artifacts/joint-related-v2-local.xml；旧analysis经JSON键规范化后全部逐值保持。首相关检查32 passed后仅旧报告比较测试因内存整数键/JSON字符串键差别失败，修正测试序列化比较，未改统计结果；失败XML保留。完整回归运行中，`git diff --check`与新增入口compileall通过。

## 2026-10-23 — 成本受控树 + ASD跨六类评估完成

- 第一阶段组合完成：六类各8道已观察development，四组自然3次/128-EOS2次，共960计时+48同题AR；两worker各480，24题有效性屏障后完成其余24，模型/预算/题集无调参。全部strict逐token/text/EOS同AR，输入、重复轨迹/账本、实际B<=8通过。
- GSM组合较线性ASD自然省7.89%、等128省10.95%（N7），两组答案8/8；同896token验证ASD157轮、严格树142、组合126，相对严格树等128再省9.87%。接纳该数学场景算法收益。自然输出较ASD更长，组合相对严格树完整仅省0.40%，不把前缀增益直接当完整请求增益。
- MBPP较ASD自然省5.96%、同通过集合6/8，相对严格树省1.85%；MATH自然省10.02%但正确7→6/8、等128反慢5.69%，新增448的逆元14明确错误。HumanEval较ASD自然省15.97%但通过7→6/8，严格树本身较DFlash省9.67%且组合较树慢1.31%；SQuAD/JSON零近似触发。所有正负结果保留，未实现第二步预算驱动构树。
- 最终源05ec51424cc67ad3a38d86beed98ad3c3b47347e，2058 canonical Git blobs，source SHA6b89ac60d295dc44efc469717c61042c0bf0659932798bf7937edab0edf537d1；完整本地ZIP25451573B/SHA8ba3eec035d6e75471814356781e17b0847af4be91ffdcf74ee9c58b1194a52b。实际existing engineering ZIP+243文件delta2943869B/SHA48ce8fcdd2af8888f4add14f16371251aba7b99958d5c73a1661e408b753b826重建，无EOL偏离，运行前后2058 SHA保持。
- root asd-budget-tree-v2、timeout PID1821773、13:54:42→14:15:06 UTC启动/末核查窗口。加载20.45/19.82s、校准4.92s/卡，峰allocated最大35.21GiB、每卡4图/正式新图0、cache415135B/卡，末两GPU19MiB/0%。除1008主生成，另18预热/校准生成及合成成本Target前向单列。
- 原pipeline ZIP850991B/SHAaae86dc98ccad27c76b7acfcc7693f46e6e3f214621fa9f4ee3d84f126fdcfe1；追加部署/末核查metadata的final ZIP851755B/SHA63db7b3fa1a70f5b7cf52d526a3da3e7eb37e6e79d9276bdf7bace75481f32b0，29原始成员逐SHA发布。初EOS遗漏停止的20成员264365B/SHAcadde1be3680d09cd340a9a0127eed34b533615d651518701849d4ea1a30c349单列startup-eos，不混入正式结果。
- 最终本地完整979 passed/7 skipped/136.46s，相关131 passed/2 skipped/13.29s，服务器133 passed/9.33s；六XML保留。统计JSON/MD跨Windows3.13/Linux3.11逐字节重现；canonical Git/source、raw publication、实际ledger/coverage/AR/input/budget、资源及原infra三引用核查通过。数学444/448的明确反例单列quality-review，不改冻结评分。
- 结论docs/ASD_TREE_FINDINGS.md、主表ASD_TREE_RESULTS.md、原始/复算results/asd-budget-tree。命令`python -m scripts.write_asd_tree_report`、`python -m results.asd-budget-tree.inspect_results`、`python -m results.asd-budget-tree.analyze_interaction`。下一步优先GSM的计算与近似预算联合分配，保留本轮四组对照和现有工程成果。

## 2026-10-23 — 树ASD执行启动及EOS构树修正

- 初执行源ae2c6cb0eb6e388ad8d187c01d724bb24bd25437，2058个canonical Git blobs，source SHA f074e787889c818759114e9fe5a5197209f8cf85d38c7c006d4a21fba8b56522。源ZIP25450676B/SHAecb7a5d92eb29de8af209ed2fac3fbda1bcf90a2cdfb57b69ff7b8a3fee3858f；全包传输超时，只收到部分，不使用该残包。
- 实际以服务器已有engineering源ZIP+243文件delta2942972B/SHAea49cf910b57bd95df6fa3c8c9b0427ec04a71b4dbd24906d3724a03a925fe46，在asd-budget-tree逐文件重建全部2058 canonical SHA，PID1817346。初部署系统python3缺is_relative_to，在未写任何源文件时失败；改用项目Python3.11完成，不改源内容。服务器132 passed/9.32秒，本机完整978 passed/7 skipped/148.80秒。
- 运行期间仅查看进度、节点分布与成本表，未比较方法速度/质量。代码复核发现构树surrogate错误计入EOS后代；停止本root拥有的两个worker进程组，保留部分运行和失败归档，不作为最终候选性能证据。修正EOS为叶子，新增针对性测试，其他模型/参数/题集/顺序保持；新版本将在独立v2目录执行。

## 2026-10-23 — 成本受控候选树 + ASD执行准备

- 用户授权实现第一步并跨数据评估。新增best-first路径质量构树、实测节点成本表，以及树上ASD最长可行路径/同长度低regret选择；原生16深度、Top4、8..32节点，原始B8/g.25/m2。只提交选中路径的KV、特征和regret；第二步剩余预算驱动构树尚不实现。
- 来源BASTION论文式1/3/6及MIT仓库可见commit1577b5d9ee28783e15358f1dd791844e9f966172；独立实现，实测表代替roofline，不称完整官方复现。协议docs/ASD_TREE_PROTOCOL.md。
- 冻结六类各8道已观察development，共四方法×(自然3+128/EOS2)=960计时+48 AR；先完成六类各4的24题有效性屏障，再继续其余24，无速度/质量选题。严格输出、重复轨迹、输入和预算检查贯穿全部运行。
- 本机首机制91 passed/2 skipped/15.70秒；补齐报告后相关129 passed/2 skipped/13.22秒。命令`.venv\Scripts\python.exe -m pytest -q tests/test_budget_tree.py tests/test_draft_tree.py tests/test_dflash_reuse.py tests/test_asd_cpu.py tests/test_asd_tree_report.py --basetemp artifacts/tree-tests-2 --junitxml=artifacts/tree-tests-2.xml`。尚无本轮GPU性能结论。
- A6000预检两卡各19MiB/0%、D盘余402GiB。后续在新隔离目录使用原模型/.venv-dflash，不改旧实验和面试引用；source导出直接读Git blobs，避免上一阶段git archive的CRLF偏离。

## 2026-10-23 — ASD候选统一比较与匹配缓存组合完成

- 用户授权一次完成多步。原reserve96按六类各4/12分成screen24/confirm72；14组1008计时+24 AR之后按冻结规则选各族两个候选，再完成1440计时+72 AR。全部4个worker正常完成，严格各臂与同题AR逐token/text/EOS相同，重复轨迹/账本稳定，实际ASD预算<=8，正式新图0；source前后1964部署SHA一致。没有新训练、seed、下载或参数扫描，默认strict和独立面试版保持。
- GPU实际源34f6a2eb72d35142700fa0bbf13c867f416de27a，source SHAe648c352b141ede94846a4e9a7a8bab10727544db5035f2870f9f52b91fdc374；成功root asd-comparison-v2/PID1702283于服务器本地18:50:35启动。结果ZIP2009611B/SHAf306554454a57df0952b764ea25b0eeca20bc6d6b67fc83c7a72921e98442054，33成员原始SHA核对、JSONL无损gzip公开。首缺models链接的加载前失败8成员单列startup-failure，原失败不重写。
- 新12 GSM：位置适配等128省1.42%（调用256→253、正确集合11/12保持），自然反慢0.79%，输出长度比均值+4.05%/调用密度-2.46%；保留数学等生成量小收益，不冒称完整请求增益。新12 MATH：基础ASD相对严格D自然省17.29%/128省10.95%、正确11对10/12；Utility自然慢32.04%、正确10/12，位置+Utility慢22.01%、正确9/12。局部等128快1.61%/3.11%被后续路径增长抵消。MBPP位置适配初筛快11.83%在确认仅0.05%；HumanEval tree32较严格D仅快0.04%、严格输出保持，新增节点抵消减轮。Confidence仅screen、未入选确认；SQuAD/JSON零近似激活。不改写旧局部正结果，降低固定局部预算规则优先级，优先保留Draft适配小收益及infra执行/缓存路线。
- 按预登记，GSM确认选出的position+ASD与D/ASD匹配测8个已观察公开4K请求×off/cold/hit×2，共144自然+8 AR。三组各自缓存配对token/text/EOS相同，正确D6/8、ASD两组7/8保持，全部EOS、正式新图0。hit/off时延分别低54.75%/58.04%/57.97%，TTFT约4.86→.312秒；position在相同hit条件比base ASD慢0.04%，没有额外叠加收益。最大spent差6.29425e-5、实际<=8，未改接受器容差。
- 工程源34d049946b0975661ea26ccc0d4861001e414f8a/1967部署文件，source SHA5d5312ddf9f797873fda78dffc6bb9e39a9fa38083f310ff19be32c87a79b159；source ZIP23338151B/SHA5491556b08e0687d951e0471c1442d192d1b8aeaa1899c263f5ae1e95c413f2f。root asd-comparison-engineering，owned timeout PIDs1743989/1743990，11:50:41–12:01:44 UTC执行/审计窗口；source前后1967 SHA一致、两个旧adapter SHA保持。结果ZIP183489B/SHA3eb99594c4d748ef49fbef2baff240683feeaa239df15d0e7fd87321bcd4a0a7，18原始成员公开。allocated峰43.67GiB、compiler cache418595/418627B，两GPU最终均19MiB/0%。
- 部署复核更正：实际使用Windows git archive，有148文件转成CRLF；主源1816/1964、工程1819/1967与Git对象逐字节相同，其余全部仅换行差异。前述准备记录“canonical Git blobs”不准确，保留原记录并在结论中明确偏离；运行前后核的是实际部署字节。两份source-line-endings.json和audit_postrun记录完整差异，业务源内容/参数/数据未变。主模型加载未单独计秒，不做冷启动ROI宣传。
- 本机最终完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/comparison-final-full-tmp --junitxml=artifacts/comparison-final-full-local.xml -x --tb=short`：948 passed/7 skipped/138.14秒，仅既有starlette警告；主server100 passed/20.11秒，工程server10 passed/3.14秒。主JSON/Markdown两平台逐字节相同，工程统计本机逐字段重现；原始载荷、两Git源内容/换行、旧结果及infra三引用另审计。MATH194额外错误的根式不等价已给明确反例，评分和已冻结选择不改。
- 结论docs/ASD_COMPARISON_FINDINGS.md，主表ASD_COMPARISON_RESULTS.md、缓存表ASD_COMPARISON_ENGINEERING_RESULTS.md；公开results/asd-comparison{,-engineering}。复算`python -m scripts.write_asd_comparison_report`、`python -m results.asd-comparison.inspect_results`、`python -m results.asd-comparison.audit_postrun`、`python -m results.asd-comparison-engineering.inspect_results`。原reserve96现已观察，README/路线表已更新，今后独立验证需新预登记数据。

## 2026-10-23 — 统一比较执行中及工程组合预登记

- 已冻结34f6a2eb72d35142700fa0bbf13c867f416de27a/1964 canonical Git blobs，source SHAe648c352b141ede94846a4e9a7a8bab10727544db5035f2870f9f52b91fdc374；source ZIP23331209B/SHA9193fe7690a709aa1fd5e977c14a2135c2a196fabd3c02ee6c78a287e14bc9e9。本地初相关99 passed/23.15秒；补first-token EOS/真实tiny树后专项14 passed/22.58秒。已有结果1040文件/20198 token载荷对reserve96命中0。
- 首root asd-comparison服务器100 passed/19.38秒，但缺少部署models软链接，两个worker均在模型加载前失败，无正式生成；原失败ZIP下载artifacts/asd-comparison-startup-failure.zip。新root asd-comparison-v2从同一ZIP重新解包，补链接到原models，1964源SHA再核对，PID1702283；服务器100 passed/20.11秒，两卡开始screen。未改代码/题集/选择规则，旧适配器只读。
- 首本地完整回归无新增失败输出但在约76%后挂起；核对本轮PID后终止，未生成完整XML，不当作通过。开启-vv/-o faulthandler_timeout=60并在允许环境重跑：938 passed/7 skipped/134.56秒，artifacts/comparison-full-v2-local.xml及comparison-full-diagnostic.log；仅既有starlette警告。未查明首次挂起原因，不将其解释为产品缺陷或更改实验规则。
- 追加独立工程协议ASD_COMPARISON_ENGINEERING_PROTOCOL.md和runner/test；确认结束后，GSM两个候选按同规则选一，与ASD/DFlash共测8个已观察公开4K请求×off/cold/hit×2，共144计时+8 AR。只验证相同缓存条件下新组合，不重算通用融合收益、不消耗确认数据调参，不覆盖主执行源。初工程专项9 passed/7.47秒，补非有限计时检查后将复测。

## 2026-10-23 — ASD候选统一比较（执行准备）

- 用户授权一次完成初筛和确认，按特定场景实际收益判断，不要求所有任务胜出或新训练seed。新增evaluation/asd_comparison、run_asd_comparison及pipeline/report/test，复用原engine、CPU接受器、Confidence、严格tree32及两个旧LoRA；默认runtime、Target输入及面试版不改。
- 冻结reserve96：六类各前4为screen24、其余12为confirm72。原/ASD-prefix/位置适配三Draft×DFlash/ASD/Run/Utility共12，再加原Confidence/tree32，共14组。初筛1008计时+24 AR，按各类质量损失/新增截断/自然时延顺序，从原策略/搜索和适配两轴各选一候选；选择落盘后确认1440计时+72 AR。协议docs/ASD_COMPARISON_PROTOCOL.md，结果results/raw/asd-comparison。
- 首相关`.venv\Scripts\python.exe -m pytest -q tests/test_asd_comparison.py --basetemp artifacts/comparison-tests-tmp --junitxml=artifacts/comparison-tests-local.xml -x --tb=short`：13 passed/23.92秒；随后补计在线接受器构造及hook成本，添加真实tiny树入口覆盖，将跑相关/完整检查后冻结源码。
- 两A6000均19MiB/0%，磁盘402GiB可用；只读复用旧selected adapters，SHA固定，不下载/训练。每worker14400秒、46GiB/8图/backend300秒/cache4GiB，new root asd-comparison；当前尚无新GPU结果。

## 2026-10-23 — 阶段6共享公共段与新HTTP控制完成

- GPU实际源ecc44571a6096232be16a1035ab71804f7878551/1902 blobs，source SHA2bcebb23b8538579e75c43b8c22695be172493ae65467ddce7ebcb6320d67863，完整本机ZIP SHA9b2e491825ecdad0fa93209bdbb4ccb583e300165c5a70963b2cd064d14ff5ee；实际4454359B delta SHAc14b9e82a94689f5af1b0cc5c778da6078b7a7a0faf5e2ac303a6e06f8e087db。新root asd-shared-prefix/PID1522788于05:57:32 UTC启动，06:18:48检查完成、两卡19MiB/0%。1902源前后SHA一致，成功ZIP211506B/SHA9c8ed2c250a56a1cc1e99b7b931f48b68a9b842d5df79af50cdae62790ec9040，40成员原始字节核验。
- 新registry保留三个短entry，把4073/3937两个公开4K前缀换为3934公共段，不改请求、Target/Draft或在线PublicPrefixCache/接受器。4K原8题shared下96正式（D/ASD×自然/128且EOS×off/cold/hit），全部自然EOS、无OOM/截断/正式新图，strict逐token匹配旧完整AR，当前输出均重现旧参考。ASD原7/8、strict原6/8保持；验证总轮数/汇总接受量相同，不冒称逐轮trace或bitwise一致，spent最大差6.29425048828125e-5且实际预算≤8。
- shared ASD自然命中题均比off .4230/低57.70%，TTFT4.8023→.3090秒，首token后3.1039→3.1019；cold/off1.0017，cold+1hit/2off成本估算.7167。ASD共同完整128为5题、比.4144，D为6题/.4290，其余EOS短题保留。未新增训练/seed、reserve96未生成。
- 共120新HTTP负载（shared各卡20off+20cache、whole新进程各20cache），另16个JSON/SSE/断连/恢复请求。GPU0 shared→whole/GPU1 whole→shared，同题同卡、并发2/四题轮换/空缓存起步。交替流whole157.4184→shared72.5130秒/低53.94%，hit0/20→19/20，淘汰19→0；平均queue7.4929→3.4734秒，固定2335token。单一前缀流whole54.5367→shared56.4706秒/慢3.55%，均19/20hit，固定2405token；保留原whole配置，公共段较短需多算139token。全部输出/输入/快照/汇总量一致，协议检查通过，无失败/超时，不将串行worker称连续批处理。
- shared只填充1482457088B/1.3806GiB公共段，交替whole累计填充30184243200B；这是累计GPU入库量而非显存峰。原两个完整表示共2.8111GiB超2GiB，统一公共段解决覆盖容量问题；在线仍是整条登记entry，通用块/分支存储未实现。四进程模型准备20.259–20.858秒，各3图/backend3.256–3.428秒，首4K DFlash16冷wall8.491–9.161秒保留编译；正式峰43.6739GiB、HTTP峰43.6373GiB≤46，编译/测试cache共2725400B。
- 本机unit34 passed/27.49秒、相关60 passed/7.86秒、完整924 passed/7 skipped/137.79秒，报告补充校验后专项7 passed/0.32秒；服务器116 passed/8.97秒。独立主统计JSON/Markdown两平台逐字节相同，审计1902 Git源、87冻结执行/数据文件、1323旧结果、上轮69文件发布清单及infra三引用通过，两个历史CRLF例外保留。在线缓存/引擎/ASD/旧registry/config与de1d4e4逐字节相同，无KV/features/权重/raw trace发布。
- 报告docs/ASD_SHARED_PREFIX_FINDINGS.md、主表ASD_SHARED_PREFIX_RESULTS.md、数据results/asd-shared-prefix，入口configs/runtime-dflash-asd-shared-prefix.json用于常驻engine，默认与独立面试版不变。复核`python -m scripts.write_asd_shared_prefix_report --input results/asd-shared-prefix --output artifacts/shared-prefix-report-recheck`、`python -m results.asd-shared-prefix.inspect_results`、`python -m results.asd-shared-prefix.audit_postrun`；完整audit需artifacts/asd-shared-prefix-results.zip。后续可评估共享主干+短分支，减少单前缀额外后缀计算，本轮不冒称已经实现。

## 2026-10-23 — 阶段6共享公共段修复（执行准备）

- 用户授权下一步。新prepare_shared_prefix只对已固定公开registry求token最长公共段，将4K的4073/3937两个entry换为3934，保留三个短entry，父SHA/ID/长度入新registry；不读测试答案、不改请求。新SHA f124b3a0791ed5fc2bcb15486925e8898d3f9a0bef2016ac7faf419b2fe6add1。显式配置runtime-dflash-asd-shared-prefix.json，原registry/配置/默认/面试版不变，在线PublicPrefixCache及模型/接受器未改。
- 原runner增加向后兼容的config/levels/study/variant/mode/load-states参数，复用原HTTP helper。shared新跑4K原8题×D/ASD×自然/128且EOS×off/cold/hit共96；whole新进程只跑同流20cache HTTP控制，shared各卡20off+20cache，共120负载请求，另有JSON/SSE/断连/恢复。GPU0 shared→whole/GPU1 whole→shared，2GiB/3 entries、每worker46GiB/2400秒、总磁盘cache16GiB。协议ASD_SHARED_PREFIX_PROTOCOL.md先于GPU冻结。
- 首unit34 passed/27.49秒；新增报告后相关shared_prefix/shared_prefix_report/prefix_cache/prefix_cache_study/runtime_asd共60 passed/7.86秒，一个既有starlette警告。完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/shared-prefix-full-tmp --junitxml=artifacts/shared-prefix-full-local.xml -x --tb=short`运行中；无GPU结果。测试构造同容量交替后缀，whole持续miss/淘汰、shared命中且真实tiny模型logits/features接近、token argmax及KV边界一致。
- 准备新root asd-shared-prefix，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_shared_prefix_pipeline`，不覆盖旧目录/结果；源码前后SHA审计，测试与输出保存独立路径。主比较为同卡相同HTTP请求流，记录窗口次序/命中/填充/淘汰、冷成本、EOS质量及spent浮点差异，不冒称通用块存储或逐轮trace一致。
- 本机完整回归924 passed/7 skipped/137.79秒，一个既有starlette警告；报告HTTP控制前缀/显存等补充校验后专项7 passed/0.32秒，XML分开保留。`git diff --check`通过，准备冻结canonical Git blobs部署，尚无本轮GPU结论。

## 2026-10-23 — 阶段6后续公共前缀缓存与真实排队完成

- 实际GPU源c2903a7ab256f9916d9be05733a53946cae0200c/1852 blobs，source SHA4f7ba9bc94e2622cd7cf98de4b21f97a0defcc276d2ac63dcea0ce42b413eaad，完整本机ZIP SHA f3f4b35087098c53581bcb9fa952599759d282672e9718a9b28855c45641a061；实际4125303B delta SHA490f4dc1f6562a2914b5fdfc66c6fc21f818b66acdcdfc496bc9f33d8a7316f9。新root asd-prefix-cache-v2/PID1470334于04:33:26 UTC启动，04:52:20检查全部完成/两卡19MiB/0%；1852源文件前后SHA一致，成功ZIP196649B/SHA8ff757a282107b933c01c433bdb83b4ee15ea7d4207a03030062ccd9752d06ef，24原始成员逐字节核验。
- 288正式（三状态×D/ASD×24输入×自然/128且EOS）全完成，全部自然EOS，无OOM/截断/正式新图。strict逐token匹配旧完整AR，off也重现上轮；缓存输出、验证总轮数和ASD汇总量相同。未保存逐轮trace，不冒称每轮接受边界相同。48 ASD配对spent末位不同，最大9.5367431640625e-5，实际预算仍≤8，未放宽接受器。ASD三档均原7/8，strict7/8、7/8、6/8，无新增质量变化。
- ASD命中自然题均时延比.8631/.5747/.4090，TTFT均.4867/1.9899/4.8204→.1341/.1616/.2021秒；首token后均2.3471/2.4199/3.1077→2.3461/2.4153/3.1067秒，支持收益来自重复Target prefill减少。cold自然基本持平；cold+1hit / 2off实测成本估算.9373/.7923/.7102，不冒充另测到达轨迹。ASD共同完整128三档N6/6/5，hit/off题均比.8206/.5366/.4008，其余EOS短请求保留。
- 各卡20off+20cache、并发2/四道4K轮换，共80真实HTTP负载请求，另JSON/SSE/断连/恢复请求。相同4073token前缀流19/20hit，wall143.6329→53.9338秒/省62.45%，平均队列6.8259→2.5738秒；3937/4073前缀交替流0/20hit、19次淘汰，wall157.1194→157.2842秒/慢0.10%，两者分别约1.382/1.429GiB，合2.811GiB超2GiB限额。两前缀有3934共同token但整项存储重复，下一步共享公共段/块去重有明确工程动机，尚未实现。不跨卡比较因果，不把串行worker排队称连续批处理。
- 每worker模型准备20.100/20.378秒，各3图/backend3.402/3.371秒；首DFlash16冷wall4.249/4.263、图0→2编译2.960/2.929秒保留，第三图在后续warmup。正式总峰43.7216GiB≤46，编译/测试缓存合计1918142B。只在同一engine生命周期缓存，独立CLI每次均冷，使用入口docs/ASD_RUNTIME.md，默认及面试版引用保持。
- 本机修复前全量906 passed/7 skipped/125.62秒、相关42 passed/7.12秒，修复后相关64 passed/12.60秒；服务器修复后99 passed/8.63秒。首源98 passed后启动热点主动终止的失败ZIP/日志保留，不算正式性能。主统计冻结c2903a7源，两环境逐字段相同；本机CRLF JSON另存analysis-local.json，发布LF来自analysis-server.json。主表手动补汇总字段定义，原生成Markdown另存results-generated.md。
- 审计1852 Git源、85冻结执行/数据文件、1283旧结果、上一发布72文件和infra三引用通过；两个既有结果CRLF例外保留，无私有缓存/权重/raw trace上传。报告ASD_PREFIX_CACHE_FINDINGS.md、主表ASD_PREFIX_CACHE_RESULTS.md、数据results/asd-prefix-cache。复核`python -m scripts.write_asd_prefix_cache_report --input results/asd-prefix-cache --output artifacts/prefix-cache-report-recheck`、`python -m results.asd-prefix-cache.inspect_results`、`python -m results.asd-prefix-cache.audit_postrun`。当前未做共享段缓存/块管理/新模型训练，reserve96未生成。

## 2026-10-23 — 阶段6后续公共前缀缓存与排队评估（执行准备）

- 按用户授权继续多个阶段。实现显式 opt-in PublicPrefixCache：只读登记公开 worked-example token 前缀，Target KV + Draft所需选层features独立存储，命中逐请求克隆KV，Draft KV仍每请求新建；LRU 2GiB/3 entries，未命中完整prefill后仅复制允许部分入库。默认配置和独立面试版本不变，ASD验证器/预算/Target输入未改。
- `.venv\Scripts\python.exe -m scripts.prepare_public_prefixes`生成5个公开前缀，token数459/1867/4073/323/3937，registry SHA9bfda318cd757144b64961e8cc05d6a29f95c00e9c65229b2ab8ca2a365bae82；不含最终问题/测试答案，原24请求逐字节复用。模型revision/tokenizer/template及registry SHA绑定，实例限制Target/选层布局，配置变化要求reload。
- 协议docs/ASD_PREFIX_CACHE_PROTOCOL.md：288正式生成（24输入×D/ASD×自然/128且EOS×off/cold/hit），首次填充和命中复制计入时延；保留spent浮点差异，不放宽线上接受。通过输出/轮数检查后，各卡做跨问题复用、直接取消、真实HTTP JSON/SSE/断连恢复，空缓存起步20off+20cache、并发2/四个4K请求轮换；HTTP顺序混杂明确保留，不声称continuous batching。
- 首测试`.venv\Scripts\python.exe -m pytest -q tests/test_prefix_cache.py tests/test_runtime.py --basetemp artifacts/prefix-cache-unit-tmp --junitxml=artifacts/prefix-cache-unit-local.xml -x --tb=short`：53 passed/17.00秒。相关prefix_cache/prefix_cache_study/runtime_asd：42 passed/7.12秒，一个既有starlette警告。完整回归进行中；尚无GPU性能结论。
- 新目录asd-prefix-cache，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_prefix_cache_pipeline`。分片奇偶同题同卡、46GiB/2400秒、编译缓存总8GiB。源码及既有结果前后SHA核验，不上传KV/features/权重/raw trace。
- 完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/prefix-cache-full-tmp --junitxml=artifacts/prefix-cache-full-local.xml -x --tb=short`：906 passed/7 skipped/125.62秒，一个既有starlette警告；`git diff --check`通过。准备冻结canonical Git blobs再部署；此时尚无GPU结果。
- 首冻结3d5385226ee3c40404c7a1158b03c3e3d917c5df/1844 blobs，source SHA26ce11a9771d22086ac7a882234c470e3c59eb0a6c61957182db80cc5b608c39，完整ZIP SHAca0686195d316fed6a854ddae37b3c15f4098d322a822d78a5eb9437edee2737；4032389B delta SHA24368755e80458e3c28f53de5c4bed7c6dde6c09a50cc1a1921d3b1269a9e99d。原root asd-prefix-cache/PID1466226于04:28:28 UTC启动，服务器98 passed/8.63秒；两个worker登记校验迟迟未完成、无warmup/正式行，定位逐token len(tokenizer)热点（独立10次0.415010467秒），04:31:19 UTC仅SIGTERM已核对PID/PPID/session/cwd/argv的两个自有worker，未修改旧source/解码/环境。
- 最小修复词表长度在初始化只读取一次，补1000token登记表的有限调用回归；相关64 passed/12.60秒，报告10 passed/6.24秒。首次失败ZIP90098B/SHA b2de98751d4d179abf5354bd1704d48e987cd3a8fbc55e41a0a3e02a84212855，7成员逐字节下载验证，另存pipeline.log于results/asd-prefix-cache-failed-startup；保留失败现场，新root asd-prefix-cache-v2再次冻结执行。本机906全量属于修复前源，不冒称新增回归已在全量执行。

## 2026-10-23 — 阶段6长输入单请求完成

- GPU执行77d4646f07b87b25d6b0b51eb2cbb1c817556106/1778 blobs，source SHA09ec377a23cfcb10023ca09014115ba01d39634e19293a642e907c8f88e2989d；PID1419258于03:15:47 UTC启动，全部四预检与四正式进程成功，源码前后1778 SHA及原适配器不变。成功ZIP251671B/SHA4aee7662aa1deb1933d2643c57ddafaa361ea724d04689f5c33cf4446d696b8b，38原始成员逐字节核验，末查两卡19MiB/0%。
- 三档全部准入，216正式（三臂×24输入×自然2+128/EOS1）、24完整AR及9独立profiler完成。全部strict逐token匹配AR，重复输出/轮数一致，所有自然EOS，无截断/OOM，正式新图0。两ASD三档均同样7/8，无新增错答；1075仍错。4K DFlash/AR341额外错，strict6/8，ASD7/8。旧短输入成果及reserve96保持。
- 位置适配相对base ASD自然题均比1.0114/1.0146/1.0031，未带来净自然提速；共同128三档N6/5/4，比1.0161/.9768/1.0050。保留2K约2.32%等长收益：113→108验证轮，主要526少4轮。自然526少6轮，但1195输出182→236、24→35轮，抵消局部改善；不将前缀收益当完整解答提速。
- 原ASD TTFT均0.486/1.992/4.815秒，占相应平均请求17.15%/45.22%/60.82%。独立同题Target prefill GPU约.49/1.92/4.70秒，Draft首次约.017/.038/.071秒。优化重点转向公共示例前缀的Target KV+选层hidden features缓存；尚未实现或宣称缓存收益。借鉴vLLM APC原理，需检查缓存生命周期、FP32分段数值、完整成本，再考虑连续批处理。
- 各正式进程模型准备20.02–20.31秒、4编译图/backend3.34–3.47秒，首AR16冷wall5.32–5.59秒，编译0→4成本留在请求内；正式峰42.292GiB，8个进程cache合计3363660B。engine.generate计时含推理链路，不含HTTP/排队/文件读取/模型加载；wall−TTFT仍含Draft首次上下文处理，profiler不混入计时。
- 主结果results/asd-long-context，完整表ASD_LONG_CONTEXT_RESULTS.md，解释ASD_LONG_CONTEXT_FINDINGS.md。服务器Python3.11/本机3.13主统计逐字节相同；审计1778 Git源、81执行/数据文件、1223旧结果、上轮74文件发布清单及infra三引用通过，两个既有结果CRLF例外保留。新增4K CLI示例和CI检查，不声称远端Actions已通过。
- 复核命令`python -m scripts.write_asd_long_context_report`、`python -m results.asd-long-context.inspect_results`、`python -m results.asd-long-context.audit_postrun`。本机完整883 passed/7 skipped/127.89秒，相关32 passed/23.31秒，服务器74 passed/7.53秒；早期夹具/目录分类两次失败XML保留。只发布代码/配置/公开题目示例/JSON/XML/log，不传权重/hidden/raw trace。

## 2026-10-23 — 阶段6长输入数学小检查（执行准备）

- 用户授权下一步。新增long_context评估模块、prepare/run/pipeline/write_asd_long_context脚本、test_long_context及GPU前协议ASD_LONG_CONTEXT_PROTOCOL.md。复用原runtime两配置，未改Target/验证器/接受规则；不训练或补seed，不迁移serving。
- 选原followup独立文件顺序前8道已观察题；官方GSM train的48个固定SHA排序解题示例，排除全部test与适配40题，当前题答案不入prompt。24请求实际输入447–542/1929–2000/4061–4156token，完整嵌套示例，不作纯长度因果或新盲测声明。准备命令`.venv\Scripts\python.exe -m scripts.prepare_asd_long_context --source artifacts/asd-adaptation-gsm-train-source.jsonl --test-source artifacts/asd-attribution-gsm-source.jsonl`，已完成。
- 两卡两配置先预检并取资源准入交集；同题同卡，GPU0 base→position/GPU1 position→base平衡先后，分进程独立cache。全准入时216正式（三方法×24输入×自然2+128/EOS1）及24 AR。TTFT/首token后耗时/自然路径/共同128分别报告，正式后最多9次独立profiler聚合，不上传raw trace。每进程46GiB/2400秒/8图/backend300秒/cache4GiB，旧权重与canonical Git源前后核SHA。
- 初次相关单测1 failed/2 passed/4.90秒，测试夹具缺examples字段，已补；随后15 passed/4.52秒。首全量1 failed/25 passed/2 skipped/10.46秒，因旧独立题检查把已复用工作负载当新增research题集。将数据以完全相同5文件字节归到data/workloads，不改旧题集或测试；相关test_long_context+test_adaptation_followup共32 passed/23.31秒。各失败XML保留，第二次完整回归进行中，GPU尚未开始。
- 第二次完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/long-context-full-v2-tmp --junitxml=artifacts/long-context-full-v2-local.xml -x --tb=short`：883 passed/7 skipped/127.89秒，一个既有starlette警告。历史研究/运行时未修改，数据归类后兼容检查通过。准备冻结canonical源码/输入/协议再部署，尚无GPU结论。
- 服务器新root asd-long-context，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_long_context_pipeline`。本条尚无本轮性能结论。
- GPU执行冻结77d4646f07b87b25d6b0b51eb2cbb1c817556106/1778 blobs，source SHA09ec377a23cfcb10023ca09014115ba01d39634e19293a642e907c8f88e2989d；完整本机ZIP SHA5e8a950aaae4be8aab2472fa44345268007aa8b01265c335a10c26744e246740，实际旧完整ZIP+3565842B delta（SHAb3f4c49ca99d078792e6139aa68508a1e0f3c33bbdcae639102c15755c633bc0）逐SHA重建。PID1419258于2026-10-23 03:15:47 UTC启动。服务器74 passed/7.53秒；四个资源预检全部准入512/2048/4096，峰约42.233GiB，无OOM。正式216+24 AR正在运行，未提前读取性能结论或按结果调参。

## 2026-10-22 — 数学ASD工程接入完成，冷CLI账本例外保留

- GPU实际冻结353f6756c5d758fc2bbe0e66e337a8c2f25e0efb/1715 blobs，source SHA79d5ca2043fc6206af87b30e94287fcb208524a6219bf8121acdbf9dccf53759；PID923189于13:36:35 UTC启动。两卡各16直接/4 AR、真实HTTP JSON/SSE/断连取消恢复/4请求并发2、严格benchmark隔离和两独立CLI均已执行。服务器相关129 passed/9.56s。32直接结果token/text/EOS/轮数/ASD账本与冻结研究完全一致，strict另与本轮AR一致；旧450错答保留。
- 最终原pipeline零容差检查失败：冷CLI累计spent比原参考分别多7.62939453125e-6/3.814697265625e-6，token/EOS/轮数/例外数全部相同，实际支出均<8。原status=failed、日志和150899B ZIP/SHA67cc41e08ccde8f966748d46e9f209dccd0a8dd36ed4f0add566d757100c03d8原样保留，33成员逐字节核验；不把原协议追改成通过。另补completion-audit.json核1715源/旧权重不变、缓存和退出状态。
- 最小后处理修正仅冷CLI的spent比较允许1e-5绝对误差，普通直接/HTTP默认仍零容差、实际spent<=8和其余字段仍精确；不改Target/ASDVerifier/预算，不重跑GPU。worker4图、CLI3图与FP32编译路径舍入相符，但未逐算子定位；说明见ASD_RUNTIME_NUMERICS.md。后处理19 passed/5.49s，XML artifacts/asd-runtime-rounding-local.xml；原全量863 passed/7 skipped属于修正前核心工程，未将新增检查追算进去。
- 原/适配模型准备21.239/21.410秒，其中适配合并0.451秒；首AR32冷wall4.829/4.692秒，独立CLI冷请求6.673/6.175秒（不含模型加载），所有冷编译计入请求。各worker4图、正式新图0，GPU峰34.903/34.672GiB，worker+CLI cache814867/818191B。两卡最后19MiB/0%，没有训练或新seed。历史16题128收益3.23%仍引用原研究，本轮只证明工程可用。
- 比较/分析版本单独冻结ec7e2720a370687e76771ec0098cd32b1663ad7a，仅在新asd-runtime-analysis目录执行CPU复算，旧1715文件保持。首个分析测试18 passed/1 failed，因证据目录漏放原source manifest；补入原字节后19 passed/3.29s，失败XML不覆盖。服务器Python3.11与本机3.13 analysis逐字节相同（SHA4d64cfdf7b7b9b46a830ce0000be08fc42eb5600d4e18f2b14bc8d271cb15cc4），原GPU目录/失败记录未变。CI加入ASD运行时测试；尚未声称远端Actions已通过。
- 发布前审计1715 canonical Git blobs、185未改执行文件、5分析文件、1174旧结果及上轮66文件发布清单通过；原main/面试branch/tag均27e3619。两个旧结果和两个模型清单的Windows CRLF仅作规范化比对，原字节不重写。`python -m results.asd-runtime.audit_postrun`可复核；原33个ZIP成员逐字节相同，original_zero_tolerance_gate_passed=false保留。
- 结果results/asd-runtime，运行指南ASD_RUNTIME.md、主报告ASD_RUNTIME_RESULTS.md。重建命令`.venv\Scripts\python.exe -m scripts.write_asd_runtime_report`；本轮仅JSON/XML/log/代码/配置与报告，不上传adapter/hidden/weights。发布清单独立于既有研究publication，未开启阶段6。

## 2026-10-22 — 数学ASD工程接入（GPU验收准备）

- 用户授权落实接纳方案的工程入口。runtime显式增加ASD B8/g.25/cap2和startup-only位置适配dense合并；复用原ASDVerifier/AdapterBank及RMSNorm，Target输入和原strict接受规则保持。每请求预算/KV独立，结果/SSE携带近似标记、账本及adapter SHA；严格benchmark拒绝近似方法，原eager/面试版保留。
- 新configs/runtime-dflash-asd{,-position}.json、问题示例、prepare/validate/run/write_asd_runtime脚本及docs/ASD_RUNTIME{,_PROTOCOL}.md。位置权重SHA42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e，约2.1MB只读旧checkpoint并复制到新目录，不上传权重。新root asd-runtime，两卡各16直接/4 AR、一个独立CLI冷启动、HTTP JSON/SSE/取消恢复与4请求并发2；这是工程验收，不重新估计历史3.23%收益。
- 本机相关112 passed/11.95s（artifacts/asd-runtime-related-v3-local.xml）；之前两次受限Windows环境在16项后停留并中断，不算通过。允许本机通信后完整`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/asd-runtime-full-tmp --junitxml=artifacts/asd-runtime-full-local.xml -x --tb=short`：863 passed/7 skipped/123.83s，一个既有starlette警告。报告/矩阵守卫后续独立相关检查另记，不追算进已完成全量。
- 报告及完整16格矩阵判定追加后，`.venv\Scripts\python.exe -m pytest -q tests/test_asd_runtime_acceptance.py --basetemp artifacts/asd-runtime-report-tmp --junitxml=artifacts/asd-runtime-report-local.xml -x --tb=short`：16 passed/4.71s。固定协议和执行判定在GPU前冻结，canonical Git blobs部署到全新目录。准确入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_runtime_pipeline`；本条尚无GPU验收结果，不启动新训练/seed或阶段6。

## 2026-10-22 — 用户接纳数学ASD位置适配收益

- 用户明确接纳已测场景收益，不要求尝试不同训练seed；本阶段据此接纳ordinary-position作为数学ASD的有效优化方案，取消此前补seed的后续安排。贡献表述为已实现、已实测的位置匹配严格前缀离线适配：在固定Qwen3-8B/DFlash b16/ASD配置的新16题上，128 token时延降低3.23%，验证调用331→319，适配未新增答案错误。旧ASD前缀的局部正结果也保留。
- README、ASD_ADAPTATION_FOLLOWUP_FINDINGS和研究路线表改为突出该场景成果；本阶段后续优先整理训练配方、适配器身份和工程使用入口。原ASD内容增量门槛未通过的记录、自然7.32%中的错误路径因素、14/16与严格15/16质量差异及单seed范围均如实保留。历史进度中的补seed建议由本条决定替代，不追改原GPU前协议或原始统计。
- 本次仅文档及发布元数据更新，无新增训练/计时。保存bd993e2的原64文件发布清单为publication-bd993e2.json，新增acceptance-decision.json，更新当前publication.json；以原发布清单核验所有实验数据、执行代码和统计字节保持一致，四个解释性文档单独列出。本次检查为`git diff --check`和发布文件工作区/暂存区/Git字节SHA核验，不重复运行已有GPU或全量测试。

## 2026-10-22 — 阶段5后续：同位置控制及独立16题评估完成

- GPU来源411c221/1651 blobs（完整SHA见下方准备记录），新root asd-adaptation-followup，PID852273于11:44:14 UTC启动；12:14:31检查已完成。80新teacher状态892671808B/约0.831GiB，两卡收集54.246/56.030秒；新严格位置臂64状态/32题/532480参数/128更新，训练段11.166秒，初始校准CE1.643746→1.592008、接受3.75→4.0625，选末128，首梯度0.14908，Target无grad/Target与Draft base版本未变。选中/末权重均2132408B/SHA42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e。原两组权重和160features只读复用，前后SHA一致。
- 四模型×D/原ASD×24题，960正式（自然3/128且EOS两重复），另24完整AR和18短warmup。全部strict输出与AR逐ID/text/EOS一致，旧8base策略重现旧参考；全部重复轨迹/账本稳定、输入/快照一致、无4096截断、正式新图0。旧8所有模型8/8；新16 strict15/16、四ASD均14/16，训练无新增对→错/错→对。ASD仍在450将AR正确4变为错误7或3，近似质量损失不能隐去。
- 旧ASD-prefix自然比base省4.41%、四模型共同128的7题省3.38%，与前轮4.46%/3.34%接近；旧正结果可重复。新16 ASDp自然仅省0.83%、128省1.05%；比同位置严格组慢7.77%/2.85%，预定ASD内容增量点门槛未通过。新ordinary自然慢0.31%/128省1.57%；同位置组自然省7.32%/128省3.23%，但不在strict D上显示稳定收益（自然基本持平/128慢1.12%）。保留位置/近似策略交互候选，停止当前ASD-prefix小LoRA扩大扫参，不把位置解释成旧收益的唯一原因。
- 新自然base/position/ASDp共5307/4518/4855 token，871/704/784验证；450旧错答1125→385/463 token，仍错，自然总收益受其影响。事后四ASD共同正确14题，position自然省3.70%/128省3.41%，ASDp约0.64%/0.23%；仅排除450的15题，ASDp自然反慢3.09%。这些只作敏感性分析，不替换全部16题主结果。完整128均2048token，base/ordinary/position/ASDp验证331/325/319/326。近似预算固定，实际接受token/支出变化单列。
- 预指定请求bootstrap10000：新ASDp自然/base比区间[.8855,1.0962]、128[.9435,1.0428]；/position自然[1.0105,1.1604]、128[.9722,1.1129]。position/base自然[.8348,.9905]、128[.9355,.9974]，但/ordinary区间均跨1；单seed、旧题反例和近似质量损失保留。下一次小成本应检查独立初始化seed及不依赖ASD采集的固定位置对照，尚未运行，不进入默认或面试版。
- 每卡4图/backend3.44966/3.31189秒/cache417205B、正式新图0；评估峰37476341760B。首AR32冷wall5.81727/5.66429秒、TTFT3.54100/3.43118秒，编译0→4成本在请求内；离线dense合并20.0606/20.9785秒另列。成功ZIP768701B/SHA1f42320a13b75717747bd7ff5b38e91e52aa9274292c535b076c358fcb7c4cd4，24成员原始字节核验；末两卡19MiB/0%。实际传输2347345B delta SHA1ee08de7e1fd922d0f3f99e8020f6dcee06bbaceabb0791a9d5c1c207255f054，完整canonical ZIP SHA8c6e19822a401428a0a5aec919101a85fbd819fe95caef37ee200e8074f0bfdf只在本机。
- 统计仍GPU执行411c221版，服务器Python3.11/本机3.13逐字段完全一致。仅prepare重导出修复单独冻结8ed9ee42eb12f0e77d31b3df7631d12f9890d099，实际records/manifest/license字节不变；失败单测与修复18 passed记录保留。本机GPU前全量829 passed/7 skipped，服务器104 passed，不将修复后新增测试算作已在旧全量执行。审计1651 Git源、1126旧结果（两个历史CRLF例外）、80位置/960KV/训练评估隔离/旧adapter身份及infra三个引用通过。主报告docs/ASD_ADAPTATION_FOLLOWUP_FINDINGS.md、主表ASD_ADAPTATION_FOLLOWUP_RESULTS.md、数据results/asd-adaptation-followup；reserve96未生成。
- 复核命令：`.venv\Scripts\python.exe -m scripts.write_asd_adaptation_followup_report`，以及`python -m results.asd-adaptation-followup.inspect_results`、`python -m results.asd-adaptation-followup.sensitivity`、`python -m results.asd-adaptation-followup.audit_postrun`逐项执行。服务器新目录入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_followup_pipeline`，已运行目录不可覆盖；原权重/hidden留服务器，不上传。

## 2026-10-22 — 阶段5后续：GPU完成及数据重导出修复

- GPU实际源411c2211e6cdfe9d20365d40986f41de4590fdf3，1651 blobs，source SHAbc433a997c85ded26a40342e90f4ea41b3612a19e0b185ad38660072b9f70190，PID852273于11:44:14 UTC/北京时间19:44:14启动。双卡80 teacher状态、单新增ordinary-position适配器、960正式+24 AR参考全部完成，1651源文件前后SHA一致，旧160features和4个adapter SHA前后相同；正式新图0、全部strict逐ID与AR相同。成功ZIP768701B/SHA1f42320a13b75717747bd7ff5b38e91e52aa9274292c535b076c358fcb7c4cd4，已下载并逐member校验；后查两卡19MiB/0%。
- 运行后发现prepare脚本导出到新目录时会把已有自身题集加入排重，导致重导出题集漂移。仅修复prepare排重时始终忽略canonical自身目录，不改变本轮records/protocol/GPU执行源码。先单测真实复现1 failed/3.89s，修复后两选择/机制测试文件18 passed/23.46s；actual官方数据重导出records SHA仍d0f3d12e6f0d92c4a1ed1fbd1e5b86d3785d9d63192cf5e662253b5342cbbfc2，记录与manifest逐字节核验。此修复单独冻结，服务器保持原411c221不变，不重跑训练或计时。
- 本地829 passed/7 skipped/128.95s属于GPU前全量；服务器104 passed/24.94s包含真实PEFT/tiny DFlash。后续数据重导出修复只跑相关18项，不能把新增测试声称已在先前全量/服务器运行。当前服务器与本地统计已生成，下一步完成独立题质量/位置对照归因及发布。

## 2026-10-22 — 阶段5后续同位置控制与独立GSM（执行准备）

- 用户授权继续上一轮建议。复用旧80源/160teacher状态和两固定adapter，不重训ASD组；全部80个ASD采样位置都能在严格源EOS前到达，新增ordinary-position截取严格内容至相同generated_before，64训练/16额外诊断状态，同prompt/长度/anchor位置，80对内容均不同。原32题/8内部校准、seed、128更新/累积2、lr/rank/loss/shuffle及原32状态混合校准选模规则不变；新16状态不参与选模。
- 新16官方GSM test按固定SHA排序、排除既有GSM ID/规范化问题及source_index<32，问题记录SHA d0f3d12e6f0d92c4a1ed1fbd1e5b86d3785d9d63192cf5e662253b5342cbbfc2；旧8单列。data/research/asd-adaptation-followup/manifest.json保留来源、排重文件SHA与ID。suite reserve96不生成，新题不参与训练或选checkpoint。
- 新模块adaptation_followup.py与prepare/run/write_asd_adaptation_followup脚本薄复用原teacher、LoRA、dense bank、ASD、RMSNorm和评分。四模型×两策略×24题，自然3/128且EOS两重复=960正式，另24 AR参考、18短warmup。主对比ASD-prefix/同位置严格适配，位置混杂得到控制，仍不能代替多seed或证明上游未见。协议docs/ASD_ADAPTATION_FOLLOWUP_PROTOCOL.md在GPU前冻结。
- 首轮本机`.venv\Scripts\python.exe -m pytest -q tests/test_adaptation_followup.py tests/test_adaptation.py tests/test_asd_adaptation_report.py --basetemp artifacts/followup-tests-tmp --junitxml=artifacts/followup-tests-local.xml`：37 passed/2 skipped/31.37s。准备完整CPU回归和服务器真实PEFT前置；尚无本轮GPU结果。目标新目录asd-adaptation-followup，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_followup_pipeline`。原执行源码/私有features/adapters运行前后SHA审计，不覆盖旧目录。
- 完整本机回归`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/followup-full-tmp --junitxml=artifacts/followup-full-local.xml -x --tb=short`：829 passed/7 skipped/128.95s，一个既有starlette弃用警告。额外核对5个既有JSONL请求文件，新16题均未出现；旧公开结果/训练文件56项SHA及两adapter常量与既有provenance一致。准备冻结源码后部署，未产生新GPU结论。

## 2026-10-22 — 阶段5等预算真实训练与开发评估完成

- GPU执行4d7ac6eca2ab9970e9f71314fc77b9b9de79e140/1584 blobs，source SHA4b5109f72f1da81084e59b14533594e1ac26c8b684a47156e4774a7c0ead1e84；本地完整ZIP SHAc14712655b4b70b88136922ddfa2d3c8c61977663f2653e0d83b8a6382626deb，实际旧ZIP+1233414B delta SHA11f264bbce26c4390bb9721839a64ef0573e52821e1c7bb74f7da220dfcde091在新asd-adaptation-v2重建逐SHA核验。PID697777于07:31:56 UTC/北京时间15:31:56启动；07:48:12后处理/末核查完成，两卡19MiB/0%。成功ZIP889588B/SHAb4178f0a920155077e00ad2a1633ccf131e9192d21a9dbd399ca9e45fce7f837，前后1584源SHA一致。
- 40新GSM train问题（32训练/8内部校准）各strict/ASD源，共80源、80对/160唯一teacher状态、2400标签，无源状态缺失。39源达256采集上限、41 EOS，不当自然质量证据。两卡收集211.516/201.260秒，features1719637760B/1.60154GiB。普通/ASD-prefix各64训练状态/532480参数/128更新累积2，训练段11.714/11.648秒（此前加载/读取另计），校准CE1.643746→1.602276/1.604261，均选末128，梯度非零、Target无grad及Target/Draft base版本守卫通过。两adapter各2132408B/20张量，SHA/shape/finite服务器核验；hidden/权重不推Git。
- 六臂（三模型×D/ASD）×18开发题，自然3重复324+128/EOS两重复216=540正式，另12短warmup。所有strict输出与旧完整DFlash/RMSNorm逐ID/text/EOS相同，base ASD也重现旧自然，全部重复轨迹/账本稳定、无自然截断；strict16/18、ASD17/18均保持旧质量，无新增对→错。GSM8是预指定训练域，另五类各2诊断迁移，不把18题当新独立测试。
- GSM自然：ordinary+D比base D慢0.78%，ASD-prefix+D省1.27%；ordinary+ASD省0.83%，ASD-prefix+ASD省4.46%（相对ordinary省3.43%），最终答案均8/8。ASD-prefix的8题时延点估计均更低，但3题无减轮且有接近噪声差异；不称全部实质改善。三模型共同实际达128的7题，ASD-prefix+ASD/base0.966559、/ordinary0.950393；ordinary/base1.021797。支持非全由输出缩短的计算收益，固定128与自然不同集合不能相减归因。
- GSM自然ASD输出base/ordinary/ASD-prefix共1648/1654/1634 token，Target验证276/273/262；近似接受token均54，regret60.24634/58.59283/58.66578。没有增加预算或例外数。1196输出197→211更长却调用37→32，52变短例也保留。MATH两组自然均省约2.4%而非ASD源独有，其余家族基本持平；全18题ASD-prefix+ASD自然省2.27%，不能泛化到所有任务。
- 机制限制：64训练对全部prefix不同/此前有ASD例外，但只有3对生成边界相同，源内容与进度混杂；CE降低同时内部校准严格接受长度3.75→3.59375，两组均如此。校准是未合并eager，部署是dense/RMSNorm，不能凭loss或接受proxy断言训练因果。附加完整模型unmerged/dense数值重放的准备只读检查自动审批两次超时（非安全拒绝），未执行新增GPU复验；optional-verification.json明确所做tiny/完整dense/资产审计与未做项，不算通过。
- 按请求事后bootstrap10000：GSM自然ASD-prefix/base比95%区间[.9253,.9828]；/ordinary[.9231,1.0040]，共同128/base[.9205,1.0071]、/ordinary[.8831,1.0077]，后几项跨1。保留窄域正信号，但不声称额外增量稳定独立证实。未在18题扫参/继续训练，下一步等生成位置控制及独立GSM确认，不先加复杂模块；validation/reserve未训练，reserve96未生成。
- 原统计共同128集合缺乏三模型统一检查，按原协议修复并独立冻结f8e68035f6ae3f01e8e0fa531260a248e03fe7f8；生成源码和原目录保持4d7ac6e，没有重跑有效计时。本批新旧统计差0（反例已覆盖），修正版服务器代码SHA20a7d56864df54607429e38e8ac33b766b839a08f9ef40a8dd226878a39f8440；Python3.11服务器与3.13本地主统计差0。严格区分GPU执行与分析版本。
- 评估每卡3图/backend3.18786/3.19195秒/cache398054B、正式新图0，计时请求峰35.1619GiB；首次base D32 wall4.74335/4.76201s、TTFT3.44322/3.45287s，图0→3编译成本请求内。离线dense合并总10.8298/10.7387s另列，不宣传线上每请求切权重免费。本地完整811 passed/7 skipped/114.38s、兼容20 passed/2 skipped/8.88s、统计12 passed/6.55s，服务器87 passed/10.03s。失败PEFT源9bb81be/82100B ZIP及2 failed/84 passed完整保留。
- 审计两份1584 Git来源清单、11执行冻结文件/2分析文件、1073旧结果SHA、12批478历史发布文件、数据隔离/80源前沿/160配对样本/540实际KV轨迹及原infra三引用通过，两个既有CRLF例外单列。主结果results/asd-adaptation，主表ASD_ADAPTATION_RESULTS.md、结论ASD_ADAPTATION_FINDINGS.md。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_adaptation_pipeline`；本机`python -m scripts.write_asd_adaptation_report`、`python -m results.asd-adaptation.{inspect_results,audit_statistics,audit_postrun}`分别执行。默认及面试版保留。

## 2026-10-22 — 阶段5等预算Draft适配（执行准备）

- 实际GPU冻结4d7ac6eca2ab9970e9f71314fc77b9b9de79e140/1584 blobs，source SHA4b5109f72f1da81084e59b14533594e1ac26c8b684a47156e4774a7c0ead1e84，本地完整ZIP SHAc14712655b4b70b88136922ddfa2d3c8c61977663f2653e0d83b8a6382626deb，1233414B delta SHA11f264bbce26c4390bb9721839a64ef0573e52821e1c7bb74f7da220dfcde091。新asd-adaptation-v2/PID697777于07:31:56 UTC/北京时间15:31:56启动；修复后服务器87 passed/10.03s。两卡收集各80样本，均覆盖20问题/两源轮4/12；各臂训练64状态/32题/532480参数/128更新，初始内部混合loss1.643746，选末128普通1.602276、ASD-prefix1.604261，训练段11.714/11.648秒，Target/Draft base版本守卫通过。540计时仍运行中，没有预判速度收益。
- 在未读取正式时延结论前发现报告共同128筛选实现只对当前variant/base/ordinary检查，导致base/ordinary行可能未要求ASD-prefix也达到128；修正为同一策略三模型共同达到128，严格符合原协议，不改GPU训练/生成/计时。新增反例断言全部三行使用同一集合。GPU源保持4d7ac6e不动，分析版本另提交并审计，不重复有效GPU实验，不声称执行和分析为同一源码版本。
- 首次冻结9bb81bed254761efd4cef30d2845a59cfa71126d/1584 blobs，source SHAcfd25ba4da9e7c0dd0beed31d0c804abda403f5a3b49f8f06db0ec2b4b2d058a，本地完整ZIP SHA6098cbffee0155719ce5d41d9fc7aa8b9ccaa485fd9214bd9e0a124580975327；1232175B delta SHAe31988f5f71b11a1a053346d1cbe1b472f86a1ea7606ef9717fd0066ea8ebe51。新asd-adaptation/PID695345于07:28:44 UTC启动，但服务器CPU前置2 failed/84 passed/9.91s，GPU worker未启动。PEFT0.11探测系统旧bitsandbytes而缺cuda124.so，非PyTorch CUDA故障；失败ZIP另存asd-adaptation-failed-peft-results.zip，82100B/SHA1032258be91b46cc780f2fea789ed7c4b2b99577bf28d7cb26ea6de935fdc015。
- 最小修复只让已验证原生FP32 Linear在get_peft_model期间跳过两项不使用的bnb量化探测，并恢复全局探测函数；不升级共享环境或改精度。测试补检查探测恢复与非FP32拒绝。训练预算/数据/统计不变，将新source/新目录重试，保留原失败现场。
- 用户授权阶段5。新实现复用PEFT0.11.1和已审计DFlash，原推理引擎未改。旧16状态检查只是不支持当时的必要性假设，不能代替训练。检索DFlash原训练、AdaFlash2607.19223v3、Draft-OPD2605.29343，选择一次离线硬标签可行性，不假称完整KL/on-policy在线算法复现。
- 新官方GSM train固定commit3101c7d5072418e28b9008a6636bde82a006892c，raw SHA17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465；32训练/8内部校准按固定SHA排序，排除与完整官方test的规范化精确重合（0），只保存问题/来源元数据，没有参考答案。questions SHA a2455b6fb3571d3a2e689d392e771c2d3278fb99f0beefa4fb14cdecef82dab5，MIT notice保留。Windows首准备命令遇UTF8默认GBK读取错误，显式UTF8后成功，未生成GPU数据。
- 协议docs/ASD_ADAPTATION_PROTOCOL.md：原strict/ASD各256源轨迹，在真实Draft轮4/12取两源共同存在的状态；重新prefill实际上下文提取anchor前features，teacher新生成至多15标签，EOS后padding不计loss。禁止复用错误Target状态或把ASD旧候选当正确标签；source缺失/进度差异和ASD此前预算均记录。hidden缓存限16GiB，不上传。
- LoRA q/v rank8/alpha16/dropout0，原Target/Draft base冻结；各128更新/累积2、AdamW1e-4/.01、clip1/seed20261022。每32步用相同内部混合校准池按teacher位置CE选训练checkpoint；不按评估题继续训练。两卡各训一组、单组训练段900秒，GPU46GiB。训练/eager可微，评估统一原强RMSNorm profile，合并dense投影并单列离线准备成本。
- 评估原开发18题：GSM8+五类各2，三模型×strict/ASD，自然3次/128且EOS前缀2次=540正式，另12短warmup。严格输出核对旧完整参考，base ASD也重现旧自然；近似组单列质量/长度/TPS/共同128，不拿更短输出当吞吐。validation48不作新盲测、不训练或选ckpt，reserve96不生成。
- 新文件memorydraft/evaluation/adaptation.py、prepare_asd_adaptation及run_asd_adaptation_{collect,train,eval,pipeline}/write_asd_adaptation_report、tests/test_adaptation.py与test_asd_adaptation_report.py。初始tiny7 passed/2 skipped/14.56s，相关59 passed/4 skipped/11.42s；本机无PEFT，服务器有0.11.1，将在服务器先验证真实tiny DFlash梯度再启动GPU流程。完整本地回归运行中，尚无本轮训练/加速结论。
- 本地完整811 passed/7 skipped/114.38s，XML artifacts/adaptation-full-local.xml，`git diff --check`通过；最小测试和报告测试覆盖监督/预算矩阵/质量损失及fresh部署。完整命令`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/adaptation-full-tmp --junitxml=artifacts/adaptation-full-local.xml -x --tb=short`。准备提交冻结源/数据/主统计并部署；尚无GPU训练结果。

## 2026-10-22 — 阶段4有限多候选/后缀修复检查完成

- 冻结2949b9ed7dda32678ef0484194554cbc43365d74/1540 blobs，source SHA3fbbfe1da9e33f728d542f6ea9835eca1e0fb1323337fa344e79d3ed251eedce；本地完整ZIP SHAe2503bf149c76b5918dc5300bbb0363b5f5ecaba1f0e1cf5eb1cb95e89f1626c。旧完整ZIP+819188B delta（SHAac40da7d4e45da56a49f9bbd98d03ef21c12890a753427552adde1fa66a9a4be）在独立asd-candidate-pilot重建1540目标blob。PID618342于2026-10-22 05:23:43 UTC/北京时间13:23:43启动；约13:30完成和末审计，前后源码SHA全同，两卡最终19MiB/0%。成功ZIP307959B/SHA61753a35323d691f87f33b1418b05a47b65a2f01802a7d5f3fedc166d7d9ca36。
- 服务器121 passed/9.47s后12题×六方法×3=216正式完整自然请求，另12短warmup。各方法1987输出token、12/12自然EOS，无4096截断；全部token/text/EOS及输入/快照与旧完整DFlash/RMSNorm同，每臂重复轨迹稳定，评分字典也全同。保留旧HumanEval:134失败和SQuAD四token旧EM失败，不把严格一致当全题答对。仅strict D隔离候选机制，没有改ASD接受或将其冒充ASD完整收益。
- 题均自然wall/native：packed-linear1.008754，tree32 1.018533，tree48 1.188727，repair4 1.176796，repair8 1.178011。每题一条稳定轨迹：native Target/Draft278/278、4448验证节点；tree32 265/265、8480节点；tree48 264/264、12672节点；repair4 380/216，repair8 379/216。Target列不含12相同prefill，但所有prefill/构造/同步/选路/KV/特征/EOS成本均在wall内。
- tree32有55/265轮走分支，43次只多推进1；宽48只再少一次调用。repair4 164次/115零接受（70.12%），repair8 163次/112零接受（68.71%）：省Draft约22.3%却多Target约36%。raw tree_gain和EOS之后已付工作在interpretation.json分列，不能当多输出token或孤立kernel成本。
- 正线索保留：tree32在GSM两题时延比分别0.968742/0.962267、家族0.965504，三重复方向一致；HumanEval家族0.974818但一题退化，其余家族均更慢。总体未省3%、任何家族未省5%，按原协议不自动扩48，不临时调门槛/预算/挑题。此有限固定树/直接token修复阶段完成，不断言所有树或表示复用失败；完整BASTION、ReTrace训练/表示融合、ASD树账本及独立数据未做，reserve96未生成。
- 每卡3图/backend3.27113/3.24606s/cache398883B，peak35.1599GiB，正式新增编译0。首次空cache native32 wall4.83495/4.82669s、TTFT3.52471/3.51903s，图0→3/编译在线成本未扣。结束即时一次GPU19MiB/1%随后0%，completion保留两次观察，仅复核空闲，未重复模型生成。
- 本地完整791 passed/5 skipped/109.86s后报告14 passed/6.74s；服务器121通过。后审计1540 Git blobs/10冻结文件/1044旧SHA/11批434历史发布文件/轨迹前沿与树路径/原infra三引用通过，两个既有CRLF例外显式列出。Python3.11服务器与3.13本地主统计完全相同（差异0），后验机制脚本没有改变冻结选择规则。
- 结果results/asd-candidate-pilot，主表docs/ASD_CANDIDATE_PILOT_RESULTS.md，结论ASD_CANDIDATE_FINDINGS.md。服务器命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_candidate_pipeline`；本机`python -m scripts.write_asd_candidate_report`、`python -m results.asd-candidate-pilot.inspect_results`、`python -m results.asd-candidate-pilot.audit_statistics`、`python -m results.asd-candidate-pilot.audit_postrun`。只新增可选研究入口，不升级默认；原infra冻结引用保持。

## 2026-10-22 — 阶段4多候选与局部修复（生成前准备）

- 用户授权阶段4。检索DFlash、BASTION(arXiv2605.29727及官方repo)、OPT-Tree(TACL2025)、ReTrace(arXiv2608.29748v2)原始来源，选择免训练有限方案；没有复制新上游源码，不增加许可依赖，不宣称原创或完整论文复现。新增draft_tree.py和candidate_repair.py，adapter仅可选严格tree入口，默认路径/Target输入/权重/接受规则保持。
- 冻结协议ASD_CANDIDATE_PROTOCOL.md：六类原开发题各2，native、打包线性控制、固定Top-2单替换树32/48、一次旧后缀重验4/8，三重复共216完整自然请求4096上限/EOS，另12短warmup。树单Target调用、祖先隔离/深度位置/所选KV及feature提交；修复只重验token，不提交错误前缀的Target状态。全部在线构造/同步/KV成本计入。原完整DFlash/RMSNorm输出与质量参照保留。
- 准入整体净省>=3%或家族>=5%且两题都快，三重复方向一致、每题严格输出/输入/轨迹安全与资源满足才扩全部48开发题；不使用已观察validation作新独立，不生成reserve。完整BASTION自适应树及ReTrace表示融合未做，有限负结果不能否定这些方法。
- 首轮tiny检查81 passed/2 skipped/12.45s，相关94 passed/2 skipped/12.28s；随后补强制分支切换的跨轮feature测试，将完整回归。记录artifacts/stage4-{tiny,related}-local.xml。服务器两卡19MiB/0%；新目录asd-candidate-pilot、各worker3600秒、图8/backend300秒/cache4GiB/allocated46GiB；主协议/统计GPU前提交冻结，旧结果和infra引用保护。尚无本轮GPU性能结论。
- 完整回归791 passed/5 skipped/109.86s（含HTTP回环，artifacts/stage4-full-local.xml）；随后报告准入修正为必须全题安全，另14 passed/6.74s。`git diff --check`通过。执行`.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/stage4-full-tmp --junitxml=artifacts/stage4-full-local.xml -x --tb=short`；报告追加`tests/test_asd_candidate_report.py --basetemp artifacts/stage4-report-final-tmp --junitxml=artifacts/stage4-report-final-local.xml`。准备commit导出canonical Git source，尚未GPU计时。

## 2026-10-22 — 阶段3计算分配检查完成（按门槛停止扩测）

- 冻结cea9ebb92db4879144b79b0338cd1272b38b0a35/1491 blobs，source SHA5dc11ff081b1c81798e2f6ee4bd5c8e227bc5b8d828b178948261538a9af3bf0；asd-allocation-v2/PID568165于04:02:59–04:09:43 UTC完成，ASD ALLOCATION COMPLETE，前后1491 SHA相同，两worker0退出、两卡19MiB/0%。旧完整ZIP+404622B delta重建记录保留deployment-delta.json，原前置失败目录/ZIP未覆盖。
- 六类各2开发题，12 native512源逐ID/prompt/snapshot重现旧完整DFlash/RMSNorm前缀；144请求128/EOS计时；27状态×4动作×3重复=324实际KV分支64/EOS，共480主生成，另10warmup，9缺失状态不补位。144请求/27状态全部跨动作token一致、每动作重复轨迹精确；审计全部共同轮/KV前沿及之后native。128/64/512不冒充完整自然质量，validation不作新独立，reserve未生成。
- 请求等权native归一成本native/half/verify-half/AR=1/1.224244/1.182876/3.748638；局部一次动作及native后续为1/1.016271/1.015223/1.076463。短块单调用略省，但Target调用数题级中位增19.52%/17.52%，每调用摊销中位降约2.97%/2.18%，不能相乘中位数作总体因果分解。全部逐题指标保存interpretation.json。
- 请求oracle/跨重复/留题预测比最好固定为0.990718/0.991221/1.041159，无oracle合格场景；局部总体0.951273/0.951369/1.045080。局部MBPP oracle0.928661、短问答0.836896，跨重复相近，两个场景过空间门槛；但留题选择器分别1.003124/1.298426，没有合格场景再省>=2%，continue_selector=False。按冻结协议结束当前四动作/四特征检查，不扫特征/阈值、不加复杂头、不启动闭环或完整48自然扩测；保留局部终止机会，不否定全部计算分配路线。
- 局部机会例：mbpp:262轮4只剩EOS，Target0.05022s对native0.07204s；两token问答0.05003对0.07568。另一四token问答剩3token，Target先单步还须后续D，0.13517对0.08687s，错误选择反而慢55.6%；留题模型在这里误选。源轮0历史特征相同、prompt长度不能可靠预测终止，是本批机制解释，不是免费EOS oracle。
- 每卡4图/warmup后稳定、正式新编译0，outer3.804/3.824s与backend3.426/3.432s嵌套，cache414716B/卡，peak35.158GiB，资源满足原预算。成功ZIP264854B/SHAf3c835e254d39259771824e1488a65f34891391adfc5ee426c44b56731140684。1491 Git blob/8冻结文件/1006旧SHA/10批387发布文件/真实KV及infra引用审计通过，两既有CRLF例外单列。
- 本地相关60 passed/2 skipped/11.58s，完整731 passed/5 skipped/112.80s；后续资源/部署报告13 passed/6.65s，服务器修复后90 passed/9.76s。原合成夹具/服务器缺父目录的失败XML和ZIP保留。执行/协议/主统计GPU后未改，只有只读interpretation新增；不重复已通过GPU测量。
- 原服务器Python3.11从未压缩行内存重建主统计，与本地3.13仅11个浮点字段最大3.33e-16末位差，模型/选择/continue判定相同，statistics-rebuild-server.json保存全部差异。留题4.12%/4.51%是按预测回查实测动作成本的离线估计，未计在线选择开销，未假装实际跑过闭环策略。原前置失败ZIP73598B/SHA1bf52dc8e04065b162ba0ed7d08638e9e3a5bbbf669524d14569b7d2de4d0f45一并保留。
- 结果results/asd-allocation；主表docs/ASD_ALLOCATION_RESULTS.md，解释ASD_ALLOCATION_FINDINGS.md。运行`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_allocation_pipeline`；重建`python -m scripts.write_asd_allocation_report`，描述`python -m results.asd-allocation.inspect_results`，后审计`python -m results.asd-allocation.audit_postrun`需原ZIP/Git历史。默认执行/面试版保持。下一路线为阶段4多候选/局部修复成本预检，尚未实现，不把本轮严格局部结果冒充ASD新收益。

## 2026-10-22 — 阶段3真实计算分配（生成前冻结）

- 修复版已运行：cea9ebb92db4879144b79b0338cd1272b38b0a35/1491 blobs，source SHA5dc11ff081b1c81798e2f6ee4bd5c8e227bc5b8d828b178948261538a9af3bf0，本地完整ZIP SHA1e0cd79e5476e1c1284e80da4be82f99550c6313e3926e4882a7003657ac28cc。404622B delta SHA6b5b909f89a0e9cf1258fbf629cf6450db43aad2e61d0937234cebdcf0b0479c，结合旧完整ZIP在新asd-allocation-v2核验1491目标blob。2026-10-22 04:02:59 UTC/PID568165启动；服务器90 passed/9.76s后两worker已正式生成。旧失败ZIP服务器另存asd-allocation-failed-preflight-results.zip；生产动作/样本/主统计与门槛未改。

- 首次服务器前置81 passed/8 errors/9.94s，GPU worker未启动。新root未创建artifacts父目录，pytest --basetemp artifacts/test-tmp在部分tmp_path测试初始化失败；仅pipeline补mkdir，并新增空部署目录/首子进程之前父目录存在的测试。相关13 passed/6.65s；生产解码、样本与统计/门槛未改。原失败ZIP及pipeline.log保留artifacts/asd-allocation-failed-preflight-*，后续新目录/新manifest重试。

- 已启动：冻结d7c067bf57b6acbc2c434ad1f43a3337a881b335/1491 blobs，source SHA1f08f58f38c280040ea1f02abcc1200747401d3929933ea5d201f7f5e0dbaa27，本地完整ZIP SHA9af4058570a3f7cbb98966ec110d023e5093341d4a7d83bc0f7b66535144578f。旧已校验完整ZIP+403927B delta（SHAcff3198ff666753222d7559e7cca26c83a791aca8d441a532f6112fe12793f21）在新asd-allocation重建1491目标blob全部SHA通过；DEPLOYMENT_DELTA.json列45个变化/新增文件。2026-10-22 04:00:30 UTC/PID566285启动双卡pipeline，最终结果待完成，未预判门槛。

- 用户要求完成阶段3。协议docs/ASD_ALLOCATION_PROTOCOL.md：六类原开发题各前2，strict/native16/FP32/RMSNorm；不消费reserve，不把已观察validation当独立。动作native16、实际Draft8、Draft16/verify7和Target单步；单步累积真实feature后Draft追赶，pending达到1024强制native。仅可选round_action薄回调，默认旧轨迹不变，未改接受规则/Target输入/权重。
- 每题1源native512、四请求动作128上限各3次（其中AR为原生generate），源轮0/4/12各四动作×3次保留实际KV的64新token局部检查；均遵守EOS，缺失状态不补。最多12+144+432=588主要生成，另10短warmup。局部clock从当前起草前起算，包含该动作及实际后续，前缀共同重放排除；请求clock全在线。原完整dev DFlash/RMSNorm token/prompt/snapshot先核对，全部数值分歧与失败保留。
- 先比最好固定动作的有限oracle，再留一次重复防选噪声；总体或任一>=2请求家族须oracle省>=5%、跨重复省>=3%，全题输出安全且同范围留题单层stump再省>=2%才进入实际闭环候选。四固定廉价特征/每叶至少3状态，训练不见本题；家族报告沿用全体其他题的同一个留题模型。若不通过结束本四动作阶段，不扫更多特征/长度；通过再按协议做完整生成与质量检查，不把局部oracle当线上提速。
- 依据DISCO(PMLR2024)、DFlash(arXiv2602.06036v1)、BlockPilot(arXiv2606.31315v1)，已检索官方论文，未引入上游新代码。并行Draft成本不随裁短候选线性下降，所以两种裁短分开实测，非重复旧例外接受/拒绝预测。
- 首轮`.venv\Scripts\python.exe -m pytest -q tests/test_allocation.py tests/test_dflash_reuse.py --basetemp .test-tmp-allocation-first --junitxml=artifacts/allocation-tests-first-local.xml -x --tb=short`：52 passed/2 skipped/31.15s。相关60 passed/2 skipped/11.58s；报告合成测试一度因每侧3题留一后不足每叶3状态而失败，增加为每侧4题保持最小叶规则，原失败XML保留。
- 完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-allocation-full --junitxml=artifacts/allocation-tests-full-local.xml -x --tb=short`：731 passed/5 skipped/112.80s。随后补充资源审计反例，报告相关12 passed/6.70s，未重复无关GPU性能。真实tiny Target/KV覆盖混合动作/追赶、EOS/上限/stream、默认trace相同、fork边界与报告矩阵/不稳定/NaN/数值风险。
- A6000探测torch2.5.1+cu124/Transformers4.57.3，两卡19MiB/0%。部署独立asd-allocation，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_allocation_pipeline`；worker2400秒，图8/backend300秒/cache4GiB/allocated46GiB，先服务器相关检查。源码、协议和主统计在GPU前提交导出Git blobs，原infra引用及全部旧结果保护。

## 2026-10-21 — 显式RMSNorm运行时真实验收完成

- 冻结52fe5be256a9982d4201c24a92b498a5e9cc0e2f/1453 blobs，source SHA59b878abbab371be65d061898812483c3c18aa220a2f575ae69f0d0802d72627；rmsnorm-runtime-v2/PID12263于12:47:08–12:49:01 UTC运行同一pipeline，RMSNORM RUNTIME ACCEPTANCE COMPLETE，前后1453 SHA相同，两卡结束19MiB/0%。旧完整ZIP+delta重建记录见deployment-delta.json，原目录/失败结果保留。
- 服务器84 passed/8.93s后实际单卡：2公开示例×3方法×2重复=12计时token/prompt/snapshot全同同profile AR；R接受2历史token、快照未变；直接取消后恢复、实际Uvicorn 6请求并发2全成功、三路SSE token/profile正确。独立profile记录13527 CUDA kernel/10阶段，原大trace/DB不上传。前进程退出后新CLI进程45token成功。
- 空cache首次AR4.78660s/TTFT3.65257s，图0→4/backend3.49202s计请求内，outer3.87917s嵌套；加载setup0.00402s另列。紧接DFlash0.19941s无新图，不能当独立空cacheD冷启动。新CLI已有磁盘cache仍3.99864s/图0→3/backend1.65654s；不声称cache存在即零编译。正式benchmark无新图，验收4图/cache417641B/peak allocated34.6704GiB。
- warmed示例AR/D/R+D总8.9742/4.3535/4.8169s、均186token；R+D比D慢10.64%，仅证明R确实触发兼容，不作新增R提速或norm服务性能消融结论。研究性能仍用独立48题约2.7%–2.9%；默认eager/原面试版保持，reserve96未生成。
- 结果ZIP98910B/SHA825e38ae8e3ea608f8479d6330df8c02960e2ce530782281a57d8676cc38eb4c；原失败72760B/SHAb4b0d6b8fea3f3bdeee486d81508f182e77cc0790a5dbb7dee0312617b11dd25。results/rmsnorm-runtime及docs/RMSNORM_RUNTIME_RESULTS.md；1453 Git blob/9核心文件/9批347旧发布文件/infra引用核验通过。本地完整711 passed/5 skipped（仅后改测试夹具，23相关再过），服务器84通过；不重复无关性能测试。
- 最小修复为测试获取当前Transformers延迟导入模块，生产guard未改；validate_infra生命周期统一到末尾实际HTTP单lifespan。未放宽数值/质量条件，未覆盖旧失败。复现入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_rmsnorm_runtime_pipeline`，后审计`python -m results.rmsnorm-runtime.audit_postrun`需Git历史及两个原ZIP。新增使用说明列真实infer/serve命令和冷成本。

## 2026-10-21 — 显式RMSNorm运行时真实接入验收（执行中）

- 修复版52fe5be256a9982d4201c24a92b498a5e9cc0e2f/1453 blobs，source SHA59b878abbab371be65d061898812483c3c18aa220a2f575ae69f0d0802d72627；本地完整canonical ZIP SHAd34e475b67b7ede5354bff2324e1863c214a2cb48266af5bcdfada08c5a96d53。使用已校验旧完整ZIP+161138B delta（SHA1f2766979b88b1a7dd452cb1ee84d9e0a5f903a9cbc79a3492da11b0beb9410d）在新rmsnorm-runtime-v2重建，1453目标blob逐SHA核验；不是声称服务器收到了新完整ZIP。DEPLOYMENT_DELTA.json列出仅三个变化文件。2026-10-21 12:47:08 UTC/PID12263启动同一pipeline；原失败ZIP服务器另存rmsnorm-runtime-failed-tests-results.zip，新目录/空缓存，无生产代码变更。

- 首次前置检查83 passed/1 failed/8.92s，GPU尚未启动：Transformers延迟导入后sys.modules对象与测试顶层旧引用不同，模拟版本修改落在旧对象，运行时正确读取新对象4.57.3而进入伪模型。单例复现相同。仅test_runtime_rmsnorm.py改用import_module获取当前对象，生产guard/执行/验收门槛未改；本地`python -m pytest -q tests/test_runtime_rmsnorm.py --basetemp .test-tmp-rmsnorm-runtime-lazy-fix --junitxml=artifacts/rmsnorm-runtime-tests-lazy-fix-local.xml -x --tb=short`：23 passed/9.36s。失败ZIP及pipeline.log保留artifacts/rmsnorm-runtime-failed-tests-*，后续新目录/新manifest重试，不覆盖原目录。

- 独立验证发布9b8176e78e3463b0573782dd570eb71045691022；50发布文件在Git及工作区逐SHA一致。通过7项门槛后cherry-pick隔离655e700为96952d31927bf4184b53911d9f5db0931e30bb70，默认eager保持。候选先前本地711 passed/5 skipped，未在合并时重复GPU性能。
- 接入源码96952d3/1453 blobs，source SHAe0f3ea6f4b5bf8d7c9c650800a5b52433aeda0d059142023128e9a86fc0a9bfd，ZIP18695322B/SHA625d6ae67bdc10afb85e4bfbafac751f61d40d161f9b7a3354a35bb82060be05。首次上传仅收到14450688B，SHA阻止部署；改10块传输后重组总SHA通过，原partial保留，不用不完整源启动。
- 独立服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/rmsnorm-runtime，PID9062，2026-10-21 12:42:42 UTC启动；两卡确认<100MiB后运行，正式使用GPU0。命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_rmsnorm_runtime_pipeline`。先原runtime/serving/regional测试，再空cache冷调用、12示例计时及R/cancel/CUDA profiler/6 HTTP/三路SSE，最后前进程退出后新进程CLI。每子进程1800s、测试300s，仅终止自己创建的会话。
- 原研究验证数据、准入与源码固定；本轮只工程验收，不将示例性能等同约2.8%研究收益。结果待完成保存results/rmsnorm-runtime，完整trace/DB/cache/权重排除发布。原infra三引用仍27e3619，reserve96未生成。

## 2026-10-21 — RMSNorm48独立验证完成

- 冻结2c8c51106cd01208756de5458abb9e46889f59be/1417 blobs，source SHA27188c05eb8a7f0b609f22b9109d43712a82627aba0bf4790c3b3458af2b0c5b；服务器asd-rmsnorm-validation PID4169165于11:53:36–12:30:37 UTC完成，REGIONAL RMSNORM_VALIDATION COMPLETE，两worker0退出、两卡19MiB/0%，源码前后1417 SHA相同。
- validation48六类各8、1152配对计时+96 AR，另20短warmup；reserve96不生成。四方法自然时延比D0.973216 [0.971616,0.975672]、ASD0.972030 [0.971376,0.972684]、Run0.972200 [0.971696,0.972704]、U0.971433 [0.969198,0.972971]，低2.68%–2.86%；六类均<1，D47/48、其余48/48更快。D唯一慢题的一次0.2984s对0.2615s保留，不筛除、不重复追求全赢。
- 240完整输出token/text/EOS全同、每profile重复精确，两profile严格D/AR各48/48，无新增主/secondary/F1/未评估/截断变化；全部7工程门槛True。数学原AR/D在math500:232仍4096截断，近似ASD有GSM/MBPP/SQuAD相对AR质量下降，不作norm改进或全答对声明。58账本末位差/max5.14984e-5，5318轮除regret外相同。
- 每卡167norm/4图，正式/AR无新编译，outer3.734/3.684s、backend3.346/3.298s嵌套，cache416796B、peak35.926GiB、CPU各11319fast/0fallback；平均配对省71.97/63.99ms，条件摊销约52/58请求。8个codegen摘要仍各单融合norm/无GEMM，不新增微回放。
- ZIP911053B/SHAdbf7c7f183fcecf35f15e760f0cefbbd1fb0580fad96e075c547031e306936a1，results/asd-rmsnorm-validation；报告ASD_RMSNORM_VALIDATION_RESULTS/FINDINGS.md。25冻结文件/946旧结果SHA/8批297发布文件/infra引用审计通过。服务器130 passed/48.40s，本地完整688 passed/5 skipped/107.24s。
- 本地重建完整评分因Python3.13给3条mbpp:124 NameError附加导入建议而失败；与服务器3.11的正确性/代码SHA完全相同，差异保存scoring-portability.json。未改原评分器或审计规则，改在原服务器环境运行`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -m scripts.write_asd_rmsnorm_validation_report`成功，重评105唯一文本。本地另复核纯统计/解释；GPU后冻结执行/统计代码未改。下一步合并已隔离测试的显式runtime profile并做单卡真实服务验收，默认eager/面试版保持。

## 2026-10-21 — 显式RMSNorm运行时接入（隔离准备）

- 独立验证仍在运行，冻结候选/协议未改；未提前宣称准入。另建artifacts/rmsnorm-runtime-worktree，分支codex/rmsnorm-runtime，候选655e70082ee4ec11b553897655789db13774d129；仅当validation预登记admission全部通过才合并。原main/infra面试分支与标签保持27e3619。
- 新execution_profile默认eager，显式rmsnorm-inductor限已测Linux/torch2.5.1/Transformers4.57.3/CUDA FP32；不支持明确拒绝。加载时绑定原norm forward，close恢复；lazy编译计原请求时钟，元数据分列加载setup和编译子阶段。单owner锁内校验关闭/改profile，保留stream提交与取消清理；benchmark拒绝跨数值profile回归比较。
- 修正validate_infra生命周期：原TestClient关闭后曾复用同一engine，现将SSE移到末尾实际HTTP服务唯一lifespan。新增空缓存冷调用记录、真实R接受/历史不变断言；复用既有benchmark/cancel/CUDA profiler/HTTP负载。run_rmsnorm_runtime_pipeline先查validation门槛和源码SHA、单卡串行验收与新进程CLI，独立缓存、每进程1800秒；不在两卡验证运行时叠加GPU作业。
- 相关runtime/serving检查72 passed/11.25s（新增pipeline反例前）；完整命令在隔离工作树：`E:\PycharmProject\MemoryDraft\.venv\Scripts\python.exe -m pytest -q --basetemp E:\PycharmProject\MemoryDraft\.test-tmp-rmsnorm-runtime-full-fixed --junitxml=E:\PycharmProject\MemoryDraft\artifacts\rmsnorm-runtime-tests-full-local-fixed.xml -x --tb=short`，711 passed/5 skipped/106.19s。新测试覆盖配置兼容/拒绝、取消/关闭、冷编译、失败释放、profile比较及六类无效验收记录；是CPU语义证据，不是Inductor GPU性能。
- 首次完整检查36 passed后因新工作树configs/asd-attribution-policies.json被Windows转CRLF触发旧报告SHA审计；核实Git blob正确后恢复未改文件为Git原始字节，未改旧报告或放宽审计，并补configs/*.json LF规则。失败XML保留artifacts/rmsnorm-runtime-tests-full-local.xml；仅提交11个实际修改文件，旧证据内容不变。真实GPU验收待验证门槛和卡释放后进行。

## 2026-10-21 — RMSNorm冻结候选独立验证（执行准备）

- 已启动：冻结2c8c51106cd01208756de5458abb9e46889f59be/1417 blobs，source SHA27188c05eb8a7f0b609f22b9109d43712a82627aba0bf4790c3b3458af2b0c5b，source ZIP SHAbe33ae68cf30d2da16419e7a24f17f885a418864f46d56099aa3cdcab12ee999。服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-validation，PID4169165，2026-10-21 11:53:36 UTC。最终结果尚待完成，不提前宣称通过。
- 用户授权持续推进。候选固定e958532的RegionalRMSNorm/CPU接受/模型配置/输入及评分11核心文件SHA，新增validation runner/pipeline/report。共用自然EOS框架增加可选新题对照与split钩子，开发默认行为保持；旧48题analysis内存逐字段复现，旧结果未改。
- 协议docs/ASD_RMSNORM_VALIDATION_PROTOCOL.md生成前登记：原validation48六类各8，reserve96不生成；1152相邻配对计时+96 AR质量对照，另20短warmup，4096上限。无旧答案，逐profile重复自身首轮；输入与本轮匹配eager/AR一致。首次编译计线上，不筛错答或截断。
- 候选源码校验通过；本地初始隔离扫763结果文件/15521生成载荷，validation与reserve命中0，服务器生成前再查。ID集合SHA及扫描范围/限制见协议；不声称外部预训练数据隔离。未读取验证生成结果来改参数。
- 显式profile工程门槛冻结：240完整输出对全部token/text/EOS相同，无新增主/secondary/F1/未评估/截断退化，两profile严格D/AR均48/48，四方法整体95%时延比上界<1及六类点估计<1，并通过资源/源代码/隔离/重复审计。未通过保留研究工具，不在validation调参；通过也不称任意输入无损或统计质量非劣。
- `.venv\Scripts\python.exe -m pytest -q tests/test_asd_rmsnorm_validation.py tests/test_asd_rmsnorm_suite.py --basetemp .test-tmp-asd-rmsnorm-validation-first --junitxml=artifacts/asd-rmsnorm-validation-tests-first-local.xml -x --tb=short`：33 passed/35.95s。完整回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-rmsnorm-validation-full --junitxml=artifacts/asd-rmsnorm-validation-tests-full-local.xml`：688 passed/5 skipped/107.24s，只有既有python_multipart提醒。提交并导出Git blobs。
- 独立/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-validation，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_validation_pipeline`；每worker7200秒，图8/backend300秒/allocated46GiB/cache4GiB。根据结果决定后续显式运行时配置及stream/cancel/检索路径工程检查，默认infra保持。

## 2026-10-21 — RMSNorm完整48开发题自然EOS完成

- 源码冻结ca51a6338b039b557b80cb919be4ec15c98a9dd3/1383 blobs，source SHA9502d8b40a9229d6fd105d97ea8b5ff049e0f2c489e14bf311103809779f4c86、source ZIP SHAfeeb0faa047a386668c605af7e809c02e4cd8fe81702f66dcbc9e11755cdcdd0；服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-suite，PID4069094。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_suite_pipeline`；09:10:55 UTC启动，完成日志最后写入09:41:59。两worker0退出，REGIONAL RMSNORM_SUITE COMPLETE，前后1383源码SHA相同，两卡恢复19MiB/0%。用户额度中断期间后台继续完成，没有重跑。
- 六类各8开发题，原FP32最高精度/TF32关闭、eager attention、16块、相同CPU接受；只RMSNorm编译，MLP仍eager。1152正式相邻配对+96 AR质量对照=1248主要生成，另20短warmup；首重复同时评分，480质量条目不再加算生成。实际97唯一题/文本重新评分；validation48/reserve96未运行模型。
- 自然编译/eager题级时延比：D0.972056 [0.971392,0.972712]、ASD0.972249 [0.971607,0.972898]、Run0.972017 [0.971337,0.972641]、U0.971827 [0.971221,0.972385]，时延低2.78%–2.82%，各48/48题更快。新增profile评估36题分别0.972071/0.972108/0.971823/0.971673；两组都属开发，未称独立验证。六类点估计均改善，SQuAD约1.8%，其他约2.8%–3.3%。
- 240跨profile完整输出全部逐token/文本相同；eager240首轮重现旧完整48题轨迹/评分，所有重复精确重现；两profile D/AR各48/48。54对账本浮点不同，最大spent差9.5367431640625e-5；5058轮除regret外一致，长度/调用密度不变。无主/secondary/F1新增变化，仍有原MATH待复核和Utility math500:444到4096截断，不宣称全答对或全输入无损。
- 167norm（Target145/Draft22）每卡4图，warmup至结束不变，正式及AR对照0新增编译。outer3.829/3.839秒、backend3.434/3.452秒（嵌套），controller约0.0036/0.0039秒；warmup wall13.22秒另含推理。cache各414736字节/0.396MiB，峰值allocated两profile均35.964GiB；切换中位0.345ms计在线。CPU各11361 fast/fallback0。当前自然混合平均省71.73/60.88ms，条件摊销outer+controller约54/64请求，不作通用ROI保证。
- 运行后只读8个生成文件各一次融合norm kernel、无外部GEMM，未重复微回放；解释与前轮192状态相符，不虚构纯kernel占比。收益是DFlash/ASD共同执行选项，不相加前轮CPU/MLP百分比、不改旧策略结论。冻结当前候选，下一步预登记进入未生成答案的48验证题；默认infra保持原实现。
- 结果ZIP863831字节/SHA6eeef14c8bc0ded8683acf1e4b2d2b62d6ccf17d153ce5c26b13b21ba5fb7bd2；results/asd-rmsnorm-suite，表docs/ASD_RMSNORM_SUITE_RESULTS.md，解释docs/ASD_RMSNORM_SUITE_FINDINGS.md。重建`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_suite_report`；后描述`.venv\Scripts\python.exe -m results.asd-rmsnorm-suite.inspect_results`；后审计`.venv\Scripts\python.exe -m results.asd-rmsnorm-suite.audit_postrun`。21冻结文件/919旧SHA/7批254发布文件/7旧runtime与infra引用通过，仅两个既有CRLF例外。
- 本地相关37 passed/29.32s，完整671 passed/5 skipped/90.12s；服务器113 passed/30.19s。GPU后执行/主统计/评分源码未改，旧RMSNorm实际analysis内存逐字段重现，不重复已通过测试或GPU计时。只发布小型记录/报告，排除模型、hidden/logits及缓存。

## 2026-10-21 — RMSNorm完整48开发题自然EOS（执行准备）

- 已启动：源码ca51a6338b039b557b80cb919be4ec15c98a9dd3/1383 blobs，source SHA9502d8b40a9229d6fd105d97ea8b5ff049e0f2c489e14bf311103809779f4c86，source ZIP SHAfeeb0faa047a386668c605af7e809c02e4cd8fe81702f66dcbc9e11755cdcdd0。服务器独立目录/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-suite，PID4069094，2026-10-21 09:10:55 UTC启动。服务器113 passed/30.19s后启动两worker；warmup各10，开始完整自然输出。最终数据尚待完成，不能以局部计时宣称收益。
- 用户授权进入下一步。新增scripts/run_asd_rmsnorm_suite_probe.py、pipeline及report，复用RegionalRMSNorm、CPU接受和原评分器，不叠加MLP、不改原规则/输入。48开发题六类各8，原偶/奇GPU分配，validation/reserve不运行。协议docs/ASD_RMSNORM_SUITE_PROTOCOL.md冻结自然上限4096、三次配对；96 AR质量对照+1152正式计时=1248主要生成，另20短warmup。首重复同时评分，不能再额外加算或扣掉编译成本。
- 全部跨profile输出/质量变化保留；eager首轮精确重现旧完整48题原16块自然轨迹及质量，每profile后两次精确重现。评分在一对两次生成之后、engine时钟外，相同题/文本缓存；480质量条目是已有生成，非额外推理。单列质量主/secondary/待复核/未评估/F1与长度、轮数、成本分解；整体48/各类8题bootstrap，旧12/新增profile评估36预先登记为开发子集，不称独立验证。
- 按旧自然记录估算每卡约1813/1696秒，固定worker5400秒；图8/backend300秒/allocated46GiB/cache4GiB保持前轮，正式新编译仍计线上。独立/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm-suite；入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_suite_pipeline`。
- 先`.venv\Scripts\python.exe -m pytest -q tests/test_asd_rmsnorm_suite.py tests/test_asd_rmsnorm.py tests/test_asd_compile.py --basetemp .test-tmp-asd-rmsnorm-suite-first-fixed --junitxml=artifacts/asd-rmsnorm-suite-tests-first-local.xml -x --tb=short`：37 passed/29.32s。首版测试发现原生DFlash轮记录不含regret字段，审计器改为按原native结构读取，解码/接受不改。之前RMSNorm实际analysis在内存逐字段复现，旧文件未改。
- 完整回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-rmsnorm-suite-full --junitxml=artifacts/asd-rmsnorm-suite-tests-full-local.xml`：671 passed/5 skipped/90.12s，只有既有python_multipart提醒。提交源码、协议与统计，导出Git blob到独立服务器目录执行。结果、准确源码SHA和审计完成后另补。

## 2026-10-21 — RMSNorm-only完成（120预检/288计时/192同输入状态）

- 冻结39f99df5352796f7c2d02bc2b1d030cc19bd5657/1349 blobs；source SHA bbb852bff12a23779e7a1cad658396c5ae0c62c14ea38417faaf0c116e688d90，source ZIP SHA de57293d2ba00b1005cc42d1b62c72946e3670d865e6162998270b31b552dd54。服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm，PID4048957，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_pipeline`；两worker0退出、REGIONAL RMSNORM COMPLETE，执行前后1349文件SHA一致，两卡恢复19MiB/0%。
- 原六类各前两题，FP32 highest/TF32关闭、eager attention、16块、相同CPU接受，仅RMSNorm编译，MLP仍eager。120预检+288配对计时+12独立DFlash诊断，另20warmup；诊断覆盖16角色/调用×12题=192真实状态，原始/回放stride全部相同，各20次计时。不生成validation/reserve或新自然4096答案。
- 编译/eager题级时延比：DFlash0.973247 [0.972584,0.973876]、ASD0.973270 [0.972207,0.974349]、Run0.972470 [0.971559,0.973363]、Utility0.972586 [0.971686,0.973514]，时延下降2.67%–2.75%，各12/12题更快。六类点估计均改善，是固定开发子集结果；成本含策略构造、profile绑定/恢复（中位0.346ms），正式0新编译。
- 60跨profile对token全部相同，eager重现旧轨迹，每profile重复完全重现，同profile DFlash/AR各12/12；48非AR案例输出长度/调用密度相同，581轮除regret外一致。10对账本浮点变化，最大spent差9.5367431640625e-5；192模块输出13逐位相同，最大abs4.57763671875e-5/RMS6.55e-7。输入/规则未改，不宣称全输入或完整自然答案无损。
- 运行时发现167模块（Target145/Draft22），每卡共享4图，诊断后仍4图。backend3.463/3.607秒、outer3.839/3.962秒（嵌套不相加）、controller约0.0035秒；完整准备wall107.51/120.18秒还含推理。cache各412257字节/0.393MiB，正式allocated峰值两profile均35.158GiB，预算满足。当前请求混合平均省约31ms，仅outer+controller条件摊销约123/127请求，不作生产ROI保证。
- 同输入norm回放按角色约快5.1%–9.0%；运行后只读代码生成8文件（每卡4）各一次融合mean/pow/rsqrt/mul kernel、无GEMM。与相同轨迹在线对照共同支持norm执行收益，不虚构原版kernel次数/带宽或精确因果占比。它是DFlash与ASD共同的执行优化，不是Run/Utility独有算法贡献；不与前轮MLP/CPU百分比直接相加。
- 结果ZIP346034字节/SHA d310ffe98b9268b913de5bb1a61d4b46ebf6c7666e013c89e9b197340a527719；results/asd-rmsnorm，完整表docs/ASD_RMSNORM_RESULTS.md、解释docs/ASD_RMSNORM_FINDINGS.md。重建`.venv\Scripts\python.exe -m scripts.write_asd_rmsnorm_report`，后描述`.venv\Scripts\python.exe -m results.asd-rmsnorm.inspect_results`；后审计17冻结文件、892旧SHA、此前6批215发布文件、7旧runtime/接受文件及infra引用通过，两个既有CRLF例外保留。
- 本地完整655 passed/5 skipped/77.17s，服务器97 passed/17.53s。旧MLP实际analysis内存重算逐字段完全相同，原结果不改，面试infra引用仍27e36199。保留RMSNorm实验工具，停止12题norm扫参；下一项优先扩48开发题自然EOS质量/时延，之后再进入未观察验证集，不先叠加MLP。GPU后执行/统计源码未改，不重复已通过性能运行。

## 2026-10-21 — RMSNorm独立编译检查（执行准备）

- 按用户授权继续RMSNorm-only，不叠加MLP编译。共享可恢复区域绑定及既有测量/审计边界，新增RegionalRMSNorm、独立runner/pipeline/report入口。实际transformers4.57.3和DFlash源码已核验：隐藏norm、Q/K norm、Draft hidden_norm均为Qwen3RMSNorm；运行时记录全部模块名称/形状/eps。
- 冻结协议docs/ASD_RMSNORM_PROTOCOL.md：六类各前两道已观察dev，120预检/288正式配对/12独立诊断，192实际norm状态，另20短warmup；128上限尊重EOS。相同CPU接受实现、FP32 highest/TF32关闭、原16块；数值profile差异全部保留，每profile重复精确重现，输入/规则不改。8图/backend300秒/worker1800秒/allocated46GiB/cache4GiB，与前轮相同。
- 先`.venv\Scripts\python.exe -m pytest -q tests/test_regional_compile.py tests/test_asd_compile.py tests/test_asd_rmsnorm.py --basetemp .test-tmp-asd-rmsnorm-first --junitxml=artifacts/asd-rmsnorm-tests-first-local.xml`：26 passed/17.29s。本机Dynamo/eager只测捕获/恢复，非CUDA性能。旧MLP的实际raw重新在内存审计/计算，与已发布analysis.json逐字段完全相同，旧文件未改。
- 完整检查`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-rmsnorm-full --junitxml=artifacts/asd-rmsnorm-tests-full-local.xml`：655 passed/5 skipped/77.17s，只有既有python_multipart提醒。提交源码及统计、导出Git blobs到独立/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-rmsnorm。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_rmsnorm_pipeline`，运行前后源码/旧结果SHA及infra引用核验。完整结果另补，不能以测试通过冒充性能。

## 2026-10-21 — MLP区域编译完成（120预检/288计时/36同输入状态）

- 冻结`5aa8c79336656a6447f0f6b9b5e22b53e3fc0f77`/1316 blobs，source SHA`585549eb91372b7dd2c432881e3412b1e60de5b7638b33883ceb10ab27ac414e`、源码ZIP SHA`47c07f3af4ad01c6f745eac790626d3e36ffbb862acf49c279cbb33f31914842`。服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-compile` PID4026974，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_compile_pipeline`；两worker0退出，REGIONAL COMPILE COMPLETE，前后1316文件SHA相同。
- 六类前两道已观察开发题，原FP32 highest/TF32关闭、eager attention/16块、相同CPU接受。120预检（含24 AR）+288正式计时+12独立DFlash诊断，共420生成，另20短warmup；捕获Target首层prefill/首验证及Draft首层首起草36输入，同一clone各5warmup+20次回放。不生成自然4096/验证/保留答案。
- 编译/eager题级时延比：DFlash0.997100 [0.996158,0.998072]、ASD0.997401 [0.996232,0.998572]、Run0.997976 [0.997406,0.998552]、U0.998262 [0.997497,0.999027]，下降0.290%/0.260%/0.202%/0.174%。固定开发子集的分层bootstrap，不是通用生产保证。各11/12题更快，48非AR案例输出长度/调用密度不变，581轮除regret外相同；构造与切换/恢复均计在线。
- 60跨profile预检对全部token相同，严格D/同profile AR各12/12，eager重现旧轨迹、每profile重复精确重现；13近似账本浮点数不同，最大spent差6.103515625e-5，36模块输出均非逐位相同，最大绝对差4.8828125e-4/RMS4.46e-6。未放宽接受规则，不升级默认infra或宣称全输入无损。
- 每卡37模块共享2图，正式计时0新增编译；outer3.183/3.562秒，backend2.849/3.232秒（嵌套不相加），controller约0.0035秒，完整准备wall108/121秒另含推理。独立缓存各195032字节，编译组正式峰值allocated35.140GiB；预算均满足，两卡结束19MiB/0%。按当前混合请求平均省时粗估1276/1811请求摊销outer+controller，不作生产ROI保证。
- 同输入MLP回放prefill/验证/Draft编译/eager分别1.0032/1.0255/1.0242，没有模块本体显著变快的证据。运行后只读codegen-inspection确认每图仅SiLU*mul融合、3次GEMM仍在；这解释有限空间但不唯一归因带宽，异步流水/内存与包装因素未单独干预。不继续MLP-only扫参，下一项单独检查RMSNorm，其他执行/精度/训练方向仍未做完。
- 结果ZIP275781字节/SHA`07cf61ae4a6016a1ec0476714f1314a57df7757e6f3797b17db72c382d2394bd`；结果results/asd-compile，完整表docs/ASD_COMPILE_RESULTS.md、解释docs/ASD_COMPILE_FINDINGS.md。重建`.venv\Scripts\python.exe -m scripts.write_asd_compile_report`；后审计`.venv\Scripts\python.exe -m artifacts.audit_asd_compile_postrun`通过13冻结文件、866旧SHA、此前181发布文件、7原运行时/验证文件和infra引用，两个既有CRLF例外单列。codegen为主运行后附加，不改冻结统计。
- 本地完整645 passed/5 skipped/69.61s；服务器87 passed/13.74s。GPU后执行/统计源码未改，不重复性能复验；发布小型trace/测试/报告，排除模型、完整hidden/logits及编译缓存。

## 2026-10-21 — Target/Draft MLP区域编译（执行准备）

- 用户授权进入主要模型执行优化。新增可撤销RegionalMLP，只torch.compile现有Qwen3MLP.forward，Target/Draft共享按类编译入口；fullgraph/dynamic、关闭CUDA Graph/max-autotune、FP32 highest和原eager attention。原解码/默认infra不改，四策略近似组三者均沿用相同CPUASDVerifier。依据PyTorch2.5区域编译说明及v2.5.1源码，服务器Triton3.1.0/Inductor配置可用，不新增依赖/权重下载。
- 协议docs/ASD_COMPILE_PROTOCOL.md先冻结：六类各前两题，120预检（含24AR）、288正式相邻配对计时、12独立DFlash诊断/36实际MLP输入，另20短warmup。保持每profile自己的重复轨迹，跨profile输入相同、token/账本差异如实报告，不能因编译数值变化删题或称无损。策略构造及切换成本加回线上；正式重编译也计入且标记。
- 每worker最多8图/backend300s、总1800s、allocated46GiB、准备cache4GiB，独立缓存/两编译线程；超时仅终止新建worker进程组，保留失败。准备wall含推理与编译，backend和outer计时不相加。
- `.venv\Scripts\python.exe -m pytest -q tests/test_regional_compile.py --basetemp .test-tmp-regional-compile --junitxml=artifacts/asd-compile-tests-first-local.xml`：5 passed/9.82s；新统计/轨迹/计时边界检查相关16 passed/13.21s。多个真实MLP实例的Dynamo/eager图捕获/参数不复制/恢复通过，但这不是CUDA Inductor性能证据。
- 冻结前完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-compile-full --junitxml=artifacts/asd-compile-tests-full-local.xml`：645 passed/5 skipped/69.61s，只有既有python_multipart提醒。将源码/协议/统计一起提交，再导出Git原始blob在授权服务器独立目录执行。

## 2026-10-21 — 短块CPU判定完成（420计时/430影子块/77状态）

- 冻结`c75328fa8e03da09a262754a2e9e9207d6bdc5e2`/1285 blobs，source SHA`9df10dd95a6fbfcfbc7b31730635bd36358a73b59fb3c6059a2e1197d39506c1`、源码ZIP SHA`3ecc20f11788fb82bb53c5286307ff511e32a559408df2c1bb330cff7ca4bd2c`。服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-cpu` PID4004961，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_cpu_pipeline`；两worker均0退出、CPU VERIFIER COMPLETE，运行前后1285源码SHA相同，两卡最后19MiB/0%。
- 原六类前两道开发题，7组/5重复/EOS前128，共420正式计时；36独立生成的430实际块逐轮持久CPU影子decision/summary精确相等，77真实0/4/12轮resident状态各Tensor/CPU 5warmup+30次；14生成warmup单列。456生成全量重现旧输入/轨迹/预算/停止原因。2150次正式CPU判定及430影子块均fast，fallback0，真实候选EOS独立标量审计通过。
- CPU/原Tensor题级配对几何比：ASD0.996963 [0.996007,0.997900]、Run0.993261 [0.992593,0.993935]、Utility0.993639 [0.993130,0.994148]，下降0.304%/0.674%/0.636%。区间为固定六类各2题的分层bootstrap，不视为独立验证或生产保证。每类两题只列点估计；SQuAD基础ASD略慢0.120%，其他多数小幅改善。构造成本另测并计入主时延，全部摘要复制/精确性检查/判定在线计入。
- 同轮resident原Tensor约0.771/0.872/1.053ms，CPU约0.157/0.165/0.160ms，约80%–85%成本下降；不是生成快5–7倍。原版跨轮绝对成本较前阶段低，未做单因素归因，收益仅采用本轮交错配对。生成主要成本及验证轮数没变，与上一阶段约百分之一量级空间一致。
- 原验证器、Target/Draft、解码/评分/默认infra均未改，CPU为显式实验模块，边界回退单元覆盖。未生成新的4096自然质量/验证/保留答案，不把128相同推广为全输入或无损ASD。保留这个小幅执行选项，停止验证器扫参，下一阶段主要做模型编译/融合；专用训练/多候选仍未实现。
- 结果ZIP323902字节/SHA`5396f837f69a94d3c004f4cdabcd139a9ce1f8baf97d4dc197318f51d0a7786f`，结果results/asd-cpu，完整表docs/ASD_CPU_RESULTS.md，解释docs/ASD_CPU_FINDINGS.md。重建`.venv\Scripts\python.exe -m scripts.write_asd_cpu_report`，后审计`.venv\Scripts\python.exe -m artifacts.audit_asd_cpu_postrun`：13冻结文件、844旧结果SHA、原四批151发布文件及infra三引用通过，只有两个既有CRLF例外。
- 本地完整629 passed/5 skipped/70.22s；服务器98 passed/9.46s。GPU后执行/统计/评分源码未改，不重复已通过性能复验；小型原始计时、完整候选trace、测试和源码审计发布，不上传完整词表logits/权重。

## 2026-10-21 — 短块ASD CPU判定精简（执行准备）

- 用户授权继续上一阶段的小成本检查。新增显式CPUASDVerifier：一次传输4K FP32摘要，CPU完成原ASD/Run/Utility规则；整数ID范围、FP64精确求和格点检查及不支持设置回退原Tensor。Target、解码、默认验证器与infra不改，不新增预算/接受容差。
- 协议docs/ASD_CPU_PROTOCOL.md：仍六类各前两道开发题，7组/每题5次，420正式计时，原生DFlash完整保留；同策略Tensor/CPU相邻、随机先后。36独立生成逐轮精确影子比较，捕获原0/4/12轮用于resident回放。验证/保留题不生成答案。策略构造单列并计入主时延，摘要复制/精确性检查/回退均在线计时。
- 最初`.venv\Scripts\python.exe -m pytest -q tests/test_asd_cpu.py tests/test_asd.py --basetemp .test-tmp-asd-cpu-first --junitxml=artifacts/asd-cpu-tests-first-local.xml`：37 passed/9.05s。加入本机3090真实CUDA参考比较后`tests/test_asd_cpu.py`：29 passed/3.94s，逐轮summary/events与GPU Tensor相同；相关55 passed/11.63s，追加构造成本审计后报表12 passed/7.68s。这里只是单元检查，尚无在线速度结论。
- 冻结前完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-cpu-full --junitxml=artifacts/asd-cpu-tests-full-local.xml`：629 passed/5 skipped/70.22s，既有python_multipart提醒。下一步在独立服务器目录运行，源码/协议/配对统计均先提交冻结。

## 2026-10-21 — 原生16块执行成本诊断完成（144计时/48profile/77状态）

- 冻结`4290e2b7a6277709f8c8b90517ea2bdbe93cbabe`/1251 blobs，source SHA`6f36eda11a904354000cc50f50e58c211e1b146bcf9fdf1e92c5c71350a13eaf`、源码ZIP SHA`01ce9e139edf76608260c9b9ed46c1ec12dd5bea90f832a02c18617ce89e1f33`。服务器独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-execution` PID3983505，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_execution_pipeline`；两worker均退出0，EXECUTION COMPLETE，运行前后全部部署SHA相同。
- 六类各前两道development，原四策略/16块/FP32/eager/EOS前128；144正式计时、48独立CUPTI、36捕获，共77真实状态（worker0/1各38/39）。每状态5warmup后策略/严格核心/空同步各30回放；8短生成warmup另计。228生成全量重现前阶段16块输入SHA、snapshot、文本/token、轮次、预算及停止原因。短回答不补轮、不续写，本轮未重做自然质量，验证/保留题仍未生成模型答案。
- 模型三范围占观测GPU活动99.6817%–99.9659%，不是总时延占比。ASD/Run/Utility resident跨题中位约1.081/1.221/1.487ms，严格核心约0.129ms；按样本轮数摊销proxy中位1.185%/1.329%/1.625%，全范围0.158%–1.948%，不是可兑现速度预测或上限。在线decision host约5–6ms包含约3.3–3.7ms前序GPU同步等待，不能直接归给判定算法。
- 基础ASD数学GPU验证约75%；两道SQuAD预填充约81%，只输出2/4 token各一轮，解释接受策略收益小。HumanEval134的ASD/U轨迹100/107 token、23/22轮，对D的82 token/19轮，精简验证器不能消除这部分增长。12题只用于定位，48题自然结论继续作为主依据。
- profiler膨胀中位1.124倍，不进入速度统计；23份含30个未关联CPU设备事件，共343.868us，单份最大约0.0204%。事后仅修正报表硬等号检查，显式保留unlinked残差及测试；GPU执行/样本/回放/协议未改。报告与解释docs/ASD_EXECUTION_RESULTS.md、ASD_EXECUTION_FINDINGS.md；重建`.venv\Scripts\python.exe -m scripts.write_asd_execution_report`。
- 结果ZIP317827字节/SHA`a6785cc0fea5a808f55077d85dc9959dd523f28bd73c61bd37cf57ab116c6eef`，小型原始结果results/asd-execution。后审计`.venv\Scripts\python.exe -m artifacts.audit_asd_execution_postrun`：819旧结果SHA、原三批118发布文件、10冻结执行/协议文件及infra三引用均通过；去mark后解码AST与d72f6b6完全一致。没有上传完整logits/大trace/权重。
- 服务器69 passed/10.71s；报表残差相关13 passed/9.93s；最终完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-execution-final --junitxml=artifacts/asd-execution-tests-final-local.xml`：588 passed/5 skipped/63.00s，既有python_multipart提醒。后续先做一次小成本验证器精简；主要执行投入转Target/Draft编译融合，训练/多候选方向仍未完成。
- 发布逐Git blob审计发现Windows生成Markdown的CRLF与Git LF字节不同；报告写出改为显式UTF-8/LF，重建原审计及发布SHA。只改生成文件换行方式，不改统计结果；追加`tests/test_asd_execution.py`发布检查13 passed/9.31s，33个发布文件按最终Git blob逐字节核验后推送。

## 2026-10-21 — 原生16块执行成本诊断（执行准备）

- 下一阶段固定原四策略与六类开发题前两题，共12题；原生16块、FP32/eager、EOS前128。144次非插桩重复计时、48次独立CUPTI profile、36次独立状态捕获（至多108状态），验证/保留题不生成答案。协议docs/ASD_EXECUTION_PROTOCOL.md，接受判定/KV/发布与停止新增范围仅在profile=True启用，默认规则未改。
- 本地相关命令`.venv\Scripts\python.exe -m pytest -q tests/test_asd_execution.py tests/test_asd.py tests/test_dflash_reuse.py --basetemp .test-tmp-asd-execution --junitxml=artifacts/asd-execution-tests-local.xml`：61 passed/2 skipped/11.82s。覆盖真实tiny Target插桩轨迹一致、设备annotation去重、直接kernel归属、重叠区间并集、缺CUPTI失败、捕获预算与数量边界。
- Windows3090 profiler烟测缺少CUPTI kernel链接，按预定检查明确失败，不报告零GPU成本；Linux A6000同一烟测取得真实kernel和DtoH事件/链接、CUDA annotation与同步事件。正式诊断仅在Linux原运行时进行；两卡初始19MiB/0%。没有新的模型下载或算法性能结论。
- 全回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-execution-full --junitxml=artifacts/asd-execution-tests-full-local.xml`：581 passed/5 skipped/65.02s，既有python_multipart提醒；随后补报告审计篡改测试并运行`tests/test_asd_execution.py`：12 passed/9.15s。部署冻结包含统计/审计脚本，不以插桩延迟宣传提速。

## 2026-10-21 — 真实长块完整48题完成（3120计时/144机制）

- 用户额度恢复后接回原实验，没有重启：双worker均正常结束，各1560计时/72机制；冻结`b107e53213137268415d44bab0e954ae704fd727`/1216文件前后SHA一致。原目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-suite` PID3884160，pipeline.log结束BLOCK SUITE COMPLETE，两卡各19MiB/0%。仍是六类各8开发题，验证48/保留96没有模型生成。
- Utility24/ASD16自然时延题级几何平均变化：GSM+4.07%、MATH+8.68%、MBPP+16.41%、HumanEval+2.86%、SQuAD+4.18%、合成JSON+8.29%。共同等128的GSM7题/MATH8题/JSON6题分别慢4.14%/15.89%/13.29%；短代码和问答无共同128，不强制续写。每类仅8题，部分自然区间含1，保留全部原始比较，不宣称所有差异都已统计确认。
- 预检正例保留：GSM885 Utility24/Utility16等128仍快13.68%；MATH329快23.31%，但该题Utility24/ASD16慢10.88%。另外42道非预检开发题的Utility24/ASD16六类自然点估计均更慢，约+2.72%至+21.27%。它们不是独立验证数据，不能把重复或方法数算成更多题。
- 长块未持续减少调用，还增加摊销成本。Utility24/ASD16在GSM密度-0.97%但摊销+4.77%；MATH密度+2.75%/摊销+5.48%；MBPP密度+14.68%。独立同token前缀探索：24块773状态中更早拒绝190/更晚82；32块724中更早226/更晚76。机制记录可观察的共同候选ID改变149/164及145/158；状态相关、隐藏特征未逐位核对，不唯一归因训练长度外推。
- Utility24最终主正确数与ASD16各类相同；基础ASD32额外错MATH448（49而非44，17*49 mod83=3）及MBPP504（sum_of_series而非测试请求的sum_Of_Series）。MATH444全13组错误表达式以k=3辅助复核，Utility16仍截断；不重命名输出、不改主评分。代码无受限未评估项，所有预算守恒。
- 重建`.venv\Scripts\python.exe -m scripts.write_asd_block_suite_report`通过：全部3120矩阵/重复/质量重算，16块含AR1200行重现，严格D三长度自然/128共288对与AR一致，自然/前缀裁短前无分歧；144机制3294块完整候选/真实EOS独立标量核验。3264行输入SHA/token数/隔离空snapshot经服务器CPU重新tokenize一致。17冻结执行/统计函数未改，791旧结果核验（两个已知CRLF例外），前两批83发布文件按发布提交重核，infra三引用仍27e3619。
- 首批15 passed/22.57s、相关30 passed/1 skipped/11.07s、完整574 passed/5 skipped/61.97s；服务器87 passed/8.87s。GPU后只有解释文档与独立离线诊断，没有修改执行、评分或冻结统计函数，不重复同一性能复验。
- 结果ZIP2302258字节/SHA`f6473e15fabfad055bf051fd54cbb79cc11009ded5de0c98678a0e59b58115bb`，原始trace/输入与源码审计/质量复核位于results/asd-block-suite；完整表docs/ASD_BLOCK_SUITE_RESULTS.md，解释docs/ASD_BLOCK_SUITE_FINDINGS.md。预览GSM与最终汇总一致，192比较成本恒等式通过。后加diagnostics为运行后探索，与预注册主分析分开保存。
- 当前b16权重直接外推24/32不升级，不继续长度/价格扫描，也不消费验证池确认负候选。下一阶段先用独立短profile和非插桩对照量化执行开销再设计优化；专用训练、多候选、调度与并发仍未完成，不能据此否定整条方向。

## 2026-10-21 — 真实长块完整48开发题（执行准备）

- 用户授权下一步，保留六类各8题及D/ASD/Run/Utility三长度全部对照，再加AR。自然三次/遵守EOS前128两次，共3120计时，另144机制；验证48/保留96不生成模型答案。两卡空闲且磁盘405GB，继续只读使用原模型，无新增下载/训练。
- 新独立runner/pipeline复用原experimental_block/run/execute，不改旧预检或生产推理源码；全部16块继续与第一阶段输入/轨迹/账本/质量重现。记录完整候选及Target top IDs，只在额外机制运行开启，独立标量检查使用实际EOS而非缺失位置枚举。
- 协议docs/ASD_BLOCK_SUITE_PROTOCOL.md在GPU前冻结，3120次不是3120独立题。各类别、截断/错答、长度/调用密度/摊销成本、128有效样本、重叠六题与其余42题均分开报告，不据本批选择最终赢家或扫描参数。
- 本地最先检查`.venv\Scripts\python.exe -m pytest -q tests/test_asd_block_suite.py --basetemp .test-tmp-asd-block-suite --junitxml=artifacts/asd-block-suite-tests-first-local.xml`；通过相关检查后再冻结提交/部署，尚无本轮GPU结论。
- 首批15 passed/22.57s；加入独立完整候选审计、EOS/预算/矩阵篡改及重叠分组检查后，相关30 passed/1 skipped/11.07s。配对估计复用原5000次题级bootstrap，并在GPU前冻结Utility24/ASD16及Utility32/ASD16跨块强参照；短答案不强制续写。
- 完整回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-block-suite-full --junitxml=artifacts/asd-block-suite-tests-full-local.xml`：574 passed/5 skipped/61.97s，只有既有python_multipart提醒。GPU部署将复查固定DFlash真实tiny KV/位置边界及新审计。
- 已部署并运行：冻结`b107e53213137268415d44bab0e954ae704fd727`/1216 blobs，source SHA`cdf22b442b5aff8c98059b480b36eb8047daa5e7aa9c71eed646c3d19ea3d481`、源码ZIP SHA`21c8c0d730af72fef5ad5b416cae96354ba3cb48f75a1e6afe9c9a80f30c9c7f`。服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-suite` PID3884160，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_block_suite_pipeline`。服务器87 passed/8.87s后双卡启动；每卡1560计时/72机制，完成后取父目录asd-block-suite-results.zip及pipeline.log。当前仍在运行，不重启或覆盖目录。

## 2026-10-21 — 真实更长Draft预检完成（228次，继续完整开发集）

- 冻结`852a08e0112252799021a4e50319c3f140437deb`/1184个blob，源码ZIP SHA`f2e2a37fe2bbe65c6376703c04070e2aaba0218916505e720c1cf1d80d29cfe2`，source SHA`77fdea7117fca6d3b97912b38376cd7e4783eb9425e753ba4671cc8092e064a5`。独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-block-preflight` PID3852852，入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_block_pipeline`。两卡各3题/114行，228全部完成，结束各19MiB/0%。
- 六类各首个开发题，真实16/24/32、四原策略，自然1次/前128两次、AR自然/128各1次。全部78条自然输出答对并EOS；这批原本就容易、每类只有1题，不能证明质量非劣，也不作为适用场景确认。16块含重复84行与第一阶段相同；严格D三长度自然/128共36对均与同轮AR逐ID相同，自然/前缀在裁短块之前无分歧。
- 长块并非统一变好：严格D24/32在本批六题自然时延均更慢；GSM885的Target调用38→40/41，JSON12为11→12/13。Utility24在GSM885/MATH329等128时间相对自身16低14.12%/23.05%；Run24在两题一退一进，基础ASD24为快10.76%/慢55.89%。长块同时改变预测及近似门控/路径，不能直接解释为Draft预测变准；完整数值由报告重建，不由1题/类选赢家或否定方向。
- 服务器69 passed/8.76s，包括固定DFlash源码的tiny结构前向16/24/32及eager/SDPA；这不冒充真实8B预测质量。最终本地`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-block-final --junitxml=artifacts/asd-block-tests-final-local.xml`：556 passed/5 skipped/58.75s；报告审计2 passed/6.34s。初始测试参考错误的失败XML保留，之后按实际近似前缀重prefill核验。
- 执行前后源码SHA一致、768旧结果核验、原执行6函数未改；第一阶段54发布文件按其发布提交dcfc01d重核，路线表为后续更新的活状态文档。infra三引用仍27e3619。结果ZIP192974字节/SHA`ab9bde15246e55f548411d0f8604c2e3d7511ec6a316d039632e00f5b2d69835`。
- 原始压缩trace/provenance/测试和审计在results/asd-block-preflight，报告docs/ASD_BLOCK_RESULTS.md，重建`.venv\Scripts\python.exe -m scripts.write_asd_block_report`。新报告为执行后分析，原runner/Target输入/评分/生产路径未改变。
- 下一阶段保留16/24/32和全部四策略，将比较扩到同样六类48道开发题并增加自然计时重复；验证48及保留96继续留出。暂不根据当前两道数学题筛选类别、改预算或选24赢家。设备账本、调度、多候选、训练和并发仍未完成，按路线表逐项继续。

## 2026-10-21 — 继续阶段1：真实16/24/32 Draft预检（准备）

- 第一阶段已推送`dcfc01d`，继续更长真实Draft单因素方向。协议ASD_BLOCK_PROTOCOL.md固定六类各第一个开发题/三长度/原四策略，自然1次/前128两次，另AR自然/128各1次，共228生成；每类1题仅结构和成本预检，不作为方向否决或新适用场景确认。
- 研究context临时覆盖解码block_size属性并恢复，训练config仍16；生产适配器/RuntimeConfig不改。每个长度真实前向；16块与上阶段token/账本/轮次/输入完整重现。更长块同时影响Draft预测与ASD门控取值，同长度严格对照保留。
- 初次单测将近似前缀与严格AR特征比较，3项失败（8通过/1环境跳过），这是测试参考错误；改为重prefill实际已提交前缀后11 passed/1 skipped。初始失败XML保留。新增长度边界检查后再冻结执行；真实tiny DFlash结构在服务器核验，不能用随机小权重单测宣称质量或性能。

## 2026-10-21 — 跨任务基线第一阶段完成（48新题，1200计时）

- 新池192题/六类，48开发题进入本轮，48验证和96保留题尚未生成模型答案。五方法自然EOS上限4096三重复、遵守EOS前128两重复，共1200计时；另48机制回放单列。数据/许可/来源/划分固定，前192参考预检故障与修复均保留。
- 实际执行冻结`be1251b24f7a53f38fd2e817d68d62d4ab8e30d6`，1147源码blob，source SHA`57f1e8fd9d0e8d6d38213abec2601bc430ade8bfeee2e753632653f5c3ddcbee`，源码ZIP SHA`cdfee8f529d67c895b59a19a248657fe3eaeea59e94ad7de0aefc42caee249e7`。服务器独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-suite-stage1` PID3824147正常完成，运行前后源码一致，737旧证据核验通过（两个既有CRLF例外），infra三引用仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。
- 精确GPU入口：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_suite_pipeline`，两子进程固定`CUDA_VISIBLE_DEVICES=0/1`，各600计时/24机制，命令和分片记录见pipeline.log。结束两卡各19MiB/0%。结果ZIP683670字节、SHA`4550bfdc066bb2e2e7ec483008f94c6cb40f2072f36d42a64cb7f09e72f754a8`，压缩trace无损保存，模型权重/原整库未上传。
- 严格D相对AR题级几何平均加速GSM3.56×、MATH5.14×、MBPP3.23×、HumanEval3.90×、SQuAD1.34×、合成JSON6.97×；完整自然/128共96对token一致。基础ASD在GSM/MATH等128时延低11.45%/18.80%，本批Run/Utility未进一步改善。SQuAD/JSON各8题均零近似支出、轨迹未变；代码组均不足128，未强制续写。
- MATH各方法7/8正确，Utility在错题444截断4096，其他方法正常结束但变量表达式错误；保留全部主结果，人工辅助复核不改主评分。HumanEval AR/D/Run6/8、ASD/U7/8；134实现变化修复提供测试，60重复docstring导致39→112 token，各组仍通过。不能把计算效率、输出缩短和质量变化混为一个收益。
- 自然重复token/轮次/账本一致，题级bootstrap5000次。1200行Target输入SHA/token数/隔离空snapshot经CPU重tokenize通过；48机制回放相同，1064块标量相容性通过，缺失未提交候选ID的EOS位置限制明确保留。480组逐题重复秒数/IQR、错答、截断、质量升降、共同EOS诊断、数值检查在results/asd-suite-stage1。后加分析仅增强分解/审计，配对估计、聚合、输入、评分和执行12函数与冻结相同。
- 检查：服务器70 passed/7.71s；报告审计7 passed/7.05s；最终`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-suite-final --junitxml=artifacts/asd-suite-tests-final-local.xml`为542 passed/4 skipped/56.53s，只有既有python_multipart提醒。重建`.venv\Scripts\python.exe -m scripts.write_asd_suite_report`，完整表docs/ASD_SUITE_RESULTS.md，解释docs/ASD_SUITE_FINDINGS.md。
- 这只是阶段0完成，不将旧小样本结果当整个方法已否决。下一项是真实16/24/32 Draft结构/KV预检，再做同长度方法对照；其余设备账本、调度、多候选、训练和并发仍依路线表逐项推进。

## 2026-10-21 — 按用户要求建立跨任务研究基线（执行准备）

- 用户要求更丰富数据、细致逐项评估，允许慢慢推进。新增ASD_RESEARCH_ROADMAP.md逐项区分未做/仅诊断/局部初测，不再把4/8旧题负结果当整个方法已否决。第一阶段仅补六类基线，下一阶段按单因素推进更长Draft、GPU执行、调度、候选结构、训练和并发。
- 固定新池192：GSM8K/MATH-500/MBPP/HumanEval/SQuAD/合成JSON各32，8开发/8验证/16保留；仅开发48进入本轮GPU。GSM排除旧0..31，MATH排除0..23并覆盖7学科，MBPP test19..510，HumanEval保守排除0..15且核对旧Spec-Bench未见相同函数签名，SQuAD每文章一题。数据来源/提交/哈希与各源许可记录在data/research/asd-suite。
- 获取新源时GitHub API公开限流403，改用git ls-remote固定HumanEval b9453e8f1865df0a0057a1383767caa400acfbcf、SQuAD eee5fdbf62f8613a7812b03419e6b29617b74fd1后下载原blob。MBPP沿用f82046ba5aabbbb427dbfd38a254d26bff08b533。原文件位于artifacts/asd-evaluation-sources，不上传整库/权重。
- 新评分模块asd_suite.py：代码提供测试、SQuAD EM/F1、严格类型JSON、MATH保守主评分及有限标量等价。原192参考预检191通过，MBPP124因cmath被白名单阻挡；补齐普通算法模块后全参考通过，不删题，初始失败JSON保留。旧评分器/旧结果不改。
- 冻结协议ASD_SUITE_PROTOCOL.md：原模型FP32/eager、greedy、batch1、非思考，AR/DFlash/ASD/Run/Utility五组；48题自然3次/遵守EOS前128两次=1200主生成，另48机制，全部自然上限4096，两个物理GPU按每类奇偶分片。AR也完整生成，不拿短AR对照完整D；统计按题、不把重复当更多质量样本。
- 本地准备`.venv\Scripts\python.exe -m scripts.prepare_asd_suite --sources artifacts/asd-evaluation-sources --output data/research/asd-suite`；评分/隔离测试`.venv\Scripts\python.exe -m pytest -q tests/test_asd_suite.py --basetemp .test-tmp-asd-suite-first --junitxml=artifacts/asd-suite-tests-first-local.xml`：11 passed/4.31s。后续相关检查加入bootstrap、重复合并及最后裁短块边界审计。

## 2026-10-21 — ASD剩余优化排查完成（420主生成，执行组合有收益）

- 首轮冻结`37f15d0ff9842f87e87cb208c621f38c2dac2bec`/1035文件，追加归因冻结`389737323d5e88e3284be979f8cd4203f784d2ac`/1038文件，均执行前后逐blob SHA一致。首轮独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-optimization` PID3777460、追加`.../asd-optimization-extension` PID3787125均结束，两卡19MiB/0%。
- 8道已观察任务，A204、冻结迁移72、看到BF16正信号后另立协议的归因144，共420主生成；另32单token严格分支、28AR128、6独立profiler，短warmup单列不计主结果。自然每臂每题仅1次，前128两次，不能作稳定性或质量非劣结论。追加不是原预注册，不把重复旧题当新增样本。
- 执行组合：BF16/SDPA的D-fast等128秒较FP32/eager下降38.30%（开发4题，14.876269→9.178037）、41.19%（迁移4题，15.669007→9.215361）。同BF16基础ASD较严格D另降11.73%/21.11%；这些属于组合执行与基础ASD，不能当作新增长度或Utility独有收益。
- FP32与BF16各6个长度臂全部未达冻结门槛。BF16同轮固定短验证慢15%–19%、短Draft慢25%–29%、近期/预算动态长度约慢1%，停止参数搜索。增量EOS逐ID/账本保持、无确认时间收益，默认仍旧路径。Utility相对基础ASD等128开发快2.82%、迁移慢4.18%，新增规则无稳定独立增益。
- BF16/eager归因的D/ASD/U等128秒9.636287/7.633023/8.531475；同轮BF16/SDPA9.071516/8.028093/7.820026，轮数与自然轨迹可随attention改变，不作纯内核因果分解。BF16/eager自身已明显快于原FP32，支持精度变化是执行收益的重要来源。
- 主BF16-SDPA数值检查6/8题与同profile AR128逐ID相同；math500:7第118、gsm8k:25第9出现分歧。追加BF16/eager math500:4第63也有分歧，短验证还可改变数值轨迹；全部保留，不作为无损profile晋升生产。生产RuntimeConfig、默认严格FP32/eager与面试infra引用保持。
- Draft检查16对/15等128对：第一块严格接受92/88，后续轮268/281，未见近似前缀退化，不追加训练。8条原FP32完整D/Utility基线的token、输入及Utility账本重现；651旧证据SHA核验，只有两个既有Windows CRLF例外按旧规则归一化。冻结`summarize/draft_summary/select_transfer/choose_lengths`源码逐函数一致，记录postrun-audit.json。
- 首轮结果ZIP709963字节/SHA`e4ab854b6491907ad25931a525bd2641e8e60e30ee77eb118080b70af65b8650`；追加351383字节/SHA`881cdf4a1c893cf07485d22ea920727c2bcb0ea569f35bfe37e08134fe0235a1`。追加源码ZIP SHA`ff5d0f87f093409f17f2ee8adef68d43bfbd466f62cd78e7c99756224b6d8f79`，source SHA`f3b0b4ca4350a2a3f4805e3aeceaf26f1b7e5b0d05f3c87ac6473777c84b5141`。
- 服务器首轮80 passed/6.74s、追加28 passed/5.34s；最终本地`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-optimization-final --junitxml=artifacts/asd-optimization-tests-final-local.xml`：524 passed/4 skipped/51.02s，仅既有python_multipart提醒。发布审计重算评分、矩阵完整身份、EOS/重复/预算、原基线与选择规则；测试拒绝质量字段篡改。
- 结果`results/asd-optimization/`及`extension/`；报告`docs/ASD_OPTIMIZATION_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_asd_optimization_report`。准确GPU入口分别为原目录`../.venv-dflash/bin/python -u -m scripts.run_asd_optimization_pipeline`、追加目录`../.venv-dflash/bin/python -u -m scripts.run_asd_optimization_extension`，均`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.`；完整argv、GPU划分、配置见各pipeline.log和冻结协议。
- 阶段决定：保留基础ASD/简单Utility与执行优化证据；停止本轮长度/预算调度及ASD前缀适配训练投入。未穷尽其他模型、大并发引擎或所有训练方法，不宣称全局无优化空间。

## 2026-10-21 — ASD优化排查追加归因（执行准备）

- 原执行冻结`37f15d0ff9842f87e87cb208c621f38c2dac2bec`，1035文件，ZIP SHA`0cd5dfe87089923c08e5b742b4100284236cf5e6f083894b7a44628811c8e8e9`、source SHA`4b0a0aa3c3e7c2ee405d46236cd6320141dcf9313e3cca123997a05cfdcebafa`，独立pipeline PID3777460。服务器80 passed/6.74s；本地完整回归522 passed/4 skipped/50.18s（服务测试需允许loopback，沙箱内停滞运行已终止，相关XML均保存）。
- 初轮A全部204主生成完成：FP32/eager固定短验证/短Draft及两动态长度均未达速度门槛；增量EOS输出/账本不变但无时间收益。FP32/SDPA与原输出/轮数相同，速度持平。32个Draft分支/15等128对的第一块接受92/88、后续轮268/281，未见近似前缀退化，不进入训练。
- BF16/SDPA四旧题D-fast等128约9.178s，对照FP32/eager14.876s，显示组合执行收益，已由原冻结规则选择4其他旧题迁移。看到这个结果后才追加`ASD_OPTIMIZATION_EXTENSION_PROTOCOL.md`，不冒充预注册：BF16下重测全部原长度选项，增加BF16/eager归因，不调阈值；若长度仍不通过即停止。
- 新独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-optimization-extension`，等待原pipeline结束后运行，不修改原执行目录/manifest。追加144主生成及8AR128，选择性迁移仍用原5%速度/质量/截断条件。实验loader新增profile与attribution入口，生产RuntimeConfig仍FP32/eager。

## 2026-10-21 — ASD剩余优化空间排查（执行准备）

- 用户授权尝试剩余方向。冻结执行成本/验证长度/Draft前缀适应三个检查轴；不重新扫描置信度、价格或未来二选一模型。协议`docs/ASD_OPTIMIZATION_PROTOCOL.md`，独立服务器目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-optimization`。
- 显式可选`draft_block_size`/`verify_length`/`incremental_stop`保留默认路径。增量EOS只扫描新提交token；分别测试缩短实际Draft和仅截短Target验证，不能退还已付Draft成本。生产RuntimeConfig继续只允许FP32/eager，其他精度/attention由独立实验loader启用并记录实际配置。
- A固定4旧题、11/3/3臂、自然1/128两次共204次，另AR128及独立profiler；冻结规则决定是否迁移4道其他既有题，不使用迁移结果调参。32单token严格分支检查Draft是否有明确前缀适应退化，未通过门槛则不训练。
- 首次相关测试75 passed/2 skipped/1 failed：新测试错误假定ASD加入EOS后与无EOS轨迹相同，实际Utility会按EOS调整收益。修正为同EOS配置下旧/增量输出与流式逐ID一致；保留失败XML，不改GPU数据（尚未执行）。
- 本地命令`.venv\Scripts\python.exe -m pytest -q tests/test_asd_optimization.py tests/test_dflash_reuse.py tests/test_asd.py --basetemp .test-tmp-asd-optimization-second --junitxml=artifacts/asd-optimization-tests-second-local.xml`。模型配置确认原Draft块16，服务端支持BF16，两卡空闲。

## 2026-10-20 — 实际计算收益预测（100次完成，停止扩大）

- 冻结执行`3132838503d0130d78726c5e52b1d28c6bcafa13`，独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-compute`，PID3335336已完成；1010个Git原始文件执行前后SHA一致。源码ZIP SHA`fdcfc1020d1fe7402a24fc26330a29a9a0739d682058ea929a2683eca60f007f`、source SHA`9ffdff87763d40ffe300cc9b2c441de1b675b32810640abc8021121ec286eea7`，推理package未改，仍为`4097fa46857e96b785fd4a9d2b88ef94dfd9dd0d700d4fe42d83ac537d1c934f`。
- 4完整baseline精确重现原Utility输出/账本；16个不同源块、32自然分支、64范围分支，两次重复的token/动作/验证轮完全一致，共100生成。分叉保留真实KV，A保留当前Utility接受，B截到首例外前恢复Target，后续均严格；旧原完整Utility的gsm8k:21错误仍保留。新32自然分支最终答案均正确并EOS，不是原完整Utility质量已改善。
- 15个双方均达128token且自然答案均正确的状态、覆盖4题，符合预定覆盖门槛；gsm107的B提前EOS单列排除等范围统计，原数据不删。固定A未来255轮/局部18.979368s，固定B278/20.647496，首例外Run二选一规则262/19.523525，有限二选一oracle249/18.509340，固定四特征留题岭回归259/19.276312。
- oracle相对最好固定动作只省6轮（2.35%），局部秒数约降2.48%，未达运行前冻结的5%轮数门槛；预测器反而多4轮（+1.57%）、局部秒数+1.56%，MAE轮差2.845。计时排除共同验证/前缀重放/当前选择核查及预测器，已经是乐观局部空间，不能发布端到端加速比。Run规则不是完整ASD-Run在线对照。
- 阶段决定：停止继续这条128token/有限二选一未来收益预测路线，不扫参数、不加复杂value head、不扩大同类开发题。仅有4旧题，不能证明所有ASD/其他工作负载无优化空间；现有Utility与infra保留。置信度方向继续停止投入。
- 初版离线审计把strict_accepted（EOS裁短前）与实际接受长度直接等同，导致Strict future报错；逐条核实全部为最后EOS轮裁短、后续regret0。修正审计为只在最后EOS处裁短，加回归；不重跑GPU、不改原数据/预测器/门槛。发布相关13 passed/9.51s，预测函数会与冻结Git源码逐函数核验。
- 回传ZIP349568字节/SHA`16917117766f9fe84f0aa137941a7343a6528ce8a938b9ff159ae7c5eff11604`；原始证据`results/asd-compute/`、报告`docs/ASD_COMPUTE_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_asd_compute_report`。准确GPU命令见冻结协议及pipeline.log、run_asd_compute_pipeline.py。服务器相关28 passed/4.95s，本地首次28 passed/9.45s；结束后两卡均19MiB/0%。
- 完整回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-compute-full --junitxml=artifacts/asd-compute-tests-full-local.xml`：496 passed/3 skipped/49.61s，仅既有python_multipart弃用提醒。96分支/4631轮另做推进/真实缓存前缀核对；`predict`/`summarize`与冻结Git源码逐函数完全一致，记录于postrun-audit.json。632个旧证据全部核验，两个既有Windows CRLF例外按原规则归一化且明确列出，旧内容未改。

## 2026-10-20 — 实际计算收益预测（执行准备）

- 按用户授权停止继续优化置信度定价，检查有限动作oracle空间与廉价计算特征的可预测性。4道旧开发题，每题4个不同源验证块；按源位置/源长度和等间隔索引冻结，不按新分支结果筛选。
- 新runner `scripts/run_asd_compute_probe.py` 从原prompt重放到同一个Target验证块，在实际KV上比较原Utility完整接受动作与当前严格截断；之后均严格续写。原始账本保留，未来regret为零；不重prefill，不用改token后的旧suffix logits。4 baseline+32自然+64等范围，两次重复，共100生成。
- 128token范围从共同验证块边界开始，计入本轮缓存多推进；局部计时在共同验证/原选择/源核查之后，包含KV提交与后续，排除重放与预测器，因此只是乐观计算空间，不作线上加速结论。协议 `docs/ASD_COMPUTE_PROTOCOL.md`。
- 冻结四项非置信度特征和alpha=1的按题留出岭回归；先检验oracle能否比最好固定动作省>=5%未来验证轮，再判断预测器是否优于固定动作与Run。质量/长度/EOS全部单列；不扫超参、不增加样本或特征。
- 首次检查命令`.venv\Scripts\python.exe -m pytest -q tests/test_asd_compute.py tests/test_asd_utility_causal.py tests/test_asd.py --basetemp .test-tmp-asd-compute-first --junitxml=artifacts/asd-compute-tests-first-local.xml`。
- 首次28 passed/9.45s，覆盖真实tiny Target/KV上两动作与各自实际前缀的严格AR续写一致、后续零regret、原始账本保留、按题留出无本题标签泄漏、质量/EOS过滤及oracle最好固定对照。没有修改生产推理模块。

## 2026-10-20 — Utility 错误轨迹与计算收益（62次完成）

- 冻结执行提交`80f4a001671c409d0fdf93728b4b54fc7d3f71a1`，服务器独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-utility-causal`，PID3313804已正常完成。975个Git原始文件在执行前后SHA一致；源码ZIP SHA`2747fc75d2b16c580a3e5959d61f83ff3fdd452535f51d338f97afe37c067f55`、source SHA`229f38b015414d5ef81fece9a4c759be46fa3cbb9ab5a5b2b807d26160f8a9fe`。runtime package未改，继续为`4097fa46857e96b785fd4a9d2b88ef94dfd9dd0d700d4fe42d83ac537d1c934f`。
- 2次完整Utility基线、12次完整单次拒绝、48次同前缀严格自然/128续写分支，共62生成全部完成。gsm8k:21全部9个实际例外（包含块内第二个）与math500:7固定3状态；两个baseline精确重现旧token/账本，全部完整自然分支EOS，无自然上限截断。独立标量重算751策略块的原可行域/Utility最优选择、单次干预、原始预算/EOS/前缀/计数通过。
- gsm旧错答23有8/9个单次拒绝恢复14；这些是完整策略轨迹效应，不能把8处都标为单token有害。严格自然proposal/recovery为11/12与12/12正确，只有gsm位置211直接出现质量差异：proposal分隔符让模型进入Final Answer，recovery为`$$`，继续算37−23=14。该源状态regret1.8164、Run6，r/Run0.3027>0.25，已有Run局部门控会拒绝；仅同状态计算，不冒充完整Run轨迹重跑。
- 位置211之前已花4.5570、剩余3.4430，足以支付1.8164，原完整Utility最终6.3734<8；本例不支持预算耗尽解释。没有同题regret差<=.25/Run差<=2且proposal质量不同的状态对，尚无新增置信度独立信号，不上线新置信度门控或训练头。
- 6对均达128 continuation token的分支，proposal后续轮更少3/更多2/相同1。math7位置10的Run仅2，后续128轮18对30；gsm14/17同为Run2却为24对20/24对22。局部Run不是未来验证效率的直接估计。不同文本/重prefill且每条件仅一次，不发布时延加速比。
- 回传ZIP565052字节/SHA`5ba09867dcec9ff69346f79ca444cb94dcc7497f8268844ce1b10f9278402bb9`；原始公开证据`results/asd-utility-causal/`，报告`docs/ASD_UTILITY_CAUSAL_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_asd_utility_causal_report`。两GPU完成后均19MiB/0%，旧证据未变；infra的main/面试分支/tag仍指向`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。
- 准确GPU命令见冻结协议与`results/asd-utility-causal/scripts/run_asd_utility_causal_pipeline.py`、pipeline.log：GPU0跑gsm8k:21、GPU1跑math500:7，`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_utility_causal_probe --config configs/runtime-dflash.json --task-id ... --output results/raw/asd-utility-causal/{gsm,math}`。GPU完成后仅用冻结Target tokenizer离线解码追加semantic-context.json，重建入口`scripts/export_asd_utility_context.py`，不把事后文本观察当预定在线特征。
- 本地首次相关30 passed/9.34s，增加真实tiny Target/KV/流式干预覆盖后31 passed/9.28s；服务器31 passed/4.91s。发布初审14 passed/10.44s。语义解码包含Unicode，Windows报告曾因默认GBK读取失败；已显式UTF-8读取，修复后含预算重置/重复分支篡改审计16 passed/11.18s。GPU数据未重跑，失败XML保留。
- 完整回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-utility-causal-full --junitxml=artifacts/asd-utility-causal-tests-full-local.xml`：483 passed/3 skipped/49.32s，仅既有python_multipart弃用提醒。正确性沿用最终答案评分；位置211恢复分支保留前缀旧错误陈述，不能称为整段推理无错误。报告显式UTF-8/LF导出，避免Windows换行改变发布SHA。
- 离线decoder按公开冻结tokenizer重建semantic-context字节一致，SHA`544f4f681ddec3364f879d175b802c3a0c0f05c23c7b33f8a51ef4f4bbe57201`，脚本/命令/tokenizer文件SHA记录在context-decoder-provenance.json；追加报告来源审计7 passed/7.00s。26个发布证据/协议/报告的原始SHA在publication.json；607个旧证据核验，只有两个既有Windows CRLF文件需按旧规则归一化核对（manifest明确列出），无旧结果内容修改。

## 2026-10-20 — Utility 错误轨迹与计算收益（执行准备）

- 用户授权落实置信度/Utility后续验证。先验证已知Utility错误中的关键例外，暂不盲目增加新在线价格规则；原严格运行时与infra冻结引用不改。
- 新runner `scripts/run_asd_utility_causal_probe.py`：覆盖gsm8k:21全部9个实际接受例外，含块内第二例外；math500:7按第一个/最大regret/最后一个取3状态。先复现原Utility，再从原prompt单次截断接受并正确恢复原始请求账本，之后继续原Utility；另做同前缀proposal/recovery严格续写的自然EOS与128 continuation两范围。
- 协议 `docs/ASD_UTILITY_CAUSAL_PROTOCOL.md` 固定62次生成、2旧开发题，不训练预测器、不扫描参数。全部质量/截断保留；插桩与重prefill时延仅诊断，不作为在线加速。原结果SHA和执行源码执行前后核验。
- 首次命令 `.venv\Scripts\python.exe -m pytest -q tests/test_asd_utility_causal.py tests/test_asd_confidence.py tests/test_asd.py --basetemp .test-tmp-asd-utility-causal-first --junitxml=artifacts/asd-utility-causal-tests-first-local.xml`，后续增加真实tiny Target/KV流式干预覆盖后再冻结执行。

## 2026-10-20 — ASD 置信度小检查（156次完成）

- 冻结四道已观察开发题math500:1/4/7、gsm8k:21；七组含原四组、固定价格1/3与alpha=.5的Confidence-Utility。28机制+56完整自然计时+56遵守EOS的128前缀计时+16单例外干预分支全部完成，无错答/截断剔除。四个旧策略在旧未裁短块之前的前缀保留，机制/计时轨迹、账本与计数完全一致；1901机制块通过独立标量重算，完整proposal/EOS观测通过核对。
- 三道MATH所有七组均3/3正确并EOS。Utility35.440130s/4376token，Confidence34.105988s/4010token（两重复）：时延−3.76%、输出−8.36%、自然TPS−4.78%、每token轮密度+4.72%、每轮成本+0.28%。等128前缀Confidence9.361886s、Utility8.726646s（+7.28%），不支持总体吞吐增量。单题math7有轮密度改善（77/793→71/765），math4则退化（79/601→83/576）。
- gsm8k:21的Utility仍真实错答，Confidence答对；固定价格1/3及其他基线也答对。Confidence6.725093s，Utility6.700870s（+0.36%），价格3为4.343506s（Confidence比其慢54.83%）。不是置信度独有纠错证据。
- 8个干预状态/16分支全部答对且同状态Target回放通过。math1/7满足预定regret/run配对，math4/gsm21无匹配对、按协议标记unmatched；没有有害/无害质量标签差异，无法验证风险预测能力，也未定位Utility错答轨迹的致错例外。严格分支均重prefill，未跨分支使用旧logits，诊断时延不作线上加速。
- 指标局限实例：原ASD math4/45的Target首选概率0.9296、proposal0.0683，pair_mass0.9979、regret2.611。集中度高不代表Target犹豫或认可proposal；一律降价会削弱原regret惩罚。这是指标定义局限，未称其为置信度组退化的唯一因果源。阶段决定停止扩大此简单定价规则，保留采集/干预资产；不在四题扫描alpha，不否定所有置信度方法。
- 正式执行commit`e7ec17b5f3b832b91f1c68cceb74395ac7607f85`；服务器独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-confidence-v2`，933文件执行前后SHA核验。源码ZIP SHA`9c55a0f1c40d567c7618fae7a4d9d1918042a22e476621a7a55e0c4cb5af6c9e`、source文件清单SHA`aea26518d2dd031aa4432bc7c7c738cf78a8a8819dea460384ef0aa6fc2dd489`；runtime package `4097fa46857e96b785fd4a9d2b88ef94dfd9dd0d700d4fe42d83ac537d1c934f`。正式回传ZIP1117088字节/SHA`aea5ad867f7108e7361162387aa16481af964eeb453f7faff610ed877d224ed6`。
- 首次路径失败保留在`results/asd-confidence/failed-v1`，失败ZIP SHA`91584aaeca4900ce506fa3e7616b18e5e2808328e52b13cea9b4992c6975ff50`；不计入有效140计时/机制生成。正式服务器39 passed/4.75s；本地修复39 passed/9.97s；发布审计15 passed/7.92s。最终完整命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-confidence-final --junitxml=artifacts/asd-confidence-tests-final-local.xml`：471 passed/3 skipped/49.19s，仅既有multipart弃用提醒。GPU完成后双19MiB/0%。
- 准确GPU命令见冻结协议`docs/ASD_CONFIDENCE_PROTOCOL.md`、源码`results/asd-confidence/scripts/run_asd_confidence_pipeline.py`及pipeline.log；原始证据`results/asd-confidence`，可重建报告`docs/ASD_CONFIDENCE_RESULTS.md`，命令`.venv\Scripts\python.exe -m scripts.write_asd_confidence_report`。面试infra分支与tag未改。

## 2026-10-20 — ASD 置信度小检查（执行准备）

- 用户授权代码与小规模检查。冻结4道已观察开发题math500:1/4/7、gsm8k:21；七组含原四组、固定价格1/3与alpha=.5的Confidence-Utility。请求预算按原regret扣除，不按概率价格打折；原ASD可行域不变。协议`docs/ASD_CONFIDENCE_PROTOCOL.md`。
- 新增Utility可覆写代价入口、`asd_confidence.py`、独立runner/pipeline和行为测试。机制记录完整proposal/EOS、Target/Draft概率，修复旧trace无法还原未接受EOS的观测限制；默认严格运行时不启用。
- 上限140次机制/自然/128前缀生成，加最多16同前缀单例外干预分支。两GPU独立目录执行，不按质量筛结果，分支重算缓存，不跨分支复用旧suffix。
- 本机首次相关检查：`.venv\Scripts\python.exe -m pytest -q tests/test_asd.py tests/test_asd_confidence.py --basetemp .test-tmp-asd-confidence --junitxml=artifacts/asd-confidence-tests-first-local.xml`：25 passed/35.23s。服务器双A6000均19MiB/0%，尚无本轮GPU结论。
- 首次冻结`405040e0106efcce485905791ebfd488546848cc`，933文件核验；服务器38 passed/4.80s。runner错误假定旧补全参考在math500目录，实际为part-a/part-b，两个worker在机制读取参考时失败、尚未自然计时；保留`asd-confidence`目录与失败日志。修正为加载前读取真实三目录，并加参考覆盖测试；新目录`asd-confidence-v2`重新执行，策略/任务/参数不变。
- 首次完整本机回归`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-confidence-full --junitxml=artifacts/asd-confidence-tests-full-local.xml`：466 passed/3 skipped/56.61s（在参考路径修复前）；仅既有multipart弃用提醒。

## 2026-10-20 — ASD 新题收益归因（600次完成）

- 固定GSM8K test24..31（8题）、MATH-500 8..23（16题），原四组加固定gate0.0625的Conservative。120机制+240完整自然EOS计时+240遵守EOS的128前缀计时全部完成；不按质量/截断剔题，不扫描新题参数。四分片math-a/b、gsm-a/b分别同题同卡交错，精确命令见`results/asd-attribution/run-pipeline.py`与`pipeline.log`，冻结协议`docs/ASD_ATTRIBUTION_PROTOCOL.md`。
- 新MATH原ASD293.699300s、Run254.355233s（−13.40%）、Utility264.796971s（−9.84%）、Conservative262.111812s（−10.76%），两重复。Run自然TPS反而−2.10%，Utility−2.20%；Run输出token−15.22%、每token验证轮+1.73%、每轮成本+0.42%。主要自然时延收益伴随输出路径缩短，未复现每token执行效率增量。Run/ASD跨题配对bootstrap时延比区间[0.6668,1.0492]含1。
- 严格DFlash260.130839s；Run相对它仅−2.22%时延/−1.29%自然TPS，Utility+1.79%时延。Run与Conservative实际总regret75.527/76.270、例外233/229接近；Run相对该对照时延−2.96%、输出−7.02%、TPS−4.19%，跨题区间也含1，不支持实际后缀门控独有稳定吞吐收益。
- MATH五组保守主评分依次DFlash/ASD/Run/Utility/Conservative为11/8/12/13/10（共16题）。math500:23五组均输出5，与参考x=5等价，单列`quality-audit.json`后为12/9/13/14/11，不修改raw或主评分。截断依次2/3/2/2/2；12题五组均EOS、1题均length、3题mixed。全EOS12题Run时延仍−8.80%，但全题和终止分组不可互相替代。旧Utility gsm8k:21真实错误继续保留。
- 新GSM全部五组8/8正确、无截断。ASD45.560562s，Run49.213269s（+8.02%、自然TPS−7.65%），Utility45.568200s（+0.02%基本持平）。Conservative56.544874s（+24.11%）；原ASD相对严格DFlash时延−16.09%、自然TPS+14.24%。后缀选择的增量没有跨任务复现，ASD本身仍有本切片收益。
- 所有24题五组均达到128前缀，共同等token集合即全部新题。MATH Run比ASD慢3.65%、Utility慢0.67%；GSM Run慢4.11%、Utility快0.18%。全部短上限裁短块之前与完整轨迹一致，两次前缀输出一致；没有越过EOS续写。
- 同状态诊断最初漏考虑被拒绝的proposal EOS：gsm8k:30/Utility event32可行上限因EOS裁到3，策略只接受2并继续生成。修正独立标量审计，新增专门测试；原推理/账本/输出/计时不改。因旧trace没有保存未接受proposal EOS位置，反事实比较收窄为原ASD非终止状态（MATH1838/GSM288块）；Run和Conservative都更严，其前缀不会跨越源ASD实际可行上限。其他四条源轨迹不报告不安全的无界后缀收益。门槛开发校准同样只是近似启发式，保持冻结值，不事后重选。
- 在MATH相同原ASD状态/剩余账本上，Run局部348额外proposal/23.679 regret，Conservative324/28.709；存在局部支出效率信号。但独立块重算不推进替代账本/模型轨迹，不能当成全请求节省验证轮。这与完整计时的吞吐负结果分列。
- 执行commit`38a87cef2d37192719c2fefe282eb574d19ca09e`，独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-attribution`，889源码文件执行前后SHA一致。源码ZIP SHA`e3bb035680481abdec9d019b2193e065ab448e0bee06492011cf37b2ee7bf441`、清单source SHA`47ebcf7da57984861de94af4ce0432a09cf304c3f8554e0a4bd71a846e764392`；runtime package仍`8924a107833ad260fb0dac41a3ebdfb73c4b747dc5fdfcdf11aa4a09cc200e67`，与旧预算/三题补全相同。回传ZIP1587598字节/SHA`4812684cf2e118d4ccaf0e3911304e780cb02d6babc088df7d8cbef8c4a4833b`。
- 本机相关32 passed/10.22s，服务器32 passed/4.81s；发布后相关18 passed/10.98s。最终完整命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-attribution-final --junitxml=artifacts/asd-attribution-tests-final-local.xml`：455 passed/3 skipped/47.68s，仅既有multipart弃用提醒。报告重建`.venv\Scripts\python.exe -m scripts.write_asd_attribution_report`，数据与报告`results/asd-attribution/`、`docs/ASD_ATTRIBUTION_RESULTS.md`。
- 阶段判断收紧：旧三题输出缩短和轮密度改善均真实存在，新24题未显示Run/Utility对原ASD稳定每token增量。保留实验资产，不接默认严格运行时；不在新题继续扫参数。若继续，先在旧开发集建立原ASD门槛/预算的速度—质量对照，再做独立任务验证，不优先增加复杂预测器。两GPU收尾均19MiB/0%；main、infra冻结分支和tag仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。

## 2026-10-20 — ASD 新题收益归因（执行准备）

- 用户授权继续，重点区分实际后缀信息、保守门槛和输出长度。新题固定GSM8K test24..31、MATH-500 8..23，五组包含原四组及只收紧原ASD gate的Conservative；完整自然EOS/2048上限，并做遵守EOS的128短前缀，不强制越过EOS。协议`docs/ASD_ATTRIBUTION_PROTOCOL.md`，计划120机制+240完整计时+240前缀计时=600生成。
- 仅用旧24题1115个Run非终止状态，在相同状态、相同剩余账本上匹配总regret选择固定gate0.0625：Run47.585608/107例外，对照50.556339/141例外。网格及全部候选保存在`data/research/asd-attribution/calibration.json`；这是离线开发校准，不能当真实rollout或新题预算精确匹配。
- runner新增显式data/policies/prefix参数，默认旧四组路径不变；独立配置`configs/asd-attribution-policies.json`。使用原评分器，MATH保守精确匹配之外的等价性审计单列，不修改原输出。推理core、默认CLI/服务与infra冻结版本保持。
- 公开GSM源749738字节/SHA`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`，沿用已固定source commit；MATH完整源既有446564字节/SHA`35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。准备命令与相关测试见协议；目前尚无本轮GPU性能结论。

## 2026-10-20 — ASD 截断MATH补全（36次完成）

- 用户授权下一步，预先固定上一轮math500:1/4/7三个截断请求。保留四组与B8/g0.25/M2、Utility价格2；输出上限改为2048，不在这些已观察题上调整阈值。36次正式生成，另计预热；协议`docs/ASD_COMPLETION_PROTOCOL.md`。
- 复用runner增加任务选择/输出上限/旧前缀核对。旧上限裁短最后验证块可能改变局部门槛，因此核对首次裁短块以前的完整前缀，并保存实际512前缀是否相同；固定Target输入和runtime package。旧budget发布证据也加入执行前后SHA保护。
- 12机制+24自然EOS计时全部完成，三题四组全部正常结束并通过评分，无截断，输出586–838token；12/12机制和24/24计时保留各自旧512输出完整前缀，计时轨迹与机制完全一致。旧512结果不改。
- 原ASD41.070915s/4662token（两重复）、自然TPS113.511；Run34.570983s/4262token，时延−15.83%、自然TPS+8.61%；Utility35.255236s/4376token，时延−14.16%、自然TPS+9.35%。严格DFlash41.016759s，两新组相对严格DFlash时延分别−15.71%/−14.05%。质量均3/3；仍有输出长度变化，不能把全部时延收益解释为等工作量执行效率。
- 机制验证轮ASD261、Run219、Utility223；regret23.434→6.310/20.375。Run在math500:1/4时延−17.69%/−29.34%，math500:7反而+6.26%；Utility三题均改善。收益具有逐题差异，不宣称通用加速。旧gsm8k:21的Utility真实错误继续保留，不升级默认策略。
- 冻结提交`07663d23117f122a620d417fadc483b85abb2015`，服务器独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-completion`，858源码文件执行前后SHA一致。源码ZIP SHA`3f26a3bba1e8df54be7ddaed756e31e34e95a78f2632ee138f75cd9929ff7325`、清单source SHA`3f06e4cdc497acc41c7cafd3709da5eea4c169468220b612fc72a7c36b3303ba`；runtime package继续为`8924a107833ad260fb0dac41a3ebdfb73c4b747dc5fdfcdf11aa4a09cc200e67`，与原512实验一致。
- 准确命令在`docs/ASD_COMPLETION_PROTOCOL.md`和`results/asd-completion/run-pipeline.py`：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0/1 ../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark math500 --task-ids math500:1 math500:4`（另卡`math500:7`）`--max-new-tokens 2048 --previous results/asd-budget --protocol docs/ASD_COMPLETION_PROTOCOL.md --output results/raw/asd-completion/part-a`（另卡`part-b`）。新进程从原始prompt生成，没有中途重置预算。
- 本地相关33 passed/15.52s；服务器相关27 passed/4.28s。本地完整命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-completion-regression --junitxml=artifacts/asd-completion-tests-regression-local.xml`：431 passed/3 skipped/48.42s，仅既有python_multipart弃用提醒。真实结果发布后相关审计16 passed/7.55s；最终报告审计6 passed/7.34s，覆盖缺失/重复、预算、前缀、轨迹和政策篡改。
- 回传ZIP228192字节/SHA`e7010f384f8736523625230f83c13772e8606580ff45f8d333763041311421f3`；原始结果`results/asd-completion/`、报告`docs/ASD_COMPLETION_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_asd_completion_report`。执行结束两GPU均19MiB/0%，旧结果SHA不变。完整解答下总体正结果保留，但仅3道已观察题，不作为新题泛化或质量非劣性证明。

## 2026-10-20 — ASD 实际前缀收益预算验证（288次完成）

- 冻结执行提交`88d8143f6ece7316942d03efdea7d3221161ad64`，独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-budget`。833个Git原始文件执行前后SHA全一致；源码ZIP SHA`9b744c2400c46657248d704eb74f32313d937d2fc3675692c99986b21c00cf46`，清单source SHA`942be35c802b6c4d1d3edd84b95e66c83d06d0c6745970429fdabed74b58fe58`，运行package SHA`8924a107833ad260fb0dac41a3ebdfb73c4b747dc5fdfcdf11aa4a09cc200e67`。
- 24个新请求、DFlash/ASD/ASD-Run/ASD-Utility四组，96机制+192自然EOS计时全部完成。ASD B8/g0.25/M2；Utility价格2在新生成前冻结。全部计时输出与机制一致，输入/快照/预算检查通过，旧发布证据SHA不变。GSM8K16题GPU0、MATH8题GPU1；不跨卡混算任务质量。
- GSM8K原ASD118.929300s、Run119.216686s（+0.24%）、Utility113.728458s（−4.37%）；正确数15/16、15/16、14/16。Utility轮772→735、regret123.831→80.204（−35.23%）、129块主动缩短，但gsm8k:21把应算37−23=14的答案输出为23，人工核对为真实推理错误，不修评分、不排除该题。
- MATH原ASD63.230965s、Run56.771632s（−10.22%）、Utility56.953907s（−9.93%）；均5/8，三道全组上限截断。轮411→367/368；regret60.822→16.105/34.640。四组均完整结束的5题，Run/Utility时延−7.26%/−9.00%，自然TPS仅+0.03%/+0.25%，耗时下降伴随输出缩短；总体TPS收益主要来自3道512token截断题，未验证完整长解答的稳定每token加速。
- 实际命令保存在`results/asd-budget/run-pipeline.py`及`docs/ASD_BUDGET_PROTOCOL.md`；两个进程分别设置`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0/1`，调用`../.venv-dflash/bin/python -u -m scripts.run_asd_budget_probe --config configs/runtime-ot-benefit.json --benchmark gsm8k/math500 --output results/raw/asd-budget/gsm8k或math500`。输出上限512，自然EOS，没有强制续写吞吐阶段。
- 服务器相关23 passed/4.28s。本地完整回归命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-budget-regression-network --junitxml=artifacts/asd-budget-tests-regression-network-local.xml`：426 passed/3 skipped/43.31s；需允许本机回环的服务测试，沙箱内的首轮在服务测试附近停滞后中断。真实结果发布后相关审计56 passed/12.40s；终止原因拆分报告6 passed/6.95s。最终完整命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-budget-final-regression --junitxml=artifacts/asd-budget-tests-final-regression-local.xml`：427 passed/3 skipped/43.10s，含真实报告重建审计；仅有既有python_multipart弃用提醒。
- 回传ZIP797360字节/SHA`579ec36622bea055ca764fcabb3ddc2dd371f0f3e2d15866116a83de4ae56278`；结果`results/asd-budget/`，报告`docs/ASD_BUDGET_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_asd_budget_report`。两GPU已恢复19MiB/0%。当前保留实验实现，Utility不进入默认运行时，不在本批继续扫描价格；研究结论是局部预算效率改善，尚无稳定质量下的完整长解答每token优势。

## 2026-10-20 — ASD 论文任务迁移（320次完成）

- 64机制+128自然EOS+128固定128token全部完成。GSM8K strict/ASD最终答案均8/8；MBPP公开测试均5/8；全部自然运行无截断、计时输出与对应机制一致。非零ASD允许轨迹变化，GSM8K 8/8轨迹改变，MBPP 3/8改变。
- GSM8K自然EOS总秒 DFlash66.675068、ASD54.463856（−18.31%），固定128token同为2048输出，TPS62.537677→69.167596（+10.60%）；输出2250→1986、轮436→353、例外53。Splice-ASD55.464486秒，比单独ASD慢1.84%，内部拼接0提议。
- MBPP自然EOS DFlash10.718594、ASD10.466526秒（−2.35%），轮55→53、例外5。固定吞吐26.265640→66.631536看似+153.68%，但约71.8%/72.3%为自然EOS后强制续写，64/64固定前缀与对应自然输出一致；该大增幅不当作正常代码任务收益。
- 原评分有48条机制/重复记录漏解析`<answer>`/boxed货币数字，离线quality-v2仅修正明确输出数字；原rows与评分不改，数据、prompt、模型、阈值不改。新grader SHA与原执行package双列，score修正不能冒充Target质量改善。
- 802个源码文件执行后SHA全通过，runtime package `4d92d736d78a5c1962022922443853282ad661edddc9582d91b453b0a8a19e0e`，原证据SHA不变。紧凑结果ZIP338559字节/SHA `fab077b022d4bd0377c75d919741a3bd7707c310185b02b7b261616ad5a6e27d`；结果 `results/asd-bench/`，报告 `docs/ASD_BENCH_RESULTS.md`。
- 服务器32 passed/4.18s；本机完整可用409 passed/3 skipped/40.44s，新报告评分保真/预算篡改2项另测。报告重建命令 `.venv\Scripts\python.exe -m scripts.write_asd_bench_report`；GPU已恢复双19MiB/0%，无后续模型进程。
- 本轮支持ASD在该数学小切片上的收益，不支持内部拼接增量或长期历史独有价值；未重启OT/旧风险预测器，未按新结果筛题或加大预算。

## 2026-10-20 — ASD 论文任务迁移（机制完成与评分修正）

- 执行冻结 `129e6193b37fb2e0bd74a8ef4adf18ed010c0b67`，独立服务器 `asd-bench`，802文件SHA核验，ZIP SHA `4dc27209a4cacb17971fca1c3a1ad295356b30e693ea12d6548bfdff347395c9`。服务器32 passed/4.18s；计划320运行未因质量提前停止。
- 64机制完成：ASD在GSM8K放行53例外、验证436→353轮、输出2250→1986token；MBPP放行5例外、55→53轮、289→284token。内部拼接0提议，少量普通R另计，不把ASD收益归因于拼接。
- 初版GSM数值解析漏识别实际输出的 `<answer>` 标签/boxed货币数字，造成格式假失败。只修正离线评分，原rows/原评分保留，报告以quality-v2修正为准，不改提示、输出、标准答案或ASD门槛。新解析下GSM strict/ASD均8/8；MBPP原始公开测试均5/8。
- 完整回归首次402 passed/6 failed/3 skipped：旧风险报告把后来新增ASD结果也算入历史覆盖，非模型/解码失败。改为只检查当时source-manifest中存在的证据，仍逐文件核验全部旧记录，新增“未来结果不影响旧报告、缺旧证据仍拒绝”回归。相关27 passed/10.05s。
- 修复后完整可用回归（仅待GPU数据的本轮report测试单独执行）：`.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_asd_bench_report.py --basetemp .test-tmp-asd-bench-regression-final --junitxml=artifacts/asd-bench-tests-regression-final-local.xml`，409 passed/3 skipped/40.44s。失败XML保留。

## 2026-10-20 — ASD 论文任务迁移（执行准备）

- 用户建议使用ASD论文任务，选官方GSM8K test前8条、MBPP官方test编号11–18；来源提交与MIT/Apache-2.0许可保存 `data/research/asd-bench/`。数据在本批模型生成前固定。
- 仍用现成Qwen3-8B/DFlash、FP32/eager、非思考；使用ASD官方stable B=8/g=0.25/M=2。四方法DFlash、DFlash-ASD、Splice、Splice-ASD；每题独立空历史，R只查当前输入，不放入答案或其他题目的生成。本轮不作DSpark复现或长期记忆价值声明。
- 计划64机制+128自然EOS（最多512token）+128固定128token，共320生成，质量错误不提前停止。GSM8K数值答案评分，MBPP执行公开3测试，评分不计入生成时延；来源、模式与成本分别记录。
- 新评分模块 `memorydraft/evaluation/asd_bench.py`、runner `scripts/run_asd_bench_probe.py`、协议 `docs/ASD_BENCH_PROTOCOL.md`。相关32 passed/8.52s：`.venv\Scripts\python.exe -m pytest -q tests/test_asd_bench.py tests/test_asd.py --basetemp .test-tmp-asd-bench-fixed --junitxml=artifacts/asd-bench-tests-local-fixed.xml`。
- 初次测试9 failed/23 passed，Windows受限系统temp目录清理失败；改用项目artifacts/mbpp-eval临时目录后全部通过，失败XML保留。没有根据模型表现改题目/提示/阈值。
- GPU命令：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_bench_probe --config configs/runtime-ot-benefit.json --output results/raw/asd-bench`，隔离目录 `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-bench`。

## 2026-10-20 — ASD 与历史拼接（360次运行完成）

- 执行提交 `112c18fb62a5e9aef984e2360ad34fa80f12f282`，773个Git原始文件在服务器执行前后SHA全通过；独立目录 `asd-splice`。源ZIP SHA `703b2c524624d3ab42b913e7a786752af762b24bf2d354347d029b2c91aaf51c`，runtime package `baca9c8119a1d08499d7e146b4e35ba13728d3acec25e2986d016a7a43a5b5a9`。
- 8已观察请求/5项目、9方法，72机制+144自然EOS+144固定64token全部完成，无质量提前停止。自然EOS全部216输出匹配旧AR且任务合同正确；固定阶段每方法1024输出token，忽略EOS，仅作等工作量计时。两阶段均无近似例外，ASD未改变轮数。
- 自然EOS两重复总秒：DFlash15.626821/ASD15.657451，Suffix12.146994/ASD12.162636，Guided11.593674/ASD11.597269，Splice11.652066/ASD11.685018/Run11.722166。Splice-ASD比Splice慢0.28%、Run慢0.60%；固定吞吐+0.03%/-0.25%不能归因于决策改善。
- 首次阻断（排除已结束EOS之后）的最小regret9.058；局部regret/剩余长度最低1.071，仍高于g=0.25。三个失败桥段regret53.902、49.950、26.785。旧分数检查B=8也零可放行，不伪称重跑B=8；本批缺少ASD所需低差距例外机会，不提高阈值强造收益。
- 本机完整测试387 passed/3 skipped/43.77s；服务器ASD/输入/真实tiny解码55 passed/5.92s；报告与预算篡改审计2 passed/6.44s。实际命令 `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-asd-full --junitxml=artifacts/asd-tests-full-local.xml`，服务器运行命令见 `results/asd-splice/run-pipeline.sh`。
- 原始紧凑包117592字节/SHA `35c26a5639be5547e8e99a9a3718200d000cd2a945aede70e0d452000869f99b`，新证据 `results/asd-splice`、报告 `docs/ASD_SPLICE_RESULTS.md`；旧证据执行前后不变。GPU完成后双19MiB/0%。部署第一次使用系统旧Python缺少Path.is_relative_to，在解压/推理前停止；改用现成Python3.11虚拟环境完成部署，不改实验源码。
- 重建：`.venv\Scripts\python.exe -m scripts.write_asd_splice_report`。初始模型/参数与原严格Target保持；非零ASD仅由实验runner显式接入，未启用到CLI/HTTP默认路径。

## 2026-10-20 — ASD 与历史拼接（执行准备）

- 用户授权按合理机制直接实施，独立分支 `codex/asd-splice`，旧严格路径和 infra 引用保留。实现 ASD 请求级 B=2/g=0.25/M=2；唯一变体以实际连续 Target 匹配后缀替代剩余块长作为价值。近似输出单列，不宣称无损。
- 新模块 `backends/asd.py`，解码器可选 verifier，实验脚本 `run_asd_splice_probe.py`；官方 ASD 提交 `9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6` Apache-2.0 已核验，本次独立实现论文公式，不复制源码。
- 旧8请求/5项目，DFlash、Suffix、Guided、Splice 各 strict/ASD，加 Splice-ASD-Run；72机制+144自然EOS+144固定64token，无质量/速度提前停止门槛。协议 `docs/ASD_SPLICE_PROTOCOL.md`，已观察任务不作泛化。
- 已运行 `.venv\Scripts\python.exe -m pytest -q tests/test_asd.py tests/test_dflash_reuse.py --basetemp .test-tmp-asd --junitxml=artifacts/asd-tests-related-local.xml`：52 passed/2 skipped/10.46s；`tests/test_asd_probe.py tests/test_runtime.py --basetemp .test-tmp-asd-inputs --junitxml=artifacts/asd-tests-inputs-local.xml`：41 passed/12.44s。
- 服务器计划命令：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_splice_probe --config configs/runtime-ot-benefit.json --output results/raw/asd-splice`。GPU 实验尚未执行；隔离目录 `/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-splice`。

## 2026-10-19 — 风险拼接完整续测与已知机制覆盖（完成）

- 用户授权放宽任务质量/零触发的停止条件后，主12请求已完整60机制/120计时；错误Target输出保留并单列，未重跑AR、不改prompt/权重/阈值。180/180严格匹配AR，120/120计数路径一致。五组同为64轮/32Draft前向，Risk评估/改选0；主Risk相对Splice-Edit−0.02%、相对Guided+0.17%，全量14.155447s vs14.154433s的微差无风险机制变化，不能算收益。
- 补充旧payment/shipment训练项目10机制/20计时，全部30严格一致、20路径一致。执行`00bc845542cd6ab54778131a472a9e7945e8fd4e`，747文件核验，源zip SHA`1061fab2d3a6c4c77c3e456196ff5a34dd8b6db13217fb6b36f7054557bdbb67`，独立服务器`prefix-risk-controls`、PID2415173已完成；先8报告/输入测试通过/3.50s。原主数据和旧结果执行前后SHA一致。
- 两次执行相同runtime package SHA`098e4ec770108a3b4da457f16f46d86ac58f9dd0e6c46d146accb1ab4f2ff44c`，算法未调整。payment/15风险将边界14→7、接受7→32，全请求验证8→7轮；Risk两次总1.519649s，Splice1.669519s，Guided1.484506s，无置信度1.520009s。无置信度也作同一改选，两个控制的实际逐轮候选完全相同；只能证明已知例中边界修复有效，未显示置信度独立增量，且仍慢于Guided。shipment保持8轮，Risk1.711838s、Splice1.712927s、Guided1.671890s。
- 这两例特意选择自旧训练项目，只作机制覆盖，不合入12请求主结果，不称新holdout。控制原始zip73519字节/SHA`94344763aa0898514acbc7bea5cfc6b63fa4088a9d8a21d7705952ced3aa5779`，证据`results/prefix-risk/continuation/controls`；主zip109037字节/SHA`b2af856cf52177ed7d624924b4afe1b84691911a9ebdee06a2f341856176f097`。GPU双19MiB/0%。
- 实际命令见三个新协议及各目录run-pipeline.sh。报告`PREFIX_RISK_CONTINUATION_RESULTS.md`、`PREFIX_RISK_CONTROLS_RESULTS.md`，分别`python -m scripts.write_prefix_risk_continuation_report`/`write_prefix_risk_controls_report`重建；只新增续测目录，原停止现场/报告/publication不变。后续不把细小计时差、训练例修复或相同无置信度行为称为新的算法优势。
- 最终报告/输入审计10 passed/8.53s；完整373 passed/3 skipped/40.91s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-risk-continued-full --junitxml=artifacts/risk-continued-tests-full-local.xml`。主/控制共210次严格匹配；payment Risk较Splice快8.9768%、较Guided慢2.3673%。三个infra引用保持`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。CI dflash补充续测与控制报告审计。
- 最终交付`74a791dffa32a725c8d0bfd273aba4cbed1f430c`，独立`prefix-risk-continued-delivery`，766文件SHA全通过；zip SHA`dc785bcc3747b63b27580eda59333cfcec4cd6dbe6e7c1f05c46c068563dc7bf`，source SHA`1cccc2dfdd36346eb21a2494ac690fdfe10a60bd2379427b77e83a42b078309b`。服务器报告/输入10 passed/3.60s，两报告CLI跨Python3.11/3.13重建后766文件仍全部SHA一致，runtime package仍`098e4ec770108a3b4da457f16f46d86ac58f9dd0e6c46d146accb1ab4f2ff44c`。receipt/XML归档于续测目录，没有再次运行GPU计时。

## 2026-10-19 — 风险续测12请求完成，补充分支覆盖检查（执行准备）

- 执行冻结`41ecab699695c90653454c360516cbef816e3fd8`，729文件核验，源zip SHA`a429342d081f5477d1573b9ff023f9f28e40f21a4898d39d66630cfc68637dc1`，独立服务器`prefix-risk-continuation`。94相关测试通过/8.48s；流水PID2408502已完成，GPU双19MiB/0%。
- 实际60/60机制、120/120交错计时全部严格匹配旧12AR，120计数路径一致；复用全部12参考，未重跑AR或排除任务错误例。五组机制均64轮/32实际Draft前向，风险评估和改选均0：唯一拼接池也没有合格候选，风险模型未被调用。全部时延Suffix14.110045s、Guided14.139104s、Splice14.154433s、NoConfidence14.167224s、Risk14.155447s。小差异无候选/路径变化，不能归因为风险提速。
- 续测证据在`results/prefix-risk/continuation`；完整zip109037字节/SHA`b2af856cf52177ed7d624924b4afe1b84691911a9ebdee06a2f341856176f097`。初次打包忽略已压缩参考gz，补齐原字节后重建完整归档；生成结果未更改，无GPU重跑。主报告`PREFIX_RISK_CONTINUATION_RESULTS.md`，按全部12/主8/控制4及合同正确11/错误1分别报告。
- 为覆盖风险分支，单独补旧payment-runbook:1、shipment-contract:1两个训练项目，10机制+20计时，不并入主12请求、不声称泛化。原系数CPU核对payment/15边界14存活0.109、边界7存活0.868会改选；shipment/22原边界6存活0.875应保留。协议`PREFIX_RISK_CONTROLS_PROTOCOL.md`事先冻结；不改阈值/权重/候选，不增加新随机案例。
- 主报告首次跨进程重建因保存的diagnostic键排序导致描述性gate JSON行序不同，1 failed/7 passed；固定sort_keys，仅改报告渲染，不改结果或策略，失败XML保存。补充脚本py_compile通过；将冻结独立controls源码，原12请求数据按SHA保护。

## 2026-10-19 — 前缀风险续测（用户授权放宽评测停止条件，执行准备）

- 用户明确要求继续，认为原停止条件过严。新增`--continue-evaluation`显式模式，任务质量标签不再阻断解码检查，Risk零改选也完成120计时以测开销；硬性token/prompt/snapshot/来源/计数检查仍保持。原strict模式和首次停止证据保留。
- 不改风险策略、56状态训练系数、0.5阈值、候选池/长度/查询预算/Target输入；复用旧12个AR的全部参考（包括vector-ingest:0错误正文），不重新采样、不选11个成功例。校验旧参考SHA、输入/校准、固定推理设置和当前tokenizer prompt身份。
- 新协议`PREFIX_RISK_CONTINUATION_PROTOCOL.md`冻结60机制+120交错计时；报告全部12、主8/控制4，以及合同正确11/错误1的子集。已观察答案，称开发续测而非新holdout。独立服务器`prefix-risk-continuation`，新发布目录`results/prefix-risk/continuation`，旧文件SHA单独保护。
- 冻结前相关13 passed/7.67s：`.venv\Scripts\python.exe -m pytest -q tests/test_prefix_risk_continuation.py tests/test_prefix_risk.py tests/test_prefix_risk_probe.py tests/test_prefix_risk_report.py --basetemp .test-tmp-risk-continuation-related --junitxml=artifacts/risk-continuation-tests-related-local.xml`。runner/helper通过py_compile，diff检查通过，A6000空闲双19MiB/0%。

## 2026-10-19 — 前缀风险辅助拼接（实现完成，独立检查止于Target合同）

- 冻结执行`ad231f712858868c418ce33494dd967d5de8230c`，702个Git原始文件SHA核验，源zip SHA`315904bb2076b8ad328e04f44ac4b70de4b97111ef91eec91f0f10e9a0389255`，实际package SHA`ba8782a55669b9bf4647d99b729c1f5023bd7ea78a87165d0688f5c20986a943`。新独立服务器目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/prefix-risk`，部署即建立`models -> ../models`。
- 服务器相关88 passed/8.18s。流水PID2393802完成，实际执行`run_prefix_risk_probe --stage mechanism --config configs/runtime-ot-benefit.json --output results/raw/prefix-risk/mechanism`；先生成新12AR，共52.955987s，11/12合同正确、12/12 EOS。vector-ingest:0要求返回完整配置，Target仅输出`batch_size = 224`、漏掉全部其余内容，非截断。按冻结协议`stopped_target_contract`，未执行60机制、未执行120计时，不挑11个成功请求继续、不改提示重跑。
- 新策略尚无性能结论；没有把Target输出合同失败解释为风险策略无效。默认suffix/独立infra保留，三infra引用仍`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。GPU已恢复双19MiB/0%，无运行中的生成进程。
- 源码执行前后全部702文件SHA一致，旧401证据在服务器执行前后SHA一致。紧凑原始zip65207字节/SHA`86cef10850e129a91230d6223d26a625d5dbfa5615578740a6f6dbf8585fb746`；证据`results/prefix-risk`不含DB/权重/完整logits，命令脚本与新AR失败输出保留。
- 报告首次重建发现两份早期research-audit JSON本机CRLF与Git原始导出LF不同，报表仅对这两份只读核对规范Git blob，不改原文件或其他SHA检查；来源、输入、权重及协议准确核验。报告`PREFIX_RISK_RESULTS.md`，重建`.venv\Scripts\python.exe -m scripts.write_prefix_risk_report`。
- 追加风险改选/拒绝/KV的真实tiny Target组合测试；首版夹具空索引使候选分支未进入，1 failed/48 passed/2 skipped（输出仍严格一致），已提供非空测试历史，失败XML保留。此项是正确性测试，非GPU性能结果。
- 非空夹具进一步暴露真实边界bug：均匀分布的FP32归一化熵可因求和舍入略大于1，被范围检查错误拒绝；新增熵clamp[0,1]及均匀logits回归。此修复不改Target接受规则；旧ad231f7实验停止现场不重写，本轮未进入在线风险路径，最终交付代码单独验证。
- 修复后相关49 passed/2 skipped/10.17s；最终完整365 passed/3 skipped/39.69s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-prefix-risk-final-2 --junitxml=artifacts/prefix-risk-tests-final-local-2.xml`。此前完整回归1 failed/364 passed/3 skipped的熵边界现场保留。报告重建与归档6文件SHA核验通过；CI的dflash组补入全部prefix-risk测试。
- 最终交付执行`ea85be6a0dffea5fd570bf3684d3a691eb373c1f`到独立`prefix-risk-delivery`，723文件SHA全通过，zip SHA`7f8475bf396825d32d0ceb0cebaf87a1b6e6efdcabbb34d2eb5feff632cab548`，source SHA`50b91fbacb8424ce904e54b432db324ec698a649a09f2e25f2b5289fedee0195`，最终package SHA`ae2580992a6e364aebc4f40bd4254a38e745af1ebb73b3c41bf695fad022cd82`。服务器五份相关测试91 passed/8.32s，报告重建后723原始文件仍全部SHA一致，报告SHA`b709db48ec7c2312bf284700dd11ab1aa977a5f6a7f17baa614186bf129f0f34`。
- 交付复核首次漏建basetemp父目录导致82 passed/9 setup errors，创建artifacts后以新basetemp重跑通过，XML现场保留；未重跑新AR或风险GPU性能路径。交付receipt与两份XML保存`results/prefix-risk/`。相关命令：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -m pytest -q tests/test_prefix_risk.py tests/test_prefix_risk_probe.py tests/test_prefix_risk_report.py tests/test_runtime.py tests/test_dflash_reuse.py --basetemp artifacts/test-tmp-v2 --junitxml=artifacts/delivery-tests-server.xml`。

## 2026-10-19 — 前缀风险辅助拼接（执行准备）

- 用户授权执行一次小验证。新增`retrieval/prefix_risk.py`，原Edit候选边界生存<0.5且存在≥0.5候选时才按Edit顺序改选；同边界或单候选跳过置信度。固定原32上限、top8/4查询预算、严格Target及旧冷却，不预测尾部、不求解OT、不改动态长度。
- `RuntimeConfig.prefix_risk`与`edit-risk-splice`显式选项；matcher只接收当前实际Draft logits，不额外Target前向，在线成本全部计入。权重仅用旧56状态/9项目按原固定正则拟合；无置信度消融独立保留。
- 新作者构造6项目/12请求，4项目稀疏修改+长复制、2控制，均与训练项目隔离；不是生产负载或长期历史必要性证据。协议`PREFIX_RISK_PROTOCOL.md`冻结12AR合同门槛、60机制、若发生风险改选才120交错计时，未达预定净收益门槛就停止本轮，不按新输出调参。
- 准备脚本首次显式传.gz路径与旧read_rows的自动压缩回退约定不一致，修为传原jsonl路径；尚无新Target输出。首批相关测试2 failed/81 passed/2 skipped，原因是新增测试插入位置将旧benefit计数断言移入风险测试，推理输出检查均通过；最小修复将断言归位并检查risk计数，失败XML保留。
- 修复第一次归位时上下文不足又匹配到通用流式测试，第二批9 failed/76 passed/2 skipped；核对完整diff后恢复旧断言原位置。最终相关85 passed/2 skipped/14.47s，完整361 passed/3 skipped/40.36s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-prefix-risk-full --junitxml=artifacts/prefix-risk-tests-full-local.xml`。后补live tuple/JSON decision审计测试，prefix单测8 passed/2.75s。两个runner通过py_compile，diff检查通过；A6000双卡19MiB/0%。

## 2026-10-19 — 接受长度条件生存校准（完成，局部改善未达门槛）

- 冻结执行提交`36798f32c2992511ea89479be7fd70279dfc4f5f`，671个Git原始文件SHA核验；源zip SHA`2aae5e51ae81c41bd9842296c586c1c27efb4f7511d5a8d42abecfdb1979580f`，实际package SHA`b6ec034a21ab76ca2d57912a8599214d9943aa22b792377610f68fd6b7c5a435`。独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-acceptance`，原文件执行前后均SHA一致。
- 首次漏建models软链接，load_engine在生成前退出；失败日志保留。补`models -> ../models`后同源码运行完成，服务器相关45 passed/5.72s（首次无模型缓存时43 passed/2 skipped）。生成进程2327090已结束、GPU恢复双19MiB/0%。
- 实际运行协议中的`python -m scripts.collect_acceptance_probe --config configs/runtime-ot-benefit.json --output results/raw/acceptance/collection`和`python -m scripts.evaluate_acceptance_probe --input results/raw/acceptance/collection --output results/raw/acceptance/evaluation`，环境OMP_NUM_THREADS=4/HF_HUB_OFFLINE=1/PYTHONPATH=.；完整脚本保存`results/acceptance/run-pipeline.sh`。48/48生成严格匹配AR token/prompt/snapshot，24/24旧matrix路径计数一致，全部旧证据递归SHA不变。
- 56去重Draft状态/69shadow候选/9项目，完整匹配14原状态；每次排除当前整个项目，固定特征、正则与参数。旧/无置信度/有置信度MAE分别10.472736/13.917830/8.423701，Brier0.234529/0.276227/0.155672；新比旧MAE降低19.57%，未达预定20%。按项目等权MAE改善0.962952，5000次bootstrap区间[-5.042062,6.926308]跨零，开发门槛失败，不接入默认路由、不调参凑过线。
- 有效信号：56状态Draft MAE无置信度5.042924→2.122956；payment/11预测32→5.57实际3；shipment/14预测30.89→8.03实际7；shipment/22预测30.56→7.34实际6。payment/15恢复TailOT优于Edit、shipment/22恢复Edit优于TailOT的候选排序。
- 尚未修复：原14动作平均偏差+4.18→−7.50，对原本正确的长接续明显低估。Draft概率仅覆盖前15位置，16–32尾部依赖少量跨项目条件标签；样本、正则和尾部特征各自作用尚未因果区分。保留前缀风险模块，未来优先独立项目风险排序/条件尾部校准，不将本次离线改善当作在线加速。
- 原始归档116833字节/SHA`1479a35f1c04cee9cc61005c1dcc463d1009f7d91fa606688c2bfcfc2cb1921a`，紧凑证据在`results/acceptance`，报告`docs/ACCEPTANCE_RESULTS.md`。不含数据库、权重、完整词表logits；报告按服务器已执行系数重算，不跨SciPy版本重新拟合。报告/审计10 passed/6.92s，重建命令`.venv\Scripts\python.exe -m scripts.write_acceptance_report`。
- 最终完整回归351 passed/3 skipped/48.27s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-acceptance-final --junitxml=artifacts/acceptance-tests-final-local.xml`。此前一次自动审批因额度耗尽未启动该命令，用户恢复额度后运行完成；不是测试失败。原infra三引用仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。

## 2026-10-19 — 接受长度条件生存校准（执行准备）

- 针对旧14状态MAE10.47，新增 `retrieval/acceptance.py`：Draft前缀生存 × 边界通过后的历史条件生存，候选分别预测，固定正则、逐项目排除；不使用完整OT矩阵，不更改默认路由。
- decoder新增默认关闭的只读 `on_draft` 诊断接口；读取现有logits，不额外Target前向。`collect_acceptance_probe` 固定16旧请求×3旧路径48次，全部strict检查；旧matrix24路径计数对照，原14状态须完整Draft和候选匹配。shadow候选越过线上调用预算仅作离线诊断，标签strict通过后添加。
- `evaluate_acceptance_probe` 对比旧预测、无置信度、有置信度；原14固定动作MAE/Brier、全部状态、候选排序、按项目bootstrap。协议 `ACCEPTANCE_PROTOCOL.md` 预先冻结通过条件，不按结果反复调参。本轮无速度声明，无独立holdout。
- 首轮模型/decoder测试41 passed/2 skipped/17.02s：`.venv\Scripts\python.exe -m pytest -q tests/test_acceptance.py tests/test_dflash_reuse.py --basetemp .test-tmp-acceptance-unit`。将使用新服务器目录 `ot-r-acceptance`，保护全部旧证据和独立infra版本。
- 冻结前相关43 passed/2 skipped/9.83s；完整347 passed/3 skipped/60.79s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-acceptance-full --junitxml=artifacts/acceptance-tests-full-local.xml`。两个runner通过py_compile，diff检查通过，14旧Draft重建无歧义。完整回归有一条既存python_multipart弃用警告。

## 2026-10-19 — 预计收益OT门控/动态长度（完成）

- 按用户要求保留Draft内部拼接，停止完整矩阵可靠性路线，新策略只沿用既有TailOT标量排序。正式执行冻结`7c10c5dad39ffd0d909ae577b87a8026d39e2523`，635个Git原始文件SHA核验；源zip SHA`fa8e5f843b3956ed51b662ff8be0b9aa1905c5f6559b478b280b8f562ca48d44`，实际package SHA`1d7947ea6371e8c54c63ff382f46f6eee9de413cd87531c3cf6ccbc6eabbdd43`。独立服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-budget-v2`，流水PID2305668已结束，双GPU恢复19MiB/0%。旧结果含matrix全部递归SHA不变。
- 实际依次运行`python -m scripts.profile_benefit_costs`、`probe_benefit_offline`、`run_benefit_probe --stage mechanism`、`run_benefit_probe --stage timing`，完整配置/命令见`OT_BENEFIT_PROTOCOL.md`与`artifacts/run-benefit-pipeline.sh`（服务器）；输出`results/raw/ot-benefit/{cost,offline,mechanism,timing}`。共享基础配置`configs/runtime-ot-benefit.json`；跨项目校准数据由runner注入，不自动启用旧训练数据。
- context256/768/1536、候选15/16/24/32、2预热/7重复核测量；Draft约12.48–13.86ms，Target验证约59.64–89.97ms，共享shortlist/policy中位约3.368ms、每候选TailOT2.260ms。14状态/6项目的旧数据逐一排除当前整个项目：OT调用2，缩短0，回退Draft0；预计接受376.53，实际318，平均绝对误差10.47。策略/阈值/长度未按这些结果调参。
- 正式56/56机制、112/112未插桩交错计时均严格匹配原AR token/prompt/snapshot，112/112计时路径与机制计数相同。总时延Suffix14.177480s、Guided-Edit13.515785s、Splice-Edit13.631645s、TailOT13.614736s、Gate-OT13.828701s、Budget-Edit13.633580s、Budget-OT13.828517s。Budget-OT比Suffix低2.46%，比TailOT高1.57%，比Guided-Edit高2.31%；不把基础拼接收益归因于新策略。
- 门控OT候选求解31→7、调用9→2，但总验证65→67轮，实际Draft前向仍31；Budget-Edit与Edit相同66轮，三个收益组均没有真正缩短候选。shipment/22预测OT接受30.56实际6而Edit32；payment/15选Edit仅7，漏掉TailOT32。减少求解成立，条件收益预测不准；动态长度未体现额外收益。下一步应先补独立的廉价Draft前缀风险/接受生存预测校准，而非扩大当前邻居策略或在同批换阈值。
- 证据`results/ot-benefit`，报告`OT_BENEFIT_RESULTS.md`。原始归档170881字节/SHA`f2cac0ac41b755e5fb9ce62b67391b2579f481e8df2bedeaed05bce607a9298d`，包括首版250ab31的56行汇总失败现场；不含权重、DB、大型矩阵或私有历史。报告重建`.venv\Scripts\python.exe -m scripts.write_benefit_report`。跨Python3.11/3.13浮点sum出现1e-14量级差异，报告统计重算采用1e-12容差并保留原始记录字节；路线、长度、token IDs仍精确检查，Target验证完全未放宽。
- 正式服务器相关92 passed/8.16s；本机策略/报告15 passed/7.24s；最终完整340 passed/3 skipped/42.53s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-benefit-final --junitxml=artifacts/benefit-tests-final-full-local.xml`。首版审计失败与一次新增单测夹具错误均保存/说明，修复不改推理策略。
- `main`、`codex/infra-interview-v0.3`与标签`infra-v0.3-interview^{}`仍为`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`；默认suffix不变。本轮为开发检查，未增加独立holdout或通用性能声明。

## 2026-10-19 — 收益路由汇总口径修复

- 首版执行提交`250ab31a9f10dcf6f9c7ba470d3d5a9e5bc7c1bb`，服务器 `ot-r-budget`，634文件SHA核验；91项相关测试通过。成本/14状态离线阶段完成，56次生成均通过逐请求strict检查，但机制最终audit将拼接归因恒等式错误应用于Guided-Edit，抛 `Invalid benefit accounting`，未进入计时。
- 最小修复仅限定该恒等式检查到真正拼接组，新增包含Guided-Edit非零接受量的审计回归。策略特征、邻居、阈值、长度和运行范围不变。保留首版失败现场；以新冻结版本/独立 `ot-r-budget-v2` 重跑正式小检查，不把首版汇总失败混入成功计时。

## 2026-10-19 — 保留内部拼接，预计收益调用/动态长度（执行准备）

- 用户决定舍去完整矩阵可靠性路线；旧实验不删除，新路由不使用矩阵特征。新增 `retrieval/benefit.py`，用历史已验证接受长度预测 Draft/Edit/TailOT 三个动作的推进量，并按实测成本比较预计rate；OT需要比最佳廉价动作高2%才调用，单候选不调用；拼接长度16/24/32。保留精确suffix优先与原始Draft/历史span、严格Target验证和预算。
- 校准固定14个已观察状态；每次排除当前整个项目，其他每项目最多一个邻居，最近3项目加权。不读取当前答案作在线决策、不动态调参。因数据稀少且全部已观察，只是交叉项目开发验证，不称独立holdout。协议 `OT_BENEFIT_PROTOCOL.md` 写明局部rate/成本档/pending近似限制。
- 新增成本微基准、离线检查、七组GPU runner，以及runtime显式可选入口；默认suffix和独立infra版本不变。首轮发现缺失校准抛TypeError及pytest默认临时目录不可写，已改明确配置检查/项目内basetemp。随后15 passed；相关86 passed/2 skipped/13.88s，命令 `.venv\Scripts\python.exe -m pytest -q tests/test_benefit.py tests/test_runtime.py tests/test_dflash_reuse.py tests/test_splice_reuse.py --basetemp .test-tmp-benefit-2`。
- A6000双卡19MiB/0%，使用新独立 `ot-r-budget` 目录；先冻结源码、完整回归及成本测量，再固定八个旧请求×七组机制56次、交错计时112次。不用本轮结果反复换参数。
- 冻结前完整回归334 passed/3 skipped/38.90s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-benefit-full --junitxml=artifacts/benefit-tests-full-local.xml`。三份新runner通过py_compile；`git diff --check`通过。

## 2026-10-19 — OT完整矩阵三个小改动（完成）

- 冻结执行提交`81a0de103f95cf3c990f9fcbd4947276e70723c0`，601个Git原始文件SHA全通过；源归档SHA`7f89cbd431503d11d6c27feac409a6261456ae19b8f044b47447aae0db8445c0`，实际package SHA`a5e135401d0f7558a100d3cd1c4fa19a83e70c43c05e6099183c228df6011c33`。服务器独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-matrix`，相关94 passed/8.26s；原始已发布结果含两轮拼接保持递归SHA不变。
- CPU命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.probe_matrix_offline --tokenizer models/qwen3-8b --output artifacts/offline-v1`。135次旧调用及提议全部重现，14个去重可评分状态/39候选。相对原OT，B选择变化0，C4，组合2；同状态连续接受总和Edit344、OT318、Tail339、B318、C301、组合314。该总和不是在线新增token或速度，不作独立样本统计。
- 按`OT_MATRIX_PROTOCOL.md`实际运行64次机制检查、128次未插桩交错开发计时，192/192输出token/prompt/snapshot严格匹配原AR；128/128计数路径与机制回放相同。流水PID2285057已完成，双GPU恢复19MiB/0%；CPU/GPU阶段相同package，未按新评分改参数。所有任务早已观察，不是新holdout。
- 总时延Suffix14.164274s、Guided-Edit13.496111s、Splice-Edit13.627771s、原OT13.912373s、TailOT(A)13.609542s、OrderOT(B)14.088728s、PlanOT(C)14.051844s、All13.890790s。B相对原OT+1.27%、C+1.00%、All-0.16%；组合相对TailOT仍+2.07%、相对Guided-Edit+2.92%。新三个组没有超过已测TailOT，保留局部正负结果而不继续在本批换阈值。
- B总轮与原OT同为67但多一次求解；C提议9→5、选中却未接受历史的状态4→1，却误放弃两处池内可用接续，总轮67→69。具体旧JSON可接受32的候选连续性0.469被0.5阈值拒。runbook正确32-token候选连续性0.397被拒，错误候选保留14个Draft但仅接受7、连续性反而0.704。既有未校准阈值误拒，也有上下文对齐与Target接受的排序错位，不能只靠降低阈值宣称解决。组合总轮66，仍高于Tail65；C在JSON负例减少部分浪费但未胜过便宜Edit。
- 结果`results/ot-matrix`，报告`OT_MATRIX_RESULTS.md`。原始zip99659字节/SHA`f956951905b89cb2339117a45375d5324c1aba1fbfca4edbf6ea55b7f8dad532`，仅发布紧凑特征、调用与来源，不含大型矩阵/权重/数据库/私有历史。重建`.venv\Scripts\python.exe -m scripts.write_matrix_report`。
- 本机完整314 passed/3 skipped/34.15s；追加报告/状态审计16 passed/13.83s；补充逐候选解释后最终报告8 passed/10.70s，含真实CLI跨hash-seed证据字节保留。默认suffix、独立infra保持原版本；本轮未实现预计收益调用或动态长度，不把固定启发评分称为接受概率。

## 2026-10-19 — OT完整矩阵三个小改动（执行准备）

- 用户授权测试接续可靠性、完整矩阵与三个小改动。新增`retrieval/matrix.py`：A沿用已有尾部加权；B用初始正确匹配流的相邻行1–4步顺序支持修正真实token代价，再求解一次；C用锚点前末8行支持度/集中度/连续性/dustbin质量排序或放弃；组合A+B+C独立。均为固定词面OT启发，不是校准接受概率/硬单调约束/OPW复现。参数先写入`OT_MATRIX_PROTOCOL.md`，不按新评分结果调参。
- 旧`splice.py`仅抽出可覆盖的`candidate_score`，原Edit/OT/Tail公式不改；新增显式`ot-order-splice`、`ot-plan-splice`、`ot-all-splice`和`runtime-ot-matrix.json`。候选池、长度32、预算4/top8及严格Target验证不变，未实现按预计收益调用OT。
- CPU脚本先恢复原16请求及五请求修复的三种Splice实际状态，核对旧提议/用户时间快照后才进行同池逐候选评分；只保存紧凑矩阵特征，答案只用于事后连续接受评价。若新组产生选择/放弃差异，再固定八个既有请求×八组=64次GPU机制检查；达门槛才做128次交错未插桩开发计时。
- 首批48 passed/15.69s；相关91 passed/2 skipped/12.95s，命令`.venv\Scripts\python.exe -m pytest -q tests/test_matrix.py tests/test_matrix_probe.py tests/test_splice_reuse.py tests/test_runtime.py tests/test_dflash_reuse.py --basetemp .test-tmp-matrix-related --junitxml=artifacts/matrix-tests-related-local.xml`。新增同质量逆序/锚点自证排除/两次求解边际/同池Edit记录评分测试。完整回归正在完成。
- A6000双卡19MiB/0%，将用独立`ot-r-matrix`目录，全部已发布旧结果含冷却修复递归SHA保护，独立infra不覆盖。
- 冻结前完整回归314 passed/3 skipped/34.15s：`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-matrix-full --junitxml=artifacts/matrix-tests-full-local.xml`。未发生本轮测试失败；下一步以冻结Git原始文件部署CPU检查。

## 2026-10-19 — 报告跨平台重建的字节保留修复

- 首次最终交付在Linux重建首轮报告后，修复报告被旧证据SHA保护拒绝。确认诊断JSON语义完全相同，但字典键顺序变化：原SHA`cd81e60688775121a25ca2e4cffd57fa9557bd7b1f0acbf46303d4957e73b118`，重建SHA`bc3ab55d18d00ff1e68c7dac01ffd17d16e7662631e07c35e66938f7f047cd14`。不是模型输出或数值结果变化；原GPU目录和Git原证据不受影响。
- 两个报告命令现在对语义相同的诊断保留已有原始字节，新文件采用固定排序JSON；不改历史出版物。增加实际CLI子进程、不同hash seed/等价JSON序列化的回归。`python -m pytest -q tests/test_splice_report.py tests/test_splice_cooldown_report.py --basetemp .test-tmp-splice-rebuild --junitxml=artifacts/splice-report-rebuild-tests.xml`：12 passed/15.24s。只修复报告写入，不改已计时package、runner或协议；无需重跑GPU实验。

## 2026-10-19 — 拼接冷却最小修复（五请求检查完成）

- 修复执行提交`c98bb83e3e50a0ba1c01268317b074a2fcfcbbd4`，572个Git原始文件SHA全部通过，归档SHA`01568feab9c0d830900ec0bae95e8ac8f5b94feaa710e9645f50d803986bc545`。服务器`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice-cooldown`，预执行90 passed/8.20s。实际package两阶段同为`89063c6e3007034cb72823d5a42494f26258baf35b7cbf9beaa74dbd0ee2513c`；全部旧证据含首轮拼接SHA保持不变，不覆盖原执行源码清单。
- 实际命令见`OT_SPLICE_COOLDOWN_PROTOCOL.md`，使用`--cooldown-check --stage mechanism/timing`和独立`artifacts/mechanism-v1`/`timing-v1`。流水PID2194458完成；30/30插桩检查、60/60未插桩计时严格匹配既有AR，60/60轮数/接受/候选和Draft前向计数匹配插桩。双GPU恢复19MiB/0%。没有新holdout或进一步GPU扩展。
- 冷却边界修复正确，但只恢复部分机会：payment-runbook:1中TailOT总轮8→7，普通Edit/OT仍8；shipment-contract:1中Edit总轮9→8，两OT仍9。三个正例各方法轮数保持。剩余问题有直接记录：runbook位置15普通Edit/OT保留14个Draft但仅前7个正确，TailOT保留7个并接受32；JSON位置22 Edit接受32，两OT保留8个但只接受6。可匹配内部锚点不保证其前面的Draft正确，不把剩余失败继续归于冷却。
- 五请求两重复总时延Suffix8.926047s、Guided-Edit8.482442s、Guided-OT8.522107s、Splice-Edit8.540418s、Splice-OT8.780651s、TailOT8.474506s。TailOT相对Guided-Edit仅-0.09%（基本持平），普通OT+3.52%。SQL两请求均保留各少1轮的局部收益，Edit同样有效；旧JSON TailOT保持单例少1轮。不能外推修复后的16请求整体性能或宣称稳定OT优势。
- 结果`results/ot-splice-cooldown`，报告`OT_SPLICE_COOLDOWN_RESULTS.md`。原始zip65857字节/SHA`01c55d562ad25f7418265a68e6280adba3a8582c7a6abfbe5f3cc07a7eab2370`；报告重建`.venv\Scripts\python.exe -m scripts.write_splice_cooldown_report`。追加两个报告/来源审计18 passed/5.77s；补充逐位置解释后最终10 passed/5.51s，原完整290 passed/3 skipped记录保留。main、infra分支和tag仍均`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。
- 本阶段结论：保留内部桥接和尾部加权为显式研究选项，默认suffix；若继续，应先离线解决保留Draft前缀的风险和同状态排序，而不是再扩大同类GPU负载。

## 2026-10-19 — 拼接前缀拒绝的冷却修复（五请求检查准备）

- 首轮代码/原始证据已独立提交`0000e5c`/`bc6b39e`。共享解码器只改一行冷却条件：`draft_prefix <= accepted < len(proposal.tokens)`时才冷却；尚未尝试历史的Draft前缀拒绝不触发冷却，普通R/原Guided默认prefix=0仍保持规则。匹配器此前已避免把前缀拒绝映射成历史端点失败。
- 协议`OT_SPLICE_COOLDOWN_PROTOCOL.md`固定两个负例、两个SQL正例、一个旧JSON正例；runner新增`--cooldown-check`限定这5个已观察请求，六组30次机制/60次未插桩开发计时，不做新场景确认。首轮`results/ot-splice`也加入递归SHA保护；服务器新目录`ot-r-splice-cooldown`，证据独立`results/ot-splice-cooldown`。
- 新增冷却0/1、Draft前缀0/1位拒绝、首个历史token拒绝、EOS等真实小Target回归，相关56 passed/2 skipped/10.78s；完整290 passed/3 skipped/25.00s。命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-splice-cooldown-full --junitxml=artifacts/splice-cooldown-tests-full-local.xml`。仅相关策略被修复，未改检索阈值或Target验证。

## 2026-10-19 — Draft内部锚点拼接（首轮完成，定位冷却缺陷）

- 冻结执行提交`0000e5c9f78df0d762d2dba981daafa8a59bd2bb`，550个Git原始文件逐项SHA通过；源归档SHA`c09491b515eddd94b4895a0e5b6b1a8daf28e20c353a9ff51ff7482c1900fa96`。服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice`，执行前相关72 passed/7.52s。实际机制/计时package同为`44f733dea98734d16c18a3b447207a197fba1e7ec4e14255b36e4e376048cd1a`，runner/协议/全部旧证据SHA未变。
- 16个已观察请求×6组一次机制96/96严格一致。三个Splice均在inventory-query两个请求各少1轮，达到预定小成本门槛；收益只有一个项目，不是独立场景确认。按同一提交进行192次未插桩交错计时，192/192输出token/prompt/snapshot严格一致，192/192轮数/接受量/候选与机制回放相同。实际命令见`OT_SPLICE_PROTOCOL.md`；机制PID2185996、计时PID2187541均完成，双GPU恢复19MiB/0%。
- 总时延Suffix26.000739s、Guided-Edit25.018151s、Guided-OT25.153341s、Splice-Edit25.285204s、Splice-OT25.421422s、TailOT25.258074s。三个Splice分别比Guided-Edit慢1.07%/1.61%/0.96%。SQL项目两重复四观察：Guided-Edit3.221949s，Splice-Edit3.014994s；三种拼接都恢复32token，第二请求来源是当前prompt，不能归为历史独特价值。普通OT/Edit总轮/接受量相同，15/16请求完整候选token调用序列相同；TailOT在一个旧JSON请求少1轮，总轮119→118，但整体仍无净优势。
- 逐轮定位退化：payment-runbook:1位置11保留11个Draft token但仅接受3；shipment-contract:1位置14保留9/15但仅接受7，均在Draft前缀内拒绝。原D也接受3/7，但Splice被记为R触发1轮冷却，跳过下一轮位置15/22的32token机会，各多1轮。保存首轮原始负结果，下一步只修复此反馈策略并检查5个既有关键请求，不改Target或调检索参数。
- 证据`results/ot-splice`，报告`OT_SPLICE_RESULTS.md`；原始zip80948字节/SHA`cf741ecfd7088c84de6f319a405b6472b3bd9098b4b56d7d8fb913109ca7f17f`。报告/归因审计14 passed/5.41s，追加解释后纯报告6 passed/4.91s；完整272 passed/3 skipped和原始XML均保留。重建`.venv\Scripts\python.exe -m scripts.write_splice_report`。

## 2026-10-19 — Draft内部锚点拼接（小实验准备）

- 用户授权小成本实验。新增`retrieval/splice.py`与`edit-splice`、`ot-splice`、`ot-tail-splice`显式选项，复用现有DFlash post-draft hook；保留Draft到内部4-token锚点的前缀，再拼原始历史尾部，最多32token。三个Splice使用同一Edit去重/top8池、4次查询预算、0.65阈值；TailOT仅末8位置加权，不声称单调OT。原检索/解码算法和默认suffix不改，独立infra版本保留。
- `OT_SPLICE_PROTOCOL.md`预定16个已观察请求×6组一次插桩机制检查；同一个Splice至少2请求比两个Guided基线都少轮且存在内部连续接受增益，才运行相同实现的两重复未插桩开发计时。无新holdout，不据此宣称泛化。参考仅生成后评分，全部旧证据递归SHA保护，不写回答案。
- 相关测试首批62 passed/2 skipped/35.00s；加入真实小Target拼接反馈、EOS/KV及来源/归因审计后70 passed/2 skipped/13.21s。命令`.venv\Scripts\python.exe -m pytest -q tests/test_splice_reuse.py tests/test_splice_probe.py tests/test_dflash_reuse.py tests/test_runtime.py --basetemp .test-tmp-splice-final --junitxml=artifacts/splice-tests-final-local.xml`。
- 服务器已连通，双A6000各19MiB/0%。将部署独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-splice`，沿用父目录模型和`.venv-dflash`，先完整测试再冻结执行源码。尚无本轮GPU结果。
- 冻结前完整回归272 passed/3 skipped/25.25s，命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-splice-full --junitxml=artifacts/splice-tests-full-local.xml`；新增模块/脚本compileall与`git diff --check`通过。

## 2026-10-19 — DFlash前缀引导的OT/Edit（完成）

- 执行提交`6f9303625ddf01b1c82938b2f1184a08e7c83141`，517个Git原始文件部署SHA逐项通过；原始归档SHA`3799e0d8649a8e0d85dbe0b3010afbbef7dbea965955ba8e515fe5572dbdb9de`。服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-guided`；开发/新验证实际package同为`58dc6fdbd73ca3d404b521147ad3e8227df5f05bc022fda79084fe79ae88a466`，runner相同，未在新答案上调参。任务SHA`0097a10eaa102db6c866326ca145c3842657103ee400c892c110b99e4cf2c2b4`，协议SHA`0f1eca2375b6e5bf19023cabd87d403913565d541597445b33b569eb4a1cdc80`。
- 开发命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_guided_validation --stage development --config configs/runtime-dflash.json --output artifacts/development-v1`。PID2136237退出，32/32输出token/prompt/snapshot严格一致；两旧长正例Guided-OT/Edit均保留32token接受。旧插入负例的长接续恢复；短输出/未完成新内容误调用减少。开发计时不当独立确认。
- 新验证命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_guided_validation --stage holdout --config configs/runtime-dflash.json --development artifacts/development-v1 --output artifacts/holdout-v1`。PID2137526退出，先16/16新AR契约正确/EOS，再160/160五组两重复严格一致。全部在线成本计入，历史不写回，旧结果SHA不变，两GPU恢复19MiB/0%。
- 新主场景：DFlash16.784681s、Suffix12.724529s、原OT12.928572s、Guided-Edit12.400227s、Guided-OT12.421961s。引导OT比Suffix低2.38%、比原OT低3.92%，比引导Edit高0.18%；对Suffix仅2/4项目更快，项目bootstrap比值[0.9483,1.0025]，预定工程门槛未通过。新全部请求：引导OT较Suffix低1.25%、较原OT低3.77%；control较Suffix高0.74%。这些是作者受控任务，不宣称生产泛化。
- 主场景总轮120→114，实际Draft前向50→50；引导R已经支付Draft，不能把D路线44说成只算了44次Draft。较长接续减少Target验证轮。引导OT/Edit在新32/32请求-重复配对的轮数、接受量、调用/候选/提议和Draft前向相同，均256对齐接受token；主场景各192。没有新证据支持OT排序增量；TOML/JSON两个项目的局部收益为事后格式拆分，不能升级为独立场景确认。
- 本机最终完整回归242 passed/3 skipped/24.53s，服务器执行前相关61 passed/7.16s；追加报告篡改/配对/来源审计最终11 passed/5.27s。原始失败和修复XML一并保存。报告命令`python -m scripts.write_guided_report`，Windows/Linux正文SHA均`b1924812d8d974940f27bbbad96f3131c35e473d981b0b3dddd792d79c9aaaca`，diagnostic SHA均`41bc4d7b55c3a6ab326b5cf7991a0e8aa784de28ca7839fef72cd8e943dbf000`。
- 结果`results/ot-guided/{development,holdout}`，报告`OT_GUIDED_RESULTS.md`；原始下载zip65728字节/SHA`4b85b66e58f1837a5da8e3e8334a8c028fbc237e4893df9e453cf027b1e58d64`。只发布紧凑受控任务结果、来源与测试，不含权重/数据库/私有历史。保留前缀引导为显式工程实验选项，默认suffix，独立infra面试版不改；不扩大同机制OT试验或把引导收益归为OT独有贡献。

## 2026-10-19 — DFlash前缀引导的OT/Edit（执行准备）

- 用户授权进行下一步。新增`retrieval/guided.py`，精确suffix仍优先；未命中后复用已有DFlash前4个候选作为接续锚点，最多64后缀项/8候选/4查询，候选需长于现有Draft块。OT/Edit使用同池同预算，未改Target输入或逐token验证。`draft_forward_rounds`单列实际Draft计算，避免把引导R计为跳过Draft。旧`transport.py`不改，默认仍suffix，独立infra保留。
- 机制参考LogitSpec（arXiv:2507.01449v3）预测token引导检索与SAM-Decoding模型/检索组合；没有复制新增上游代码。固定协议`OT_GUIDED_PROTOCOL.md`，先旧8请求×4组一次开发，再新8项目16请求×5组两重复验证。新任务在首个新Target答案前生成，全部方法共享给定源文，不写回当前输出。任何新AR契约不符或截断均停止正式计时。
- 首轮相关测试51 passed/2 skipped/2 failed，发现suffix库无匹配返回空位置，已补`count==0`处理；随后53 passed/2 skipped/11.78s。新增fixture/audit检查曾因tuple/list直接比较失败，已仅在测试中标准化JSON表示，任务字节不变；对应6 passed/0.65s。首次完整回归241 passed/3 skipped/1 failed为同一fixture测试，修复后将运行完整检查。原始XML在artifacts/guided-tests-*.xml保留。
- 服务器双A6000空闲；下一步部署独立`ot-r-guided`目录，不覆盖原诊断/实验/infra。完整在线成本纳入，hint只是未确认候选，不作参考答案或已确认上下文。

## 2026-10-19 — OT-R 在线错误定位（完成）

- 固定执行提交`8a40def7e0b87a3d5090a5d11201ff90c2fd4d33`，491个Git原始文件部署逐项SHA通过；归档SHA`cd8286531f1d6cc9f9d72f90fdc4ec6e0789611aa7bd4fe6daa0b70695aa1e4e`。服务器新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-diagnosis`，仅使用既有模型与`.venv-dflash`，原实验和infra不覆盖。首次部署检查遇系统Python不支持`Path.is_relative_to`，尚未解包/启动模型；改用既有Python3.11后校验通过。
- 实际命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.diagnose_transport_online --config configs/runtime-dflash.json --output artifacts/diagnosis-v1`。PID2116743已退出，24/24输出token/prompt/snapshot严格一致；全部计数与原未插桩两重复逐项一致。执行package SHA`35603a53734dde44673730c53abacbe5756e27dab24534149742dcd3120bcfc5`，原GPU package文件全保持原SHA。参考答案仅在生成后离线评分，生产检索/验证代码未修改。
- 新五请求六次OT调用：四次合法原始span最好也不足8token、一次pool漏召回、一次32token全接受。parcel-queue新值/新字段尚未完成即复制旧文本；同状态DFlash能推进更多。payment-runbook插入Region行后的长Severity接续不在池内，三个稀有新词只命中修改指令，早期Owner锚点受当前倒排最小上下文范围限制，固定±2偏移未能恢复端点。没有将该组问题简单归因于OT池内排序。
- 单点事后召回反事实：候选数8→32仍无长span；偏移2→5后池中出现32token，但OT/Edit仍选0token。该CPU诊断不改变线上参数，也不作速度/独立验证证据。可复算`python -m scripts.diagnose_transport_shortlist --source results/ot-diagnosis --tokenizer models/qwen3-8b --output artifacts/shortlist-analysis.json`，服务器重建JSON语义完全相同。
- 原未插桩逐请求拆分：payment-runbook:0确实OT较suffix降低4.78%延迟（Edit也成功），另一个插入请求增加22.09%，整体平均掩盖了局部成功。confirm-schema-0:0的Edit在端点129/130分数同为0.9375，按位置取早端点接受0；OT约0.934891/0.941865，选130并接受32。旧正例有真实消歧，但不证明OT优于更好的便宜平分规则。不能用强制末token精确匹配作简单保护，该成功OT的exact_tail=0。
- 本机完整回归216 passed/3 skipped/35.09s；服务器执行前22 passed/2.37s；追加报告/比较/篡改拒绝后本机17 passed/5.32s。命令和XML见`results/ot-diagnosis/tests-*.xml`。报告复算`python -m scripts.write_transport_diagnosis_report`，Windows/Linux正文SHA均`f7088bd4203c82120a23bc5e6ed2b9396b721463bcbac7ff9da46cbc0a8c976e`、diagnostic SHA均`2019028e0829c1a946fb300d2e967d8b6f21f66cb8b0924a9a190c4b38442003`。压缩的紧凑诊断与来源清单保存于`results/ot-diagnosis`，不含权重、数据库或私有内容。
- 结论：先减少编辑区/结束处无收益调用，再处理插入后端点召回及接续排序，保留旧正例回归；求解器提速排在后面。本阶段完成错误定位，未实现新路由，也未重新宣称性能提升。两张GPU已恢复19MiB/0%。

## 2026-10-19 — OT-R 在线错误定位（执行准备）

- 用户要求先定位错误。新增独立诊断模块/脚本和协议`OT_R_DIAGNOSIS_PROTOCOL.md`，只包装既有解码入口捕获实际提议与轮记录；生产检索/解码文件不改。范围为新验证五个实际触发请求、首轮两个局部正例及copy负例，八请求×Suffix/Edit/OT一次，共24次插桩运行，不作性能测量。
- 输出先与已保存AR的token/prompt/snapshot严格核对，再离线重放同预算、阻塞状态下的实际提议；分开枚举合法原始span、合格端点及实际候选池。参考答案只用于事后评分，>=8 token是诊断分层而非收益阈值。
- 本机命令`.venv\Scripts\python.exe -m pytest -q tests/test_transport_diagnosis.py tests/test_transport.py --basetemp .test-tmp-ot-diagnosis --junitxml=artifacts/ot-diagnosis-tests-local.xml`：22 passed/14.70s。服务器双A6000均19MiB/0%，将用新隔离目录运行；结果尚未产生。

## 2026-10-18 — OT-R新场景验证（完成，按门槛停止）

- 修复后在`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-validation-v2`运行，冻结提交`45ec2ef995bd7521c579c2fe998fcd35c9c20c31`，PID1658928已结束，两GPU恢复19MiB/0%。24个新AR输入/快照核对、192/192计时逐token严格一致；两次24个AR的tokens/text/prompt/snapshot/finish_reason/contract逐项相同。任务22/24正确，两个格式控制失败；两个候选适用组均8/8正确，无截断，不剔除失败请求。
- 四组总秒数DFlash43.902004、Suffix35.750338、Edit36.243279、OT36.375919，OT相对Suffix+1.75%、相对Edit+0.37%。local-edit分别13.357009/13.752354/13.904043（Suffix/Edit/OT），OT+4.10%；ambiguity15.294088/15.276154/15.249713，OT-0.29%，但对齐调用0且解码计数相同，不能归因于OT；control7.099240/7.214771/7.222163，OT+1.73%。两个预先指定候选场景均未通过工程门槛，停止扩大同类测试/在本批答案上调参，保留显式实验选项，默认suffix。
- OT/Edit的48个请求-重复配对中D/R轮数、R接受量、对齐调用/候选/提议/接受量全部相同；两者对齐均接受66token，查询0.211308s对0.005951s。仅parcel-queue/payment-runbook/short-settings三个项目触发对齐。Suffix→OT：D146→142、R180→190、总326→332，R接受4366→4388。66个对齐接受token不能当成相对Suffix新增66token，更不能据此声称加速；新增验证与查询没有换来净收益。
- 实际命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_transport_validation --config configs/runtime-dflash.json --output artifacts/ot-validation-v2`。实际package SHA`7201d6615cd38402d74aefb82c6d941e9048863c65bd8271fa5b1eca2a3ce03c`，首轮42个既有package文件SHA全部保持；本轮只新增评测模块。461文件部署逐项核验。运行前任务/协议来自226399c，修复仅评测字段；失败目录和原执行清单均保留。
- 本机修复后完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-validation-final --junitxml=artifacts/ot-validation-tests-final-local.xml`：205 passed/3 skipped/20.71s；服务器87 passed/8.03s。之后纯报告与失败参考审计新增检查，本机23 passed/2.20s，实际XML另存。冻结修复提交CI成功39141490756。原始下载ZIP103252字节，SHA`40f9e589e238980593dbd19c1c1bef200c883e18e662d83ac3cdef84455e81df`。
- 原始计时/参考/失败记录、执行清单、测试XML和发布SHA保存在`results/ot-validation/`，报告`OT_R_VALIDATION_RESULTS.md`，复算`python -m scripts.write_transport_validation_report`。这些是12个作者编写受控项目，不能当生产流量或宣称OT普遍无效。完整infra面试版本423文件及三个固定ref重新核验仍为27e3619，研究不覆盖main或原infra服务器目录。
- 最终报告审计两平台各23 passed（本机2.14s、服务器1.41s），独立重建正文SHA均`c9572d3f26ac99d6526d2fee018c5b97fef59036ff09ff74aa4f4607c6da0545`，diagnostic SHA均`7071844474062ce4c6f77f581e794feb43852e05fe586b317820e8001380c322`。发布24份小型证据加publication.json，总约296KB；最终package与实际GPU package逐文件完全相同。之后只有报告/测试/交付文档更新，执行源码包、原失败包和最终交付清单分别保留。

## 2026-10-18 — OT-R新场景验证（运行前冻结）

- 用户授权开展上一轮提出的24新请求验证。新增12个作者编写的受控项目，每项目两次独立v1修订，分local-edit/ambiguity/control各4项目；不是自然生产流量或12个真实外部仓库。24请求、13条合法旧记录，未生成Target答案、没有把新答案写入历史。
- `python -m scripts.make_transport_validation --output data/research/ot-validation/cases.json`已导出固定任务，SHA`6dea703af88c9e428d130dd967d81a21662713acac77f50dff97531a1c34e802`。保留全部当前源文及明确编辑指令，expected_text只用于任务评分，不进入检索。512预算、24新AR、四组两重复共192计时；原OT/edit参数、Target输入/验证保持固定。
- 协议`OT_R_VALIDATION_PROTOCOL.md`在输出前冻结：适用候选组至少5%优于Suffix、2%优于Edit，分别3/4项目更快，8/8任务正确且无截断，才通过继续投入工程门槛；区间单独报告，不当统计显著性。先测试/audit，再部署隔离研究目录。默认suffix、面试分支/标签/main及服务器infra-v0.3均不改。
- 本机新评测审计21 passed；完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-validation-full --junitxml=artifacts/ot-validation-tests-full-local.xml`为204 passed/3 skipped（21.92s）。最初一项测试把Python tuple直接与JSON list比较而失败，规范化比较后通过，任务文件未改。首轮相关44项中其余43项当时已通过。原Starlette multipart弃用提示未改。运行脚本另在首个严格分歧保存failure.json并停止；启动时核验首轮全部package文件保持原SHA。
- 首次冻结`226399cb4c7ad14cf2fe85a891f6b25d8f78307f`，461文件服务器SHA一致，86项服务器测试输出通过（SSH在测试打印结束后的10秒边界超时，XML已保存），GitHub CI成功39141060160。A6000 PID1654831在24个AR参考后、第一条计时结果审计时暴露新评测字段误用：`GenerationResult`真实字段是`finish_reason`，不是`stop_reason`。该运行status=failed、有效rows=0，目录`ot-r-validation/artifacts/ot-validation-v1`完整保留，不计入性能结果。仅修正评测字段并增加真实运行时数据结构回归；已看过AR的22/24任务正确（两格式控制失败），但任务/输入/参数/门槛均不修改，重新冻结并在新目录运行全部AR与计时。

## 2026-10-18 — 独立保存 infra 并接入 OT-R（首轮完成）

- A6000 PID1622185已结束，两GPU恢复19MiB/0%。10/10新AR与旧token/prompt/snapshot一致，独立耗时130.186340s；随后80/80配对计时全部严格一致。旧JSON：DFlash28.028885s、Suffix-R22.584828s、Edit-R22.479903s、OT-R22.171767s；OT比Suffix降低1.83%、比edit降低1.37%。收益集中在事后拆分的confirm-schema-0（两请求各两重复）：4.759366→4.320043s、-9.23%；其余三项目没有明确改善。不能把已看过数据上的子组当独立确认。
- 旧JSON里Suffix/Edit/OT的Draft轮数82/76/72，R轮数164/168/166，对齐接受token分别0/64/128，OT额外查询0.117405s计入总时间。copy负例Suffix1.373279s、OT1.532735s（+11.61%），两组Draft均10轮，OT额外2个提议接受0token；不能仅凭减少Draft轮数或离线覆盖宣布加速。全部十请求总计Suffix27.747403s、OT27.489606s（-0.93%）。默认仍suffix，不在旧任务上继续调参或扩大重复。
- 运行前代码/协议已提交推送`f30b60a7e908307c5f398b88ffb28a1c5eead712`，433文件部署逐项SHA通过。实际GPU package SHA`27070f0f5d720641ef8f250c59188e33c3ab4bc9a8c5ac5f4169d0dc2367df43`，与本机离线分析及最终package相同；后续只补报告/测试与交付文档。服务器命令：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_transport_probe --config configs/runtime-dflash.json --output artifacts/ot-r-v1`。隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/ot-r-probe`，原`infra-v0.3`未覆盖。
- 本机最终完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-ot-final --junitxml=artifacts/ot-tests-final-local.xml`：183 passed、3 skipped（19.18s，原Starlette multipart弃用提示）。服务器相关65 passed（6.55s）。报告LF输出补充检查两平台各11 passed，原始XML分别保存。`python -m build --no-isolation`因本机未安装build前端失败；替代pip wheel首次遇Windows沙箱临时目录权限，正常临时目录下`.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation . -w artifacts/ot-wheel`成功，wheel SHA`82ca3f6f07566eaf16795e2abe7631de74e62f8e37168c3f7fa0cefc459e0727`。没有因此改代码或安装额外build前端。
- 结果发布在`results/ot-r/`，报告`OT_R_RESULTS.md`。两平台`.venv-dflash/bin/python -m scripts.write_transport_report`独立重建正文SHA均`6bb035227c3e4764996e2a78d5df79846f870ed5a7a297da8a792d04b7d862e6`、diagnostic SHA均`c84ae2a673686e547901aec192b5714ae1dbea2477624ac52db32cd25bfde832`。首次JSON字节核验暴露Windows CRLF，数值与正文已相同；仅报告生成器统一LF后核验通过。原始计时/模型输出未改；旧输入所有SHA保持原值。模型、数据库和大型trace不提交。下一步若继续，是未参与本轮调参项目的独立机制验证和退化保护，不宣称OT已能普遍替换R。

- 用户授权尝试 OT-R，并要求单独保留面试项目。infra 固定提交 `27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`，独立分支 `codex/infra-interview-v0.3`、标签 `infra-v0.3-interview`，研究分支 `codex/ot-r-probe`。本机 `artifacts/infra-interview-v0.3/project` 为独立 Git 克隆；ZIP/bundle/逐项 SHA 清单另存，423 文件原始 Git blob 字节核验通过。复现命令 `.venv\Scripts\python.exe -m scripts.archive_infra --output artifacts/infra-interview-v0.3`；归档位置与面试范围见 `INFRA_INTERVIEW.md`。服务器既有 `infra-v0.3` 不覆盖。
- 新增带预算的 `retrieval/transport.py`，沿用精确后缀路径，失败后调用 lexical OT 或同池 Levenshtein；默认运行配置仍为 suffix。OT 复用 MIT POT 0.9.6.post1，上游标签 commit `85113e9a380f5fcf684c50c73c1ff6a164a7366e`，没有复制上游源码。32-token 窗口、8 候选、双 dustbin、位置先验、4 次调用上限在首轮输出前固定；不是完整 OPW 或 SpecTr，不声称语义检索或新验证算法。实际短锚点保留在 `matched`，新增上下文合格标记只决定是否提议，Target 接受条件仍为逐 token greedy 相等。
- CPU 首轮 `.venv\Scripts\python.exe -m pytest -q tests/test_transport.py tests/test_dflash_reuse.py tests/test_runtime.py --basetemp .test-tmp-ot-first`：50 passed/2 skipped；补充运行配置与配对审计测试后相关 60 passed/2 skipped，XML 在 `artifacts/ot-tests-ready.xml`。含真实 tiny Target 的 OT 拒绝/KV/流式正确性及用户、未来历史、特殊 token 边界检查。
- `.venv\Scripts\python.exe -m scripts.diagnose_transport --output artifacts/ot-r-offline-v1` 已完成，只加载 tokenizer。对旧 8 JSON 与 2 infra 请求的所有长度>=8前缀作独立决策，不模拟在线4次预算/路由。旧 JSON 60 个后缀缺失位置，原始/候选池 oracle 均有48个>=8-token机会；OT 找回48，edit找回47，但 OT 词法总正确前缀1443、edit1491，OT查询更贵。copy 的22个缺失位置原始/池均有12个机会，OT找回0、edit6，OT零接受提议19/22。离线不是在线接受率或加速，参数保持原样，后续只进行协议限定的80次GPU计时。
- 服务器已仅新增 POT 0.9.6.post1，NumPy/SciPy 等原依赖未升级。GPU试验将运行独立 `ot-r-probe` 目录，同权重/FP32/eager/4线程、10个旧AR先核验、四组两重复交错、全在线成本计入；协议 `OT_R_PROTOCOL.md`。未将 CPU 机会结果冒充实测性能。

## 2026-10-18 — 旧正结果向infra迁移的配对验证（完成）

- A6000正式运行完成，PID1596891已退出，两GPU恢复19MiB/0%。原8个完整新AR逐token/prompt/snapshot一致，单列125.609s；之后120/120计时全部一致。Legacy-D=42.030658s、Legacy-R=33.879012s、Runtime-D=42.060590s、Runtime-R=33.881685s、Runtime-R-aligned=33.875867s。新版D/默认R=1.241396，[1.198093,1.291931]，延迟降低约19.45%；旧收益保留，不支持此前短样例退化是工程化破坏旧负载的解释。
- Legacy-D/Runtime-D累计均498轮Draft；三个R组均246轮R、123轮Draft、5418接受token，逐请求/重复计数24/24相同。对齐候选的tokens/记录边界/来源指纹8/8相同；默认system资源增量在本批没有改变这些计数，默认/对齐耗时比1.000172、区间跨1。Runtime-D/Legacy-D=1.000712（约+0.071%）；对齐Runtime-R/Legacy-R=0.999907、区间跨1；预设10%工程预算均通过，不宣称零开销或统计等价。
- 实际命令（服务器infra-v0.3目录）：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_runtime_migration --config configs/runtime-dflash.json --legacy artifacts/runtime-migration-legacy --output artifacts/runtime-migration-v1`。在新输出前冻结协议/代码的提交为`06c3c4e3effb804eefce9ee7368b0138144aa255`；部署411文件逐项哈希一致。GPU实际package SHA`94340a2485f8ab5778a58da4bdda113ae374e8de3b0026be4de73437790ab738`，脚本/协议/旧源码逐文件SHA保存在provenance及execution-source.json；原确认11文件运行前后SHA不变。
- 新增离线报告及参考/资源/汇总篡改拒绝测试，未修改生产解码/检索策略。最终本机`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-migration-final --junitxml=artifacts/runtime-migration-final-local.xml`：156 passed/3 skipped/17.60s；服务器报告与原审计相关16 passed/3.03s。Windows/Linux独立重算diagnostic逐字一致，报告SHA均`2f42fcf242db68fa5acdb221b78958abd92e2af7a9bbd52c0041f65f1cd319fb`。实际测试XML发布，不把测试模拟数据当性能证据。
- 结果`results/runtime-migration/`，报告`RUNTIME_MIGRATION_RESULTS.md`，复算`python -m scripts.write_runtime_migration_report`。保存原始/压缩SHA，不提交数据库、权重或大trace。该批仍是原合成工作负载，不能重新证明历史必要性；之前infra两短请求的负结果不改写。后续退化保护应独立开发/验证，不根据本批事后改策略冒充本次迁移结果。

- 用户授权继续验证。本轮先隔离工作负载与迁移开销，不修改R路由。冻结协议`RUNTIME_MIGRATION_PROTOCOL.md`：原4项目8请求/原合法历史，五组Legacy-D/Legacy-R/Runtime-D/Runtime-R/Runtime-R-aligned交错3重复，共120计时，先新AR8例严格匹配原参考。
- 发现新版候选索引额外包含system文本，加入只对齐候选资源的控制组，Target输入始终相同。旧解码从`dd0077f`恢复，SHA`e8a697c2397b127c78db8db7dab10b42f9b1cf9f17a65e10c6bd2e43522a7099`与原确认清单一致。索引差异最初误判为换行，恢复校验失败；进一步核对原服务器归档与Git，确认后续新增`include_user_records=False`，旧`f43239a`索引SHA`9d15c30963a1194b9ad7f58c27579d6dbb68396e49a73bebc500715080bea9be`严格匹配原运行。已在GPU运行前修正协议，并加入逐例资源指纹核对；共享matcher与请求定义SHA相同。
- 新增`evaluation/runtime_migration.py`、`run_runtime_migration.py`与快照/配对/篡改审计测试。初步命令`.venv\Scripts\python.exe -m pytest -q tests/test_runtime_migration.py tests/test_reuse_report.py --basetemp .test-tmp-runtime-migration --junitxml=artifacts/runtime-migration-tests-local.xml`：13 passed/7.33s；脚本编译检查通过。A6000两GPU空闲，模型及TF4.57.3环境可用。GPU结果将在实际完成后追加，不用旧耗时冒充新结果。

## 2026-10-18 — v0.3 单GPU推理 infra（工程交付）

- 最终GitHub CI已确认：提交`c37f3206cd8136f6db321c9fa9325294e25ceb2f`的research和DFlash两个job均success，见[Actions 39129849841](https://github.com/2223434778/MemoryDraft/actions/runs/39129849841)。Linux原冻结输入针对性2 passed/4.18s。服务器完整404文件发布SHA逐项一致，独立重建`INFRA_VALIDATION.md`字节相同，package source与delivery.json逐文件相同；两张GPU空闲。该最终文档补记不改变已验证运行代码。

- 首次GitHub Actions实际执行：DFlash profile通过测试、构建和隔离wheel运行；research profile为145 passed/3 skipped/1 failed，定位旧冻结history/request输入在Linux checkout被规范成LF，而原GPU审计SHA绑定CRLF。仅将这两份原工作区输入按原始字节固定为`-text`，不改JSON内容或既有审计期望。首次发布归档也受Windows Git的换行导出影响，改为从Git blob原始字节构建归档并核对全文件SHA；不忽略错误或放宽哈希条件。

- 用户明确授权将科研原型转为完整infra。新增`memorydraft/runtime/`的请求/结果协议、配置、模型加载、AR/DFlash/R+DFlash统一生成、配对benchmark、可比性检查与退化阈值、CPU/CUDA profiler；旧H/D/RHD与研究证据保留。运行时DFlash是关闭R的同一审计适配器，原生控制仍在研究脚本。未改变Target prompt或接受条件。
- 新增`memorydraft/serving/`：单执行worker、有界FIFO队列和token buffer、429、deadline、断连/排队取消、慢客户端背压、异常恢复及健康/指标接口。SSE只发送Target已确认token，最终统一解码文本。取消不提前释放worker；异常清除traceback/context/cause，避免保存GPU KV等局部资源引用。测试覆盖失败后下一请求恢复与资源释放。
- 基础包不依赖GPU库，`research`/`dflash`分别固定TF4.52.2/4.57.3；服务声明FastAPI0.111.1、Uvicorn0.30.2、Pydantic>=2.10。加入GitHub双profile CPU CI、CPU demo Dockerfile、示例、安装/运维/API/计时文档。README改为工程入口，原README保留为`RESEARCH_README.md`，未改写原实验结论。项目版本0.3.0，无统一新开源许可。
- 最终本机命令`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-infra-final-full`：**146 passed、3 skipped、17.19s**；服务器`.venv-dflash`相关**54 passed、6.79s**。测试计数摘自实际工具stdout，原始测试log未落盘。Windows受限环境阻止asyncio内部loopback，曾在TestClient启动前挂起；在允许loopback的执行环境通过，不以跳过HTTP测试规避。只清理了本任务挂起的pytest进程。
- wheel命令`.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation . --wheel-dir artifacts/wheels-final`。在`artifacts/wheel-env`干净环境仅离线安装wheel，无torch/transformers/fastapi；基础infer在仓库外运行，最终demo benchmark成功，明确synthetic=true。Docker CLI不可用，未实际构建Dockerfile；CI定义提交后状态以Actions为准。
- 服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-v0.3`，模型通过只读使用原模型目录，未重复下载。正式命令`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.validate_infra --config configs/runtime-dflash.json --output artifacts/acceptance-v1`。初次归档漏带`scripts/__init__.py`，启动在加载模型前失败；补齐后正式运行完成，首次错误见工具输出，发布日志为后续正式运行，不计失败尝试为GPU计时。
- 真实Qwen3-8B+DFlash、A6000 GPU0、FP32/eager：2请求×3方法×2重复，**12/12**计时token/prompt/snapshot严格一致；AR9.230410s、DFlash4.489956s、R+DFlash4.966973s，分别为AR的2.056×/1.858×。本轮R+DFlash反而比DFlash慢约10.6%，不宣称检索净收益或独立自然工作负载确认。加载、预热和参考生成不并入计时；全部在线处理计入。
- 真实HTTP/SSE每方法16token均与AR一致，直接取消R+DFlash后再次DFlash生成正确。真实loopback HTTP压测6/6成功、concurrency2、wall2.998s、P50约0.982s、客户端TTFT P50约0.556s；小样本不输出P95/P99，最后queued/running均0。HTTP断连、慢流及429的系统场景另由CPU测试覆盖，不夸大为GPU高并发验证。
- GPU profiler单独插桩：218550事件、20602 CUDA kernel事件，含7个命名阶段。索引/查找较短，矩阵乘与Target验证具有显著设备成本；完整trace保留服务器`artifacts/acceptance-v1/profile/trace.json`，不提交大型trace。不据插桩时间宣布新性能优化。
- GPU实际package源码SHA`52f5845362006bc8aacebf96f85a8274c4b305b1e43953ffe49c611dfe2801f3`；最终package源码SHA`bec65fec2c1524fabec9d2b4dbbb01fb5c4259d41a96b48828e34f926b63f64a`。逐文件差异和生成核心不变核查见`results/infra-v0.3/delivery.json`；不以最终来源覆盖原计时来源。GPU验收后新增门槛/资源清理等代码已完成最终CPU/小模型回归。
- 小型公开验收证据在`results/infra-v0.3/`，报告`INFRA_VALIDATION.md`；复算`.venv\Scripts\python.exe -m scripts.write_infra_report`。无私有对话、模型、数据库、wheel或完整trace上传。下一步从真实服务负载和profile选择一项深优化；连续批处理、BF16、采样、分布式和CUDA/Triton算子均不作为本版已交付能力。

## 2026-10-18 — 已有合成正结果的归因检查（完成，CPU阶段结束）

- 用户授权落实归因对照。新增`evaluation/reuse_attribution.py`、`diagnose_reuse_confirmation.py`、离线报告生成器、7项测试和`REUSE_ATTRIBUTION_PROTOCOL.md`。候选构造仅接收当前request，明确限于原合成benchmark语法，不读取gold、当前答案、过去回答或生成器隐藏元数据。旧4项目/8请求/96计时通过原audit，并绑定全部旧发布文件SHA，原推理来源仍为`2518e677cde7d1ee4d7c4201b560d14f7a44bd75500c95e2497c0c072e2b46c2`。
- 首轮离线结果见忽略缓存及发布`offline.json`：Structure100/2452、Static1832/2452、Global2336/2452、Compiled2396/2452；Compiled文本/token/语义全8/8，历史在其外覆盖0。随后透明追加Expanded中间对照：仅展开当前标识和默认值、故意不执行override，保持同schema两轮一致；该对照在首轮诊断后提出，不称预注册。`offline-expanded.json`保留中间结果，`offline-final.json`为最终证据。
- 最终Expanded覆盖2332/2452，历史独有4/2452（0.16%）；原Global实际接受5418个R token的位置中，Expanded也可oracle覆盖5409个（99.83%）。Expanded与历史首轮4/4逐字相同，全两轮4/8；Expanded任务0/8正确，Compiled8/8正确。Expanded/Compiled额外候选token数均与历史8/8配平；Static稍短，组合条件更多，逐例长度公开。oracle是6锚点/32接续/至少8连续正确的乐观枚举，来源/特殊token边界独立，不是速度、接受率或19.4%收益的因果百分比分摊。
- 原DFlash41.991595s、Global33.862142s（1.240×，延迟降低19.4%）和D轮数498→123保留。新证据表明这批结果可由当前明确事实展开成目标格式解释，未证明长期历史必要性。按协议在CPU阶段结束，不新增GPU生成或Static/Expanded/Compiled计时，也不宣称替代候选实际达到旧速度；H、原CI确认及新任务搜索未恢复。
- 本机命令：`.venv\Scripts\python.exe -m pytest -q tests/test_reuse_attribution.py tests/test_reuse_report.py --basetemp .test-tmp-attribution-final`，13 passed/5.66s；`.venv\Scripts\python.exe -m scripts.diagnose_reuse_confirmation --output results/raw/research-stage2/reuse-attribution-complete/offline-final.json`，最终结果复制至规范发布源目录，旧中间结果保留。完整`.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-attribution-full`：105 passed、3 skipped、18.61s。分析本身仅CPU/tokenizer，无Target或Draft权重加载。
- 服务器审核包`reuse-attribution-audit.tar.gz` SHA `63d84631dc1a06ad494c74b3769eb97a9dd2cabf2c966ba1a43323455e918df9`，18个部署文件SHA通过。`.venv-dflash`相关13 passed/2.78s；用同一旧发布证据执行`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.diagnose_reuse_confirmation --source data/raw/reuse-attribution-original --output results/raw/research-stage2/reuse-attribution-server/offline-final.json`。两平台summary/逐例/资源/代码SHA/原证据SHA全部一致，报告字节SHA均`d15186f34306b97663fc7fe748ca7bee905560e3e397f79a60c22691fa7365f5`。初次服务器日志重定向因`results/raw`尚不存在失败，创建目录后完成；该次没有执行分析或模型生成。
- 最终本机分析SHA `16c4ce5db2b8afba80bda46e592e567fdb42e15a3428086a52fc97cb8033ca35`；服务器分析SHA `d50d686819aebadfba57f6875eb92ef766161e81ba2bf44e36b165f74eef92dd`（差异为tokenizer绝对路径和离线耗时）。分析脚本SHA `23f61c710a898beb418953e722d20ce5ff94a1dfe43637875327e7a68e8fe9c0`，模块SHA `6a66ad93d8b0520285d16816455208ba02ad059a0dddb668b906a3daf2cbb2a1`。服务器两GPU仍19MiB/0%。结果见`results/reuse-attribution/`及`REUSE_ATTRIBUTION_RESULTS.md`，原发布161文件保持原内容。
- 发布命令：`.venv\Scripts\python.exe scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results`，新增8份公开CPU诊断/决定/测试证据，总169份；原161份manifest条目及发布文件SHA逐项不变。`.venv\Scripts\python.exe -m scripts.write_reuse_attribution_report`可从公开证据离线重建报告，不需要tokenizer。代码、报告及8份新证据提交GitHub供审阅。

## 2026-10-18 — 真实发布说明初筛：小成本可行性检查（完成，按门槛停止）

- 数据/协议/标签/代码在模型输出前提交并推送 `d45f942`；实际推理源码SHA `83f7e6f6b8f43965b9e916c8edd91e36cc484f85e933f25b9224763bef7cc302`，归档 `source-release-triage.tar.gz`。服务器部署全文件哈希通过，前置13 passed；原Qwen3-8B五权重分片SHA逐一验证后，A6000 GPU0执行18次正式Target AR，全部结束、0截断，累计请求217.819s，加载13.761s单列。固定预热、权重审计和离线评分不并入请求时间。作业PID1388481已完成，两GPU均空闲。
- 实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_release_triage.py`。首评分：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.write_release_triage_report`。冻结质量门槛失败：严格合同3/18、合同合格7/18、严格类别30/90。18份实际都可解析JSON且版本/类别顺序正确；整体合同失败会将五类全计错，不能把该严格得分称为纯分类准确率。
- 原预定oracle机会通过：12个有历史请求1633/2597额外独有覆盖（62.9%），6项目均达到项目门槛；它使用答案枚举最优接续，未计在线成本，不是接受率或加速比。质量未过，因此未启动最多48次DFlash/Context/Global/Arctic计时，原CI12项目确认和H训练亦未重启。
- 事后只读诊断 `scripts/diagnose_release_triage.py` / `posthoc-diagnosis.json`：只看status为88/90类、16/18请求正确；11请求证据超1–2条、9请求含不支持相应判断的证据行（重叠），关键status漏判0/误报1。两实际status错误为FIPS兼容误作漏洞修复、注解/内部typing导入误作API迁移。原始门槛/得分/标签/输出未改。
- 新增探索性静态schema资源对照：两个虚构JSON、323 token，不用当前事实/标签/答案。历史在prompt+静态资源之外独有覆盖仅96/2597（3.7%），解释大量固定格式重复。它在看过输出后追加，格式选择有事后性、资源未逐请求配平、没有在线计时，不能当新门槛或实测速度。两平台独立重算逐例诊断与oracle一致；补充回归本机15 passed（7.47s）、服务器15 passed（2.91s）。
- 结果 `results/release-triage-feasibility/`，主报告 `RELEASE_TRIAGE_RESULTS.md`，解释及结束决定 `RELEASE_TRIAGE_DIAGNOSIS.md`。完整运行前协议与标签保留。发布新增12文件（无权重/数据库），总161份，原149份逐项不变。参考原JSONL SHA `f151438d2d31d2fd84aadfcc0773c93edbfaacb58dd85c54204c7ddb0b93d7ed`；原门槛SHA `a21fe1416869abc0f0f1f0c0d38fb4514a5db3245c3e6ca9cc3012ff60050d10`；Windows/Linux主报告SHA均 `d8ddb168079eb8f65b0bec8ac386d7e5568d157c407bafd990593326fcc5d8c2`。
- 本轮候选检查结束，保留R+DFlash与负结果；不自动继续换数据集、扫提示或增加H/路由。既有静态格式就能解释大部分机会，尚无已论证的新方法贡献。用户未来若提供独立真实应用需求及明确瓶颈，再据其重新评估；不是声明所有自然结构化任务都无历史价值。

- 用户授权执行最后一个小成本候选。按 `RELEASE_TRIAGE_PROTOCOL.md` 固定6项目×3版本18请求，先Target质量/合法历史独有机会，任一门槛失败则停止加速计时。没有启动原12项目CI确认或H训练。
- 已通过只读GitHub连接获取每版真实changelog，tag逐一解析commit并校验tag/commit文件相同；保留各版完整章节及六份许可。模型当前材料充分，历史只来自同项目更早已完成回答。逐例来源标签在任何Target输出前审阅；没有独立专家或Target自评，限制明确记录。
- 推理前复核收窄三组过宽证据行范围，剔除纯类型/内部重构行；最初未运行标签草稿和准备manifest保存在忽略缓存。最终manifest SHA `e5e0b6ac9ddd1a33da4b3dadb5d11e1f3211aab93b3b34006f566c9b9c3415e7`。不按任何模型输出更改标签。
- 新增冻结采集器、分类评分/离线oracle模块、Target-only回放、审计报告和对应回归。本机针对性命令 `.venv\Scripts\python.exe -m pytest -q tests/test_release_triage.py --basetemp .test-tmp-release-triage`：11 passed。权重manifest结构已按现有文件核对，使用原Target五分片SHA校验。后续实际部署/推理来源及结果按完成情况追加。
- 补充未来答案/跨项目快照与标签篡改回归后，本机完整 `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-release-full`：96 passed、3 skipped（41.39s）；脚本语法及 `git diff --check` 通过。数据/协议/评分代码将在任何GPU输出前提交并推送。

## 2026-10-18 — CI 负结果归因与处理建议（离线分析完成）

- 用户要求分析原因、提出可行解决办法，并明确没有可靠办法的部分。新增 `CONFIG_REVISIONS_DIAGNOSIS.md`：区分 Target 原文保真、当前输入与历史重复、成熟基线贡献和实际应用必要性。新方向仅为有停止条件的候选，未实施或宣称成功。
- 本轮再次核对开发 90/90、合同诊断 4/4 的保存 token/文本与各自 AR 一致。逐例查看原 7 个失败完整文件：5 个存在文件头/未改内容缺失，2 个改变正文空白；Requests 第二轮是可解析列表而非完整 YAML 映射，不误称为语法异常。v2 两例主体恢复后的注释/末尾 LF 问题另行解释。
- 用现有 `quality(..., mode='diff')` 在隔离临时目录直接应用开发请求给定的 12 个 patch，12/12 得到期望文件（沿用末尾 LF 容差）。未计时、未执行工作流、未新增 GPU 推理。实际命令为 PowerShell here-string 管道到 `.venv\Scripts\python.exe -X utf8 -`，完整复算代码及结果在新报告末尾；首次 GBK 打印失败以 UTF8 重跑，原结果未变。
- 建议停止当前 CI 扩展和 H 调参：工程上已知 patch 直接应用；自然语言编辑可另行评估最小修改操作；研究上最多保留一次真实材料重复结构化处理的小规模可行性检查。若没有相对当前输入的独立复用机会或可解释的场景贡献，结束新方法主线，保留实现与负结果。未启动此前停止的 12 项目确认。
- 仅修改三份文档，检查 `git diff --check`；原报告、发布结果和推理代码均保持原证据。无需为文档分析重复完整 GPU/回归套件。

## 2026-10-18 — 公开CI配置修订回放（开发与单次探测完成，确认按门槛停止）

- 单次contract-v2探测已完成：2个开发首轮、4次DFlash/Global配对全部与新AR严格一致；完整文本0/2正确，YAML结构1/2相等。深入逐行及叶子诊断表明Ruff删一行注释，Requests改写一句注释；Ruff唯一YAML值差异为末尾script块少终止LF，不能自动解释为业务逻辑错误。两例均文本正确的预定门槛未通过，**12项目确认未启动、未生成任何确认Target参考**，未继续扫提示或放宽保真要求。
- 探测实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_config_revisions.py --split development --contract-version 2 --contract-probe`。源码清单 `077a04368b75f50be7a5d8a4ad2c0fd94e15763dfcc8c53e7f9e19dd4fe27be8`，与开发c6来源分开。GPU0/GPU1均已空闲。原模型输出/计时未改；两个输入版本都只在同批方法内比较。
- 报告 `CONFIG_REVISIONS_CONTRACT_PROBE_RESULTS.md`，原始探测 `results/config-revisions-contract-probe/`，包含输出SHA绑定的 `fidelity-diagnosis.json` 和预定门槛 `gate-decision.json`。本机最终完整83 passed、3 skipped；服务器`.venv-train`完整85 passed、1 skipped、1 warning（原有toy LoRA保存元数据提示），DFlash环境前置35 passed。最终服务器测试来源 `11bff98ee8bdfcb5536e61cef6c1ec6b86485a00c076e2e900bb968d1da97d30`，代码文件SHA与本机最终代码一致；测试来源与c6/077推理来源分开。
- 最终新增26份公开结果（开发13、探测13），统一发布清单149份；原123份保持不变。开发/诊断两报告在Windows/Linux复算逐字节相同，SHA分别 `770bc42af38fd7f2ae1a822ed3ab3469a251d9019c749deea508e95fa80e5297` / `f458c25f2835730f440641b0c78c19a3912b8cffadcc3c7662d7e1cc235d1fae`。原始构建日志保留终端空格并标记-whitespace，原始/压缩及Git blob SHA核验后发布。

- 开发90/90严格一致，完整组AR318.002s、DFlash60.753s、Adapted60.819s、Global36.220s、Context36.414s、Arctic37.306s。Global为DFlash的1.677×，Context/Global1.005、项目区间[0.997,1.019]，没有稳定独立历史贡献；Arctic/Global1.030仅6开发项目/单重复，不能作全面方法优势。
- 原Target合同v1完整文件仅5/12文本及YAML结构正确，10/12可解析；diff修正终止LF评分后3/6可正确应用，0/18参考截断。全部失败保留，性能不代表可用任务已经成立。报告 `CONFIG_REVISIONS_DEVELOPMENT_RESULTS.md` 已从原始/发布数据离线重建：版本/用户快照、token、KV前沿、任务检查、配对、summary全部核对。运行源码仍为c6清单，不用后补报告冒充原运行来源。
- 探测前预先声明：Ruff/Requests开发首轮，尾部重申完整文件/注释/未修改行，2个新AR参考+4次DFlash/Global配对。无答案或新事实；当时16项本机审计/合同/前置阶段测试通过。两例均正确才进入确认；不能把输入版本改变的效果归为R。结果与停止决定见本节顶部。
- 开发结果11份白名单JSON/无损gzip JSONL/LOG已发布，含原始v1评分和独立quality-v2.json更正、失败g++与成功clang构建日志、事后环境捕获；统一发布清单现134份。权重、C++二进制、SQLite和公开API原始缓存未入库。

- 用户授权进入真实项目工作负载。新增采集器、`evaluation/config_revisions.py`、`run_config_revisions.py`、离线审计报告和测试。固定6开发/12确认仓库、2025-09-01之前三版同一工作流及两轮修改，新增diff边界；这是公开项目衍生编辑API回放，不冒称自然用户对话。
- 全部18项目已冻结；数据SHA256 `944f0f77a41b08126411e8558a7b451e45341a412d2cee0460fb7c96b12f351e`。未按模型结果选取文件。API限流/服务器读取超时通过已连接GitHub只读工具缓存公开响应解决，保留原规则和跳过候选；手动复核mypy MIT/NumPy BSD三条款及scikit-learn COPYING，保留完整许可。Starlette同仓库ID迁移记录为Kludex/starlette。
- 初始用户文件与模型回答分开记录；后续真实仓库状态作为显式外部版本更新。Global增加显式授权用户文件选项，默认索引仍只读assistant。每条方法/重复独立SQLite，库创建、文件更新、冷索引、前向、缓存恢复、解码和写回计时；未来输出仅用于评分。
- 最相关基线锁定Apache-2.0 ArcticInference@aca5d9a8a62474035c15d114d40a01abc8c94b51的官方线性后缀候选组件，接入相同严格验证循环，非完整vLLM/树系统。源码七文件SHA与许可已保存。服务器`.venv-dflash`安装nanobind2.7.0；默认g++9缺少C++20 span，改用已有clang++15及官方CMake后成功构建，直接接续查询通过，未改系统环境或上游算法。
- 阶段启动协议 `CONFIG_REVISIONS_PROTOCOL.md` 原拟开发90次、确认540次；后续确认资源收敛和保真门槛见顶部与协议修订，原计划不是已运行结果。
- 首轮本机检查暴露默认GBK读取和系统临时目录不可写；统一UTF8及项目内忽略目录，隔离Git外层仓库发现/自动换行，未改冻结数据。修复后本机完整75 passed、3 skipped，服务器DFlash环境29 passed，包含官方C++候选器+真实tiny Target验证。原服务器源码清单 `c6c269ffce98cbfd2036dca7ccd612892cef52287e12a2a637da2bf91bd0215b` 全文件一致；冻结实现/协议提交6b1451c已推送。
- GPU开发运行：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_config_revisions.py --split development`，结果 `results/config-revisions-development/`。进程1223876；仅GPU0，不把部分配对当完整结论。
- 开发发现真实Target会遗漏未修改部分，任务质量与严格token一致性分别报告，不剔除失败。另发现diff评分v1未补最后LF而误判可应用性；新增评分v2及缺失LF回归，本机10 passed。原始输出/计时/评分不改，报告先按原版本审计再离线记录quality-v2.json更正；原GPU进程仍运行c6源码。该修正不改变Target或接受条件。
- 任何确认输出之前收敛确认计时：开发仍有正式AR；确认保留独立AR参考、五条加速路径及diff边界，共468次而非原拟540次，避免额外重复AR基准成本。主比较仍为DFlash/Global，参数和Target输入不变；确认不报告相对AR加速比。方法集合/缺失行审计已补测试，本机11 passed。协议修订在 `CONFIG_REVISIONS_PROTOCOL.md` 顶部。

本阶段实际准备/验证命令：

```powershell
.venv\Scripts\python.exe scripts/prepare_config_revisions.py --offline
.venv\Scripts\python.exe scripts/prepare_suffix_assets.py
.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-config-full
.venv\Scripts\python.exe -m pytest -q tests/test_config_revisions.py --basetemp .test-tmp-config-confirm-plan
.venv\Scripts\python.exe scripts/bundle_source.py
```

服务器官方后缀组件构建及GPU前检查（默认g++失败日志另保留）：

```bash
.venv-dflash/bin/python -m pip install --no-deps --timeout 15 --retries 1 nanobind==2.7.0
cmake -S models/arctic-suffix-audit/csrc/suffix_decoding -B models/arctic-suffix-audit/build-clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 -DPython_EXECUTABLE=$PWD/.venv-dflash/bin/python -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/models/arctic-suffix-audit/arctic_inference/suffix_decoding -DCMAKE_BUILD_TYPE=Release
cmake --build models/arctic-suffix-audit/build-clang -j4
OMP_NUM_THREADS=4 PYTHONPATH=. .venv-dflash/bin/python -m pytest -q tests/test_config_revisions.py tests/test_dflash_reuse.py tests/test_reuse_report.py --basetemp .test-tmp-config-dflash
```

## 2026-10-18 — R/DFlash 接入、场景开发与独立小规模确认（完成）

- 用户授权实现并运行。新增 `backends/dflash_reuse.py`，薄适配已审计 MIT DFlash 循环，复用模型前向/严格最长前缀验证/KV裁剪；支持连续R后累计有效Target特征并在回切时补齐，设待消费缓冲门槛。
- 后缀索引保留旧 `match`，增加有界位置查询及显式当前prompt候选；`retrieval/reuse.py` 增加本请求游标和局部重匹配。当前prompt不伪装成过去事件；历史仍由原MemoryStore时间快照读取。
- 本机首轮完整59 passed、2 skipped（PEFT和审计DFlash环境）；服务器4.57.3中12项新测试全部通过，包括真实小型DFlash类与官方循环对齐。补充报告/发布检查另6项通过。
- 服务器DFlash环境新增固定 `pydivsufsort==0.0.19`，不修改原环境。初次GPU A阶段暴露零命中 `sa_search` 返回left=None，已补提前返回及回归；该失败批次保留为 `results/reuse-pilot-lookup-failure/`，未进入开发计时。
- 开发实际源码SHA `57a013cadafc9859dc002bdefce72d47acaae4774bd13263788a9140ff8e947e`；A八请求×三方法24次全部对齐AR。B按 `REUSE_PILOT_PROTOCOL.md` 完成12合成开发项目/18当前请求/五方法单重复90次，全部对齐。开发Global总18.314s、DFlash23.345s；配置修订Global12.053s、DFlash16.284s（1.351×），Cursor14.191s，Context16.362s。完整文档Global1.363×，边界0.998×；不是独立性能结论。
- 实际命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_reuse_pilot.py`。源码/模型SHA已校验，A6000仅GPU0。结果 `results/reuse-pilot/`，作业日志 `results/reuse-pilot-job.log`。
- 离线报告 `REUSE_PILOT_RESULTS.md` 已从原始及公开gzip复算：重建每轮过去快照，核对固定参考、配对、场景标签和KV前沿。开发及首次失败证据13份已加入发布白名单/哈希清单，合计112份。过去资产0/12、当前参考0/18达到预算。
- 预先声明小规模确认 `REUSE_CONFIRMATION_PROTOCOL.md`：选择简单Global，在4个全新项目/4种JSON schema/两轮修订上做3重复，另加适配关闭R和同策略Context对照。数据与参数在确认生成前固定，计划96次计时；不增加概率模型或修改R参数。服务器相关18项测试通过，确认实际源码SHA `2518e677cde7d1ee4d7c4201b560d14f7a44bd75500c95e2497c0c072e2b46c2`。
- 确认命令为上述runner加 `--confirmation`；结果 `results/reuse-confirmation/`，独立复算入口 `python -m scripts.write_reuse_confirmation_report`。96次计时全部严格一致：DFlash41.991595s、关闭R适配42.000497s、Global33.862142s、同策略Context44.427281s。DFlash/Global=1.240075，[1.198,1.286]；延迟降低19.36%。Adapted/Global=1.240338，[1.199,1.286]，Context/Global=1.311998，[1.290,1.333]。4个项目都比DFlash更快，但仍是小型受控JSON工作负载。
- 确认Global有246轮R、123轮Draft，R接受5418/7329；关闭R为498轮Draft。R全部来自合法过去回答。同策略Context有21轮R但接受0/672，说明当前事实描述不能替代过去完整表达。8/8当前参考JSON可解析，且按冻结请求逐字段检查全部正确；当前0/8、过去0/4达到768预算。检查不筛选任何计时样本。
- 本机完整65 passed、2 skipped；服务器旧后端/PEFT环境完整66 passed、1 skipped、1个已有toy保存警告；DFlash专用环境相关18项通过。新增语义审计另6项通过。服务器测试原始日志及其源码来源随确认结果发布。
- 原始/压缩发布清单共123份（以前99份+本轮24份），模型/数据库不发布。开发、首次失败和确认来源分别保存，禁止以最终部署SHA替换实际运行SHA。报告 `REUSE_PILOT_RESULTS.md` / `REUSE_CONFIRMATION_RESULTS.md` 均离线重建，主结论限定在配置连续小改。
- 本阶段完成后不继续对确认项目调参；保留游标负结果，采用简单Global作为此场景的后续底座。更广的自然配置/代码/文档、并发、其他精度和采样尚未验证。

## 2026-10-17 — 用户修订：场景优先，允许复用成熟架构（方案已调整）

- 用户明确无需击败所有基线，优先找到设计适合的场景；允许借鉴已有架构并形成自己的有效差异。更新 `MEMORY_REUSE_RESEARCH_PLAN.md` 和审阅入口，作为此前通用门槛的替代。
- 第一优先场景为同项目多轮局部修订，应用需要较长完整产物且改动集中；模板任务为次级探索，短输出/diff/新话题作为边界。开发阶段选择场景，在新的测试项目上确认，不能按测试加速比挑样例或人为改变输出模式。
- 核心对照保留 AR、DFlash、一个最相关复用基线与本方案；消融围绕实际改动。成功条件限定到选定场景，允许速度/内存/维护成本的明确取舍；取消“所有场景、所有基线均优”和全局低重复退化硬门槛。
- 先实现简单可复用组合，连续接受长度模型按观测到的问题再决定；新算法复杂度不再作为前置条件。本轮只有文档修订，未执行新 GPU 实验。检查：`git diff --check`。

## 2026-10-17 — 新方向文献、接口核查与研究方案（规划完成，实验未开始）

- 用户要求详细规划新方向及 idea 运作机制。新增 `docs/MEMORY_REUSE_RESEARCH_PLAN.md`，把主线候选收窄为严格验证反馈下的历史接续长度估计及成本选择；保留 H 实现和已有负结果。
- 核查 REST、SAM-Decoding、SuffixDecoding、EfficientEdit、ReSpec、DReSD、DFlash、ReTrace。明确后缀检索、动态长度、混合回退、编辑后重匹配及位置反馈均有先例；尚未宣称新颖性或新方法有效。文献链接和比较限制在方案中。
- 读取实际 `SuffixArrayIndex`、`Router`、`MemoryStore` 和锁定 DFlash `spec_generate`。识别连续 R 后 Draft KV 落后、有效 Target 特征累积、anchor 与缓存位置等接入约束；没有修改模型/验证逻辑。
- 方案规定 A 正确性、B 开发机会诊断、C 独立项目评估及强基线消融；所有数据规模、长度和停止门槛为拟议参数，正式运行前冻结。未构建新数据、未训练或运行 GPU，暂无新结果目录。
- 本轮检查命令：`git diff --check`；仅文档变化不重复模型实验或代码测试。研究方案入口同步到 `docs/REVIEW_HANDOFF.md`。

## 2026-10-17 — H 蒸馏与 DFlash 基线（完成）

- 用户授权执行上一轮建议的两个交付；运行前协议 `docs/H_TRAINING_PROTOCOL.md`。先核验数据隔离/标签掩码，完成独立环境 DFlash 小检查及同权重 D/H 蒸馏，不修改 Target 或放宽验证规则。
- A6000 GPU0/GPU1 空闲，项目盘余 410GB；原环境 torch 2.5.1、Transformers 4.52.2、PEFT 0.11.1。DFlash 使用独立环境，不更新原环境。
- 已读取官方模型快照源码与上游 MIT 许可；锁定 HF revision、上游 commit 和权重 SHA，详见协议及 `configs/dflash-manifest.json`。模型源码/权重留在忽略目录。
- DFlash 独立基线已完成：12 请求（旧六合成+六类别公开各一条）×2方法×3重复=72 次，全部与同环境 AR 及旧 AR 严格一致。FP32/eager、Transformers4.57.3/torch2.5.1，AR=187.385s、DFlash=84.682s，2.213×，请求 bootstrap [1.713,3.263]；合成2.840×、公开1.894×，不是完整公开24请求复现。结果 `results/dflash-baseline/`；模型从 hf-mirror 传输，最终官方 SHA 核验通过。大文件直传留下的部分 tar 未用于加载。
- H 训练前增加独立 `.venv-train`（bitsandbytes0.45.5、pydivsufsort0.0.19/rank-bm25），避开全局旧 bitsandbytes CUDA124 导入问题，不覆盖原环境。服务器四项训练小测试通过；本机完整48 passed、1 skipped（本机没有所需 PEFT，服务器已实际验证）。
- 40 条过去回答先以源码 SHA `28a80ae186d2b116c40ff5e48361c739f35142717cc9c98b7c36c191b3223524` 冻结。审查发现重新加载原始 S 会恢复默认 assistant 阈值；在计时前停止资产任务，修正原始/训练后均为 constant/threshold0，并用 `--resume-assets` 只复用核验后的过去回答。正式训练/评测实际源码 SHA `89362a22632ac47397e53e189faf6175d8c16c82a3344816cec19ef8735a41b9`，不把初始未完成资产计算当正式性能结果。
- LoRA 四轮训练40.485s、1,146,880参数；开发响应loss最低为第2轮0.083185，已固定该权重。开发64条长度计时、107个stride4前缀、24条路由计时选择k4，门控任务类为空；未宣称关闭H即成功。
- 测试八项目417个全前缀：未训练H−D_format平均+0.505590，请求bootstrap [0.068376,1.021098]；训练后H−D_format为−0.009273，[-0.084270,0.056452]，新静态R弱匹配96位置为+0.125。训练D_format平均接受长度3.749，训练H为3.740；历史对弱Draft有效的信号，未转化为对齐后Draft上的稳定增量。
- 最终168次计时全部正确，加开发88次均与固定AR一致。测试AR64.075s，BASE_D62.933s、BASE_H48.837s、TRAIN_D46.522s、TRAIN_H46.099s、TRAIN_off36.149s、TRAIN_on36.482s。BASE_D/TRAIN_D=1.353，[1.079,1.537]；TRAIN_D/TRAIN_H=1.009，[0.972,1.055]；TRAIN_off/TRAIN_on=0.991，[0.953,1.052]。off/on验证450→366，模型起草93→132，节省验证仍未形成稳定净时间收益。
- 报告 `docs/H_TRAINING_RESULTS.md`，原始公开/合成记录 `results/h-training/`；完整训练前未读取测试标签。报告复算重新核验项目隔离、冻结历史快照、Target参考及summary。未根据测试反馈继续训练或调阈值。

实际环境和运行命令（服务器项目根目录）：

```bash
.venv/bin/python -m venv --system-site-packages .venv-dflash
.venv-dflash/bin/python -m pip install --no-deps transformers==4.57.3 tokenizers==0.22.1 huggingface-hub==0.36.0
.venv/bin/python -m venv --system-site-packages .venv-train
.venv-train/bin/python -m pip install --no-deps bitsandbytes==0.45.5 pydivsufsort==0.0.19 rank-bm25==0.2.2
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-dflash/bin/python -u scripts/run_dflash_baseline.py
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python -u scripts/run_h_training.py --resume-assets
```

最后一条为本次复用已冻结过去资产的实际命令；干净目录从头运行不带 `--resume-assets`。目录存在时脚本拒绝覆盖；adapter或测试标签已存在时拒绝资产续跑。本机下载使用 `python -m scripts.prepare_dflash_assets` 并核验官方SHA；服务器镜像仅传输同一固定字节。DFlash已以`cd9ac0b`推送，最终H报告与结果随本次完成提交发布。

- 本机最终完整回归48 passed、1 skipped in 12.76s；服务器独立训练环境49 passed、1 warning in 7.96s（LoRA小测试空模型名导致的保存元数据警告，测试通过）。命令 `.venv-train/bin/python -m pytest -q --basetemp .test-tmp-training-final`，日志 `results/h-training/tests-final.log`。GPU任务全部结束，服务器报告离线重建成功。
- 发布清单现共99份白名单JSON/无损gzip JSONL/LOG；DFlash5份、本轮H23份。模型权重、adapter、数据库与源码压缩包未发布。统一原始/压缩SHA核验及Git字节核验用于交付；精确配置、checkpoint SHA、前缀候选、逐轮成本均保留可复算。
- 补充直接比较：新配置未训练BASE_D/BASE_H=1.289，[1.048,1.463]，普通Draft上历史的确有正向效果；训练后H独立优势未通过。两项实验集合不同，不能直接用DFlash与训练后H两表加速比相除。没有按测试表现继续训练或改变门控。

## 2026-10-17 — Target 历史的对称 H 开关验证（完成）

- 用户授权验证更换历史后 H 是否产生净加速。协议 `docs/H_TEACHER_ROUTER_PROTOCOL.md`：复用上一轮冻结 Target 历史，双方共享同一完整 R 索引；固定旧参数，仅比较 AR/H-off/H-on/缓存 H-on，六请求、三重复，72 次计时和 8 次 smoke。
- 新入口 `scripts/run_h_teacher_router.py` 复用旧路由包装器，核验冻结原文 SHA、过去任务构造、历史版本替换和当前请求；与上一轮 AR token/prompt 逐条核对，不重新生成历史或调节阈值。结果目录 `results/h-teacher-router/`，拒绝覆盖。
- 运行前协议/实现提交 `b7e1e53`；服务器测试发现夹具只兼容发布 gzip，修复为兼容原始 JSONL/gzip 并提交 `0419b0b`，未启动失败的 GPU 运行。修复后本机/服务器针对性均通过，服务器 **8 passed in 4.81s**。
- 实际 GPU 命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_h_teacher_router.py`，只使用 GPU0。运行源码清单 SHA-256 为 `436c2daaa920df93a6c0b745da0ede3d6a2b39c3d273f3bef2df32b4a88c640a`；后补报告与测试不替换这个实际来源记录。
- 72 次正式计时 + 8 次 smoke 全部与旧 AR 的 token/Target prompt 一致，方法间快照相同，结束后历史未变。AR=81.109s、H-off=49.356s（1.643× AR）、H-on=53.934s（1.504×）、缓存 on=52.043s（1.558×）。本批 H-on 比 off 慢 **9.27%**，缓存后仍慢 **5.44%**；off/on=0.915，请求 bootstrap 95% [0.888,0.932]；off/缓存 on=0.948，[0.922,0.964]。
- H-off/on 验证轮数 **807/816**，小模型 H+D 起草 **77/96**；H 本身接受 96/120。六项目启用 H 均更慢；复制/代码修改验证轮数不变，文档修改增加 18 轮，新任务减少 9 轮但多 12 轮模型起草。缓存与未缓存路线/来源/接受计数序列 18/18 配对相同。
- 无 GPU 的追加同前缀审计将在线 H 的 60 轮起点与旧完整评分对齐：在线 H 接受 96/120，旧离线 D_native 同位置 90/120，人工指定一条 Target 历史的 H_teacher 为 93/120。此为已保存候选质量检查，不是替换路由后的速度；在线自动历史与旧单条 H prompt 不同，不能单独归因多历史干扰。
- 结果目录 `results/h-teacher-router/`，报告 `docs/H_TEACHER_ROUTER_RESULTS.md`。离线复算：`python scripts/write_h_teacher_router_report.py`；结果发布：`python scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results`。
- 本机完整回归 **44 passed in 12.29s**，服务器最终完整回归 **44 passed in 6.27s**，日志 `results/h-teacher-router/tests-final.log`。命令分别为 `.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-teacher-final` 和 `OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m pytest -q --basetemp .test-tmp-teacher-final`。服务器也成功离线重建报告，两张 GPU 均已空闲。
- 本轮导出 9 份白名单结果，统一发布清单共 71 份；全部原始/压缩 SHA-256 核对一致。原有 62 份结果不变；模型、数据库、原始私有对话与源码压缩包均未发布。报告、协议、入口和测试提交 GitHub 供 Astra 审阅。

本轮补上“同一 Target 历史下 H 是否有净收益”的缺失对照：当前固定策略仍无收益。原人工历史批次不用于跨批来源加速因果估计；没有调节模板、块长、阈值或样例。六合成项目并非独立自然长期数据，成熟 B6/B7、新触发策略、训练、采样及并发仍待验证。

## 2026-10-17 — H 来源与对称路由追加实验（完成）

- 用户授权执行两项小实验，协议预先固定于 `docs/H_CAUSAL_PROTOCOL.md`。复用已有六个合成测试项目，明确属于追加归因诊断；固定 Qwen3-8B/0.6B、FP32/eager、非思考 greedy、k=2、当前输出上限 128，没有重新挑选样例或阈值。
- 新增 `memorydraft/evaluation/history_alignment.py`：仅从原始 order=1 记录构造过去 lower()/ten-minutes 任务，保存 order=0 请求和 order=1 Target AR 回答，再冻结派生历史后读取当前测试请求。版本替换键包含 user/record/order，避免未来同 ID 版本污染过去。人工/Target 历史选取同一合法记录，全前缀评分，另报等长度截断敏感性检查。
- `TransformersBackend.generate(... enable_history=False)` 仅允许 RHD + 独立快照 R 索引；跳过 H 语义检索，R/D/AR 的探测、EMA、驻留和冷却保持相同逻辑。关闭 H 的旧 RD 保留为辅助基线，不充当严格 H 开关对照。
- 新增测试覆盖过去任务构造/未来更正、H 关闭和路由一致性、相同历史/空历史质量控制。本机针对新增模块 **4 passed**；完整回归和 GPU 结果随后记录。
- 实际运行命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_h_causal.py`。结果目录 `results/h-causal/`；先来源诊断，再两请求 smoke，再六方法三重复的路由评测，避免 GPU 任务重叠。

### 结果与复核

- 实际运行源码清单 SHA-256：`d5072df7254a4867a73e4ed2548d502d30234e2e29acec056aad15dd4b15577f`。协议与实现先以 `8c230ec` 推送；原运行 provenance 保留其实际文件清单，后补报告/CPU 审计不冒充主运行源码。
- 六条模拟过去回答均未达到 256 token 上限，Target 生成累计 32.618s。冻结历史 SHA `55483ae5160588591aed932bca5fb62e188e301f39bd064c817e3abcab0c919a`，之后才加载当前请求。全前缀 **532** 个位置，新 AR 与第二轮保存的 token/Target prompt hash 完全一致。
- H_manual 平均接受长度 1.663，H_teacher 1.753；Teacher−Manual 为 **+0.090**，请求 bootstrap 95% **[+0.008,+0.238]**，主要来自文档修改项目。首位命中 86.09%→91.17%（位置等权）；等长度截断比较 +0.153，但截断会改变信息覆盖，不将其当作纯文风因果证明。
- 108 次路由配对计时 + 6 次 smoke 全部正确。AR=81.264s、D_native=69.960s、旧 RD=57.317s、对称 RHD_off=54.550s、RHD_on=60.086s、缓存 on=58.082s。相对 AR 分别为 1.000/1.162/1.418/1.490/1.352/1.399×。H-on 比 H-off 慢约 10.15%，缓存后仍慢约 6.47%；旧 RD 不能替代对称 H-off。
- H-off/on 验证回合 914/912，仅少两轮；模型 H+D 起草回合 76/120，多 44 轮。H-on 的 150/168 接受量不是全局增量收益，需扣除被它替代的 R/D 机会；不把其 89% 接受比例直接解释为系统成功。
- 两个主 GPU 实验结束后执行 CPU 后分析：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/audit_teacher_overlap.py`。来源改善的 51 个位置中，使用同一 Target 历史的静态 R 可覆盖其中 31 个。此为接续质量审计，不是在线速度；文档项目在新 R 弱匹配处仍有局部来源增益，但只有一个项目。
- 本机完整回归 **40 passed in 12.26s**；新增模块/报告/发布针对性检查 **6 passed**。服务器主运行前针对性 **4 passed**；两个 GPU 实验结束、最终源码部署后，完整回归 **40 passed in 6.00s**，日志见 `results/h-causal/tests-final.log`。
- 已导出只含合成/公开 JSON、压缩 JSONL、TXT/LOG 的结果，模型、数据库和源码压缩包不入库。`docs/H_CAUSAL_RESULTS.md` 由 `python scripts/write_h_causal_report.py` 离线重建；发布原始/压缩后 SHA 保存在统一 publication ledger。

本轮支持“历史来源对齐影响 H 候选质量”，但没有测出原人工历史上启用 H 的净收益；两个实验不合并冒充 Target-history RHD 的在线加速。成熟 B6/B7 及新的独立用户项目验证仍未完成。

## 2026-10-16 — 架构 v0.2

- 已完成：架构复核、六阶段计划、Codex 启动任务及工程约束。
- 未完成：底座选择、代码导入、模型运行、GPU 性能测试。
- 下一步：执行 CODEX_START.md 第一部分，选择独立 Draft 友好的现有底座，建立 AR/D 基线并尽早实现 H。
- 性能结论：无。本仓库初始内容为设计，不是已运行的推理系统。

## 2026-10-16 — 可运行原型与 GPU pilot

### 已完成的工程

- 选择并锁定 Transformers 4.52.2 / commit `237c7c356cce5f8a6befc0a5f71bc952bb5adc9a`，复用上游验证、KV 回滚、模型加载和独立 Draft 生成。SAM/RAPID 仅审查接口，未复制无明确许可代码；详见 `BASE_SELECTION.md`。
- H/D 共用 Qwen2.5-0.5B 权重，各自固定 prompt、开放 assistant 前缀、独立 KV、即时拒绝裁剪和懒追赶；Target 固定为 Qwen2.5-1.5B。实现 R 接续复制、语义 L1 + token L2、成本路由与 AR 退路。
- SQLite 不可变版本事件，按明确可用顺序恢复快照；用户隔离、未来更正/删除、持久化、最终回答写回已检查。候选库只含请求前的历史，没有把当前测试答案写入过去。
- CLI 提供 env/import/generate/replay/report；冻结历史跨方法回放，相关/空/等 token 无关历史消融，逐 token 一致性检查及配对总时间比。
- 19 项关键测试在本机通过，涵盖验证偏移、全接受/拒绝/EOS/长度、H/D/R/AR 切换和真实上游 KV 使用、检索计时公平性、异常恢复、时间与用户隔离、不可变事件、持久化、路由成本退路和配对统计。

### 环境与运行位置

- 本机 RTX3090 24GB；当前 Python 3.13.5 / torch 2.10.0+cpu，因此 GPU 运行转到服务器，不是本轮模型超出 3090 显存。
- 服务器 Linux，双 A6000 48GB，实际仅用 GPU0。项目路径 `/media/a6000/D/ZhaoBinHe/MemoryDraft`，独立 `.venv`，torch 2.5.1+cu124 / Transformers 4.52.2 / sentence-transformers 4.1.0。复用现有基础 Python，不修改其他项目环境。
- 0.5B 普通指令模型从固定 revision 下载；多语言 MiniLM 从本机官方源下载、SCP 传输，两端压缩包 SHA-256 一致：`f9eec229a244d12246e90fc4a58c1499456be3fe5110829919f5d7d3fedee3c8`。
- 本机 SSH 克隆 GitHub 超时，HTTPS 克隆成功；远端 Hugging Face 直连超时、镜像曾限流，均已绕过为官方源下载后传输，没有把网络失败写成成功实验。

### 实际命令与结果

本机检查：

```powershell
.venv\Scripts\python -m pytest -q --basetemp .test-tmp7
git diff --check
python scripts\bundle_source.py
```

GPU 回放及机制检查（服务器项目根目录）：

```bash
.venv/bin/python -m pytest -q --basetemp .test-tmp4
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python scripts/diagnose_precision.py --config configs/a6000-pilot.json --output results/diagnostics/precision.json
.venv/bin/python -m memorydraft replay --config configs/a6000-validation.json --db pilot.db --requests data/synthetic/requests.jsonl --output results/validation/replay.jsonl --max-new-tokens 64 --block-size 8
.venv/bin/python scripts/run_a6000_suite.py
```

- `results/validation/`：初版路由，13 请求 × 6 方法 = 78 次，全部与 AR token IDs 一致。
- `results/ar-fallback/`：新成本退路，13 请求 × 2 方法 = 26 次，全部一致。
- `results/semantic/`：真正多语言 CPU embedding、最终路由，13 请求 × 6 方法 = 78 次，全部一致。
- `results/ablation-empty/`、`results/ablation-unrelated/`：各 5 个相同受控请求 × AR/D/H = 15 次，均一致；空历史 H/D 接受和提议计数相同。
- `results/demo/`：最终回答在 order=30 写入；另一个进程验证 order=30 尚不可见、order=31 可见、其他用户不可见，结果全部通过。同步写回单独计时并计入该 demo 的总延迟。
- 共 212 次配对方法运行通过，另有单请求写回检查；这些不是 212 个独立自然请求。初期失败的 BF16 运行不在通过次数内。
- 上述小型原始结果、summary、provenance 已同步到本机。最终套件包含运行源码 SHA-256 清单；初版词法运行只有模型/配置记录，不伪造缺失的 Git commit。

### 实测发现与修复

1. 延迟激活的 H 首轮 mask 未覆盖已确认 Y，会导致 RoPE 维度错误。已修复完整 `C_H+Y` mask，并通过路由切换回归。
2. BF16/SDPA 的 docs-3 在第 54 个 token 与 AR 分歧；诊断发现接近并列的量化 logits。相同请求 FP32/eager 一致，后续参考实验统一使用该配置，未放宽接受规则。
3. 普通小模型起草成本大，H 没有稳定提高接受量。基于真实成本添加 H/D/AR 探测及退路，仍保留 H 实际运行与独立消融。
4. 语义最终配置：AR 总耗时 **9.972s**、D **18.819s**、H **19.965s**、RHD **13.300s**。H 接受 320/852，D 接受 322/819。完整 RHD 实际走 R=2/H=21/D=21/AR=242 轮。当前组合没有净加速。

详见 `EXPERIMENTS.md`，表内数字来自对应结果文件。单次合成 pilot 不作统计显著性、通用性能或“越聊越快”声明。

### 尚未达到的研究完成条件

- B6/B7 成熟学习型 Draft 对照没有完成。已确认 EAGLE 官方 Qwen2-7B 配对，但它不匹配本轮 Qwen2.5-1.5B；不能拿普通 0.5B D 冒充。需要另选匹配 Target，并适配 Target 特征恢复。这个范围尚未实现，而非已经验证不兼容所有模型。
- 当前 pilot 是设置 A，使用明确标注的合成项目流；应用事实记忆设置 B、真实合法用户历史、更长输出及可靠重复仍未完成。
- 采样、并发、服务后端、树融合及训练未实现；CLI 明确拒绝非 greedy 参数。CPU 内存/索引增长成本与精简 trace 的正式测量仍待完成。
- 项目已具备可运行工程骨架与 R/H/D 机制闭环，不能据此宣布六阶段研究或性能目标全部完成。代码与结果保留于本机及授权服务器目录。

## 2026-10-16 — GitHub 审阅交付

- 用户已授权将代码与报告推送 GitHub，供 Astra 检查与分析失败原因。
- 本次整理核心代码、配置、测试、合成数据及小型实验结果；模型权重、虚拟环境、数据库和模型压缩包由 `.gitignore` 排除。
- 新增 `docs/REVIEW_HANDOFF.md`，区分实测事实、可能原因和缺失对照，特别标明 dtype/attention 同时变更的归因限制。
- 发布前检查远端 main、文件大小、敏感内容、结果一致性及 `git diff --check`。本次仅增补审阅文档，不重新运行已通过的 GPU 套件；既有测试与运行记录保留。

## 2026-10-16 — 按用户要求复核研究路线

- 阅读原始论文/官方项目：标准 SD、REST、SAM-Decoding、RAPID、DReSD、RASD、Transformers 官方 assisted/prompt lookup、EAGLE 模型表。新增 `docs/NEXT_EXPERIMENT.md`，架构顶部明确 R/H 独立检索和原生 D/同指令 D 的优先修订。
- 已完成离线日志审计：`python scripts/audit_pilot.py --input results/semantic/replay.jsonl --output results/research-audit/pilot_summary.json`。输入校验 token、历史快照及方法配对；纯 R 4/424 轮起草、接受 12/17 token、贡献 12/436 输出 token。没有重跑 GPU。
- 已执行 `.venv\Scripts\python.exe -m pytest -q tests/test_pilot_audit.py`，**3 passed**，覆盖接受率/覆盖率区分及错误 token/快照配对拒绝。
- GitHub API 检查上游 commit/许可；REST `154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf` 根目录 Apache-2.0、检索子模块 MIT。源码发现 u16 token 持久化与无记录分隔问题，不能直接适配 Qwen。SAM/RAPID 未见 LICENSE，提交与前次审查一致；本轮未复制第三方代码。
- Hugging Face 公开元数据核验 Qwen3-8B/0.6B，固定 revision 和文件 SHA-256 见 `results/research-audit/model_preflight.json`。两个 tokenizer 文件分别逐字节一致、配置词表大小相同；未下载权重或运行推理，尚不能宣称新组合有加速。
- GPU 下一阶段按协议推进：官方 D/prompt lookup 基线 → 独立历史 R → 同前缀 H/D → H 混合检索 → 路由与 B6/B7。长历史数据按原始版本、许可和时间过滤导入；不得用当前测试答案构造历史。

## 2026-10-16 — 第二轮实验执行记录（进行中）

- 新增原生 `D_native`、原生上下文适配 `D_adapter`、上游 prompt lookup `PLD`；旧 `D` 保留为带空 reference 的 H 同指令对照。RD/RHD 的 D 改用原生上下文，避免弱化实用基线。
- R 使用独立快照的 pydivsufsort 索引，uint32 固定宽度编码、对齐检查及记录/特殊 token 边界；动态输入复用调用上游 PromptLookupCandidateGenerator。H 的 dense/BM25-RRF 与 R 分离。新增相同前缀的 H/D 质量诊断与请求成组 bootstrap 汇总。
- 固定 Spec-Bench 提交 `fd2c1cd7d2201ef71db4c5f4e455008f017967bf`，按六类任务的既定顺序选 12 个开发、24 个测试请求，只取首轮。合成设置 B 另有 6 个开发、6 个独立测试项目，每用户 4 条预先存在的历史；详情及许可在 `data/research/`。
- 模型权重使用 `configs/qwen3-download-manifest.json` 中的公开 Qwen 官方镜像，通过完整 HF SHA-256 校验；在授权服务器目录下载。服务器中途断网，用户恢复后断点续传；下载未混入性能有效统计。
- 旧模型对照命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_research.py --config configs/a6000-validation.json --output results/research-old-bounded --stage smoke --max-new-tokens 48`。8 请求 × 6 方法均正确；AR 9.192s，D_native 13.809s（0.666×），PLD 6.395s（1.437×），R 6.111s（1.504×）。当时后台仍有下载，此结果仅作开发诊断，不作正式稳定性统计。
- 发现并修复 PLD 末尾超长（54 对 48 token）；保留原失败现场。另发现 Qwen3 新版 generation_config 会回填 do_sample=True，导致原标为 greedy 的实验实际采样。现 Target/S 均显式关闭 use_model_defaults，Target 显式 do_sample=False；`test_checkpoint_sampling_defaults_cannot_override_greedy` 禁止 multinomial。错误采样运行移至 `results/qwen3-inherited-defaults`，不计有效结果。
- 无关历史控制改为与 H_dense 等 payload token 数直接注入；人工预选历史强制使用合法选定记录，不再由语义阈值过滤。保存实际前缀长度、来源及少量候选/参考 token，测试空历史 H 与 D_format 提议相同。
- 新套件命令：`OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_qwen3_pilot.py`。在 `results/qwen3-pilot/` 保存逐阶段 provenance、SOURCE_MANIFEST、配置、首个分歧及配对记录；所有阶段先核验部署文件哈希。主运行源码清单 SHA-256 为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`。
- 修复默认采样后，BF16/eager 在 spec-121 的 index=26 仍有分歧；FP32/eager 的 8 请求 × 6 方法通过。开发校准 D_native 的 k=2/4/8 加速为 1.186/1.125/1.006，冻结 k=2、FP32/eager 后进入测试。该选择不读取测试结果。
- 本机项目 venv 已安装 torch 2.6.0+cu124、NumPy 2.1.3，并确认 RTX3090 可用；系统 Python 的 CPU PyTorch 未改动。主 GPU 实测仍在 A6000 上进行，不混合两种硬件/运行时。显式 greedy 修复后本机完整回归 30 项通过，服务器针对新对照的 8 项通过；报告统计测试另有 3 项通过。最终报告将在套件完成后从结果生成。

## 2026-10-17 — 第二轮主实验完成与成本消融

- A6000 主套件和后续诊断已完成，`results/qwen3-pilot/complete.txt`、`post-checks-complete.txt` 均存在。FP32/eager、k=2、输出上限 128，所有测试配置在开发阶段冻结；原主运行源码清单仍为 `6be842aa6afc83a1ed59c8e9129b5eaea384e58cd8b7feec6180e80c1bd95ba5`，未用新源码伪造旧 provenance。
- 合成设置 B：6 项目 × 8 方法 × 3 次 = 144 次计时运行全部正确。D_native 1.168×，R 1.487×，RD 1.424×，H 1.103×，RHD 1.356×。H 相对同指令 D 多用 3.610 秒，在线接受比例 77.59% 对 78.45%；RHD 相对 RD 多用 2.848 秒，检索累计 2.934 秒。不能把相减的推算当成缓存后实测。
- Spec-Bench 固定测试切分：24 请求 × 4 方法 × 3 次 = 288 次全部正确。D_native 1.000×，R 1.056×，PLD 1.117×。D 在数学任务 1.166×、QA 0.897×；本次不能宣称 D 已普遍快于 AR。
- 主同前缀诊断每 16 token 采样，共 35 位置，H_dense 相对 D_format 平均 +0.375、请求区间跨零。为解释与在线接受率方向的差异，追加同请求全前缀 532 位置检查，增量降为 +0.045、95% 请求 bootstrap [-0.058,+0.149]；静态 R 弱匹配处约 +0.005。H_selected 与 H_dense 相近。保留两份结果，后者标为事后探索；没有重新挑选测试项目。
- `scripts/run_post_checks.py` 在主计时完成后依次执行 `run_dense_prefix_check.py`、`diagnose_saved_mismatch.py`。BF16 的 spec-121 index=26 在相同已确认输入上复现：AR top2 同为 33.75、块验证为 34.00/33.75，最大 logits 差 0.25；query 长度 1/9，支持数值形状差异解释。该诊断不计速度。
- 新增逐文本片段、容量 4096 的 CPU embedding LRU，只给当前合法快照文本打分，查询仍在线编码。未缓存路径保留；H/RHD 使用独立空缓存做一次交错成本消融，不预先离线建库。部署清单 SHA-256：`8d0276f05a9867701466941fe2fa9f28c4bddb7d8aba98b4164e8528948f443d`。本机完整测试 **35 passed in 30.48s**，服务器 **35 passed in 5.40s**。

本机结果整理命令（纯报告，无 GPU）：

```powershell
.venv\Scripts\python.exe scripts/publish_research_artifacts.py --source results/raw/research-stage2 --destination results
.venv\Scripts\python.exe scripts/write_research_report.py
.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-final35
git diff --check
```

追加成本实验（服务器项目目录，主 GPU 运行完成后）：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_embedding_cache_check.py
```

结果路径：主报告 `docs/EXPERIMENTS_V2.md`，主配对记录 `results/qwen3-pilot/{memory-test,public-test}/replay-k2.jsonl.gz`，H 诊断 `prefix-test/` 与 `prefix-dense-check/`，成本消融 `embedding-cache-check/`。发布工具仅收集公开/合成实验的 JSON/JSONL/TXT/LOG，JSONL 无损 gzip；逐文件原始与发布 SHA 在 `results/research-v2-publication.json`。权重、数据库与原始下载档不上传。

成熟 B6/B7、自然长期用户数据、采样/并发仍未完成；下一机制重点是历史接续与固定 Target 对齐，而不是在这六个测试项目上继续换检索器挑最优结果。项目未被标记为六阶段完成。

### 缓存消融结果与发布收尾

- 追加 6 项目 × 8 条方法/缓存条件 × 3 次 = **144** 次计时运行全部正确；本轮主测试加成本消融共 **576** 次，不含开发校准/失败/离线前缀诊断，不是 576 个独立请求。
- 同批 AR=81.172s，D=69.858s，D_native=69.446s，RD=56.956s；H 未缓存/缓存为 73.331/71.414s，RHD 为 59.944/57.923s。缓存使 H 约快 1.027×、RHD 约快 1.035×；但缓存 H 仍比同指令 D 慢约 2.23%，缓存 RHD 仍比 RD 慢约 1.70%。候选增益不足的问题没有被缓存掩盖。
- H/RHD 检索累计分别从 2.980/3.000s 降到 1.008/1.015s；两个独立缓存均命中 63 次、新编码 9 个片段。逐轮路线、来源、提议数、接受数和推进量在各自 18/18 个缓存配对中相同。第一次编码也计入，没有把离线预处理当零成本。
- 完整报告由已完成结果重建，缓存为看过测试后的工程消融；对应 provenance/缓存访问记录/测试日志一并保留。已有主证据提交 `4bf7010` 已推送。新增 `.gitattributes` 固定发布文件原始字节，并用 `git show HEAD:<path>` 确认全部已提交文件与发布 SHA 相符，避免换行转换破坏核查。
# 2026-10-20 ASD 实际前缀收益预算验证（执行准备）

- 用户授权验证第一方向。新增`ASDConfig.selection=utility/regret_price=2`，在原ASD可行前缀中比较额外推进与regret支出；EOS后的尾部不计收益，请求预算不重置。原最长前缀默认保留。
- 新任务冻结为GSM8K test8..23及MATH-500 0..7，共24题；来源commit/SHA与许可说明在`data/research/asd-budget/records.json`。旧GSM8K0..7仅作已观察开发依据，不用新结果扫描价格。
- 四组DFlash/ASD/ASD-Run/ASD-Utility；96机制+192自然EOS计时，总288生成。两物理A6000各跑一个benchmark，组内同卡交错。协议`docs/ASD_BUDGET_PROTOCOL.md`，输出`results/raw/asd-budget/{gsm8k,math500}`。
- 首次本地命令`.venv\Scripts\python.exe -m pytest -q tests/test_asd.py tests/test_asd_budget.py --basetemp .test-tmp-asd-budget --junitxml=artifacts/asd-budget-tests-first-local.xml`：21 passed/8.13s。随后增加utility真实小Target/KV路径参数覆盖，将再跑相关检查；尚无本轮GPU性能结论。
# 2026-10-26 — DSpark成本树等价优化（执行准备）

- 用户授权优化DSpark后复测。新分支`codex/dspark-tree-opt`源自9677322，原DSpark/DFlash及面试版引用保留。新增请求冷LRU64行Markov偏置缓存和向上舍入的保守最优树搜索上界；保持原逐行GEMV、权重、候选块、Target输入和ASD规则，旧每卡树价格冻结。
- 新模块`memorydraft/backends/dspark_tree_fast.py`；独立runner/pipeline/report及协议`docs/DSPARK_TREE_OPT_PROTOCOL.md`。计划Alpaca16、MT4对话、HumanEval4，24源/28匹配输入、四方法448主计时；七族各2源任务真实GPU状态四变体消融先行。旧AR仅复用输入/对话历史，旧时延不作本轮对照。
- 首批`.venv\Scripts\python.exe -m pytest -q tests/test_dspark_tree_fast.py tests/test_dspark.py --basetemp artifacts/pytest-dspark-opt-first --junitxml=artifacts/dspark-opt-first-tests.xml`：24 passed/16.97s。加入报告完整矩阵/输入/选树/账本/定长/编译拒绝检查后，`... -m pytest -q tests/test_dspark_tree_fast.py tests/test_dspark.py tests/test_dspark_tree_opt_report.py --basetemp artifacts/pytest-dspark-opt-ready --junitxml=artifacts/dspark-opt-ready-tests.xml`：32 passed/11.54s。随机非单调成本、EOS及真实小Target路径通过；尚无优化GPU性能结果。
- 服务端两卡19MiB/0%且新目录不存在。待源码封存后，以`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_dspark_tree_opt_pipeline`在独立`/media/a6000/D/ZhaoBinHe/MemoryDraft/dspark-tree-opt`运行；结果`results/raw/dspark-tree-opt`，不覆盖旧目录。
