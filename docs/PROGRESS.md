# 项目进度

## 2026-10-18 — v0.3 单GPU推理 infra（工程交付）

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
