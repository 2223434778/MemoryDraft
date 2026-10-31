# 新场景扩展评测

2026-10-31。用户授权IFEval、LiveCodeBench、LongBench、CanItEdit，附带AIME25数学扩展，并优先引用可获得的文献基线。

- 选定182请求：IFEval64；LiveCodeBench v6-only24（easy/medium/hard各8）；CanItEdit descriptive32；LongBench Qasper/MultiFieldQA-en/MultiNews/GovReport各8；AIME25完整30。SHA排序/分层在模型生成前固定；官方数据修订、源码commit与字节SHA见data/research/expanded-suite/manifest.json。
- 同机仅DFlash、ASD、当前BufferedKV-ASD三组，预定546自然生成。EAGLE-3/DDTree/BASTION已有文献数据列原文报告值，不重复部署；文献硬件/精度/样本与本次不同，不将其倍率相除宣称同机胜负。当前固定Qwen3-8B、原DFlash-b16、单A6000、FP32/eager/RMSNorm、4线程、greedy。输入和EOS各组完全相同。
- 输入上限4096；LongBench沿用官方首尾截取形式，在tokenizer层将prompt限制至3840，保存原长度/截取标记。这是长度受限子集，不称完整长上下文排名；其他任务超输入上限时在模型回答前统一拒收并列覆盖率，不静默截掉代码。输出上限IFEval/LCB/编辑2048，QA128，摘要512，AIME4096；全部EOS自然停止、截断单列。
- 新成本表用合成请求校准，不使用测试答案。共享已经验证的推理代码，宽度反馈/预算/KV按请求隔离。每题确定性打乱三组顺序；全请求同步计时包括prefill/draft/构树/门控/提交/解码，质量评分、加载、预热、一次校准单列。使用DFlash实际输出作为严格参考，不新增AR生成，不把DFlash身份称作已验证AR身份。
- IFEval用官方strict/loose prompt与instruction两级指标，语言检测固定seed0；LongBench用官方answer F1/ROUGE-L，不把摘要相似度当事实正确率；LCB用全部公开+私有测试2s/test、全局70s；CanItEdit用官方hidden tests，独立Python3.11子进程，canonical/before预检。生成代码限制危险导入/文件和进程操作，CPU/内存/时间受限；这是任务专用执行限制，不称通用安全沙箱或官方Docker复现。
- CanItEdit官方仓库为BSD-3-Clause加禁止模型训练条款，本次仅推理评测；IFEval Apache-2.0，LCB/LongBench代码MIT，许可原文随隔离快照保留。官方测试依赖不满足的编辑题在模型输出前标记，对全部方法统一标为不可评分，仍保留生成时延；评分器异常与模型错误分开，不把环境异常记作算法错答。
- 上游评分代码、官方prompt、固定数据与全部答案只用于离线评分，不作为draft历史或Target提示答案。原始长文和完整测试载荷保留研究缓存，GitHub发布版本/ID/SHA与统计，不另行散发原始长文。
- A6000独立expanded-suite-v1目录，确认27B空闲后释放GPU1，finally恢复原服务；GPU0聊天和infra冻结引用保持。完成后报告实测三组、逐题配对时延/实际TPS/长度/截断/质量、严格参考一致性及文献引用适用条件。

命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_expanded_suite`。

本机：`.venv\Scripts\python.exe -m pytest -q tests/test_expanded_suite.py tests/test_tree_execution.py --basetemp artifacts/pytest-expanded-local --junitxml=artifacts/expanded-tests.xml --tb=short`，15 passed/29.85s。

## 独立复现准备

先clone官方CanItEdit、LiveCodeBench、LongBench到`artifacts/upstream-canitedit`、`artifacts/upstream-lcb`、`artifacts/upstream-longbench`，分别checkout manifest中固定commit；执行`python -m scripts.fetch_expanded_data`及`python -m scripts.prepare_expanded_suite`。准备器在已有manifest时拒绝不同题目和源码，仅恢复已记录的LF/CRLF字节；原始数据缓存不提交Git。

## 离线评分修订

正式生成前的规范答案预检发现9个编辑题被初版导入/对象方法检查拦下。在尚未查看这些题的模型输出时，冻结独立修订：允许正常局部双下划线变量、`super().__init__()`等对象方法，以及已安装的科学计算包；编辑子进程AS上限改8GiB，强制CPU、不准加载额外模型或访问文件/进程。生成结束后在独立评分目录统一重评全部32编辑题、三组输出，规范答案仍不能通过的题按相同掩码标不可评分。旧预检/旧评分保留，推理源码、输出token和计时均不改。
