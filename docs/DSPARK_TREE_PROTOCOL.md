# DSpark + 成本树 + ASD：独立接入与基线比较

2026-10-26，在查看新 DSpark 任务输出前固定本轮比较。保留原 DFlash 成果，新分支为`codex/dspark-tree`；`codex/asd-splice`保持`8e7a80b5b1af3b09886659e3549691be3ce6493e`。旧推理/构树文件、配置、实验载荷与面试版引用不修改。

## 来源和实现

- DeepSpec代码`005e03b81cec38b7da6399833d609ee89a2587f2`，MIT，版权与NOTICE分别核验；采用其Markov推理方程，未导入训练框架。原始建模/推理文件另存供骨干对照。
- 官方`deepseek-ai/dspark_qwen3_8b_block7`，revision`03326e5043815da1f81b109078b2889737c26017`；权重4742170330字节，SHA256`5c922d1f96c8d18dbdf005c40ea99878d9133d70eaa3be4bd1077ee5f2afc1e5`，config SHA256`e470e70a553616f0dce8f783f42cfd25b7bd7dfdcd57ead680a0623040725611`。官方检查点无单独model card/许可元数据，不将代码MIT自动写成权重许可；不上传权重。
- 新DSpark适配器复用已有、运算/参数名称对应的五层DFlash骨干，核验所有权重名称、形状和RoPE。DSpark检查点的embed/lm_head逐块与实际Target核对相同后共享。其anchor位置也预测未来token，因此7位置产生7个proposal；DFlash旧16窗口产生15个proposal，保持原生设置。
- Markov条件分布按`base_logits[depth] + W2(W1(parent_token))`计算，best-first扩树使用每条分支自己的父token。复用TreeCosts、原树mask、TreeASDVerifier与KV提交。仅支持官方vanilla Markov，无Joint主干预留。
- 初次比较关闭置信度裁剪，固定原生proposal长度；检验DSpark的Markov起草与成本树/ASD组合，不声称已经评测完整生产置信度调度。无新增训练或LoRA。

## 比较矩阵

每个Draft四组：严格线性、线性ASD、严格成本树、成本树+ASD，共8组。Qwen3-8B/原prompt/模板/贪心/FP32/highest/TF32-off/eager/RMSNorm/线程4相同。两个Draft均驻留时测量配对请求，记录峰值和编译；同一卡按题随机顺序及反序重复。共享Target验证计时表，每个Draft分别校准起草/构树辅助成本，最小8/最大32节点、Top4。ASD固定B8/g.25/M2。

使用已观察的`data/research/asd-paper-extension`每族前4个源任务：GSM8K、MATH-500、HumanEval、MBPP、MMLU、MT-Bench、Alpaca，共28源任务/32匹配请求。它们是开发回放比较，不能改称新的独立保留集。MT第二轮统一采用本轮AR首轮回答，另存完整消息及SHA；不混入各方法不同的历史。

每请求自然EOS/4096上限2次（正反顺序）、128/EOS一次、固定256/ignore-EOS一次；共1024主计时、32自然AR参考，预热/校准/正确性诊断另列。计时包括tokenization/prefill、Markov循环、构树/传输/mask、Target验证、选路/KV/停止/解码；加载、下载、冷编译、校准单列，不计HTTP网络/排队。评分在计时后执行，不根据速度或正确性删题/改主配置。

检查同方法重复、相同输入/停止规则、预算/cap/轮次与实际发出长度、正式新增编译、资源峰值。严格方法与AR的token差异逐项记录，不能因已知一题几何数值差异而忽略新错误；发现非数值实现问题须先修复。不同方法的近似输出差异按质量评分，不要求强行逐token相同。

## 结果报告

分别比较DSpark组合对旧DFlash组合，以及每种Draft内部树的增量、ASD的增量；报告自然时延、遵守EOS共同128、固定TPS、输出长度/截断、原评分器质量、实际发出Draft接受长度和验证轮次。开放输出保留，但本轮未运行外部裁判时不声称开放质量保持。原生长度和检查点训练来源不同，因此换底座比较不单独归因Markov结构。

输出`results/dspark-tree`，服务器隔离目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/dspark-tree`；旧结果目录保持原SHA。专项测试：`.venv\Scripts\python.exe -m pytest -q tests/test_dspark.py tests/test_budget_tree.py tests/test_draft_tree.py tests/test_dflash_reuse.py`。准确执行命令、源版本、校验及运行结果追加到PROGRESS与结果报告。
