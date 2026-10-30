# 官方树基线补全

2026-10-30。用户授权第一步：补齐DDTree、BASTION与现有方法的直接性能/质量对照。

- 官方BASTION commit `1989b80b304f01710946d7fb68af330c03d97461`，https://github.com/kaist-ai-osi-lab/BASTION ，MIT。
- 官方DDTree commit `c96427a185677bf4133ed865dd1626a5041aef9b`，https://github.com/liranringel/ddtree ，MIT。
- 外部代码只放隔离实验快照third_party，保留许可；UPSTREAM_TREES.json逐文件SHA绑定。调用官方generation/build/verify/cache，不以本项目简化成本树代替官方实现。
- 共同Target Qwen3-8B与原DFlash-b16冻结权重，单A6000/GPU1，FP32/eager、既有RMSNorm、4线程、greedy。共享加载的模型实例。这是统一执行栈的官方算法对照，不是原文BF16/SDPA/FlashAttention环境复现；官方BF16 roofline经同一FP32硬件实测重新拟合，不直接使用BF16标称时延。
- 九组：AR、DFlash、ASD、严格成本树、成本树+ASD、当前BufferedKV-ASD、官方DDTree总32/128节点、官方BASTION默认32..8192自适应。DDTree API预算不含根，分别传31/127；两档预先固定，不择优汇报。启用官方CLI默认可选C++缓存整理，若编译失败记录回退。BASTION使用官方完整默认合成校准网格与sigma_scale=30、三次官方常数profile。我们的成本表也重新校准。
- 沿用既有62源开发题/66匹配输入，GSM8K8、MATH12、HumanEval8、MBPP8、MMLU14、MT4源双轮/8输入、Alpaca8；并非新保留集。包含旧慢题和失败题，不按结果剔除。每格一次自然4096/EOS，共594计时，不强制EOS后继续生成。
- 每请求确定性打乱组别；Target输入逐ID哈希与历史快照必须匹配既有冻结输入。MT第二轮共享原冻结AR第一轮回答；新AR也在本窗口实测，不引用历史AR时延。新AR输出差异与严格方法数值差异逐项记录，不假装全部一致。
- 全请求同步计时包括tokenization、prefill、第一轮draft、构树、CPU验证、缓存提交、输出传回CPU与decode。官方内部decode计时不替代端到端。官方return_stats引入的同步计入在线成本，记录此实现差异。模型加载、编译、合成校准和预热另列，运行不调用网络。
- 质量沿用原始代码测试、数学等价答案评分；MMLU沿用指定字母生成合同，格式合规和知识能力不得混淆。MT/Alpaca本轮只提供输出与速度、AR一致性，不把一致性当开放回答质量评分；裁判评测单列后续阶段。
- 报告逐任务完整wall、tokens/s、输出长度、正确集合/截断/AR一致性、验证轮数、树节点、峰显存，同时提供逐题配对时延比与混合总量比；二者不混用。不将更短回答全部解释为每token提速。
- A6000独立目录，先检查27B空闲，GPU1服务暂时释放并finally恢复；GPU0聊天保持。旧infra冻结分支与当前线上代码不修改。

命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_external_trees`。
本机快速检查：`.venv\Scripts\python.exe -m pytest -q tests/test_external_trees.py tests/test_tree_execution.py --basetemp artifacts/pytest-external-trees --tb=short`。
