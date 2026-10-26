# Infra v0.4：DFlash + 成本树 + ASD

本版把已评测的成本树算法接入统一运行时、CLI和服务。复用原DFlash解码循环、树attention与KV提交，不在生产调用中patch研究脚本。v0.3面试版保持原固定分支、标签和独立副本，新版单独交付。

## 可选方法

|method|需要配置|行为|
|---|---|---|
|AR|Target|Target逐token生成|
|DFlash|DFlash|原严格线性候选|
|R+DFlash|DFlash，可选合法历史库|原严格检索复用|
|ASD|DFlash、asd|原近似线性候选|
|BudgetTree|DFlash、cost_tree|成本受控树，严格路径验证|
|BudgetTree-ASD|DFlash、cost_tree、asd|成本树和请求级ASD预算选路|

ASD配置示例为B8/g.25/M2。它允许改变输出，默认接口不会自动把DFlash改成ASD。严格树沿用已发表实现中的有限精度验证；跨前向形状近并列token可能不同于单token AR，严格benchmark会保存失败并停止，不放宽规则。

成本树可减少验证轮数，但每轮需要验证更多节点，收益由完整在线耗时决定。树+ASD的七任务速度与质量证据见[研究结果](ASD_PAPER_EXTENSION_FINDINGS.md)；开放质量与代码功能有已记录取舍，服务接入不消除这些取舍。

## 准备和校准

沿用[infra环境说明](INFRA_GUIDE.md)的独立DFlash依赖profile、固定Target/Draft及本地权重校验。8B/FP32组合使用48GB级显卡，已授权服务器可运行；CPU测试和HTTP demo不需要模型下载。先修改`configs/runtime-dflash-tree-base.json`中的本地模型路径。

```bash
python -m memorydraft calibrate-tree --config configs/runtime-dflash-tree-base.json --output artifacts/tree-costs.json --runtime-config artifacts/runtime-tree.json
python -m memorydraft infer --config artifacts/runtime-tree.json --request examples/request.json --method BudgetTree-ASD --output artifacts/tree-answer.json
```

第一条命令使用合成前缀，在128/1024上下文和8/12/16/20/24/32节点上测Target验证，并测Draft和构树开销；每形状首遍单列，余三遍中位数。校准是显式离线步骤，不读测试答案，不混入请求时延；所有输出拒绝覆盖。命令生成成本表及已填入SHA-256的运行配置，无需手抄哈希。

`cost_tree`配置包含`path`、`sha256`及可选`min_nodes=8`、`max_nodes=32`、`top_k=4`；当前主方案固定`spine_mode=none`。路径相对配置文件解析。加载时校验完整资产哈希及环境：Target revision、审计DFlash revision、adapter身份、GPU型号、Torch/Transformers、dtype、attention、执行profile、线程数和原生块长。环境不同则拒绝，需重新校准；不会隐式借用论文或另一张型号显卡的成本表。

配置加载后成本表驻内存，修改策略需重载engine。节点范围必须落在实测范围内；上下文沿用研究代码的线性插值/外推，长于校准范围的成本仍是估计，不是逐轮现场测速。

## API、流式输出和观测

```bash
python -m memorydraft serve --config artifacts/runtime-tree.json --host 127.0.0.1 --port 8000
```

`POST /v1/generate`继续使用原请求格式，指定`"method":"BudgetTree-ASD"`或`"BudgetTree"`。每请求新建ASD账本；客户端不能在请求中覆盖预算或成本表。`/healthz`公开支持方法、近似方法和成本表身份。

JSON及SSE complete包含`tree`配置身份，`counters.tree_rounds/tree_verification_nodes/tree_built_nodes`记录验证/构树工作量；近似结果还有`approximate=true`和`asd`预算。SSE start也标注近似方法，tokens只发布最终选路已提交的前缀，不发布候选树的其他分支。取消在轮次边界协作执行，清理同步后再处理下一请求；旧有界队列、背压、deadline和单worker所有权不变。

```bash
python -m memorydraft benchmark --config artifacts/runtime-tree.json --requests examples/requests.jsonl --methods AR DFlash BudgetTree --repeats 3 --output artifacts/tree-strict-bench
python -m memorydraft benchmark --config artifacts/runtime-tree.json --requests examples/requests.jsonl --methods AR DFlash ASD BudgetTree BudgetTree-ASD --allow-approximate --repeats 3 --output artifacts/tree-comparison
python -m memorydraft profile --config artifacts/runtime-tree.json --request examples/request.json --method BudgetTree-ASD --output artifacts/tree-profile
```

默认benchmark仍要求严格AR token/prompt/snapshot一致性。`--allow-approximate`只显式允许ASD方法，核对同输入和预算账本；其`correct/strict_correct=null`、`quality_evaluated=false`，token_match_ar只是输出身份诊断。自然时延可能伴随输出长度变化，任务正确率、代码测试和开放偏好须另评。近似结果不能用于严格回归门槛；严格回归还核对成本表哈希和树策略。

profile复用`build_tree`、`target_verify`等已有阶段，单独运行，不混作无插桩性能。运行时未增加遥测或远程推理调用。

## 验证与归档

本机完整1144项测试通过/7跳过，A6000专项150项通过；真实模型60次计时、12次同配置研究实现对照、取消恢复及JSON/SSE接口验收完成，正式新编译图0。真实模型接口使用ASGI TestClient，不是新增TCP容量测试。见[验收报告](INFRA_TREE_VALIDATION.md)、[冻结协议](INFRA_TREE_ACCEPTANCE_PROTOCOL.md)与`results/infra-tree-v04`。该验收证明部署入口和研究实现一致，不重算或扩大旧论文研究结论。

旧发布工具的`--git`源保护检查面向发布时HEAD，新版runtime会合法演进；使用以下命令核验旧载荷和对应冻结Git源：

```bash
python -m scripts.verify_frozen_releases
```

单分支或离线副本可能没有旧分支名称，工具会核验已固定的commit对象并明确标注缺失的名称；若已有分支或标签被移动则拒绝，不把缺少名字误报为旧数据损坏。

不会重写已封存的旧结果、报告或验证脚本。v0.3固定引用及本地归档不变；新版独立副本见[面试交付](INFRA_TREE_INTERVIEW.md)。
