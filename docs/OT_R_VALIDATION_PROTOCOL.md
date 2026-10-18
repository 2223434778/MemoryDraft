# OT-R：新项目场景验证（运行前冻结）

2026-10-18。继首轮探索之后，检验同一机制在新编写任务上的迁移。
这是12个手工编写的受控项目、24个请求，不是真实生产流量，也不是12个外部开源仓库。
任务未根据本轮模型答案筛选；不把旧项目9.23%的事后子组作为确认结果。

## 假设与任务

1. `local-edit`：parcel-queue JSON、artifact-store Python、catalog-deploy YAML、payment-runbook Markdown。
   每项目两个独立v1修订，检验局部改动后较长原文的接续恢复。
2. `ambiguity`：worker-pools、tiered-reader、gateway-routes、cache-operations。
   重复块和两个项目中的相似旧记录，检验对齐后是否选择正确接续。
3. `control`：format-conversion、empty-history、unrelated-history、short-settings。
   覆盖格式改变、无历史、无关历史和短输出，保留可能退化的请求。

源数据为 `data/research/ot-validation/cases.json`，由已提交的
`python -m scripts.make_transport_validation --output <new-path>` 导出。
每个当前请求包含完整旧文和修订指令，因此Target不依赖R提供知识；不主张历史独有价值。
历史为预先编写的v1和少量干扰记录，order=1/2；两个请求order=10/11，均独立修订v1。
不同项目使用不同user_id。所有AR/候选结果均不写回历史，不将第二个请求改成依赖本轮第一个答案。
expected_text仅由显式输入执行字符串替换得到，用于事后任务评分；不进入检索或Target输入。

## 固定机制与执行

沿用首轮package，不改解码、索引、matcher、阈值或门控。精确6-token后缀快速路径优先；
缺失后对最近<=32个已确认token与<=8个合法旧窗口作OT/edit对齐，每请求最多4次。
四锚点、每锚点64位置、端点移动±2、最少8-token上下文、得分阈值0.65。
OT词法/位置代价、双dustbin、位置权重0.25、熵0.06、丢弃代价0.35、80次迭代沿用首轮。
相同数值阈值并不使OT/edit得分尺度等价，本轮不在确认任务上校准。
候选只是一段历史连续原始token，Target逐token greedy验证规则不变。

四组为 `DFlash` / `Suffix-R` / `Edit-R` / `OT-R`；每请求两重复，固定种子随机交错方法顺序。
共192次计时，另先生成24个新AR参考。24个参考全部完成后才开始四组计时。
冻结512输出预算、Qwen3-8B/DFlash固定权重、A6000 GPU0、FP32/eager、4线程、non-thinking。
AR前记录每例Target输入token哈希和合法快照；所有组检查token IDs、prompt和snapshot，首个分歧即停。
任务要求只返回完整修订文，不加围栏或解释。单独报告任务契约：仅忽略结尾LF数量，其余字符逐字比较。
原始逐字检查与EOS/长度截断另报；任务失败不剔除、不补选样例、不事后缩小主表。

全部在线快照、tokenize、冷索引构建、对齐初始化/查询、起草、验证、KV追赶、同步和decode计入。
POT导入、模型加载、预热、新AR参考单列。不测HTTP/profiler，不使用暖历史索引冒充冷索引性能。
旧结果目录及任务文件运行前后SHA检查，执行源码清单独立保存；最终报告来源不替代计时来源。

## 判断标准

主比较为OT相对Suffix与同池Edit的**端到端延迟比**；同时报告DFlash对照、逐项目结果、
R/D轮数、实际接受token、对齐调用/提议/查询成本。旧首轮19.4%收益不计为新增OT贡献。
两个候选适用场景预先指定为local-edit与ambiguity，control只检验代价与失效边界。

预先设定的继续投入工程门槛：在至少一个候选场景中，OT总延迟比Suffix低>=5%、比Edit低>=2%，
且同场景至少3/4项目分别优于两者；8/8请求任务契约正确且没有长度截断。
这些是工程筛选门槛，不是统计显著性或生产SLA。
按项目整体重采样报告95%描述性bootstrap区间，每个项目内两请求/两重复保持成组；四项目组枚举4^4抽样，
全部十二项目使用固定种子4000次抽样。重复不能当独立项目，三个分组不用于显著性筛选。
若区间跨1，应明确证据不稳定，不将工程门槛通过宣布为研究确认。

若近似对齐有效但OT未优于Edit，优先保留便宜的Edit；若无场景达到门槛，本轮停止扩大同类实验。
若有清晰跨项目信号，下一阶段才独立开发成本门控并在另一个留出集验证；本轮不混入新门控。
不基于本轮答案调参、替换失败请求或反复增加重复。

## 复现

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_transport_validation --config configs/runtime-dflash.json --output artifacts/ot-validation-v1
python -m scripts.write_transport_validation_report
```

完整infra面试版仍固定 `27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`，不改main、面试分支/标签或服务器infra-v0.3目录。
