# Infra v0.4 成本树 + DFlash + ASD 验收

统一配置、CLI、服务JSON/SSE和benchmark已接入成本树，保留原严格方法；近似ASD必须显式启用。

本机完整回归 **1144 passed / 7 skipped**；A6000服务/树专项 **150 passed**。
已安装0.4.0 wheel：183025B，SHA-256 `6c914ff8a3d0381501db4581b948d202c4e4e89a8790511203018ee3fdfeea5c`。无Torch/无FastAPI的隔离环境中3项CLI/demo通过，88个包文件与本次GPU源逐字节一致。

## 真实GPU部署验收

- 单A6000、Qwen3-8B/审计DFlash、FP32/eager attention/RMSNorm/thread4、B8/g.25/M2。真实CLI在合成前缀校准成本，自动输出绑定哈希的运行配置。
- 6个已观察公开请求（GSM2、HumanEval1、MT首轮1、Alpaca2），五方法两重复，共60主计时。另6条AR参考用于运行时身份检查。
- AR/DFlash/BudgetTree共36计时行逐token、prompt、snapshot一致；ASD和组合24行校验同输入/预算，correct为null，不冒称答案质量通过。
- 严格树与组合各6次，共12次同成本表的研究实现对照，输出/轮数/ASD账本一致。两个树方法取消后恢复相同结果。
- 真实GPUengine的ASGI TestClient JSON/SSE输出、成本表身份和预算一致。它验证应用接口及单worker链路，不是新增TCP压测或容量结论。
- 正式新增编译图0；峰allocated约34.596GiB。加载、离线校准和预热另列。

## 验收批次计时（不作新研究结论）

|方法|观测/独立请求|自然输出token/s|相对AR的总时延比倒数|严格身份|
|---|---|---:|---:|---|
|AR|12/6|20.545|1.000×|通过|
|DFlash|12/6|60.031|2.922×|通过|
|ASD|12/6|65.797|3.203×|近似，任务质量未评|
|BudgetTree|12/6|72.138|3.511×|通过|
|BudgetTree-ASD|12/6|78.761|3.834×|近似，任务质量未评|

自然停止和近似路径可能改变输出长度，上表是小规模部署验收，不替代七任务独立研究或开放质量评估。完整在线成本计入，插桩不混作正常计时。
成本表依赖模型和运行环境，新设备应重新校准。原研究收益与质量取舍见[七任务结果](ASD_PAPER_EXTENSION_FINDINGS.md)，使用方式见[部署说明](INFRA_TREE_GUIDE.md)。

## 来源和复核

实际GPU源提交 `2a779207cf98e929c5574f5ef82290bf70fc45f9`，2628 canonical Git文件执行前后SHA一致；source SHA `279af9ebe7f67ad406a0ba611eaed48c5183616a41a47eb6309bc413398861bd`。
本次成本表SHA `ab72a7946fa14ac33c6bc081ff5cf683bcbae6b60ef8ca74bc7c4bcead46c30c`。原始行、校准样本、HTTP结果、测试、执行验证与源码manifest保存在[results/infra-tree-v04](../results/infra-tree-v04)。
原infra-v0.3与研究固定分支保持；新版独立归档单列，不把后续文档提交冒充GPU执行源。

```powershell
.venv\Scripts\python.exe -m results.infra-tree-v04.helpers.audit --check
.venv\Scripts\python.exe -m results.infra-tree-v04.helpers.verify_publication --git
.venv\Scripts\python.exe -m scripts.verify_frozen_releases
```
