# Infra v0.4 独立面试交付：成本树、DFlash与ASD

新版冻结为`codex/infra-interview-v0.4`和`infra-v0.4-interview`，对应独立副本位于`artifacts/infra-interview-v0.4/project/`。上级目录有`START_HERE.md`、`MemoryDraft-infra-v0.4.zip`、`MemoryDraft-infra-v0.4.bundle`及`manifest.json`。manifest记录确切release commit、全部文件字节SHA、ZIP和bundle SHA；本地project拥有独立Git库。

v0.3的`codex/infra-interview-v0.3`、`infra-v0.3-interview`、`artifacts/infra-interview-v0.3`及服务器目录保持原样。新版新增成本树与ASD的可部署接口，不用研究候选覆盖旧版。

```bash
git clone -b codex/infra-interview-v0.4 https://github.com/2223434778/MemoryDraft.git MemoryDraft-infra
cd MemoryDraft-infra
```

离线可用`git clone MemoryDraft-infra-v0.4.bundle MemoryDraft-infra`。按[部署说明](INFRA_TREE_GUIDE.md)准备独立环境和本地审计权重，先执行calibrate-tree生成本机成本表和运行配置，再使用infer/serve/benchmark。环境、模型权重及大trace不在归档内；不能直接把服务器验收配置当作其他机器的有效成本表。

面试入口依次为：

1. [部署说明](INFRA_TREE_GUIDE.md)：统一API、显式严格/近似策略、成本资产与环境绑定。
2. [工程验收](INFRA_TREE_VALIDATION.md)：1144本机测试、150服务器测试、60次真实GPU计时及12研究等价检查。
3. [算法架构](ASD_TREE_ARCHITECTURE.md)：成本树组织候选、ASD在预算内选路、树attention和KV提交。
4. [独立研究证据](ASD_PAPER_EXTENSION_FINDINGS.md)：530匹配输入的适用场景和速度/质量取舍。
5. 原[infra指南](INFRA_GUIDE.md)：单worker、有界队列、取消/背压、SSE、profiler与回归阈值。

可以讲述从研究模块到可配置运行时的迁移：成本表校准和checksum、启动环境核验、请求预算隔离、失败证据保存、流式提交与取消恢复、严格和近似评测分离。ASD允许改变输出，不宣称无损；短小验收批次不是容量规划，七任务研究收益与工程接入验收分别报告。连续批处理、分布式推理及自定义CUDA算子仍不是已实现能力。

重建另一个独立副本（输出目录必须不存在）：

```powershell
.venv\Scripts\python.exe -m scripts.archive_infra --ref codex/infra-interview-v0.4 --version 0.4 --output artifacts/infra-interview-v0.4-copy
```

实际GPU执行源为`2a779207cf98e929c5574f5ef82290bf70fc45f9`，发布结果及说明在后续提交封存；runtime包的88个源文件与该执行源一致。release与GPU源的关系、各载荷SHA可用`results.infra-tree-v04.helpers.verify_publication --git`及`scripts.verify_frozen_releases`复核。
