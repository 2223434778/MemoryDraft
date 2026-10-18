# 独立 infra 面试交付

OT-R 研究开始前，已将验证完成的 infra 版本固定并单独保存：

- Git 提交：`27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`。
- 固定分支：`codex/infra-interview-v0.3`；标签：`infra-v0.3-interview`。
- 本机独立仓库：`artifacts/infra-interview-v0.3/project/`，有自己的 `.git`，不受研究分支修改影响。
- 同目录上级含 `START_HERE.md`、源码 ZIP、离线 Git bundle、`manifest.json`；423 个受版本控制文件均按 Git blob 原始字节导出并逐项校验。
- ZIP SHA-256：`f4ecf9af4d37a44d0c087283a576dd28abf2215d99373d31e6e83138f910a2fe`。
- bundle SHA-256：`1ec8b71b965ad18161dbc8b6462c9575e5986ee7900393d8daa79197b2fe197a`。

新电脑可以 `git clone -b codex/infra-interview-v0.3 https://github.com/2223434778/MemoryDraft.git`。
离线时可 `git clone MemoryDraft-infra-v0.3.bundle MemoryDraft-infra`。
环境与权重不打包，CPU demo 不需要模型，GPU 按该版本 `docs/INFRA_GUIDE.md` 准备。
服务器既有 `infra-v0.3` 保持不变；新实验在独立目录进行。

复现归档（输出目录必须不存在）：

```powershell
.venv\Scripts\python.exe -m scripts.archive_infra --output artifacts/infra-interview-v0.3-copy
```

面试入口依次为该版本 README、INFRA_GUIDE、INFRA_VALIDATION 与 RUNTIME_MIGRATION_RESULTS。
可讲述固定 Target 正确性、KV 切换、单 worker 与有界队列、取消/背压、SSE、配对评测、profiler 和真实回归定位。
GPU 证据支持 batch=1、greedy、FP32/eager；连续批处理、分布式推理、自定义 CUDA 算子不是已实现能力。
旧负结果和受控任务约 19.4% 延迟收益均保留，不将其说成通用生产加速。

OT 代码、实验结果和依赖只在 `codex/ot-r-probe` 研究分支演进，不自动合入这个面试交付。
