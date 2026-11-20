# MemoryDraft 与树流的分工

[ShuLiu-Decode](https://github.com/2223434778/ShuLiu-Decode) 是树流的独立项目。成本树、ASD 树路径验证、保真门控、反馈调宽、执行优化、相关实验总表，以及聊天网页、账号与历史记录、部署模板，统一在该仓库维护。

2026-11-20，本仓库清理了对应副本。共享推理接口恢复为接入成本树之前的版本；MemoryDraft 继续保留原 H/R/D 研究、后缀与 OT 检索、DFlash、早期 ASD 研究、适配与公共前缀实验，以及原 infra 服务和 benchmark。早期固定候选树是历史可行性实验，并非树流的成本树实现。

## 入口

- 树流算法与最新实验：[ShuLiu-Decode 首页](https://github.com/2223434778/ShuLiu-Decode)。
- 树流网页与阿里云/A6000 部署：[网页部署](https://github.com/2223434778/ShuLiu-Decode/blob/main/docs/CHAT_DEPLOYMENT.md)。
- MemoryDraft 推理服务：[infra 指南](INFRA_GUIDE.md)。
- 原面试项目：[独立冻结版本](INFRA_INTERVIEW.md)。

这次清理不重写 Git 历史，不移动 `infra-v0.3-interview`、`infra-v0.4-interview`、`infra-v0.5-interview` 标签及对应冻结分支，也不修改线上部署、聊天数据库、模型或独立 ShuLiu 工作目录。旧实验分支与历史提交仍可用于追溯。

`PROGRESS.md` 的旧条目是当时的运行记录，其中某些相对路径已迁出。需要完整还原旧环境时，请查看 [清理前的 MemoryDraft](https://github.com/2223434778/MemoryDraft/tree/ce807c7fe275514bc019517ba0d13e9d8e734413)，或检出对应冻结版本；不要在当前 MemoryDraft 重新接回树流模块。
