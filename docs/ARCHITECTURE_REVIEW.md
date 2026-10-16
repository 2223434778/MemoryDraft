# 架构复核 — 2026-10-16

## 结论

架构机制合理，可以开始工程实现。性能仍是待测假设：关键在 H 提高接受量是否覆盖其额外 prefill、起草和状态维护成本。本轮完成了文档级复核及官方来源检查，没有执行推理或验证具体库版本的兼容性。

## 保留的核心

- 用户跨会话历史作为候选来源。
- L1 语义召回候选池，L2 在池内做前缀接续匹配，两层为级联。
- R 直接复用，H 历史辅助起草，D 普通起草；H 是必做研究模块。
- Target 固定输入、严格验证；缓存回滚与实际 proposal 保持一致。

## 本次修正

| 项目 | 决定及原因 |
| --- | --- |
| 初始底座 | 优先独立小模型 H/D，SAM 作为候选而非强制底座；避免 EAGLE 特征恢复拖延核心 H |
| Target 事实记忆 | 应用 RAG 设置 B 为主场景；各基线共享同一事实记忆，A 只作为补充 |
| H/D 对照 | 统一非历史 prompt 条件，隔离历史本身的作用 |
| Token 前缀 | 追加已确认 token IDs，避免 decode/re-encode 改变切分 |
| 历史时间 | 恢复当时版本、状态和派生信息，不能用未来更正标记筛选过去记录 |
| H 路由 | 核心模块必做不等于每请求强制启用；计入冷启动和探测成本 |
| 强基线 | 第一部分先确认成熟 Draft 的模型兼容性，第五部分完成完整混合对照 |

## 为什么在机制上成立

贪心验证不要求 Draft 与 Target 上下文相同，只要求 Target 在固定真实前缀下正确验证并提交连续一致 token。随机采样必须采用真实 q 的接受和残差规则。历史可以改变 proposal，却不能绕过 Target 改变所承诺的输出分布。

## 仍然需要 Codex 实测的边界

1. 所选主底座是否允许独立 H prompt、KV 和验证适配。
2. 目标模型与小模型词表映射及 special tokens 是否兼容。
3. H 首次 prefill 与非活跃路径 catch-up 是否抵消收益。
4. 真实跨会话数据是否有足够可复用文本，语义召回是否选中它们。
5. 相对官方可用成熟 Draft，增益是否仍存在。

## 官方来源

- [标准 speculative sampling](https://proceedings.mlr.press/v202/leviathan23a.html)
- [SAM-Decoding](https://github.com/hyx1999/SAM-Decoding)：公开描述包含后缀检索和模型 Draft 切换。
- [RAPID](https://github.com/real-absolute-AI/RAPID)：检索上下文辅助 Draft 的参考。
- [RAPID 论文](https://proceedings.mlr.press/v267/chen25s.html)：其完整算法不直接等同于本项目固定 Target 的验证。
- [EAGLE](https://github.com/SafeAILab/EAGLE)：成熟学习型基线，需匹配权重与特征接口。

复核不构成“无人做过相同架构”的证明，也不构成已验证性能的声明。
