# 公共前缀 Target KV / DFlash 特征缓存检查

2026-10-23，GPU 运行前冻结。借鉴成熟 APC 原理，不作新算法首创声明。

- 固定原 Qwen3-8B / DFlash、float32/eager + RMSNorm 编译配置和 ASD B=8/gate=.25/cap=2。不训练，不改验证器。显式配置 runtime-dflash-asd-prefix-cache.json，默认和独立面试版本不变。
- 原长输入 8 道已观察 GSM 题 × 512/2048/4096，24 个完整请求逐 token 不变；只复用公开 train worked examples。登记表 5 个前缀，不包含最终问题、当前测试答案或用户历史。tokenizer/revision/template/登记文件 SHA 绑定；实例只服务一个 Target 与 feature layer 布局。
- 命中后复制独立 DynamicCache 和拼接后的选层 features；Draft KV 每请求新建。未命中保持完整 Target forward，截取允许的公共部分克隆入库，复制开销计入本次生成。缓存上限 2GiB/3 entries、LRU；大 entry 绕过，精确输入前缀比较，禁止跨用户保存问题或答案。
- 24 输入 × DFlash/ASD × 自然 EOS(4096上限)/128且EOS × off/cold/hit，共 288 正式生成，每格一次，单请求配对、不把 token 当独立样本。每个冷填充后紧接一次命中，off 位于二者前/后按固定种子平衡；每个 cold 清空缓存，全部计时含查找、填充、复制、tokenize、解码。模型加载/首次编译另列。warmup 在正式前覆盖三长度与状态，不隐藏冷成本。
- 对比 token/text/EOS/输入/快照/验证轮数/ASD例外数，单独报告实际 spent 差值，不修改线上预算。严格 D 还核对上轮完整 AR 参考。与上轮 off 的差异也保留；发生 token 或决策差异时停止该 shard 的 HTTP 阶段、保留失败现场，不把不同输出纳入等价性能结论。分段 prefill 的浮点非逐 bit 等价不能解释为验证标准放宽。
- 自然与128分别报告；短于128仍保留并标明，不以强制 EOS 后续写制造吞吐。主报告 hit/off、cold/off 及 cold+一次hit / 两次off，冷命中混合值是配对实测成本估算，不是另行执行的请求到达轨迹。
- 检查后继续跨问题公共前缀复用、直接取消、真实 localhost HTTP JSON/SSE/断连恢复。每 shard 四个 4K 请求轮换，HTTP 串行模型 worker/有界队列，20次 off 后20次启用 cache，客户端并发2、每状态从空缓存开始；保存所有 engine/queue/client 时间和 miss/hit 数。该 HTTP 次序固定，有顺序混杂，不能视为严格性能因果或 continuous batching；未作开放到达率/生产 SLA 声明。
- 两卡按原题奇偶分片，各题始终同卡。每卡 allocated peak≤46GiB，缓存≤2GiB，worker≤2400秒；编译沿用原 8 图/backend300秒限额，磁盘缓存两卡合计≤8GiB。若资源或正确性失败保留全部记录，不据收益排除题。
- 新 root asd-prefix-cache，source Git blobs/旧结果只读，执行前后核 SHA。入口 `OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_prefix_cache_pipeline`。只导出代码/公开输入/聚合 JSON/JSONL/XML/log，不上传 KV/features/权重/raw trace。

原理参考：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/)。这减少共享输入的 prefill 工作，不直接改善后续 Draft 接受率；DFlash 另需选层 Target features，故不能只保存 KV。
