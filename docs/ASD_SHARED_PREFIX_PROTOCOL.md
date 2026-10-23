# 共享公共段：整前缀淘汰问题的最小修复检查

2026-10-23 GPU前协议。只在同一公开模板家族中合并精确token公共段，复用现有PublicPrefixCache；不实现通用块/分页存储，不训练或改变接受规则。

- 原registry SHA9bfda318cd757144b64961e8cc05d6a29f95c00e9c65229b2ab8ca2a365bae82中两个4K entry长4073/3937，其最长公共段3934 token。新registry保留三个短entry，替换两个长entry为一个公共段，保存父SHA/ID/长度；新SHA f124b3a0791ed5fc2bcb15486925e8898d3f9a0bef2016ac7faf419b2fe6add1。公共段完全由既有登记token求交，不使用测试答案或生成结果。所有原请求逐字节不变。
- 原模型FP32/eager+RMSNorm，原ASD B=8/gate=.25/cap=2；同样2GiB/3 entries，Target/Draft计算和验证器不变。新配置runtime-dflash-asd-shared-prefix.json，原配置/registry/面试版保持。
- 仅复用原8个已观察任务的4K输入，shared下两方法DFlash/ASD×自然EOS(4096上限)/128且EOS×off/cold/hit，共96正式生成。每格一次，cold后紧接hit，off前后次序沿用固定规则；查找/复制/填充/淘汰均计入请求。自然与共同完整128分别报告，短于128不强制续写。
- 检查token/text/EOS/输入/快照、验证总轮数及ASD汇总量，spent差单列、实际预算仍≤8。严格D核对旧完整AR；近似输出保持原评分。汇总相同不冒称逐轮接受边界相同。预先warmup覆盖4K off/cold/hit及两方法，正式新编译为0；冷模型加载/编译成本另存。
- shared通过后：跨问题复用、直接取消、真实JSON/SSE/断连恢复；每卡原四题4K流轮换，20off+20shared，客户端并发2、空缓存起步。whole控制在新进程加载旧配置，仅做同流20whole HTTP负载及相同协议检查，不重新做旧自然矩阵。共120 HTTP负载请求，另有验收请求。共享/整项两种cache窗口逐index核对相同输出/快照/汇总量，记录实际hit/miss、填充字节、淘汰、queue/client/engine时间，不预设命中结果。
- 两波：GPU0 shared→whole，GPU1 whole→shared，同题始终同卡；shared内部先off后cache，保留窗口先后与两卡题目不同的限制。whole与shared为独立engine/编译cache，加载20秒不藏入部署收益。只主张这两个固定请求流的工程结果，不作生产SLA/continuous batching声明。
- 主要问题是交替流能否在原容量下减少淘汰、降低总HTTP成本；同一4073前缀流则测公共段稍短带来的后缀计算代价。即使命中提高也要求全部在线成本实测更低。新KV边界和分段prefill浮点差异先通过检查，失败保留原始数据，停止该进程后续阶段，不据时延选择排除任务。
- 每worker allocated≤46GiB、2400秒、原编译8图/backend300秒；四进程磁盘缓存合计≤16GiB。新root asd-shared-prefix，源Git blobs前后核SHA，旧结果不覆盖。入口 `OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_shared_prefix_pipeline`。
- 公共数据、代码、配置、JSON/JSONL/XML/log可发布；不上传权重/KV/features/raw trace。报告复用原analyze但明确其decision字段仅为汇总量。全部质量/数值/未命中反例保留。

设计参考：[vLLM前缀缓存与块级共享](https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/)。本轮只做最小公共段登记，不将其称为完整vLLM块缓存复现。
