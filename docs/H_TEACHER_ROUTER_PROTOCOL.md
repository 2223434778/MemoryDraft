# Target 历史下的对称 H 开关验证

2026-10-17。用户授权验证。运行前固定以下方案；复用六个已测合成项目，只作追加机制诊断，不作为新的独立保留集。

- 直接复用上一轮先生成过去回答、再读取当前请求的冻结 Target 历史。原始 SHA-256 固定为 `55483ae5160588591aed932bca5fb62e188e301f39bd064c817e3abcab0c919a`。校验过去请求、替换版本、干扰记录及当前请求不变，不重新生成或筛选历史。
- R 与 H 读取同一用户、同一请求前的合法快照。R 保留完整静态接续索引及原动态 lookup；H 使用原请求级语义检索与 512-token 历史预算，R 不受 H 的 top-k 限制。
- 复用上一轮 `MatchedRouterComparison` 和同一 RHD 路由代码，比较 AR、RHD_off、RHD_on、RHD_on_cached。on/off 的 D/AR 探测、EMA、冷却、驻留及 D 原生上下文不变；开启 H 后增加的探测和全部在线成本计入。
- 固定 Qwen3-8B/0.6B、FP32/eager、非思考 greedy、块长 2、输出上限 128、六请求、三次交错重复，共 72 次计时方法运行。先跑两个请求 × 四方法、输出上限 16 的 8 次 smoke。smoke 与正式计时使用不同的空向量缓存。
- 每条输出与上一轮 AR 的 token IDs 和 Target prompt hash 核对；同请求各方法的快照必须相同。当前答案不写入历史，结束后检查快照未变；首次差异保存证据并停止。
- 主指标为配对 H-off/H-on 总时间比及按请求成组 bootstrap 95% 区间；大于 1 表示 H-on 更快。另报缓存控制、首次/后续访问、逐项目结果、验证轮数、H+D 起草轮数、接受量、TTFT、在线 L1 和 H prefill/追赶输入。host 分阶段计时不能当作独立 GPU kernel 时间相加。
- 保留原人工历史实验作为历史背景，但不同批次时间不直接作来源效应的因果估计。主结论仅来自本批 Target 历史下的 H 开关；不根据结果调整触发阈值、块长、模板或样例。

服务器项目根目录运行：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/run_h_teacher_router.py
```

输出目录 `results/h-teacher-router/`，拒绝覆盖已有结果。模型加载、公共文本预热及独立 AR 评分参考不计入请求延迟；在线快照、检索、索引、prefill、起草、验证、追赶、缓存维护和输出解码计入。模拟过去回答的构造成本已在上一轮记录，不发生于本轮当前请求，也不计作免费在线重生成。
