# RMSNorm显式运行时profile接入候选

状态：48题独立验证已通过，显式选项已合入研究分支，真实GPU接入验收中。首次服务器前置测试发现延迟导入测试夹具引用旧模块，已修复后准备新源码重试；尚不声称接入验收通过。默认配置仍eager。

配置字段execution_profile默认eager；configs/runtime-dflash-rmsnorm.json显式选择rmsnorm-inductor。该数值profile只对原Qwen3RMSNorm编译，不改变precision/attention、Target输入、接受规则或检索。使用已测Linux/PyTorch2.5.1/Transformers4.57.3、CUDA FP32 Target和Draft；不支持的环境明确拒绝，不静默回退。

编译入口与实例forward绑定在引擎加载时准备，持续到engine.close；每请求不重复切换。compile本身仍lazy，首次触发成本在原generate同步计时及TTFT内。元数据单列execution_profile、加载阶段setup_seconds、图预算；结果counters记录调用前后图数及本次backend编译秒（是请求时间内部子阶段，不相加）。benchmark显式warmup可提前支付编译，不能把warmup wall全当纯编译。

引擎保持单owner，stream只提交已确认token；取消清理后下一请求仍使用同一固定profile。关闭只在无请求拥有锁时恢复所有绑定，异常同步也尝试恢复。改变profile需要重启worker进程；不提供请求级切换。依然没有连续批处理/采样支持。

完整输出及质量一致只针对被评测数据，FP32归约末位仍可能变化。默认配置及独立infra面试版保持原样。CLI/HTTP性能不能直接引用研究runner的约2.8%数值；接入后的冷调用、stream/cancel及R路径需另做小规模真实GPU检查。

## 接入验收安排

独立验证报告的admission全部通过后，才允许运行`python -m scripts.run_rmsnorm_runtime_pipeline`。使用冻结Git源码导出及`RMSNORM_RUNTIME_SOURCE_MANIFEST.json`，在新目录连接已有只读models；离线运行，不下载模型。单A6000、独立空Inductor/Triton缓存、编译线程2，每子进程最多1800秒（CPU测试300秒），只终止本轮创建的进程会话。

先运行runtime/serving/regional相关测试，再执行`python -m scripts.validate_infra --config configs/runtime-dflash-rmsnorm.json --output results/raw/rmsnorm-runtime/acceptance`。前两个AR/DFlash冷调用保存真实请求时延和编译计数；随后2个公开示例×3方法×2重复，共12计时，要求完整token/prompt/snapshot与同profile AR一致，并至少实际接受一次历史R候选。三个方法均检查HTTP SSE完整输出，另做6请求/并发2的排队检查、取消恢复及独立CUDA profiler。所有请求结束后历史快照必须不变。

首个验收进程完全退出后，再在新进程运行实际CLI `python -m memorydraft infer --config configs/runtime-dflash-rmsnorm.json --request examples/request.json --method DFlash --output results/raw/rmsnorm-runtime/cli.json`，使用已准备磁盘缓存；这不是第二个空缓存冷启动。图最多8、backend累计300秒、缓存4GiB、记录的peak allocated46GiB，正式benchmark不得有新增编译。结果包排除数据库、原始Chrome trace、权重与编译缓存；保留紧凑operators、正确性与排队记录。示例验收只说明工程可用性，不把小样本延迟当作新的性能结论。
