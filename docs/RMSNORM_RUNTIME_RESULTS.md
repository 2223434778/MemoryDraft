# 显式RMSNorm运行时接入验收

独立48题预登记门槛全部通过后，已将rmsnorm-inductor作为显式execution_profile接入CLI/HTTP。默认eager、严格接受规则、Target输入、精度和原面试版均保持。真实单A6000工程验收通过，未用本轮两个示例替代研究性能实验；独立性能结果仍见ASD_RMSNORM_VALIDATION_FINDINGS.md（自然时延低2.68%–2.86%，240完整输出对相同）。

## 实现与实际检查

- 原RMSNorm.forward在engine加载时准备，167个实例绑定至引擎关闭；每请求不重复切换。单owner锁内检查closed和固定profile，取消后仍使用同一profile；close恢复绑定，忙时明确拒绝。配置和GenerationResult携带profile，benchmark拒绝跨数值profile回归比较；不支持的栈明确报错，不静默回退。
- 复用validate_infra：2个公开示例×AR/DFlash/R+DFlash×2重复，12正式计时全部与同profile AR的token/prompt/snapshot相同。复制历史是可追溯合成旧记录，4条R+D计时合计接受2个历史token；历史快照前后相同。此项只证明R确实触发且兼容，不证明长历史覆盖或增益。
- 直接取消回调后下一次生成精确恢复；实际Uvicorn/httpx服务6请求、并发2，全部成功；AR/DFlash/R+DFlash三路SSE token与参考及完成事件完全相同，结果profile正确。GPU profiler独立执行，记录13527个CUDA kernel事件和10类runtime阶段；原始大Chrome trace、数据库不发布。
- 首个engine进程退出后，新进程实际CLI输出45token，profile为rmsnorm-inductor。正式benchmark无新编译；CLI及冷调用编译成本不排除。服务结束后两卡均19MiB/0%。

## 冷启动与边界

加载阶段profile setup为0.00402秒，另于请求时钟记录；模型加载与权重校验属于加载阶段，未算成生成延迟。空Inductor/Triton缓存首次AR请求4.78660秒、TTFT3.65257秒，图0→4，backend3.49202秒。outer编译3.87917秒已包含backend，不相加。紧接其后的首个DFlash请求0.19941秒、图4→4，因共享norm已被AR编译，不是独立空缓存DFlash冷启动。

新CLI进程使用已有磁盘缓存，仍耗3.99864秒、图0→3，backend1.65654秒计入原请求时钟；磁盘cache存在不等于图建立或编译成本为零。验收进程最终4图、8图预算内；两进程之后cache417641字节，最大记录allocated34.6704GiB，符合原资源边界。没有将短示例收益换算为通用摊销或服务P95/P99。

示例warmed总时延AR/D/R+D为8.9742/4.3535/4.8169秒，均186输出token；D/AR约2.06×、R+D/AR约1.86×，只是2独立请求的描述。R+D反而比D多耗约10.64%，不能写成R带来额外提速。该轮没有匹配eager运行时消融，不从这两个示例推算norm的服务性能增量，也不重新挑选能让R获胜的示例。

## 测试失败、最小修复与复现

本地初版runtime相关72 passed/11.25秒，完整711 passed/5 skipped/106.19秒；首次完整检查因新工作树CRLF触发旧配置SHA审计而中止，恢复规范Git字节后通过，原失败XML保留。服务器首次84检查中83通过、1失败/8.92秒，尚未启动GPU：Transformers延迟导入替换sys.modules对象，测试顶层旧对象的版本模拟没有作用。单例复现并打印两个模块identity确认原因；仅测试改用import_module获取当前对象，生产guard和执行代码未改。修复本地23 passed/9.36秒，服务器84 passed/8.93秒。原失败ZIP、XML、日志和source manifest保留failed-tests，未覆盖旧目录或声称首次全通过。

另修复验收工具原有生命周期问题：TestClient的lifespan结束会关闭engine，旧脚本之后还复用它。将三路SSE统一移到最后实际HTTP服务的单一lifespan，直接取消/benchmark/profile都在关闭之前执行；不放宽engine.close语义。代码层CPU检查涵盖关闭恢复、忙时拒绝、编译失败释放owner、版本/配置拒绝、profile不兼容及无效验收记录。

冻结52fe5be256a9982d4201c24a92b498a5e9cc0e2f/1453 Git blobs，source SHA59b878abbab371be65d061898812483c3c18aa220a2f575ae69f0d0802d72627。部署采用旧完整canonical ZIP（SHA625d6ae67bdc10afb85e4bfbafac751f61d40d161f9b7a3354a35bb82060be05）加161138字节delta（SHA1f2766979b88b1a7dd452cb1ee84d9e0a5f903a9cbc79a3492da11b0beb9410d），新目录每个目标blob都核验。delta只变测试、进度和状态文档；服务器没有收到本地新完整ZIP，不将它的SHA冒充传输验证。

服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/rmsnorm-runtime-v2，PID12263，2026-10-21 12:47:08–12:49:01 UTC。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_rmsnorm_runtime_pipeline`。必须已有独立验证analysis的admission全部True、RMSNORM_RUNTIME_SOURCE_MANIFEST.json、只读models；新目录和空cache，CPU测试300秒/每推理子进程1800秒，只终止自己创建的进程会话。

原失败root为rmsnorm-runtime/PID9062，冻结96952d3；其完整源码ZIP首次上传不完整被SHA拒绝，分10块重传才启动，失败原因不是研究候选数值变化。新运行前后1453文件SHA相同。成功结果ZIP98910字节/SHA825e38ae8e3ea608f8479d6330df8c02960e2ce530782281a57d8676cc38eb4c；原失败ZIP72760字节/SHAb4b0d6b8fea3f3bdeee486d81508f182e77cc0790a5dbb7dee0312617b11dd25。

后审计1453个冻结Git blob、9个当前核心文件、9批347个既有发布文件及原infra三引用通过；旧结果文件字节不变，旧研究源码在各自不可变提交核验，不要求它们等于新runtime源码。面试main/codex/infra-interview-v0.3/infra-v0.3-interview仍27e36199f8a65acdb9a9c42ccb744ddfa9a9402b。reserve96未生成，本轮也未重跑验证调参。

结果results/rmsnorm-runtime；后审计`python -m results.rmsnorm-runtime.audit_postrun`需要Git历史及artifacts中的两个原ZIP。示例API使用RMSNORM_RUNTIME.md。发布仅小型公开记录、operators、测试及报告，排除模型、DB、原始Chrome trace和编译缓存。
