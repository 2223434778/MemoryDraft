# 公共前缀缓存：命中收益与容量边界

公共前缀的 Target KV + DFlash 选层特征缓存已实现，并完成单请求和真实 HTTP 检查。该方案适合重复使用公开长模板的常驻服务：4K 输入的 ASD 在命中时自然时延平均低59.10%，一个从空缓存开始的 HTTP 请求流总耗时低62.45%。另一个请求流因2GiB容量下反复淘汰、零命中，没有提速。这个适用范围比“所有长输入都能加速”更准确。

## 做了什么

显式配置 configs/runtime-dflash-asd-prefix-cache.json，沿用原 Qwen3-8B/DFlash、FP32/eager+RMSNorm、ASD B=8/gate=.25/cap=2，没有训练或调整验证规则。只登记公开 train worked examples，最终问题、私有历史和答案不入缓存。未命中完整prefill并复制公共部分；命中克隆独立Target KV、计算问题后缀、拼接选层features。每请求独立Draft KV和ASD预算，LRU 2GiB/3 entries，默认配置及原面试版保持。

使用原8个已观察任务、24个few-shot请求，三长度×两方法×自然/128且EOS×off/cold/hit，共288正式生成。每格一次测量，cold后紧接hit，off在前/后按固定顺序规则平衡；输入/快照/模型不变。全部自然EOS，无截断/OOM/正式新编译。严格DFlash逐token匹配上轮完整AR；当前off输出与上轮相同。

## 收益和来源

| 输入 | ASD命中自然时延降低，题均比 | 首token：off → hit | 首token后：off → hit | 冷填充+一次命中的总成本降低 |
|---|---|---|---|---|
|约512|13.69%|0.4867 → 0.1341秒|2.3471 → 2.3461秒|6.27%|
|约2K|42.53%|1.9899 → 0.1616秒|2.4199 → 2.4153秒|20.77%|
|约4K|59.10%|4.8204 → 0.2021秒|3.1077 → 3.1067秒|28.98%|

主收益集中在首token之前，符合跳过重复Target预填充的机制。首token后的耗时与验证总轮数基本保持；没有新的Draft接受策略增量结论。DFlash也获得命中自然时延约14.37%/41.58%/56.00%的降低，支持这是可组合的推理工程优化。不同算法若作公平横向比较，应给予相同缓存机会；本轮没有实现缓存AR，不拿它与未缓存AR比较来主张ASD算法优势。

cold/off题均比分别约.9994/1.0001/1.0025，首次填充整体基本持平；每格一次不能精确分离毫秒级复制开销。cold+hit / 2off使用实测成本求和，是配对估算，不是单独执行的到达轨迹。完整128子集按同方法三种cache状态共同达到128筛选，ASD三档N6/6/5，命中题均比.8206/.5366/.4008；其余遵守EOS的短请求也完整保留。自然和前缀不是同一个题子集，不能直接相减归因。

## HTTP及容量反例

同一单worker服务，各卡四道4K题轮换、客户端并发2、先20次off再20次启用cache，两阶段均从空缓存起步。计时包含实际冷miss、排队和SSE传输，JSON/SSE输出匹配，断连取消与后续恢复均通过；合计80负载请求，无失败/超时。另有各卡JSON、SSE、断连及恢复请求，不混入80负载样本。

| 请求流 | 命中 | 20请求总wall：off → cache | 平均排队：off → cache |
|---|---|---|---|
|同一4073token公开前缀|19/20|143.6329 → 53.9338秒，低62.45%|6.8259 → 2.5738秒|
|3937/4073token两前缀交替|0/20|157.1194 → 157.2842秒，慢0.10%|7.4789 → 7.4874秒|

第二组发生19次淘汰。两种entry各1483587584/1534836736字节，合计2.8111GiB，大于原定2GiB。当前整前缀LRU只能保留其中一个；访问顺序交替，故每次都重新prefill和入库。这是有明确机制的未命中反例，不应当把第一组95%命中率当所有流量的默认条件。两卡题目不同，不能用卡间时延差作控制；HTTP先off后cache也有时间顺序混杂，未作生产SLA或continuous batching声明。

值得继续的小改进是共享公共段：两个4K前缀的前3934个token完全相同，当前却重复保存；该公共段本身约1.3806GiB。可以先只登记这一共同前缀，用略长的后缀计算换取两种请求都能复用，再考虑真正的块存储/去重。此项尚未实现、未测收益。成熟实现可参考[vLLM按块组织、匹配前缀及LRU的设计](https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/)，同时为DFlash保留对应features；无需先扩ASD门控或引入新的预测模型。

## 一致性及工程范围

缓存三状态的token/text/EOS、输入/快照、验证总轮数及ASD汇总量全部相同。主表/JSON中的“decisions equal”仅指这些GenerationResult汇总字段；本轮未保存逐轮trace，不保证每一轮接受边界相同。48个ASD配对的累计spent非逐bit相同，最大绝对差9.5367431640625e-5；线上预算仍原值8，未引入比较容差到接受器。分段prefill改变浮点舍入是需要保留的数值边界，不能推广为任意模型/输入的bitwise保证。

质量保持原值：ASD三档均7/8；DFlash前两档7/8、4K6/8。原1075等错答没有因缓存修复，不把近似ASD当严格无损算法。跨问题缓存检查中，第一组为1次cold+3次hit；第二组因同样容量压力4次cold，不能声称它验证了跨问题命中。第一组真实服务及CPU测试验证了活动KV不共享和取消恢复。

模型准备20.100/20.378秒，各3个编译图/backend3.402/3.371秒，正式新图0。首DFlash16的冷wall4.249/4.263秒，编译0→2的2.960/2.929秒留在请求内，其余编译发生后续warmup。正式所有cache状态总峰43.7216GiB，低于46GiB预算；此总峰不是精确的off/hit显存增量消融。编译/临时测试缓存合计1918142字节。服务只在同一engine生命周期内保留缓存；每次独立CLI infer均冷启动，使用入口见[运行指南](ASD_RUNTIME.md)。

## 来源、失败记录与复现

GPU实际源 c2903a7ab256f9916d9be05733a53946cae0200c，1852 canonical Git blobs，source SHA4f7ba9bc94e2622cd7cf98de4b21f97a0defcc276d2ac63dcea0ce42b413eaad；新目录asd-prefix-cache-v2，PID1470334于2026-10-23 04:33:26 UTC启动。源码前后1852 SHA相同，成功ZIP196649B/SHA8ff757a282107b933c01c433bdb83b4ee15ea7d4207a03030062ccd9752d06ef，24原始成员逐字节核验，末查两卡19MiB/0%。完整本机ZIP仅在本机；实际传输4125303B delta，SHA490f4dc1f6562a2914b5fdfc66c6fc21f818b66acdcdfc496bc9f33d8a7316f9。

首次源3d53852的逐token词表长度查询启动热点、主动停止记录与90098B失败ZIP均保留于results/asd-prefix-cache-failed-startup；没有warmup/正式生成，不作性能证据。最小修复在初始化取一次词表长度。本机修复前全量906 passed/7 skipped/125.62秒，修复后相关64 passed/12.60秒；服务器修复后99 passed/8.63秒。没有把新增回归冒称已在旧全量运行，远端CI状态另查。

复核命令：

```bash
python -m scripts.write_asd_prefix_cache_report --input results/asd-prefix-cache --output artifacts/prefix-cache-report-recheck
python -m results.asd-prefix-cache.inspect_results
python -m results.asd-prefix-cache.audit_postrun
```

主统计使用冻结c2903a7代码，两环境逐字段相同；本机原JSON为CRLF，保存analysis-local.json，发布analysis.json为服务器LF字节，analysis-server.json保留独立输出。原始生成Markdown在results-generated.md，主表额外澄清汇总字段定义。审计1852 Git源、85执行/数据文件、1283旧结果、上一发布72文件与infra三个受保护引用通过，两个既有结果CRLF例外单列。原始数据、[完整主表](ASD_PREFIX_CACHE_RESULTS.md)和[GPU前协议](ASD_PREFIX_CACHE_PROTOCOL.md)均保留，没有上传权重、KV、features或raw trace。

完整audit_postrun还需要原ZIP位于artifacts/asd-prefix-cache-v2-results.zip；原件保存在A6000的MemoryDraft基目录同名文件。独立重算统计仅需仓库中的公开数据与结果，不需要GPU或私有模型状态。
