# 共享公共段：在固定容量下修复交替前缀淘汰

这次最小修复在目标场景有效。同一张卡、同一组交替前缀请求，公共段配置把HTTP命中率从0/20提高到19/20，总耗时157.4184→72.5130秒，低53.94%；容量仍为2GiB。原本已高命中的单前缀流则慢3.55%。因此保留两个显式配置，按模板家族与访问方式选择。

## 实现与固定条件

原两个4K登记前缀长4073/3937 token，其前3934个token完全相同。prepare_shared_prefix从原公开registry直接求最长公共段，换为一个3934-token entry，保留三个短entry，记录父registry SHA、父ID和原长度。新registry SHA f124b3a0791ed5fc2bcb15486925e8898d3f9a0bef2016ac7faf419b2fe6add1。没有读取测试答案或修改请求；模型输入hash保持原值。

在线PublicPrefixCache、Target/Draft、RMSNorm数值profile和ASD接受器均未修改。仍是FP32/eager、ASD B=8/gate=.25/cap=2，缓存2GiB/3 entries，命中复制独立Target KV并拼接选层features，Draft KV及ASD预算每请求新建。新配置configs/runtime-dflash-asd-shared-prefix.json，原整前缀配置、默认入口及独立面试版继续保留。

这是已知公开模板家族的登记合并，在线仍按整条登记前缀保存；尚未实现任意前缀的块/分页存储或自动路由。只对原8个已观察任务的4K输入评估，不作新盲测或其他任务泛化声明。

## 同请求流的新HTTP对照

两卡各四题轮换、客户端并发2，每个窗口20请求、从空缓存起步。shared各卡20off+20cache，whole在新进程各20cache，共120负载请求；另有16次JSON/SSE/断连/恢复请求。GPU0先shared再whole，GPU1先whole再shared；同题始终同卡，两个cache配置独立加载与编译，不使用上一轮计时替代新控制。

| 同卡请求流 | 关闭缓存 | 整前缀whole | 共享段shared | shared相对whole | whole/shared命中 | whole/shared淘汰 |
|---|---|---|---|---|---|---|
|单一4073-token前缀，GPU0|143.7000秒|54.5367秒|56.4706秒|慢3.55%|19/19（各20请求）|0/0|
|3937/4073交替，GPU1|157.3612秒|157.4184秒|72.5130秒|低53.94%|0/19（各20请求）|19/0|

每组输出工作量固定，分别2405/2335 token；所有配置逐请求输出、EOS、输入/快照及汇总接受量一致。JSON与SSE拼接一致，断连取消和恢复均通过，无失败/超时。shared相对同轮off总wall分别低60.70%/53.92%，不是靠缩短答案获得收益。

交替流的平均客户端时延whole15.3747→shared7.1090秒，平均排队7.4929→3.4734秒。整前缀在该流重复入库30184243200字节，shared只入库1482457088字节；这是累计GPU缓存填充量，不能当作显存峰值。减少重复prefill和填充后，串行worker也减少了等待时间。

shared窗口两卡均1次cold+19次hit，无淘汰；直接跨问题检查也均1次cold+3次hit。覆盖两个模板家族所需的缓存表示从整前缀合计2.8111GiB变为公共段1.3806GiB，落入原2GiB限制。原whole实际上一直只保留一个大entry，不能据此声称GPU显存峰值下降了两者存储差额。

单前缀流的代价也明确：whole原本就能命中，shared少缓存了139个公共token，后续请求需多算这段Target后缀。因此shared的HTTP总wall增加约1.93秒。该流可继续用原整前缀配置，不需要统一切换所有场景。

窗口先后已在两卡交错，但shared内部仍先off后cache；闭环并发的实际到达次序可受线程调度影响。两卡题目不同，不跨卡作因果比较，也不主张生产SLA或continuous batching。

## 单请求、数值及质量

shared下8题×DFlash/ASD×自然/128且EOS×off/cold/hit，共96正式生成，每格一次。全部自然EOS、无截断/OOM/正式新编译；严格D逐token匹配原完整AR，off及cache输出均重现旧参考。ASD原7/8、DFlash原6/8正确保持，旧错答未修复。

ASD命中自然题均时延比off为.4230，低57.70%；首token均4.8023→0.3090秒，首token后3.1039→3.1019秒。收益仍主要来自共享Target预填充。cold/off题均比1.0017，首次填充基本持平；cold+1hit / 2off实测成本估算.7167，低28.33%，此估算不是另外执行的到达轨迹。

ASD完整128子集为5题，hit/off题均比.4144；DFlash为6题，比.4290。其余遵守EOS的短请求全部保留，不强制续写。DFlash自然命中题均比.4531；没有增加近似预算或获得新的接受策略增量证据。

输出、结束条件、验证总轮数及ASD汇总量相同，不等于逐轮接受边界或浮点逐bit相同。本轮未保存逐轮trace。ASD spent最大配对绝对差6.29425048828125e-5，实际预算仍≤8，未把比较容差引入接受器。完整数值与质量见[结果表](ASD_SHARED_PREFIX_RESULTS.md)及analysis.json。

## 工程成本与使用

四个进程模型准备20.259–20.858秒，各3个编译图，backend3.256–3.428秒；首4K DFlash16冷wall8.491–9.161秒，编译成本留在warmup内。正式峰43.6739GiB，HTTP峰43.6373GiB，均低于46GiB限额；这些是各自总峰，不能当作严格的配置显存增量。编译/临时测试缓存合计2725400字节。模型加载单列；查找、填充、复制和淘汰计入engine请求时延，具体边界见[GPU前协议](ASD_SHARED_PREFIX_PROTOCOL.md)。

可在已准备模型的环境启动常驻服务：

```bash
python -m memorydraft serve --config configs/runtime-dflash-asd-shared-prefix.json --host 127.0.0.1 --port 8000 --queue-capacity 4 --event-capacity 32
```

单次CLI退出会清空缓存，不产生跨进程命中。只缓存登记公开前缀，用户问题/私有历史/答案不进入entry；其他输入不匹配时正常完整prefill。AR路径仍未接入缓存，不拿缓存ASD与未缓存AR比较来宣称算法优势。服务仍是单模型worker排队。

现阶段接纳共享段配置作为这类交替模板场景的工程优化。后续如果需要同时保住单前缀性能，可研究共享主干加短分支存储，避免重复计算139-token差异部分；这项尚未实现。本轮结果足以说明统一公共段能修复已定位的容量问题，不要求它打败所有配置。

## 来源与复现

GPU实际源ecc44571a6096232be16a1035ab71804f7878551，1902 canonical Git blobs，source SHA2bcebb23b8538579e75c43b8c22695be172493ae65467ddce7ebcb6320d67863；新root asd-shared-prefix，PID1522788于2026-10-23 05:57:32 UTC启动，06:18:48检查全部完成、两卡19MiB/0%。源码前后1902 SHA相同。成功ZIP211506B/SHA9c8ed2c250a56a1cc1e99b7b931f48b68a9b842d5df79af50cdae62790ec9040，40原始成员逐字节核验；实际4454359B delta SHAc14b9e82a94689f5af1b0cc5c778da6078b7a7a0faf5e2ac303a6e06f8e087db，完整本机ZIP SHA9b2e491825ecdad0fa93209bdbb4ccb583e300165c5a70963b2cd064d14ff5ee。

本机unit34 passed/27.49秒、相关60 passed/7.86秒、完整924 passed/7 skipped/137.79秒；报告补充校验后专项7 passed/0.32秒，服务器116 passed/8.97秒。只有既有starlette警告，远端CI状态另查。两环境使用冻结源独立重算，JSON和Markdown逐字节相同；审计1902 Git源、87冻结执行/数据文件、1323旧结果、上一发布69文件及infra三个引用通过，两个历史结果CRLF例外单列。在线缓存/引擎/ASD代码及旧registry/config与上一发布逐字节相同，保留全部旧结果。

```bash
python -m scripts.prepare_shared_prefix
python -m scripts.write_asd_shared_prefix_report --input results/asd-shared-prefix --output artifacts/shared-prefix-report-recheck
python -m results.asd-shared-prefix.inspect_results
python -m results.asd-shared-prefix.audit_postrun
```

完整audit还需artifacts/asd-shared-prefix-results.zip，原件在A6000的MemoryDraft基目录同名文件；独立统计重算只需仓库公开数据。原始结果位于results/asd-shared-prefix，没有上传权重、KV、features或raw trace。原阶段数据池reserve96未生成，没有新训练或seed实验。
