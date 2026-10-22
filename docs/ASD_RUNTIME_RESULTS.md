# ASD运行时工程验收

原ASD与位置适配ASD均已接入CLI、HTTP和SSE。两配置共32条直接解码、8条完整AR参考、2次独立CLI完成；流式、请求预算隔离、取消恢复和严格benchmark隔离检查通过。

这是已观察四题的工程接入检查，不是新性能研究。历史独立16题的128 token时延降低3.23%、Target验证331→319仍为本方案的已接纳收益；本轮没有重新估计这个百分比。ASD允许改变输出，原450错答继续保留。

执行源码：`353f6756c5d758fc2bbe0e66e337a8c2f25e0efb`；1715个canonical Git文件运行前后SHA一致，原适配器只读SHA不变。

|配置|模型准备秒|适配器合并秒|首次AR32 wall秒|独立CLI冷请求秒|编译图|GPU峰GiB|
|---|---:|---:|---:|---:|---:|---:|
|base|21.239|0.000|4.829|6.673|4|34.903|
|position|21.410|0.451|4.692|6.175|4|34.672|

模型准备是load_engine总耗时，包含适配器合并；合并列是其中子项，不能相加。首次AR和独立CLI均从空编译缓存开始，编译成本在请求内；正式32条请求没有新编译。CLI请求耗时不包括此前模型加载；不能把启动成本当作免费。两配置各自使用单卡，FP32/highest/TF32关闭、eager attention及显式RMSNorm profile。

直接结果的token/text/EOS、输入快照、验证轮数和ASD账本与冻结研究参考相同；strict结果另与本轮AR逐token核对。CLI两条的token/EOS/轮数/例外数也完全相同，但spent分别相差7.62939453125e-6和3.814697265625e-6，原零容差验收因此失败，execution-verification.json及失败日志原样保留。

后处理仅对独立冷CLI的spent累计比较允许1e-5绝对误差，仍要求实际spent<=8，其他字段零容差。未改变Target、ASD决策、预算或生成结果，也未重跑GPU。worker预热后编译4图、CLI冷启动3图，微小差异与FP32编译路径舍入相符，尚未逐算子定位；不能将其写成账本位级一致。详见[数值例外说明](ASD_RUNTIME_NUMERICS.md)。每次请求新建预算，取消后新请求恢复到完整参考。

|配置|HTTP成功请求/4|并发|断连取消|恢复|HTTP失败|
|---|---:|---:|---|---|---:|
|base|4|2|True|True|0|
|position|4|2|True|True|0|

HTTP是有界单worker队列。小检查验证协议和恢复，4请求不用于推断生产吞吐或尾延迟；load观测包含排队与传输。SSE start先标记approximate，complete包含账本与adapter SHA。

使用步骤和HTTP示例见[运行指南](ASD_RUNTIME.md)，固定题目和资源合同见[验收协议](ASD_RUNTIME_PROTOCOL.md)。原始结果位于results/asd-runtime，可用`python -m scripts.write_asd_runtime_report`重建本报告及analysis.json。模型、适配器和hidden不发布，默认strict配置和原面试版保持。
