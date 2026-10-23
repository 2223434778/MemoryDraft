# 数学ASD运行配置与位置适配

这两个显式配置把已测的数学ASD方案接入同一个CLI/HTTP运行时。适配器在模型启动时合并进原dense Draft，每个请求独立维护ASD预算。已测适用场景为Qwen3-8B/DFlash b16的GSM8K数学生成；CPU demo和本机RTX3090不承担这份8B/FP32配置的性能验证。

|配置|Draft|调用方法|执行profile|
|---|---|---|---|
|configs/runtime-dflash-asd.json|原DFlash|ASD|rmsnorm-inductor|
|configs/runtime-dflash-asd-position.json|位置匹配严格前缀适配后DFlash|ASD|rmsnorm-inductor|

两配置也保留严格AR、DFlash、R+DFlash。ASD使用B=8、gate=.25、cap=2、remaining/longest；请求返回approximate=true，允许输出与Target greedy不同。适配版在已有独立16题上128 token时延降低3.23%、验证调用331→319，未新增答案错误；完整自然7.32%包含错误长路径缩短，质量和事后正确题分析见[实验报告](ASD_ADAPTATION_FOLLOWUP_FINDINGS.md)。这些是已完成研究的结果，本次接入验收另列。

## 准备和CLI

使用已准备的A6000 Linux、Torch2.5.1+cu124、Transformers4.57.3及本地已审计模型。安装接口依赖沿用`python -m pip install -e ".[dflash,serve,client,test]"`；服务加载适配器不需要PEFT训练依赖。模型/adapter不随Git分发，运行时不自动下载。

在项目目录中，从本机已有的已接纳checkpoint准备标准路径：

```bash
python -m scripts.prepare_asd_runtime_adapter --source ../asd-adaptation-followup/artifacts/followup-adapter/selected.safetensors
python -m memorydraft infer --config configs/runtime-dflash-asd-position.json --request examples/request-asd.json --method ASD --output artifacts/asd-answer.json
```

第一条命令校验SHA后准备artifacts/adapters/asd-position.safetensors，重复运行只验证相同文件；已有内容不匹配则报错。若checkpoint位于其他本地目录，用--source指定。原ASD只需将--config改成configs/runtime-dflash-asd.json，不需要适配器。输出文件应使用新路径。

结果包含approximate、asd账本、draft_adapter_sha256、execution_profile、token IDs、文本、EOS/长度原因、请求耗时及TTFT。SHA固定为42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e；加载阶段校验原DFlash资产，再核对适配器dtype、投影形状和有限值。合并准备耗时在engine.describe的draft_adapter.setup_seconds单列，首次编译留在首个生成请求内。

## HTTP与SSE

```bash
python -m memorydraft serve --config configs/runtime-dflash-asd-position.json --host 127.0.0.1 --port 8000
```

另一个终端调用：

```bash
curl -N -H "Content-Type: application/json" --data-binary @examples/request-asd-http.json http://127.0.0.1:8000/v1/generate
```

Windows PowerShell可用curl.exe执行同一组参数。示例请求开启SSE；JSON请求将stream改成false。/healthz列出approximate_methods和draft_adapter_sha256。SSE顺序为start、tokens、complete；ASD的start包含approximate:true，complete附完整账本，tokens拼接必须等于complete.token_ids。ASD接受的token按其近似规则提交。

接口沿用有界单worker队列，timeout包含排队，满队列429，断连/取消在解码边界生效。不同请求不共享KV或ASD剩余预算，取消不会写回部分答案。HTTP并发请求排队执行。更换adapter或ASD策略需要重新加载engine；不支持每请求切换权重或R+ASD组合。

## 验收和架构

```bash
python -m scripts.validate_asd_runtime --config configs/runtime-dflash-asd-position.json --output artifacts/asd-runtime-check
```

该命令使用4道已观察数学题检查冻结轨迹、严格AR、直接流式、取消恢复、HTTP/SSE和小队列；输出目录须新建。完整双配置+独立CLI冷启动验收由scripts.run_asd_runtime_pipeline在全新部署目录执行，资源/题目/判定见[固定协议](ASD_RUNTIME_PROTOCOL.md)。原benchmark和check_regression只接受严格方法；ASD服务吞吐可用load-test采集HTTP观测，结果不自动宣称质量或无损加速。

```mermaid
flowchart LR
    Config[显式配置与权重SHA] --> Load[加载原模型并合并适配器一次]
    Load --> Engine[共享推理引擎与RMSNorm]
    CLI[CLI请求] --> Engine
    HTTP[HTTP / SSE] --> Queue[有界单worker队列]
    Queue --> Engine
    Engine --> Ledger[每请求ASD预算与KV]
    Ledger --> Result[已接受token与策略元数据]
```

默认严格运行配置、原面试版及历史H/D/R/OT实现继续保留。本配置是已测数学近似解码的显式使用入口。

已完成的[工程验收](ASD_RUNTIME_RESULTS.md)保留两配置32直接结果、8 AR和2次CLI，以及真实HTTP/SSE/取消恢复证据。冷CLI的累计spent存在小于1e-5的数值差异，原零容差失败与后处理比较修正见[数值说明](ASD_RUNTIME_NUMERICS.md)；这不改变实际预算或解码规则。未来pipeline只对独立CLI账本比较使用该容差，普通直接/HTTP仍精确比较。

## 长输入示例

约4K token的公开多示例数学请求保存为examples/request-asd-long.json，当前题的参考答案不在输入中。在已准备模型/适配器的全新输出路径执行：

```bash
python -m memorydraft infer --config configs/runtime-dflash-asd-position.json --request examples/request-asd-long.json --method ASD --output artifacts/asd-long-answer.json
```

同输入三方法的[长输入评估](ASD_LONG_CONTEXT_FINDINGS.md)已完成，正式峰约42.29GiB，使用48GiB A6000。该轮是模型请求基准，未测长输入HTTP排队或连续批处理；原短输入3.23%收益与长输入边界分别报告。

## 显式公共前缀缓存

```bash
python -m memorydraft serve --config configs/runtime-dflash-asd-prefix-cache.json --host 127.0.0.1 --port 8000 --queue-capacity 4 --event-capacity 32
```

该配置使用原Draft和原ASD策略，不加载位置适配器；同一服务可调用DFlash或ASD。使用已准备的本地模型，测试环境是48GiB A6000/FP32。示例请求仍是examples/request-asd-long.json；发送HTTP时加method:"ASD"、stream:true等服务字段，沿用前文API。

缓存只存在于同一个engine/服务进程中，重启清空；单次CLI infer退出后不会留下下一次可用的GPU缓存。默认运行配置未启用缓存。当前登记表只覆盖公开GSM train worked examples，不缓存最终问题、私有用户记录或生成答案。其他prompt精确前缀不匹配时正常完整prefill，AR路径保持完整计算。

缓存保存Target KV和DFlash需要的选层Target features。命中时复制独立KV并计算问题后缀，Draft KV和ASD预算每请求新建；活动请求KV不共享。取消时释放活动状态，已完整登记的公共输入缓存可以继续复用。registry SHA、模型revision、tokenizer/template和选层布局检查失败会拒绝启动或要求reload。

上限2GiB/3 entries，按LRU淘汰；当前以整个登记前缀为entry，不做相同前半段去重。首次完整prefill、存入缓存、命中复制、查找和淘汰均在engine.generate计时内。返回counters中的prefix_cache_hit、prefix_cache_prefix_tokens、prefix_cache_reused_tokens、prefix_cache_filled_bytes、prefix_cache_evictions、prefix_cache_bytes和prefix_cache_entries可用于解释请求成本。

固定请求/数值profile下已执行配对检查，见[缓存结果与边界](ASD_PREFIX_CACHE_FINDINGS.md)。分段prefill仍可能带来浮点舍入差异；当前检查覆盖输出、验证总轮数与ASD汇总账本，不是所有模型/输入的逐bit或逐轮一致性保证。服务仍是单模型worker串行排队，未实现continuous batching。

## 共享公共段配置

```bash
python -m memorydraft serve --config configs/runtime-dflash-asd-shared-prefix.json --host 127.0.0.1 --port 8000 --queue-capacity 4 --event-capacity 32
```

该配置保留原模型/ASD策略/2GiB容量，将两个已登记的4K公开前缀合并为同一个3934-token公共段，其他三个短前缀保持。在线仍使用同一个PublicPrefixCache；这是登记边界调整，尚未实现通用块存储。新registry从原公开token求公共段，保留父SHA/ID/长度，重建命令为`python -m scripts.prepare_shared_prefix`。

[共享段评估](ASD_SHARED_PREFIX_FINDINGS.md)中，交替模板流命中0/20→19/20，相对新整前缀控制总wall低53.94%；原本单一前缀高命中流则慢3.55%。频繁交替这两个模板时可用shared配置；只重复同一个长前缀时可继续使用runtime-dflash-asd-prefix-cache.json。两个配置都需常驻engine才能跨请求命中，默认缓存入口未改变。换模板需重新登记合法公开前缀并固定SHA，本轮没有验证任意新模板或其他模型。
