# Infra v0.5：高保真反馈树与执行优化

这一版把近期研究候选接入正式运行时，使用原DFlash权重、模型前向和树KV提交，不需要patch研究脚本。旧v0.3/v0.4面试版继续冻结；v0.5单独保存在`codex/infra-interview-v0.5` / `infra-v0.5-interview`，本机独立副本为`artifacts/infra-interview-v0.5/project`，同目录提供ZIP、Git bundle和逐文件SHA清单。

## 实际包含什么

`BudgetTree-ASD`在启用`tree_policy`后使用：

1. 真实Target首选兄弟覆盖、EOS保护、完整词表一步JS/ln2≤0.1，以及当前token的raw regret≤ln4。
2. 请求内最近4轮的覆盖漏选反馈，调整树的旁支惩罚；树节点仍受成本表和32节点上限约束。
3. 近似路径至少比同树严格路径多推进2个token，否则在预算和KV提交前回到严格路径。
4. 复用同次验证的最大值、ID、候选分数及token列表；Target/Draft各自按块扩容KV，attention只读取有效前缀。

这仍是显式近似方法，不承诺与AR完全相同，也不把更接近AR等同更高正确率。研究证据：[最小增量接受](GATE_MIN_GAIN_RESULTS.md)、[输出保持的执行优化](TREE_EXECUTION_RESULTS.md)。后者七任务66输入、198计时，132配对输出/停止/路径/反馈/预算相同，总wall低1.03%，峰allocated增加约0.38GiB。这里的部署验收不重新估计或扩大这些研究结论。

## 从配置启动

沿用[原infra依赖及本地模型准备](INFRA_GUIDE.md)，DFlash使用独立的`transformers==4.57.3`环境。先检查`configs/runtime-dflash-fidelity-base.json`的本地路径与设备。该文件设置最大输入8192、最大输出16384，实际请求上限由请求和模型上下文共同限制，长上下文可能需要更多显存。

```bash
python -m memorydraft calibrate-tree --config configs/runtime-dflash-fidelity-base.json --output artifacts/fidelity-costs.json --runtime-config artifacts/runtime-fidelity.json
python -m memorydraft infer --config artifacts/runtime-fidelity.json --request examples/request.json --method BudgetTree-ASD --output artifacts/fidelity-answer.json
python -m memorydraft serve --config artifacts/runtime-fidelity.json --host 127.0.0.1 --port 8000
```

校准是离线合成步骤，不使用评测答案；生成的配置已填入成本表路径及SHA。新成本资产绑定完整规范化tree_policy、Target/Draft revision、硬件、Torch/Transformers、FP32/eager、RMSNorm profile及线程数；切换策略/缓存设置后需重新校准。空tree_policy的旧配置、旧成本资产仍按原环境规则读取。

`tree_policy`完整示例（基础配置已提供）：

```json
{
  "mode": "min-gain-feedback",
  "max_js": 0.1,
  "max_token_regret": 1.3862943611198906,
  "min_extra_tokens": 2,
  "reuse_scores": true,
  "buffered_kv": true,
  "cache_block_tokens": 256
}
```

同一配置中的`asd`为`{"budget":8,"gate":0.25,"cap":2}`。策略只作用于`BudgetTree-ASD`；`AR`、`DFlash`、`R+DFlash`、线性`ASD`及严格`BudgetTree`保留各自验证方式。省略tree_policy时，BudgetTree-ASD沿用v0.4原策略。接口方法名不变，策略由服务端配置控制，客户端不能按请求覆盖。

buffered_kv目前用于单请求full-attention解码，不能与公共前缀缓存同时启用。可将buffered_kv设false使用原动态缓存，仍需重新校准。KV存储和反馈状态每请求创建，服务仍为单GPU owner、有界队列、背压和轮次边界取消；不宣称实现连续批处理。

## API与观测

`POST /v1/generate`指定`"method":"BudgetTree-ASD"`，按旧协议使用`stream:true`、`visualize:true`即可收到最终提交token和真实投机/树事件。

- `/healthz.tree_policy`与SSE start公开规范化策略。
- JSON/SSE complete的`tree.verification`标明该请求实际使用的策略。
- `asd`保留总预算、支出、JS门控、反馈和最小增量回退摘要。
- `counters.kv_allocations/kv_updates/kv_history_copy_bytes/kv_capacity_bytes`记录缓存工作量，严格方法不冒充启用该优化。

请求取消后释放独立状态，下一请求重新获得完整预算；运行中改变配置会要求重载。JSON/SSE不包含模型权重、KV内容或完整logits。

```bash
python -m memorydraft benchmark --config artifacts/runtime-fidelity.json --requests examples/requests.jsonl --methods AR DFlash BudgetTree BudgetTree-ASD --allow-approximate --repeats 3 --output artifacts/fidelity-benchmark
```

近似benchmark继续标注`quality_evaluated=false`、`strict_correct=null`，不能用于严格回归门槛。严格方法仍核对AR身份。任务准确率、开放回答质量以及容量测试应另行评估；部署小批验收只确认配置/运行时/接口一致。

## 面试副本与复现

```bash
git clone -b codex/infra-interview-v0.5 https://github.com/2223434778/MemoryDraft.git MemoryDraft-infra
```

本机`artifacts/infra-interview-v0.5/START_HERE.md`是离线入口；ZIP和bundle排除环境、权重和私聊。`manifest.json`绑定确切release commit及所有文件字节。可以重点讲述：将近似研究模块接入可配置服务、成本资产与策略绑定、请求隔离、真实流式投机过程、输出保持的缓存优化，以及失败/取消恢复。

A6000上另有独立可运行目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-interview-v0.5`，包含核验代码、`runtime/runtime.json`和`runtime/costs.json`，模型通过共享目录引用。`RELEASE.json`记录运行时源commit和144文件SHA；它是部署副本，本机project/ZIP/bundle是包含Git历史的完整面试副本。

该目录没有启动额外模型，也未切换聊天后端。物理GPU1目前由27B服务使用，运行新8B实例前需释放该卡；独立入口与具体命令见服务器目录的README。当前服务保持运行，部署验收时已临时释放并自动恢复27B。

接入结果见[INFRA_FIDELITY_VALIDATION.md](INFRA_FIDELITY_VALIDATION.md)，执行源码与公开证据位于`results/infra-fidelity-v05/`。旧研究和冻结面试版本不覆盖。
