# ASD运行时接入验收（GPU前固定）

用户接纳已测数学ASD位置适配收益，并授权把原ASD与位置适配ASD接入CLI/HTTP。本轮是工程集成验收，不重新训练、不补seed、不重复960次性能研究；历史性能及质量数字保留在ASD_ADAPTATION_FOLLOWUP_FINDINGS.md。

## 实现合同

原配置保持严格方法。新configs/runtime-dflash-asd.json显式启用ASD B8/g.25/M2、remaining/longest；runtime-dflash-asd-position.json另加载已经接纳的q/v rank8/alpha16 FP32 adapter SHA42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e。启动时校验本地safetensors SHA/形状/有限值，再复用原AdapterBank合并一次dense权重，之后不在请求内换权重。Target权重/prompt/tokenizer保持，底座仍为已审计DFlash b16、FP32/highest/TF32关闭/eager attention/rmsnorm-inductor。

同一个method ASD在两配置下分别表示原权重/位置适配权重，结果携带draft_adapter_sha256。每个请求新建独立ASDVerifier，明确approximate=true，报告budget/spent/relaxed_tokens/gate/cap；SSE start在tokens前标记近似模式。AR/DFlash/R+DFlash仍使用严格验证；R与ASD本轮不组合。运行期间改变ASD配置或adapter需重新加载engine。取消后下次请求新KV、新预算；不写回生成答案到历史。

原benchmark/check_regression继续严格合同，拒绝近似方法/记录，并核对适配器身份。ASD可使用infer、profile、serve、load-test及本验收runner。API保持既有有界队列、deadline、断连取消；CPU demo只作transport测试，不当性能证据。

## 固定GPU验收

两个新模型进程：GPU0原ASD配置、GPU1位置适配配置。每卡先AR/DFlash/ASD各32 token短warmup，首请求冷编译成本完整保留；每卡四题1196/959/450/885均来自已观察GSM记录，包含局部正例、同输出减轮、已知ASD质量失败及中性例。

每题一条完整4096/EOS AR参考；DFlash/ASD分别完整4096/EOS及128/EOS各一次。两卡合计32配对结果、8 AR参考，不按结果重选题或重复扫参。逐token/text/EOS、prompt/snapshot/profile、验证轮数与上一轮已保存的base/ordinary-position轨迹比较；ASD账本全部字段相同。strict另与新AR核对；450仍保留原质量失败，不通过改答案标记为正确。

每卡直接ASD中断一次并核对恢复后的128/EOS完整结果和新预算。另运行一题32 token严格benchmark，检查近似方法在建输出目录前被拒绝。HTTP在127.0.0.1临时端口运行：一次JSON、DFlash/ASD各一次SSE、接到首个真实token后断开一次，再做恢复请求；最后4请求/并发2的小队列检查。全部流式拼接等于complete及冻结参考，SSE近似标记正确，取消计数增加且worker恢复、失败数0。HTTP开销与队列时间单列，不把排队并发称为连续批处理。

worker结束释放模型后，每卡另起一个真实CLI进程，使用examples/request-asd.json（1196，原Target消息，没有参考答案）执行infer --method ASD。CLI使用独立空编译cache，完整token/策略支出与原轨迹一致，冷编译计入CLI请求wall。两配置和适配器SHA均记录；不把这次小接入测试用于宣称新的3.23%速度结果。

## 资源和证据

沿用服务器已准备的模型和.venv-dflash，不下载模型或修改共享环境。新root为/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-runtime。prepare_asd_runtime_adapter只读旧asd-adaptation-followup/artifacts/followup-adapter/selected.safetensors，校验后复制约2.1MB到新root/artifacts/adapters。旧权重SHA运行前后核对。CPU前置300秒、每GPU worker1200秒、独立CLI300秒；GPU allocated<=46GiB、每进程Inductor<=8图/backend300秒，每配置worker+CLI缓存合计<=4GiB。只结束本runner创建的owned进程。

运行前冻结源码/配置/请求/本协议及验收判定，canonical Git blobs逐SHA部署，运行前后核全部源码，旧main/面试branch/tag保持27e3619。本地先相关再完整回归；CPU HTTP测试需要允许Windows本机通信。前两次受限环境测试在16项后停留并手动中断，不算通过；同组在允许本机通信环境中112 passed/11.95秒，失败/中断说明保留。

入口：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_runtime_pipeline
```

原始results/raw/asd-runtime；公开results/asd-runtime及ASD_RUNTIME_RESULTS.md。发布JSON/JSONL/XML/log及源码SHA，不含adapter、features、模型权重、数据库或大型trace。完整源码及失败ZIP保持独立，不覆盖旧实验目录。主性能表引用原独立16题结果，并明确与本次工程验收分开。
