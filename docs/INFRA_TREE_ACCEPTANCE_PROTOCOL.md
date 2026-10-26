# Infra v0.4 GPU验收协议（2026-10-26，运行前冻结）

- 主线为DFlash + 成本树 + ASD，DSpark不参与。部署新目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/infra-tree-v04`，旧研究目录和infra-v0.3不变；源码canonical Git blob清单执行前后校验。
- 单A6000、Qwen3-8B/审计DFlash原生16块、FP32/eager attention/RMSNorm-inductor/thread4、B8/g.25/M2。调用真实CLI在合成上下文上生成本机成本表及带SHA的运行配置，再加载真实runtime。
- 使用旧extension固定顺序的GSM前2、HumanEval前1、MT前1源对话首轮、Alpaca前2，共6个已观察公开请求；自然EOS/上限128。AR、DFlash、ASD、BudgetTree、BudgetTree-ASD共5方法×2重复=60计时，额外6条AR输入参考。只作部署验收，不称新保留集或新任务准确率。
- 严格方法要求token/prompt/snapshot与AR一致。近似方法显式允许、校验同输入与预算；质量不评且不得标为严格正确。所有方法在线计时包含起草、构树、验证、KV提交及解码，加载、离线校准和预热另列。正式新增编译图必须0。
- 同成本表另以旧研究execute对照严格树和树+ASD，共12次输出、轮数和账本等价检查；研究wrapper仅用于验收，不在production调用中使用。再检查两个树方法取消后重跑结果恢复。
- 同一真实GPUengine通过HTTP JSON和SSE调用组合方法：流token=complete token=非流式token，成本表身份及ASD账本一致；单worker有界队列沿用。服务关闭后model engine关闭。
- 服务器先跑runtime/tree/serving相关测试，owned子进程timeout300秒；GPU验收timeout1200秒。记录模型配置、环境、成本资产、原始计时、等价性、HTTP和完整源码SHA；无权重、KV、完整logits或私有历史发布。

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_runtime_cost_tree.py tests/test_runtime_asd.py tests/test_runtime.py tests/test_serving.py --basetemp artifacts/pytest-infra-tree-regression --junitxml=artifacts/infra-tree-regression-tests.xml -o faulthandler_timeout=30
.venv\Scripts\python.exe -m scripts.package_asd_tree_source --output artifacts/infra-tree-v04-source.zip --manifest INFRA_TREE_SOURCE_MANIFEST.json
```

```bash
cd /media/a6000/D/ZhaoBinHe/MemoryDraft/infra-tree-v04
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_infra_tree_pipeline
```

服务器结果`results/raw/infra-tree-v04`，公开结果`results/infra-tree-v04`。原infra-v0.3保留，新增v0.4标签/独立源码ZIP/Git bundle/本地clone在验收完成后冻结；不覆盖旧面试交付。
