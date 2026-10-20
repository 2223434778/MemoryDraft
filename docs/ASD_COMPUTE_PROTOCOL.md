# 实际计算收益的小成本可行性检查（运行前冻结）

已停止置信度定价/风险筛选的继续优化；本轮不重新引入概率、熵或新语义模型。只检查有限动作集上的事后理想选择空间及廉价现有特征能否预测。开发题结果不能证明新题泛化，oracle不可在线使用。

## 状态与动作

- 固定4道已观察题：math500:1/4/7、gsm8k:21。源为results/asd-confidence的ASD-Utility机制记录。每个实际接受例外的块仅取首个例外，去重块；筛选源位置<512且源旧完整输出在该位置后尚余>128token。这是按已知源长度选开发状态，不按新分支结果选样。
- 每题在上述按位置排序列表中取索引`floor(i*(n-1)/3), i=0..3`，共16个不同源块。包含旧失败题，不移除分支错误或截断。
- 固定原Target输入、Qwen3-8B/DFlash版本、FP32/eager、greedy、非思考、空历史、ASD B8/g.25/M2、Utility价格2。先完整重现各题Utility token/账本。
- 每个状态从原始prompt真实重放到同一个验证块。A保留原Utility接受长度；B在该块首个例外前截断并发出Target recovery。A可能包含多个例外，这是完整当前接受动作的比较。保留实际KV、Target特征与已付验证；两分支之后都严格DFlash，不重prefill、不复用改变token后的旧suffix logits。以前已花regret保留；只扣当前实际接受例外，后续不再花regret。

## 运行与成本口径

- 每动作1次自然EOS生成（总2048上限）用于最终答案评分；另2次固定范围生成，输出上限为`源块generated_before+128`，遵守EOS。对照范围包含本轮多接受的缓存token、恢复/bonus token及后续token，不能只比较从强制token重prefill后的代价。源块前未改变长度/候选空间。
- 4次完整baseline + 16状态×2动作×(1自然+2范围) = **100次生成**。同题同卡、固定种子交错顺序，两个独立GPU进程。
- 计时起点在共同Target验证及当前原策略选择/源状态核对之后、返回当前接受决定之前；包含当前KV/特征提交和实际后续严格计算，结束于生成返回且GPU同步。此前prompt/prefill/重放成本为共同沉没成本，不纳入局部差值。排除当前选择器核查和任何预测器成本，因此是乐观局部计算空间，不能声称线上端到端加速。
- 后续验证轮数不含已经共同支付的当前轮。记录当前动作真实推进长度、原始支出、未来验证轮数、两次局部秒数、输出ID、EOS/自然长度/最终答案评分。等范围双方均达到128token才比较计算成本；提前EOS单列，不强制续写。
- 两次范围输出/动作/轮数必须完全一致；自然分支只要求范围最终裁短块之前一致（greedy严格条件下完整共同前缀应一致）。源矩阵argmax/regret(1e-3)、原接受/预算及共同前缀核查失败，按机制错误停止并保存。

## 冻结分析与继续门槛

1. 主要计算集：两个动作均达到128token，且两个自然分支均按原评分答对。质量过滤仅用于离线共同质量比较，线上不可获得。所有16状态的质量与EOS仍完整报告，不把过滤后子集当请求总体效果。
2. 对照为固定A（当前Utility动作）、固定B（当前严格动作）、Run局部门控（r/Run>.25选B，否则A）。有限二选一oracle按实测后续验证轮选少者，平局A；秒数另报，用两次均值，报告oracle轮数选择是否也改善秒数。两个固定动作取更优者作为自适应必要性对照。各状态独立，不能把它们直接串成一个完整请求的策略或宣称全局上界。
3. 不使用置信度特征。冻结4特征：当前A相对B多推进token、两动作当前regret支出差、首例外Run、此前最多4轮平均推进（没有此前轮则1）。仅依赖当前/过去数据。
4. 固定岭回归alpha=1，预测`未来轮B−未来轮A`。按题留一法，训练仅用其他题双方达到128的状态（训练不按质量筛选），标准化/均值只拟合训练题。训练<3状态时明确fallback选A。正/零预测选A，负预测选B。不扫alpha、不增删特征、不利用本题其他状态或自然答案训练。另报在全部16状态按该选择的自然最终答案质量。
5. 先看oracle：主要集至少8状态且覆盖>=3题；若oracle相对最好固定动作的总未来轮节省<5%，停止继续此小型未来收益预测路线。即使过门槛，预测器也须好于最好固定动作及Run规则、全部16状态选择后的最终正确数不低于固定A且无自然截断，才值得做独立新题验证；否则结论仅为存在oracle空间、当前特征无法利用。这里5%是预先设定的工程投入门槛，不是显著性标准。
6. 必须单列长度、最终答案与截断；两次测量和4题不足以推断显著性。即使门槛通过也不部署预测器，因为尚未计入在线预测开销或检验完整近似策略。

服务器独立目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-compute`。准确命令：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../.venv-dflash/bin/python -u -m scripts.run_asd_compute_probe --config configs/runtime-dflash.json --task-ids math500:1 math500:4 --output results/raw/asd-compute/math-a
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 ../.venv-dflash/bin/python -u -m scripts.run_asd_compute_probe --config configs/runtime-dflash.json --task-ids math500:7 gsm8k:21 --output results/raw/asd-compute/mixed-b
```

冻结commit和源码manifest；前后核验全部源码及旧证据SHA。所有原结果与面试infra保留。
