# OT完整矩阵：三个小改动的开发检查

2026-10-19。固定设计后才运行新评分；全部任务和AR答案已在此前观察，是事后开发实验。

## 三个改动及对照

- A：沿用已有TailOT；全局和末8行单位质量贡献各占一半，作为已实现的末端加权对照。
- B：局部顺序正则。先求原OT计划，将每行正确token匹配流归一化；对(i,j)，统计前一行在[j-4,j)及后一行在(j,j+4]的匹配质量，取有效相邻行的平均支持。真实token代价加0.25×(1-支持)，再求一次Sinkhorn。无匹配的相邻行不施加约束。dustbin代价、边际和求解精度不变。仅一次固定点式修正，不是硬单调约束或精确OPW复现。
- C：完整矩阵用于排序/放弃。在强制精确4-token锚点之前的末8行计算正确匹配质量/行边际、正确匹配流的最大分量/总量、相邻行1–4步递增匹配流及dustbin质量。连续性乘两行较小的匹配支持。排除锚点内部自证。
- C的支持度、集中度、连续性均需>=0.5，未匹配质量<=0.5；可靠性启发分数=0.4支持+0.3集中+0.3连续，最终分数=0.5原分数+0.5可靠性，沿用0.65提议阈值。另记录未确认Draft前缀的最弱匹配行，供诊断，不将其当作新token真值或训练标签。没有校准接受概率。
- 完整组合A+B+C同时启用。对照八组：Suffix-R、Guided-Edit、Splice-Edit、Splice-OT、Splice-TailOT(A)、Splice-OrderOT(B)、Splice-PlanOT(C)、Splice-AllOT(A+B+C)。共用既有Splice候选池/去重/top8/最长32/最多4查询和修复后的冷却，候选长度不自适应；本轮不实现按预计收益调用OT。
- 来源：OPW强调局部顺序和传输结构，https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Order-Preserving_Wasserstein_Distance_CVPR_2017_paper.html 。上述词面token/dustbin、有界一轮修正及矩阵特征是本项目实验适配，未复制上游代码，不宣称论文首创或复现其速度。

## 固定范围与门槛

1. CPU离线：从首轮Splice和冷却修复的既有三种Splice路线记录恢复实际调用状态；原合法快照、prompt资源、已确认前缀、实际Draft候选、4查询预算和阻塞状态。先核对旧提议，再对同池逐候选评分。记录所有候选的事后连续接受、特征、各组选择和放弃；答案只在评分之后离线计算，不进入在线匹配器。不保存大型矩阵trace。
2. 不按这些答案调公式/阈值。只要B/C/组合中至少一组在一个实际可评分状态改变选择或放弃，即进入小规模GPU机制检查；若全组路径完全重复则停止GPU扩展并报告。
3. GPU固定8请求：payment-runbook:1、shipment-contract:1、confirm-schema-0:0/1、inventory-query:0/1、parcel-queue:0、copy。覆盖两个负例、旧JSON正例、SQL桥接与无效调用控制；八组各一次，共64次插桩机制回放，严格核对原AR。
4. 若B/C/组合在某请求减少相对原OT或TailOT的总轮，或总轮不增而减少无效提议，则固定同一package/runner再做八请求×八组×两重复=128次未插桩交错开发计时。否则不扩大计时。所有在线成本包括第二次求解、特征计算及放弃成本均计入。
5. FP32/eager、Qwen3-8B/原DFlash、4线程、batch1、原输入/预算/历史/严格验证不变；首次token/prompt/snapshot差异停止。所有已发布旧证据逐文件SHA保护。独立infra不覆盖。
6. 这是已观察任务机制筛查，不是新holdout或接受概率校准。已有记忆以外的当前源文也可检索，不能据此宣称长期历史独特价值。矩阵结构无法证明保留Draft前缀一定正确；分别报告新增接受、误放弃、原Draft回退和总轮。

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.probe_matrix_offline --tokenizer models/qwen3-8b --output artifacts/offline-v1
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_matrix_probe --stage mechanism --offline artifacts/offline-v1 --config configs/runtime-dflash.json --output artifacts/mechanism-v1
# 仅机制门槛通过：
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_matrix_probe --stage timing --offline artifacts/offline-v1 --mechanism artifacts/mechanism-v1 --config configs/runtime-dflash.json --output artifacts/timing-v1
```
