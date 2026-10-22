# 阶段3：有限真实计算分配（生成前登记）

目标：检查当前原生b16权重下，请求级/轮级缩短起草、缩短验证或跳过Draft是否有可利用空间。先测成本和有限动作oracle，再决定低开销选择器投入。旧24/32外推、置信度定价及接受/拒绝例外的128步二选一预测保持停止；本轮动作在当前起草/验证之前选择，当前成本也实际支付，与旧共付验证之后的例外选择不同。

## 论文依据与实现边界

- DISCO，PMLR2024 https://proceedings.mlr.press/v262/mamou24a.html ：动态lookahead可优于固定长度，不能从它的自回归Draft设置推断当前block diffusion收益。
- DFlash https://arxiv.org/html/2602.06036v1 ：起草为并行单forward，适中块长下成本不随长度线性增长，因此必须区分实际缩短起草与只缩短验证。复用原MIT adapter和Qwen3/DFlash版本，不新增上游代码。
- BlockPilot https://arxiv.org/html/2606.31315v1 ：研究请求间最优块长差异。本轮只借鉴按请求比较动作的思路，不复现其训练/特征网络，也不以接受长度代替实际wall time。

固定FP32 highest/TF32关闭、eager attention、rmsnorm-inductor执行profile、greedy、空历史和原Target输入。原生Draft16含anchor/15proposal；不扩到24/32、不更换权重。不改ASD接受条件，第一步用严格D隔离真实计算动作成本；其结果不能直接冒充ASD全策略收益。默认CLI/服务和面试引用不变。

## 数据、动作和成本

使用selected_records原始六类开发题各前2，共12题；各类偶/奇位置分两GPU。已观察开发数据，validation48不作新策略独立确认，reserve96不生成。所有题保留，不按正确性/长度选题；短EOS缺失的状态单列。

四动作：native=Draft16/验证15，half=实际Draft8/验证7，verify-half=Draft16/只验证7，target=本轮不调用Draft、Target单步并保留实际hidden features，后续Draft真实追赶KV。Target单步的feature/缓存成本实际保留；pending达到原1024上限时强制native并记录实际动作，不能假装仍零Draft。回调在起草前，只见本轮序号、已确认长度和过去最多4轮；无当前Target验证、未来接受/答案。默认无回调轨迹和旧接口不变。

每题1次native512源运行，逐ID与旧完整自然DFlash/RMSNorm相同前缀、prompt/snapshot核对。取源轮0/4/12，缺失不补位。每状态从原prompt独立真实重放native前缀；在选定轮起草前同步开始clock，执行一次指定动作，之后恢复严格native，最多生成64新token并遵守EOS。保留真实KV/feature，不重prefill改变后的分支；共同重放/prefill及源状态核查是沉没成本，排除在局部clock之外。当前Draft/Target/KV提交、后续严格生成、decode和末尾同步包括在内，是乐观局部诊断而非线上加速。

每状态4动作×3重复，冻结种子allocation:fork:<id>:<round>打乱；输出/轮次每臂重复必须精确。每题另做native/half/verify-half/原生AR四动作，128上限且遵守EOS、各3次，冻结allocation:request:<id>:<repeat>顺序。原生AR不保留Draft追赶特征，与轮内target分开报告。请求engine同步wall含全部生成在线成本；缩短Target不退还已付Draft。12源+144请求+最多432分支=最多588主生成，另每卡5短warmup。实际状态覆盖/计数据实报告。

同范围完整token/prompt/snapshot必须在各动作间相同；提前EOS只要各动作一致仍可比较，不强制续写。数值分歧/不稳定全部保留并阻止相应候选准入，不删样本。512/128/64前缀不能当完整自然答案质量证据，源记录附旧完整质量；若继续候选，必须另评完整自然输出。

## 冻结分析与门槛

各状态/请求每动作三次取均值，用native均值归一。轮级先同请求内状态等权，再请求等权；请求级每题等权。总体与六类分别列四动作全部成本。最优固定动作是同范围四动作中聚合最小者，不只比较native；事后oracle逐状态取最小，只是乐观有限参照，不可串成全局上界。

为防单次抖动选赢家，留一次重复：另两次选每状态动作及全体固定动作，用被留重复评成本，三折合并。整体或任一至少2请求的家族，须同时事后oracle比最好固定省>=5%、跨重复选择省>=3%，才存在值得投入的空间；5/3%为工程门槛，不是统计显著性。

廉价选择器仅1层decision stump；四固定输入generated、prompt_tokens、过去最多4轮平均advance、上一轮advance。阈值仅训练数据相邻值中点，每叶至少3状态；常量动作也参与、相同成本优先简单模型及native顺序。目标为请求等权的实际归一成本。按请求留一，标准化/标签/本题其他状态均不可进入训练；family结果用全体其他请求训练的同一留题模型，不能用本族两个样本各自挑模型。家族/总体在通过上述oracle门槛后，留题选择还须比该范围最好固定动作省>=2%，才运行真实闭环候选。仅离线预测好不能称线上收益。

若无范围同时通过，完成此固定模型/四动作阶段，保存负结果，不扫特征/阈值或加复杂头；不据此否定所有计算分配算法。若通过，冻结所得单层策略，在相应已观察开发请求做实际闭环配对，完整计选择成本；有净收益且严格输出一致后，再扩大完整48开发自然EOS与合适的独立新数据。旧48 validation不能冒充新独立，reserve继续保护。ASD移植另保持原预算并单列质量，不能拿严格局部结果直接宣称近似策略提速。

## 工程与运行

真实tiny Target/KV检查混合动作、AR积累后Draft追赶、EOS/上限、stream、默认native等价、非法配置、fork边界及报告错误输入。先相关再完整本地回归；服务器再跑相关。源码/协议/主统计在GPU前提交冻结并导出canonical Git blobs，前后逐SHA及旧结果审计。独立cache、每卡8图/backend300秒/缓存4GiB/记录allocated46GiB，编译在线成本不扣；worker2400秒，只终止本轮创建进程会话。

独立服务器/media/a6000/D/ZhaoBinHe/MemoryDraft/asd-allocation。入口`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_allocation_pipeline`。probe指定--config configs/runtime-dflash-rmsnorm.json --shard 0或1 --output results/raw/asd-allocation/part-0或1。结果发布results/asd-allocation，报告`python -m scripts.write_asd_allocation_report`；不上传权重、完整logits/hidden、DB或cache。
