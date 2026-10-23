# 阶段6：长输入数学工作负载（GPU前冻结）

用户授权长上下文单请求检查，暂不训练、不补seed、不迁移serving。原阶段5位置适配收益已接纳；本轮研究新输入条件下的适用边界，不重复要求证明原3.23%。原面试版保持27e3619。

## 数据与固定条件

复用上一轮冻结records中independent的前8题，按输入文件顺序确定：578/959/341/1075/526/1195/1142/467。题目已观察，本轮不是新盲测，不读取模型结果选题。官方train固定3101c7d、原始SHA17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465；完整test SHA3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14。示例按固定SHA排序取48，排除全部官方test及适配训练/校准40题，移除计算器标记，保留公开解题答案。当前目标题的答案不进prompt。

每题约512/2048/4096 token：使用同一示例序列的完整前缀，选最接近目标长度的示例数，容许±25%，不截断示例或填充无意义文本。实际447–542/1929–2000/4061–4156。各方法同条件prompt、snapshot完全一致；跨长度同时改变示例内容/数量，不能声称纯序列长度因果。data/workloads/asd-long-context保存完整请求、示例来源/许可及SHA，服务器实际tokenization逐项核验。它是已观察任务的工作负载，区别于data/research的原独立题选择，旧题集和独立性检查不改。

三臂为base DFlash、base ASD、位置适配ASD。直接复用已验收runtime configs/runtime-dflash-asd{,-position}.json，Qwen3-8B/DFlash b16、FP32/highest/TF32关闭/eager attention/RMSNorm、ASD B8/g.25/cap2/remaining/longest，适配SHA42755672a816b44b727cbc5bcdbbd2b18dd2a19db42e5706f09d540d678b1f6e。Target消息仅按本预定few-shot构造改变，同条件所有策略固定；不改接受规则，不把ASD当无损。

## 资源优先与配对

新部署root /media/a6000/D/ZhaoBinHe/MemoryDraft/asd-long-context，旧模型/权重只读。每卡4道题，各题三档长度；每卡先独立base/position预检，每档各支持方法+AR16短请求，冷加载/编译计入warmup记录。输入长度必须在两卡两配置预检均满足allocated<=46GiB才进入正式；遇OOM仅按预定资源交集排除该长度并保留原因，不按质量/时延筛题。无可用长度则停止，不扩大模型/改精度。

GPU0 base→position、GPU1 position→base，分进程加载、同题同卡比较，并平衡配置执行先后；每进程独立空编译缓存，固定4 CPU线程。预检全部完成后再运行正式。正式同档warmup与AR参考另列，任务内策略/阶段顺序固定种子打乱，该种子仅控制运行顺序，不是训练seed复验。

每输入三臂：自然4096/EOS各2次、128/EOS各1次，三档全准入时共216正式结果；另24完整AR参考。两次计时不充当新增独立题。记录输入/输出token、EOS/截断、TTFT、wall、验证轮数、ASD支出/例外、显存与编译计数。strict逐token比较AR，数值分歧如实标记，不通过改Target或删除问题宣称无损。重复轨迹应一致；不同策略答案质量逐题保留。共同128只比较三臂都实际达到128的题，绝不强制越过EOS。

wall为现有engine.generate的同步请求计时，含snapshot/tokenize、预填充、生成、同步和文本解码；不含离线数据准备、请求文件读取、模型加载、HTTP传输或排队，不能称完整服务端到端延迟。模型准备和冷请求成本另列。TTFT含Target预填充/首token处理及tokenize，wall−TTFT仍含Draft首次长上下文消费，不能称纯decode。正式数据无profiler；完成全部计时后，对shard0首题×三长度×三臂独立回放最多9次128/EOS。复用原profile_summary，新增诊断范围仅标记Draft首次forward，保存聚合target_prefill/draft_first_forward等GPU工作量、CPU等待及输出一致检查，不发布raw trace、不混入时延均值。不把host时间与GPU工作量相加。

每进程时限2400秒、CPU前置300秒，compiler<=8图/backend300秒、每缓存<=4GiB；只回收runner自建owned进程。canonical Git blobs逐SHA部署，运行前后核源与原权重不变。异常/负结果不覆盖旧目录。

## 命令与结果

准备：`python -m scripts.prepare_asd_long_context --source artifacts/asd-adaptation-gsm-train-source.jsonl --test-source artifacts/asd-attribution-gsm-source.jsonl`。

服务器新目录：`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_asd_long_context_pipeline`。

公开results/asd-long-context，重建`python -m scripts.write_asd_long_context_report`。GPU前冻结本协议、输入、执行与分析代码；不上传模型/adapter/hidden/大型trace。先报告各长度质量、自然路径与共同128，再根据瓶颈提出下一步，不为超过某百分比继续扫参，reserve96保持未生成。
