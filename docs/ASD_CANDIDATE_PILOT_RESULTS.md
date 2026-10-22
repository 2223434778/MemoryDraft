# 阶段4：多候选树和局部后缀修复

FP32/RMSNorm/原生b16，严格greedy、原Target输入，4096上限并遵守自然EOS。题级三次均值相对同批native归一后等权；重复只检查计时方向，开发数据不冒充独立验证。

|范围|方法|时延/native|更快题数|严格且重复稳定|Target调用|Draft调用|验证节点|
|---|---|---:|---:|---|---:|---:|---:|
|overall|native|1.000000|0/12|True|278|278|4448|
|overall|packed-linear|1.008754|0/12|True|278|278|4448|
|overall|tree32|1.018533|3/12|True|265|265|8480|
|overall|tree48|1.188727|0/12|True|264|264|12672|
|overall|repair4|1.176796|1/12|True|380|216|4258|
|overall|repair8|1.178011|0/12|True|379|216|4760|
|gsm8k|native|1.000000|0/2|True|87|87|1392|
|gsm8k|packed-linear|1.012843|0/2|True|87|87|1392|
|gsm8k|tree32|0.965504|2/2|True|78|78|2496|
|gsm8k|tree48|1.146494|0/2|True|77|77|3696|
|gsm8k|repair4|1.213931|0/2|True|116|62|1259|
|gsm8k|repair8|1.190364|0/2|True|113|61|1407|
|humaneval|native|1.000000|0/2|True|26|26|416|
|humaneval|packed-linear|1.008758|0/2|True|26|26|416|
|humaneval|tree32|0.974818|1/2|True|24|24|768|
|humaneval|tree48|1.163217|0/2|True|24|24|1152|
|humaneval|repair4|1.174507|0/2|True|33|18|363|
|humaneval|repair8|1.185537|0/2|True|33|18|412|
|json|native|1.000000|0/2|True|27|27|432|
|json|packed-linear|1.009896|0/2|True|27|27|432|
|json|tree32|1.060076|0/2|True|27|27|864|
|json|tree48|1.265150|0/2|True|27|27|1296|
|json|repair4|1.270559|0/2|True|37|26|469|
|json|repair8|1.276316|0/2|True|37|26|493|
|math500|native|1.000000|0/2|True|127|127|2032|
|math500|packed-linear|1.009218|0/2|True|127|127|2032|
|math500|tree32|1.049904|0/2|True|125|125|4000|
|math500|tree48|1.282215|0/2|True|125|125|6000|
|math500|repair4|1.315760|0/2|True|181|102|2014|
|math500|repair8|1.316768|0/2|True|183|103|2282|
|mbpp|native|1.000000|0/2|True|9|9|144|
|mbpp|packed-linear|1.008850|0/2|True|9|9|144|
|mbpp|tree32|1.046622|0/2|True|9|9|288|
|mbpp|tree48|1.217138|0/2|True|9|9|432|
|mbpp|repair4|1.086027|0/2|True|11|6|121|
|mbpp|repair8|1.093220|0/2|True|11|6|134|
|squad|native|1.000000|0/2|True|2|2|32|
|squad|packed-linear|1.002960|0/2|True|2|2|32|
|squad|tree32|1.014274|0/2|True|2|2|64|
|squad|tree48|1.058146|0/2|True|2|2|96|
|squad|repair4|0.999992|1/2|True|2|2|32|
|squad|repair8|1.005857|0/2|True|2|2|32|

准入扩测的候选及场景：`{}`

全部节点构造/CPU与设备同步/自定义attention mask/分支选择/KV整理/特征提交均计入请求wall time。packed-linear隔离打包路径成本；tree32/48保留native链并加固定前四位置的Top-2单替换分支，每分支含4/8 token；repair4/8只重试一次真实恢复anchor之后的旧后缀。分支局部推进增加不能代替自然请求净收益。未复现完整BASTION或训练ReTrace，不据本轮否定这些方法。
