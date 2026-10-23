# 确认候选的匹配缓存组合

|方法|hit/off时延比|cold/off时延比|候选/ASD off|候选/ASD hit|正确off/cold/hit|输出一致|
|---|---:|---:|---:|---:|---|---|
|base/DFlash|0.452450|1.001977|1.064635|1.149989|{'off': 6, 'cold': 6, 'hit': 6}|True|
|base/ASD|0.419560|0.996397|1.000000|1.000000|{'off': 7, 'cold': 7, 'hit': 7}|True|
|ordinary-position/ASD|0.420299|1.001167|0.995229|1.000379|{'off': 7, 'cold': 7, 'hit': 7}|True|
