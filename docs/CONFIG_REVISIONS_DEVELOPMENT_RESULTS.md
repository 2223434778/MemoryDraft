# 公开CI配置连续修订回放结果

2026-10-18。协议：[CONFIG_REVISIONS_PROTOCOL](CONFIG_REVISIONS_PROTOCOL.md)。

阶段：development；6个公开项目，每项目两轮完整文件修订和一项diff边界，每方法1重复，共90次计时。全部与各自固定Target AR参考逐token一致。
这是基于真实仓库文件/提交构造的编辑API回放；请求和输出合同由实验指定，不是自然用户或部署服务日志。
Target输入合同版本1，同批所有方法共享；不同版本结果不作跨批加速归因。

## 完整文件：全部在线成本

| 方法 | 次数 | 秒 | DFlash/方法 |
|---|---:|---:|---:|
| AR | 12 | 318.002 | 0.191× |
| DFlash | 12 | 60.753 | 1.000× |
| Adapted | 12 | 60.819 | 0.999× |
| Global | 12 | 36.220 | 1.677× |
| Context | 12 | 36.414 | 1.668× |
| Arctic | 12 | 37.306 | 1.629× |

| 对照/Global | 配对总时间比 | 按项目bootstrap95%区间 |
|---|---:|---|
| DFlash/Global | 1.677× | [1.517, 1.779] |
| Adapted/Global | 1.679× | [1.523, 1.780] |
| Context/Global | 1.005× | [0.997, 1.019] |
| Arctic/Global | 1.030× | [1.012, 1.059] |
| AR/Global | 8.780× | [7.144, 10.385] |

Global/Context使用相同算法，仅资源不同；当前输入已含完整旧文件和所需patch。因此Global快于DFlash本身不能证明额外历史有独立贡献，须直接看Context/Global。
Arctic为官方线性后缀候选组件接入同一严格验证器；保留原生动态候选长度，但统一最大长度、准入门槛和资源。没有复现完整vLLM、树验证、在线输出树增长或其论文全部系统配置。

## 逐项目结果

| 项目 | 文件 | DFlash秒 | Global秒 | DFlash/Global | Context/Global | Arctic/Global |
|---|---|---:|---:|---:|---:|---:|
| astral-sh/ruff | `.github/workflows/daily_fuzz.yaml` | 12.924 | 7.741 | 1.669× | 1.000× | 1.030× |
| fastapi/fastapi | `.github/workflows/build-docs.yml` | 20.043 | 11.255 | 1.781× | 1.001× | 1.000× |
| encode/httpx | `.github/workflows/publish.yml` | 3.130 | 2.344 | 1.335× | 1.043× | 1.096× |
| pallets/flask | `.github/workflows/pre-commit.yaml` | 9.956 | 5.256 | 1.894× | 1.027× | 1.054× |
| psf/requests | `.github/workflows/codeql-analysis.yml` | 10.172 | 6.357 | 1.600× | 0.996× | 1.025× |
| Textualize/rich | `.github/workflows/comment.yml` | 4.528 | 3.266 | 1.386× | 0.987× | 1.057× |

## 任务完成与截断

完整文件：原文件修改后文本一致 5/12；YAML映射可解析 10/12；YAML结构相等 5/12。
diff：patch文本一致 3/6；实际应用后得到期望文件 3/6。
参考输出达到预算 0/18；所有计时达到预算 0/90。任务检查在计时外进行，不删除失败样本。
YAML结构相等不保证注释/文本保留；不执行工作流，不声明CI或部署测试通过。所有方法输出相同，速度优化不改善Target任务能力。
原运行评分版本1；本报告版本2。旧版未为无结尾LF的diff补patch终止符，离线修正3项应用检查；原参考/计时未修改，差异保存在quality-v2.json（仅旧版运行）。

未完全符合文本合同的请求：
- `astral-sh--ruff` / full / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': True, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `astral-sh--ruff` / full / turn=1：{'exact_file_or_patch': False, 'yaml_mapping': False, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': 'ParserError'}
- `fastapi--fastapi` / diff / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': None, 'yaml_equivalent': None, 'patch_applies_to_expected': False, 'parse_error': None}
- `pallets--flask` / full / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': True, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `pallets--flask` / diff / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': None, 'yaml_equivalent': None, 'patch_applies_to_expected': False, 'parse_error': None}
- `psf--requests` / full / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': True, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `psf--requests` / full / turn=1：{'exact_file_or_patch': False, 'yaml_mapping': False, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `Textualize--rich` / full / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': True, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `Textualize--rich` / full / turn=1：{'exact_file_or_patch': False, 'yaml_mapping': True, 'yaml_equivalent': False, 'patch_applies_to_expected': None, 'parse_error': None}
- `Textualize--rich` / diff / turn=0：{'exact_file_or_patch': False, 'yaml_mapping': None, 'yaml_equivalent': None, 'patch_applies_to_expected': False, 'parse_error': None}

## diff边界：单列

| 方法 | 秒 | DFlash/方法 |
|---|---:|---:|
| DFlash | 20.838 | 1.000× |
| Global | 12.784 | 1.630× |
| Context | 12.034 | 1.732× |

## 机制及资源来源

| 方法 | R/D轮数 | R接受/提议 | 来源轮数 | 写回秒 |
|---|---:|---:|---|---:|
| Adapted | 0/563 | 0/0 | {} | 0.9257 |
| Global | 173/82 | 4623/5388 | {'working-file': 95, 'context:current-prompt': 18, 'artifact': 60} | 0.9662 |
| Context | 173/84 | 4621/5496 | {'context:current-prompt': 173} | 0.9337 |
| Arctic | 201/83 | 4567/5209 | {'arctic:legal-resources': 201} | 0.9421 |

## 可复现性与边界

固定Qwen3-8B/DFlash、FP32/eager、greedy、Transformers4.57.3、A6000 GPU0。每个方法/重复独立SQLite，首轮库创建、用户文件版本更新、冷索引、检索、所有前向、缓存追赶、解码和最终回答写入均计入。
独立参考构造时间 394.416s；模型加载 15.645s，单列。
开发包含交错正式AR对照；确认阶段AR仅为独立正确性参考，没有正式AR方法或确认相对AR加速比。
历史从真实旧文件及已经完成的同方法回答重建，逐请求审核版本、用户、Target prompt、token、配对、KV前沿与summary。方法间的结果重复不增加独立项目数；项目间文件规模不同，主结果为配对总时间比。
这批工作流维护及统一diff指令不能代表自然语言修改请求、一般部署配置或真实长期用户。给定完整diff本可由git apply直接完成；本回放衡量既定LLM输出路径的性能，不证明该任务必须调用LLM。未测生产精度、并发、TTFT或稳定尾延迟。CPU索引数组载荷与C++树估计的口径不同，不能据它们宣称内存优势。
每轮快照仅包含一个当前文件版本及至多一个过去回答；未验证大规模跨项目长期记忆的容量或索引维护。
源码SHA-256：`c6c269ffce98cbfd2036dca7ccd612892cef52287e12a2a637da2bf91bd0215b`。
冻结数据SHA-256：`944f0f77a41b08126411e8558a7b451e45341a412d2cee0460fb7c96b12f351e`。

复算：`python -m scripts.write_config_revisions_report --source results/config-revisions-development`。
