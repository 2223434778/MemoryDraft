# CI 回放原因分析与处理建议

2026-10-18。基于提交 `7d9ee72` 的已保存输出和结果做离线复核；本轮没有新增模型推理或性能计时。下述新方向是建议，尚未实施，也不是已验证的改进。

**判断：直接复用的生成加速成立；当前 CI 场景不足以支持长期记忆研究主张。工程质量有明确处理办法，但没有证据表明继续调整 H、检索器或路由能挽救这条具体研究路线。建议结束该场景的扩展评测，保留 R+DFlash 实现及全部负结果。**

**证据与原因。**

| 要回答的问题 | 已有证据 | 能支持的判断 |
|---|---|---|
| 解码器是否破坏输出 | 开发 90/90、合同诊断 4/4 与各自固定 AR 的 token 和文本一致；本轮再次核对 | 当前观察到的任务错误来自相应 Target 输出，不能靠严格无损加速器纠正 |
| 是否有真实生成提速 | 完整文件 DFlash 60.753s、Global 36.220s，1.677×，延迟降低约 40.4% | R+DFlash 对这一既定生成路径有净收益；不能等同于成功完成编辑工作流 |
| 额外历史是否有用 | Context 36.414s，Context/Global=1.005，项目区间 [0.997,1.019] | 未发现稳定额外历史收益；当前 prompt 已含完整旧文件 |
| 是否已形成相关方法优势 | Arctic 37.306s，Arctic/Global=1.030 | 只有 6 个开发项目、单重复、官方线性候选组件对照，不能当作击败完整 SuffixDecoding 系统 |
| 原文件编辑是否可靠 | v1 完整文件 5/12 正确；v2 两个已失败开发例 0/2 文本正确 | 输出合同尚不满足。v2 主体恢复后主要剩下注释保真；不能说全部失败都只是注释 |
| 是否必须调用 LLM | 当前输入给出了完整旧文件和统一 diff；本轮直接应用这些输入 patch，12/12 得到期望文件 | 这一特定 API 的应用目标可由确定性工具完成；未计时，不报告工具加速比 |

性能数字来自 [开发报告](CONFIG_REVISIONS_DEVELOPMENT_RESULTS.md)，两例诊断来自 [合同报告](CONFIG_REVISIONS_CONTRACT_PROBE_RESULTS.md)。1.677× 对应约 40.4% 延迟降低，不是降低 67.7%。90 次计时也不是 90 个独立项目。

本轮逐一查看 v1 的 7 个失败完整文件，按现象归纳如下，轮号沿用从 0 开始的记录：

| 请求 | 保存的输出表现 |
|---|---|
| Ruff，轮 0 | 丢失 name/on/env/jobs 等文件头及外层结构，另删一行注释 |
| Ruff，轮 1 | 大量未改内容缺失，剩余片段无法按 YAML 解析 |
| Flask，轮 0 | 丢失 name/on 等文件头 |
| Requests，轮 0 | 丢失 name/on/permissions/jobs 等文件头及外层结构 |
| Requests，轮 1 | 只剩步骤列表等片段；可以解析为列表，却不是所需的完整 YAML 映射 |
| Rich，两轮 | 修改块字符串的缩进、空白或换行，解析后的正文值改变 |

因此，10/12 的 `yaml_mapping` 通过不能表述为 10/12 工作流正确；另外两例也并非都是语法解析异常。参考输出 0/18 达到生成预算，现有证据不支持“增加 max_new_tokens 就能修复遗漏”。这里只核验文本与 YAML，没有执行 GitHub Actions，也不推断工作流运行结果。

v2 对 Ruff/Requests 首轮恢复了主体结构，但 Ruff 仍删除一行注释，Requests 将注释中的 `file` 改成 `workflow`。Ruff 的唯一 YAML 叶子差异是块字符串结尾 LF；它不自动代表业务逻辑错误。v2 是两个已知失败例的单次提示诊断，不是泛化成功率估计。重申要求可以改善特定遗漏，但这两例不支持其可靠满足原文保真合同，也不能据 0/2 推断所有提示都无效。

在固定输入、固定 Target 和严格 greedy 规则下，`output_accelerated = output_AR`。只要任务检查只依赖输出，两者的任务得分就相同。换索引、改候选长度或给 Draft 更好的历史可以改变耗时，无法把 AR 删掉的注释强行保留下来；改变接受规则或强制保留文本会改变这个输出合同。更换 Target、修改提示、约束解码或增加后处理均需建立新的公共基线，不能算原 Target 的无损质量提升。

历史归因也已经有直接对照：Global/Context 使用同一匹配算法，R 分别接受 4623/4621 个 token，Draft 分别运行 82/84 轮。即使 Global 的来源标成 `working-file` 或 `artifact`，同一内容仍可能从当前 prompt 得到；来源标签不能证明不可替代的历史贡献。语义 RAG、增大历史库或更精细的路由都不能自动使重复的信息产生新的价值。

此前 [合成 JSON 确认](REUSE_CONFIRMATION_RESULTS.md) 的 Context/Global=1.312，当前 CI 的 1.005 不构成逻辑矛盾：前者的当前输入提供完整事实，却没有旧回答的完整表达；后者直接包含旧文件。前者说明历史有潜在适用场景，但仅 4 个合成项目，不能替代真实场景证据。若自然应用本来就需要旧文件，不能人为从 Target 输入删掉它以制造记忆优势。

**可行的工程处理。**

1. 对当前“旧文件 + 已知完整 patch”接口，直接校验并应用 patch，再返回修改后的文件。使用现有隔离路径和版本检查，冲突时显式报错；不重新生成未改文本。Git 官方文档说明 `git apply` 可直接应用给定 diff：[git-apply](https://git-scm.com/docs/git-apply)。本轮用现有安全应用检查验证开发集 12 个输入 patch，12/12 正确，仅忽略合同已允许的文件末尾 LF 差异。这个办法解决当前任务，但不是新的投机解码贡献。
2. 如果真正需求是“自然语言要求 → 文件修改”，让模型生成必要的修改操作，再由确定性工具保留未改部分并应用修改，是更合理的应用候选。需要单独验证修改意图、patch 可应用性、改动范围和必要的功能检查；不能把一个正确应用的错误 patch 当成功。它会缩短输出，可能减少 R 的可获益空间；必须与完整文件路径按任务成功和全部成本公平比较。原 CI 的 diff 组只是给定 patch 的复述，不是这一方案的有效评测。
3. 若产品必须让模型生成完整文件，先独立解决 Target 的编辑能力，再冻结所有方法共同的模型、提示及输出规则。仅换更大模型或反复强调“不要漏行”没有本项目的可靠成功证据；更强 Target 也不会自动解除历史与 prompt 的重复。

若问题限定为“既维持当前固定 AR 输出，又修复其原文错误”，没有兼容这些条件的解决办法。硬件升级或训练 Draft 也不能改变这个约束。现有证据不支持把主要问题归因于 A6000 算力或显存不足。

**研究上值得保留和应停止的部分。**

R+DFlash 的严格融合、合法历史快照和正确缓存恢复可以保留；它们已经有工程证据。当前 CI 扩大确认、同场景提示扫描、再次优化 H、在没有瓶颈证据前加入复杂反馈模型，建议继续停止。

H 的旧证据应准确理解：[H 训练结果](H_TRAINING_RESULTS.md) 中，未训练 H 相对 D 有正信号；训练后同权重 D/H 的接受长度是 3.749/3.740，k=4，H 的增量区间跨零，RHD 开关也没有稳定时间收益。它支持暂停当前 H 配方，不支持“历史辅助永远无效”。当前 CI 的质量与资源冗余问题同样不能靠重启 H 解决。

相关文献也限制了我们能够声称的新贡献：

- [REST](https://aclanthology.org/2024.naacl-long.88/) 已用检索文本提供候选；R 的基本可行性并不依赖我们证明。
- [SuffixDecoding](https://arxiv.org/html/2411.04975v3) 已使用当前和过去序列、匹配相关的候选长度与模型 Draft 回退。简单的“历史复用 + 成熟 Draft”不能自动当作新算法。我们的 Arctic 对照仅覆盖候选组件，完整系统差距尚不明确。
- [EfficientEdit](https://arxiv.org/html/2506.02780v1) 已有复用、生成、重新匹配的编辑路线；其生成阶段还采用可能接受非 top-1 token 的动态验证，不能直接把该收益移入本项目严格 greedy 对照。
- [DFlash](https://arxiv.org/abs/2602.06036) 用 Target 特征条件化并行起草。我们的收益必须相对这个已存在的强底座计算，不能把底座自身提速归给历史。

这些是 2026-10-18 核对的论文机制，不是本项目对所有论文的完整复现或全面新颖性审查。

**唯一建议保留的小规模研究候选：真实自然语言材料的重复结构化处理。**

例如同一软件项目每次发布时，将真实发布说明和当前约束转换成固定 schema 的升级检查清单。当前请求给 Target 完整的本次材料、事实和 schema；过去已完成的产物作为额外候选资源。输入材料需确实要求信息抽取、归一化或推理，如果字段已经全部机械可填，应该直接用程序生成，不再人为引入 LLM。

这个场景的假设是：过去产物包含可重复的表达和结构，当前输入足以确定答案，但没有直接包含那些完整输出片段。旧值变化时仍由 Target 严格拒绝，再由 DFlash 生成新接续。它沿用现有架构，不需要先增加 H 或新检索器；也不是保证会成功的新 idea。schema 本身可能已让 Context 足够好，值变化也可能打断所有长接续，公开数据还可能缺少真实连续重复需求。

如继续，只建议先设一次有明确退出条件的可行性检查：

1. 在任何生成前固定一个自然应用合同和公开来源，最多 6 个独立项目、各 3 个时间点，共 18 个请求。不得根据 Target 输出、重复率或速度删换项目；当前事实不能只藏在 Draft 历史中。先保证许可、时间顺序和独立的质量评分可行。
2. 先只运行 Target，检查 JSON/格式、必需事实、遗漏与过期事实；评分不能只看 schema，也不能由同一个模型无监督地自证正确。具体质量门槛在看输出前按应用需求冻结。合法历史仅来自已经完成的过去请求。
3. 离线比较已保存 Target 输出能从当前 prompt 和额外合法历史分别获得多长的精确接续。此处使用当前答案只是事后机会分析，不能用于在线路由或宣称实测加速；匹配机会不足就停止。
4. 质量与独立复用机会均成立，才跑 DFlash、Context、一个同资源的相关检索基线和当前方案。验证完全一致，计入索引维护、预填充、回切、写回及应用新增校验。建议以至少约 10% 的端到端延迟降低作为投入参考，明确这是工程门槛而非统计证明；单独观察额外历史增益和低重复请求损失，再到未见项目确认。
5. 若 Context 已覆盖收益，或同资源 SuffixDecoding 组件效果相当而没有可定位的场景瓶颈，应承认现成方法已经足够。只有观察到具体瓶颈后才考虑例如版本更新或有限容量下的改进，并为该改动设置消融；不为形式上的区别增加机制。

无需胜过所有基线，但需要解释相对最贴近现有方案的具体价值。如果这个小检查仍缺乏独立历史机会，我建议结束 MemoryDraft 的新方法主线，整理为可复现的工程实现和负结果，而非继续轮换数据集或堆模块。当前没有证据支持承诺更好的原创算法一定存在。

**本轮检查与复算。**

本轮在本机现有环境完成以下离线检查；没有改推理代码、旧报告、原始结果或确认门槛，没有启动未运行的 12 项目确认。初次诊断打印遇到 Windows GBK 无法显示公开文本字符，改用 `-X utf8` 后完成；该控制台问题不影响模型输出。

在项目根目录使用 PowerShell：

```powershell
@'
import json
from pathlib import Path
from memorydraft.evaluation.config_revisions import load_projects, quality
from scripts.write_config_revisions_report import DATA, fidelity_differences
from scripts.write_reuse_report import read

projects = load_projects(DATA, 'development')
checks = [quality(item['patch'], p, turn, 'diff')['patch_applies_to_expected']
          for p in projects for turn, item in enumerate(p['turns'])]
print('supplied_patch_direct_apply:', sum(checks), '/', len(checks))
definitions = {p['id']: p for p in projects}
for name in ['development', 'contract-probe']:
    source = Path('results/config-revisions-' + name)
    refs = read(source / 'references.jsonl')
    rows = read(source / 'replay.jsonl')
    labels = {(r['project_id'], r['mode'], r['turn']): r for r in refs}
    aligned = sum(r['token_ids'] == labels[(r['project_id'], r['mode'], r['turn'])]['token_ids']
                  and r['text'] == labels[(r['project_id'], r['mode'], r['turn'])]['text']
                  for r in rows)
    print(name, 'aligned:', aligned, '/', len(rows))
    if name == 'development':
        for r in refs:
            if r['mode'] == 'full' and not r['quality']['exact_file_or_patch']:
                print(json.dumps(fidelity_differences(definitions[r['project_id']], r), ensure_ascii=False))
'@ | .venv\Scripts\python.exe -X utf8 -
git diff --check
```

直接应用检查使用请求中已有的 patch，不用期望答案生成文件；期望答案仅作比较。它在项目忽略的隔离临时目录内操作文本，不执行任何工作流。输出为 `12/12`、`90/90`、`4/4`，并列出原 7 个失败完整文件的差异。该检查没有测量工具延迟，不能加入原性能表或冒充新 GPU 试验。
