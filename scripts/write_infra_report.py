"""Rebuild the small infra acceptance report without models, torch or a GPU."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/infra-v0.3"
    acceptance = json.loads((directory / "acceptance-v1.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "benchmark-summary.json").read_text(encoding="utf-8"))
    delivery = json.loads((directory / "delivery.json").read_text(encoding="utf-8"))
    methods = summary["methods"]
    lines = ["# MemoryDraft v0.3 工程验收", "", "2026-10-18。用户授权将研究原型整理为推理 infra。", "",
             "本轮交付可安装包、独立依赖 profile、统一生成入口、有界队列 HTTP/SSE 服务、取消/超时/背压、配对 benchmark、性能退化门槛及 CPU/CUDA profiler。H 与原实验保留。", "",
             "## 正确性与交付", "",
             f"- 最终本机完整回归：{delivery['local_tests']['passed']} passed、{delivery['local_tests']['skipped']} skipped，{delivery['local_tests']['seconds']:.2f}s。",
             f"- 最终服务器 DFlash profile 相关检查：{delivery['server_tests']['passed']} passed，{delivery['server_tests']['seconds']:.2f}s。",
             "- 干净虚拟环境离线安装 wheel，未安装 torch/transformers/fastapi，基础 CLI 与 demo 可在仓库外执行。demo 明确不是推理性能。",
             "- A6000 真实 Qwen3-8B + DFlash、FP32/eager：2 请求 × 3 方法 × 2 重复，12 次计时全部与固定 AR 的 token、prompt 和快照一致。",
             "- 三种方法的 HTTP/SSE 各返回16 token，逐 token 与 AR 及 complete 结果相同；真实模型直接取消后再次生成与 AR 相同。",
             "- 断连取消、排队取消、429、慢客户端背压、超时、异常恢复及异常帧资源释放由 CPU/小模型测试验证；不将这些测试改称真实高并发 GPU 验证。", "",
             "## 小规模端到端计时", "",
             "| 方法 | 总秒数 | 相对 AR | 计时观测 / 独立请求 |", "|---|---:|---:|---:|"]
    for method, row in methods.items():
        lines.append(f"| {method} | {row['total_seconds']:.6f} | {row['speedup']:.3f}× | {row['observations']} / {row['independent_requests']} |")
    slowdown = methods["R+DFlash"]["total_seconds"] / methods["DFlash"]["total_seconds"] - 1
    load = acceptance["http_load"]
    lines += ["", f"本轮 R+DFlash 比关闭 R 的 DFlash 适配器慢 {slowdown:.1%}；不宣称检索普遍有净收益。DFlash 名称指同一审计适配循环关闭 R，上游成熟起草器的贡献明确保留。仅两个示例，主要用于工程接入与计时检查，不是自然工作负载研究确认或生产容量承诺。",
              "", "加载与预热不计请求时间；在线快照、tokenization、索引、生成、KV 维护、GPU 同步和解码计入。模型参考生成与计时生成分开；输出不自动写回历史。", "",
              "## HTTP 与 profiler", "",
              f"真实 loopback HTTP 压测：concurrency={load['concurrency']}，{load['successful']}/{load['requests']}成功，wall={load['wall_seconds']:.3f}s，P50={load['p50_seconds']:.3f}s，客户端TTFT P50={load['p50_ttft_seconds']:.3f}s。包含排队与传输，只有6次，P95/P99不输出。服务末尾running/queued均0。",
              "", f"单独插桩捕获 {acceptance['profiler']['events']} 个trace事件，其中 {acceptance['profiler']['cuda_kernel_events']} 个CUDA kernel事件；阶段覆盖：" + "、".join(acceptance["profiler"]["stage_names"]) + "。",
              "", "示例中矩阵乘与Target验证有显著设备成本；检索阶段较短。host时间与CUDA重叠，插桩有开销，本轮不根据单个trace声称某项优化已带来收益。完整trace保留服务器与忽略目录，仅发布小型算子表。", "",
              "## 来源与可复核性", "",
              f"GPU 实际运行 package source SHA：`{acceptance['source']['sha256']}`。",
              "", f"最终交付 package source SHA：`{delivery['runtime_source']['sha256']}`。",
              "", "两者不同：GPU验收后补了性能退化阈值检查、异常traceback清理、profiler设备字段、较小预算预热和demo结束标记。生成后端、引擎与HTTP路由的文件SHA未改变，差异逐文件记录在delivery.json；没有用最终来源覆盖原GPU来源。最终源码已通过服务器相关回归。",
              "", "原部署manifest起初未包含scripts/__init__.py，首次启动因此在模型加载前失败；补齐文件后正式运行。首次错误仅记录在工具输出，发布日志为后续正式运行。运行时实际package源码清单由运行程序重新计算，不依赖该初始归档清单。最终部署清单包含67文件并逐项校验。",
              "", "`delivery.json`的测试计数摘自工具实际stdout，未保存原始测试日志；GPU原始日志、结果行、provenance及小型profile表均发布。发布文件字节SHA见`publication.json`。私有历史、模型、数据库、wheel和大型trace不提交。", "",
              "## 运行与复算", "", "```bash",
              "# A6000: /media/a6000/D/ZhaoBinHe/MemoryDraft/infra-v0.3",
              "OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -m scripts.validate_infra --config configs/runtime-dflash.json --output artifacts/acceptance-NEW",
              "# 只用已发布JSON重建本报告，无需模型/GPU",
              "python -m scripts.write_infra_report", "```", "",
              "## 交付边界", "",
              "单GPU、单执行请求、greedy、FP32/eager。HTTP并发是排队，不是连续批处理。没有BF16/采样/分布式支持。CI配置已提供，实际Actions状态另行查看；Docker CLI在本机不可用，CPU demo Dockerfile尚未执行构建。没有用未完成工作宣称生产就绪。", "",
              "下一项有价值的工程工作应从独立真实负载和profile出发，选择一个瓶颈深入优化。现阶段已经形成可使用和复核的单GPU工具链，但尚未新增CUDA/Triton算子或成熟高吞吐服务引擎移植。", ""]
    (ROOT / "docs/INFRA_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
