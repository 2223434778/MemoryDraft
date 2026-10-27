# 单次长输出验收

网页上限从1024提高到16384 tokens，网关任务期限从600延长到3600秒。底层请求契约、HTTP生成与计数接口、runtime配置、可视化轮次的上限同步调整，保留整数与越界校验。研究runtime的默认1024及旧冻结版本不改。

27B 原生窗口从12288提高到32768，适配器改从原生 `/props` 获取真实窗口。输入上限仍为8192；输出上限不代表模型会每次写满，也不等于字符数。EOS、用户停止及任务期限仍生效。

`gpu.json` 来自真实Qwen3-8B成本树ASD和27B普通生成的合成长输出；记录实际生成长度、终止原因、流式计数与完整结果是否一致，以及GPU显存峰值，不保存原始大trace。另有8B其他三种模式的短请求恢复检查。这是稳定性检查，不是算法吞吐比较，也不是所有最大输入加最大输出组合的压力承诺。

`gateway.json` 使用阿里云临时SQLite和原网关代码连接真实A6000，覆盖两个模型的全部已部署模式，确认16384及3600确实转发、回答保存原文一致。没有创建生产测试账号或修改用户聊天。临时SQLite随隔离测试自动清理。

相关测试107通过；完整CPU回归1202通过、7跳过。`tests-full.xml` 保留完整回归记录，`acceptance.json` 关联发布源码、ZIP与部署检查。

命令：

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_output_limits.py tests/test_llama_serving.py tests/test_chat.py tests/test_serving.py tests/test_runtime.py tests/test_speculation.py --basetemp artifacts/pytest-long-output -o faulthandler_timeout=45
.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/pytest-long-output-full --junitxml=artifacts/long-output-full.xml -o faulthandler_timeout=60
```

服务检查在实际部署机器执行，环境文件只保存在私有目录：

```bash
PYTHONPATH=. python /path/to/validate_chat_output_limits.py --env /private/inference.env --url http://YOUR_A6000_TAILSCALE_IP:8010 --large-url http://YOUR_A6000_TAILSCALE_IP:8011 --output /private/long-output-check.json
PYTHONPATH=. python /path/to/validate_chat_output_gateway.py --env /private/gateway.env --output /private/long-output-gateway-check.json
```

部署命令和准确结果见 [PROGRESS](../../docs/PROGRESS.md)，设置见 [CHAT_DEPLOYMENT](../../docs/CHAT_DEPLOYMENT.md)。
