# 完整文件输出合同：开发失败例小探测

只使用已观察失败的Ruff/Requests首轮，统一在输入末尾重申完整文件、注释和未修改部分。没有新事实/答案或候选参数；所有方法共享新Target输入和新AR参考。原开发结果保留。
这是单一提示诊断，不是独立泛化/性能结论。

| 项目 | 文本一致 | YAML结构相等 |
|---|---|---|
| astral-sh--ruff | False | False |
| psf--requests | False | True |

4次DFlash/Global计时全部逐token对齐各自AR；仅用于检查接入及合同。
完整文本正确0/2；预定两例均正确门槛未通过，停止12项目确认。

## 原文保真诊断

忽略文件末尾换行后，逐行diff及YAML叶子差异保存在fidelity-diagnosis.json，绑定原输出SHA。YAML结构相等检查保留字符串末尾LF，因此文件末尾块标量会因缺最后LF出现值差异；它不能自动解释为业务逻辑错误。

`astral-sh--ruff`：变化行均以#开头=True；YAML叶子差异1项。
- `/jobs/create-issue-on-failure/steps/0/with/script`：仅字符串结尾LF不同。

```diff
--- expected
+++ AR
@@ -42,7 +42,6 @@
       - uses: Swatinem/rust-cache@98c8021b550208e191a6a3145459bfc9fb29c4c0 # v2.8.0
       - name: Build ruff
         # A debug build means the script runs slower once it gets started,
-        # but this is outweighed by the fact that a release build takes *much* longer to compile in CI
         run: cargo build --locked
       - name: Fuzz
         run: |
```

`psf--requests`：变化行均以#开头=True；YAML叶子差异0项。

```diff
--- expected
+++ AR
@@ -1,7 +1,7 @@
 # For most projects, this workflow file will not need changing; you simply need
 # to commit it to your repository.
 #
-# You may wish to alter this file to override the set of languages analyzed,
+# You may wish to alter this workflow to override the set of languages analyzed,
 # or to provide custom queries or build logic.
 name: "CodeQL"
 
```

上述diff记录Target的改写/遗漏；严格验证与AR一致不能等同于满足编辑原文保真合同。没有放宽质量门槛来继续确认，没有改变Target接受条件或按测试结果筛样本。
源码SHA-256：`077a04368b75f50be7a5d8a4ad2c0fd94e15763dfcc8c53e7f9e19dd4fe27be8`。
