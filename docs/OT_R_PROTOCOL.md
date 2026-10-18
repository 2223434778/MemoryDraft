# OT-R: bounded alignment pilot

2026-10-18. Exploratory engineering experiment, not an independent confirmation.
The existing infra interview version is frozen at `27e36199f8a65acdb9a9c42ccb744ddfa9a9402b`
on branch `codex/infra-interview-v0.3`, tag `infra-v0.3-interview`. Local independent
Git/source/ZIP/bundle are in `artifacts/infra-interview-v0.3/`. `main` is not the research branch.

## Hypothesis and mechanism

After local edits, an approximately aligned observed prefix might identify a
long exact continuation that the current six-token suffix lookup misses.
Retain the existing exact route; invoke a bounded fallback only on its misses.
The fallback aligns the last <=32 observed output tokens to <=8 history windows.
Its shortlist uses rare observed token anchors, <=64 occurrences per anchor,
four anchors, and endpoint shifts of +/-2. It never reads the unknown output.

OT uses POT's `sinkhorn` with token-identity cost plus 0.25 relative position cost.
Each side allocates half its mass to a dustbin with cost 0.35 (bin-to-bin zero),
so insertions/deletions need not match. Entropy is 0.06, <=80 iterations;
nonfinite or marginal-residual >0.001 plans fail closed. Score is the fraction
of exact-token transported mass minus the weighted positional displacement.
Threshold 0.65, minimum 8 observed tokens, at most four fallback calls/request.
This is lexical entropic OT with a positional prior, not a semantic embedding
model, strict monotone alignment, the full OPW algorithm, or SpecTr verification.
An equal-budget Levenshtein fallback shares its pool, gate and call cap.
The same numeric threshold is a frozen engineering default, not a calibration
that makes the two scoring scales statistically equivalent.

Output is one original contiguous span. Record/special-token boundaries remain
hard boundaries. `matched` records the actual exact suffix length; an explicit
context qualification flag permits testing shorter anchors without falsifying it.
Target greedy token equality, first-mismatch stopping, input, KV handling,
user/version snapshots, model assets and stopping rules remain fixed.

## First probe and limits

Use the eight already-inspected frozen JSON revision requests and the two
already-inspected infra demo requests. Their answers are scoring references,
never shortlist/ranking input. Offline prefix decisions may inspect all sampled
positions and ignore the four-call online cap; label them isolated decisions,
not online acceptance rates. Oracle enumerates raw spans regardless of context
and is only an opportunity ceiling, never a latency bound.

CPU comparisons: suffix, the shared pool's oracle, edit and OT; record exact
continuous prefix lengths, recovered opportunities and measured lookup costs.
The old eight cases have substantial repeated structure; no claim of natural
workload generalization or history necessity is permitted.

Then run a bounded real-model integration pilot: the same ten requests, four
arms (`DFlash`, `Suffix-R`, `Edit-R`, `OT-R`), two randomized interleaved repeats,
80 timed generations. Refresh all ten AR checks first, check original token,
Target prompt and legal snapshot equality, and abort on any mismatch. No tuning
after viewing pilot results. Warmup/model load/AR checks separate; all online
tokenization, snapshot, cold index, alignment setup/lookup, proposal verification,
KV catch-up, synchronization and decoding count. POT is explicitly imported
during engine load for either aligned arm before timing, like model libraries;
per-request matcher/index construction still counts. No HTTP or profiler.

Report per scenario and aggregate time, R/D rounds, aligned proposals and actual
accepted tokens, alignment CPU time. Fewer requests or repetitions require an
explicit explanation; no cherry-picked successes. No speed or novelty claim
before measurement. If OT has no incremental opportunity or costs dominate,
retain the negative result and avoid expanding the same experiment.

## Sources

- POT 0.9.6.post1, MIT, tag/commit `85113e9a380f5fcf684c50c73c1ff6a164a7366e`:
  https://github.com/PythonOT/POT/tree/85113e9a380f5fcf684c50c73c1ff6a164a7366e
  Solver is imported, no upstream source is vendored.
- Cuturi, Sinkhorn Distances (2013): https://arxiv.org/abs/1306.0895
- Su and Hua, Order-Preserving Wasserstein Distance (2017), positional alignment motivation:
  https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Order-Preserving_Wasserstein_Distance_CVPR_2017_paper.html
- SpecTr (2023), distinct probability-coupling formulation:
  https://arxiv.org/abs/2310.15141
