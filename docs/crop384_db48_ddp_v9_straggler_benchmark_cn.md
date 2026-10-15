# crop384 / DB48 / DDP v9-straggler 优化与 matched benchmark

## 结论

本仓库是 v8 训练的独立 v9-straggler 实验副本。科学训练配置保持不变，只改变
两个与 128 卡长尾等待直接相关的运行时行为：

1. training DataLoader 使用 `num_workers=8`、
   `persistent_workers=true`、`prefetch_factor=2`、`pin_memory=false`；
2. loss-health 在每个 micro-step 做本地 finite 检查，但跨 128 rank 的
   `MAX all_reduce` 只在每 4 个 micro-step 的 update boundary 执行一次。

`epoch_size` 有意保持 `10240`：每 rank 每个 sampler epoch 仍是 80 个
micro-step/20 个 optimizer update。这样 v8/v9 的 dataset、sampler seed、
global sample positions、模型、loss、优化器和学习率轨迹保持 matched。

本页的 matched benchmark 仍指 PDB-only profile。AFDB + MGnify 蒸馏数据使用独立的
`distill_v1` profile、launcher 和 checkpoint contract；运行方式见
[AFDB + MGnify 蒸馏训练接入](afdb_mgnify_distillation_cluster_index_cn.md)。

## v8 基线证据

基线 run：

```text
/mnt/project_abd/klp/proteinx_output/
protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026
```

首个日志窗口覆盖 25 个 optimizer update，也就是 100 个 micro-step、全局
12,800 个 target：

| 指标 | v8 |
| --- | ---: |
| 窗口 wall time | 1,965.63 s |
| target throughput | 6.512 target/s |
| 折算 update time | 78.63 s/update |
| forward avg | 3.240 s/micro-step |
| backward avg | 3.341 s/micro-step |
| H2D avg | 0.081 s/micro-step |
| micro-step wall P95 | 71.442 s |
| micro-step wall max | 178.444 s |
| peak allocated | 45.93 GB（42.77 GiB） |
| loss avg | 13.1917 |
| window grad norm avg/max | 28.7798 / 45.3270 |
| NaN / nonfinite grad / optimizer skip | 0 / 0 / 0 |

证据文件：

- `logs/train_metrics.jsonl` 的 `global_step=99` 记录；
- `logs/runtime_rank*.jsonl` 的 128-rank 明细；
- `manifest/run_status.json`；
- v8 仓库的 16 个 AIStudio node 日志。

这些数据说明 GPU forward/backward 本身不是主要瓶颈：单次计算通常约 6–8 秒，
但偶发 rank 的数据准备或 barrier 等待把全局 micro-step 拉到 70–178 秒。

## v9 唯一变量

| 项目 | v8 | v9-straggler |
| --- | --- | --- |
| `data.num_dl_workers` | 8 | 8 |
| persistent workers | false | true |
| prefetch factor | PyTorch 默认值 | 2，显式锁定 |
| pin memory | false | false |
| loss-health global collective | 每 micro-step | 每 update boundary |
| loss finite 本地检查 | 每 micro-step | 每 micro-step |
| `epoch_size` | 10240 | 10240 |
| accumulation | 4 | 4 |
| 模型、数据、loss、seed、LR | v8 合同 | 不变 |

persistent workers 会改变跨 epoch 的 worker RNG 生命周期，因此 v8/v9 在首个
20-update epoch 后不承诺 MSA 随机行抽样等 worker 内增强逐 bit 相同。稳定
occurrence gate 和 sampler sample positions 仍保持一致。吞吐 benchmark 应比较
相同 target stream 的统计分布，不应把它描述为逐 tensor 数值复现。

deferred health sync 不改变有效 batch 和 update 决策：窗口中任一 rank 的任一
micro-step 报告非有限，update boundary 的全局 `MAX` 仍使所有 rank 一致跳过该
update。正式 benchmark 前必须通过多卡 NaN 注入测试，覆盖非 boundary
micro-step。

## v9 training contract

正式 contract ID：

```text
crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_dlpersist1_prefetch2_deferredhealth_v9_straggler
```

所有 v9 `RUN_NAME` 必须包含 `v9_straggler`。入口在 `torchrun` 前检查该标记，
防止 v9 输出和当前 v8 run 混在一起。v8 checkpoint 与 v9 contract 不兼容，
不能 strict-resume 到 v9；v9 正式训练从随机初始化重新开始。

## AIStudio 16 节点正式命令

AIStudio 建立 16 个 worker、每个 worker 8 张 GPU。每个 worker 执行完全相同的
命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v9_straggler.node${LOG_NODE_ID}.log" 2>&1
```

不需要额外写 `set -euo pipefail`；锁定入口自身已经启用。不要直接执行
`finetune_demo.sh`，也不要传科学配置覆盖。

v9 checkpoint 的 16 节点 strict full-state resume 使用同一仓库的专用入口。
下面以完成 update 1,000 后保存的 `999.pt`/`999_ema_0.999.pt` 为例；源 run
必须同样是 v9-straggler，v8 源 checkpoint 会被合同校验拒绝：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_resume1000_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<v9_source_run_dir> \
RESUME_STEP=999 \
bash resume_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_resume1000_v9_straggler.node${LOG_NODE_ID}.log" 2>&1
```

## 100 / 250 update matched benchmark

先跑 100-update safety/performance gate。它覆盖 5 个 sampler epoch boundary，
并在 update 25/50/75/100 形成 4 个完整指标窗口：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

PROTENIX_STRAGGLER_BENCHMARK_UPDATES=100 \
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_v9_straggler_bench100 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_v9_straggler_bench100.node${LOG_NODE_ID}.log" 2>&1
```

通过 100-update gate 后再跑 250 updates。它覆盖 12 个完整 epoch boundary 和
第 13 个 epoch 的前 10 updates，能够更稳定地估计 P95/P99 长尾：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

PROTENIX_STRAGGLER_BENCHMARK_UPDATES=250 \
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_v9_straggler_bench250 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_v9_straggler_bench250.node${LOG_NODE_ID}.log" 2>&1
```

benchmark 模式只接受 100 或 250，保持随机初始化，关闭周期 checkpoint 和最终
checkpoint，避免共享存储写入污染速度结果。它不能与 smoke/checkpoint-source
模式同时启用。

## 判定标准

100-update gate 必须同时满足：

- 128 rank 全部完成，未出现 OOM、NCCL timeout、Traceback；
- NaN、nonfinite gradient、optimizer skip 均为 0；
- MSA 三分类和 template gate 比例仍接近 0.2/0.3/0.5 与 0.2；
- `train_dl_persistent_workers=true`、`train_dl_prefetch_factor=2` 出现在解析后
  `config.yaml`；
- loss-health 多卡注入测试证明第 1–3 个累计 micro-step 的异常会在 boundary
  使所有 rank 一致 skip；
- sampler `sample_global_position` 与 v8 对应范围一致。

性能比较同时报告：

- 全窗口与去掉首 25 updates 后的 target/s、s/update；
- `step_wall_seconds` 的 P50/P95/P99/max；
- forward/backward/H2D/data-wait/loss-health-sync 分段；
- epoch boundary 与非 boundary 分组；
- 峰值 allocated/reserved；
- 最慢 rank、对应 PDB ID、MSA/template/token/atom 特征。

不要只比较平均 forward/backward。v9 的目标是压低 128-rank barrier 的长尾，
主要成功指标应是 wall P95/P99、epoch-boundary penalty 和 target throughput。

每 25 update 的 rank-0 `logs/train_metrics.jsonl` 中，
`runtime_window.timings` 会直接给出 `p50/p95/p99/max/max_rank`。其中包括：

- `data_wait_seconds`：主进程等待下一批数据（首批还包含 iterator/worker 启动）；
- `data_process_seconds` 及 bioassembly/crop/MSA/template/feature-label 子阶段；
- `loss_health_sync_seconds`、forward、backward、optimizer 和 EMA；
- `runtime_window.top_outliers`：最慢记录对应的 rank、global step、PDB ID、
  dataset、MSA/template/token/atom 信息。
- `runtime_window.sampler_epoch_boundary_comparison`：sampler 新 epoch 首批与其余
  micro-step 的 data-wait/data-process/end-to-end 对比，可直接验证 worker
  常驻是否消除了每 20 update 的重启尖峰。

兼容字段 `elapsed_seconds` 仍只累计拿到 batch 后的 GPU-step wall time；评估真实
训练吞吐必须使用新增的 `end_to_end_elapsed_seconds` 和
`end_to_end_samples_per_sec`，它们把每个 global micro-step 中最慢 rank 的
DataLoader wait 一并计入。

任务完成或产生首个 25-update 窗口后，可以直接生成只读 Markdown 汇总：

```bash
python scripts/runtime/summarize_v9_train_metrics.py \
  /mnt/project_abd/klp/proteinx_output/<v9_run_dir>/logs/train_metrics.jsonl
```

需要机器可读结果时追加 `--format json`。工具会严格检查 v9 所需字段，按窗口展示
E2E 吞吐和各阶段 P50/P95/P99/max，并把同一慢样本在 data-wait、MSA、
step-wall 等类别中的记录合并。

100-update 通过但结果噪声较大时才进入 250-update；如果出现 correctness failure，
不要用 250-update 延长来“观察是否恢复”。先修复并重新从相同 seed 开始。
