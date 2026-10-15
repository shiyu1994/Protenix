# DB40 paired-MSA local-attention A/B benchmark

## 目标与比较边界

这个 benchmark 用相同的 FSDP2 训练负载比较两个代码库：

- baseline：`Protenix_train2_fsdp2_crop1024_nuc128_db40`；
- candidate：`Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention`。

候选实现只允许
`protenix/model/modules/primitives.py` 不同。baseline 在 local-attention
rearrange 时构造较大的 dense mask/bias；candidate 直接构造局部索引和局部
mask/bias。模型参数、checkpoint 格式和 attention 的局部窗口语义不变。本实验只回答
训练显存和速度是否改善；它不能代替长程训练质量或最终推理精度评估。

## 锁定负载

两个 profile 的共同设置如下：

| 项目 | 锁定值 |
|---|---|
| 分布式 | 1 节点、8 GPU、FSDP2 full shard |
| 模型 | Protenix-v2，`c_z=256`、`c_m=128` |
| crop | `train_crop_size=1024`、`max_nuc_tokens=128` |
| diffusion | batch 40、chunk 4 |
| MSA featurizer pool | 最多 16,384 行 |
| MSA 模型采样 | 每次 recycle 固定 2,048 行 |
| MSA chunk | 2,048，即一个完整计算块 |
| MSA pairing | species paired；两个 unpaired/shuffle 开关均为 `false` |
| activation checkpoint | Pairformer 每 2 层；Pairformer CPU offload cadence 2；MSA cadence 1 |
| Transition | chunk 128，并开启 chunk checkpoint |
| EMA | CPU EMA，decay 0.999 |
| sampler | seed 77、800 条索引、无放回 |
| raw/EMA 起点 | 同一个旧训练 `999_ema_0.999.pt` |
| optimizer | `load_params_only=true`，新 optimizer/scheduler |
| 学习率 | `3e-6`，warmup 2,000 |

checkpoint 为：

```text
/mnt/project_abd/klp/proteinx_output/protenix_finetune_20260101_weightedpdb_lr1e-5_20260531_071805/checkpoints/999_ema_0.999.pt
SHA256: 9e0df397d9deb18ea46921c678a62afc4a587241aacd01acd27887729e3a0c3e
```

数据由两个仓库共同读取下面的 canonical artifact，避免复制目录中
`summary.json` 的旧绝对路径导致 stale-path 校验失败：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload/data/benchmark_long_protein_context_1024_800_msa2048
```

这里的 crop 1024 是 crop 上限。包含核酸且受 `max_nuc_tokens=128` 约束的样本可能
少于 1024 token，因此 profile 会逐步记录并校验实际 token/atom 组成，而不会伪称
每条样本都恰好为 1024 token。

## 为什么 benchmark 固定 MSA 2048

正式 immune-heavy 训练会在每个 recycle 独立地从
`1..min(实际深度, 2048)` 随机选择真实 MSA 高度。实现 A/B 的首要目标是隔离
local-attention 改动，因此 benchmark 故意把 `min_size` 和 `sample_cutoff` 都锁为
2048：

- baseline 与 candidate 获得相同且最大的 MSA 计算负载；
- recycle 间随机高度不会掩盖 atom-local-attention 的显存和时间差异；
- runtime trace 会验证每次 MSA-module 调用确实采到 2048 行。

固定 2048 是归因 benchmark 的设置，不表示正式训练已经取消随机深度。候选实现
通过后，还应使用 production-random MSA 做单独的真实性回归。

## 两个阶段

| profile | warmup + measured | 阶段显存采集 | 用途 |
|---|---:|---:|---|
| `..._smoke` | 0 + 1 | 开启 | checkpoint、forward、loss、backward、optimizer 生命周期与 OOM 定位 |
| `..._matched` | 10 + 20 | 关闭 | 正式显存和吞吐量比较 |

smoke 开启同步式 phase CUDA memory instrumentation，因此它的 step time 和吞吐量
不可用于性能结论。smoke 覆盖 8 个全局样本，matched 覆盖 240 个全局样本；两者
都不是 800 条数据的完整 epoch。

## AIStudio 执行顺序

每次执行必须使用新的 `BENCHMARK_ID`。为了让硬件签名严格相同，建议在同一个空闲的
8-GPU AIStudio 节点上顺序执行 baseline 和 candidate，不要并行运行，也不要覆盖
`NNODES=1`、`NPROC_PER_NODE=8` 或其他锁定变量。

### 1. Baseline smoke

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40/

BENCHMARK_ID="db40_localattn_baseline_smoke_$(date +%Y%m%d_%H%M%S)" \
MASTER_PORT=29571 \
bash run_db40_paired_localattention_ab_smoke_fsdp2_benchmark.sh
```

### 2. Candidate smoke

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

BENCHMARK_ID="db40_localattn_candidate_smoke_$(date +%Y%m%d_%H%M%S)" \
MASTER_PORT=29572 \
bash run_db40_paired_localattention_ab_smoke_fsdp2_benchmark.sh
```

两个 smoke 的 `summary.json` 都显示 `status=passed` 后，再执行 matched。

### 3. Baseline matched

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40/

BENCHMARK_ID="db40_localattn_baseline_matched_$(date +%Y%m%d_%H%M%S)" \
MASTER_PORT=29573 \
bash run_db40_paired_localattention_ab_matched_fsdp2_benchmark.sh
```

### 4. Candidate matched

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

BENCHMARK_ID="db40_localattn_candidate_matched_$(date +%Y%m%d_%H%M%S)" \
MASTER_PORT=29574 \
bash run_db40_paired_localattention_ab_matched_fsdp2_benchmark.sh
```

单节点运行完成后会自动生成：

```text
/mnt/project_abd/klp/proteinx_benchmark/db40_paired_localattention_ab/<BENCHMARK_ID>/summary.json
/mnt/project_abd/klp/proteinx_benchmark/db40_paired_localattention_ab/<BENCHMARK_ID>/summary.md
```

## 严格 A/B 比较

把下面两个占位符替换为实际 matched ID：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

BASELINE_PROFILE=/mnt/project_abd/klp/proteinx_benchmark/db40_paired_localattention_ab/<BASELINE_MATCHED_ID>
CANDIDATE_PROFILE=/mnt/project_abd/klp/proteinx_benchmark/db40_paired_localattention_ab/<CANDIDATE_MATCHED_ID>

python scripts/benchmark/compare_implementation_benchmarks.py \
  --baseline-profile "${BASELINE_PROFILE}" \
  --candidate-profile "${CANDIDATE_PROFILE}" \
  --experiment local_rearrange_v1
```

smoke 也使用同一个比较器：只需把 `BASELINE_PROFILE` 和 `CANDIDATE_PROFILE`
替换成两次 smoke 的结果目录。smoke comparison 用于并排查看
`phase_cuda_memory`、整步 allocator peak 和 NVML peak；由于同步式阶段采集会扰动
执行时间，结果必须显示 `timing_comparable=false`，禁止据此得出速度结论。正式速度和
吞吐量结论仍只使用 matched comparison。

比较器会从原始 artifacts 重建两份 summary，并且要求：

- 两侧单次运行都通过完整校验；
- 数据、checkpoint、配置、样本顺序、token/atom 数和每个 recycle 的 MSA trace 一致；
- GPU UUID、驱动、显存容量和拓扑一致；
- 除 `protenix/model/modules/primitives.py` 外，benchmark-critical code hash 全部一致；
- baseline/candidate 的该文件 hash 分别为锁定的旧/新实现；
- loss、梯度和 optimizer 没有 non-finite 或 skipped update。

正式结论以 matched 的 `implementation_comparison.json` 和
`implementation_comparison.md` 为准，重点看 PyTorch allocated/reserved peak、NVML
peak、step-time change 和 throughput ratio。若比较器返回 `failed`，先处理 mismatch，
不能只读取相对性能数字。
