# Pairformer checkpoint CPU offload 单变量实验（2026-07-16）

## 结论

在 8×H100 80GB、FSDP2、crop 1024、MSA 2048、diffusion batch 32 的
31-step matched benchmark 中，仅关闭 Pairformer checkpoint-input CPU
offload 后：

- step time 从 53.563942 s 降到 52.708598 s，降低 1.5969%；
- token throughput 从 151.972349 增至 154.438523 tokens/s，提升 1.6228%；
- PyTorch max allocated 从 54.9189 GiB 增至 60.9277 GiB，增加 6.0088 GiB；
- measurement-window NVML 峰值从 60,817 MiB 增至 66,945 MiB，增加
  6,128 MiB（5.9844 GiB）。

因此，在这组 H100 80GB 配置上关闭 Pairformer offload 可以稳定训练，但
只获得约 1.6% 的速度提升，代价是约 6 GiB/卡显存。若显存安全余量优先，
继续保留 offload 更稳妥；若配置固定且优先吞吐，可以关闭。

## 唯一变量

| 配置 | 基线 | 实验组 |
|---|---:|---:|
| Pairformer blocks per checkpoint | 2 | 2 |
| Pairformer checkpoint CPU offload every N regions | 2 | disabled (`None`) |
| MSA blocks per checkpoint | 1 | 1 |
| MSA checkpoint CPU offload every N regions | 1 | 1 |
| EMA shadow device | CPU | CPU |

关闭值必须是 `None`，即不向训练命令传入
`--model.pairformer.checkpoint_cpu_offload_every_n`。`0` 和 `false` 都不是合法
关闭值。Pairformer activation checkpoint 仍保持每两层一个 region；本实验只
取消 `save_on_cpu`，没有取消反向重算。

其余配置保持一致：FSDP2 full shard、BF16、`c_z=256`、`c_m=128`、
crop 1024、`max_nuc_tokens=128`、diffusion batch/chunk 32/4、MSA
min/cutoff/max 2048/2048/16384、MSA chunk 128、global seed 42、train sampler
seed 77、10 warmup + 21 measured steps，以及官方 `protenix-v2.pt` raw/EMA
初始化。

## 配对验证

两组结果均为 `passed`，并满足：

- 完整 31 步样本顺序哈希相同：
  `d93a3eb522df15016b360864c520429b49ff9e3a642a2af86ca47277f5e3fe48`；
- 8 个 rank 的 measured workload 哈希逐项相同；
- measured workload 均为 168 samples、170,945 tokens、1,262,150 atoms；
- MSA trace 均为 664 次调用，每次实际采样 2048；
- `7d1a` 均位于 rank 5、zero-based step 30，截断后为 521 tokens
  （393 protein + 128 RNA）；
- 无 CUDA OOM、NaN、non-finite gradient 或 optimizer skip。

## 复现命令

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_msa2048_offload

BENCHMARK_ID=<new-unique-id> \
bash run_msa2048_crop1024_nuc128_db32_31step_7d1a_no_pairformer_offload_fsdp2_benchmark.sh
```

脚本会显式清理可能从登录 shell 继承的 Pairformer offload 环境变量。CUDA
extension/Triton 编译属于启动阶段，不计入 warmup 后的同步训练计时窗口。

## 结果位置

- 基线：
  `/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_offload_crop1024_nuc128_db32_31step_7d1a/20260716_train2_fsdp2_crop1024_msa2048_nuc128_db32_31step_7d1a_r1/summary.md`
- 关闭 Pairformer offload：
  `/mnt/project_abd/klp/proteinx_benchmark/train2_fsdp2_msa2048_msaoffload_no_pairformer_offload_crop1024_nuc128_db32_31step_7d1a/20260716_train2_fsdp2_crop1024_msa2048_nuc128_db32_31step_7d1a_no_pairformer_offload_r1/summary.md`
