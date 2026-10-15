# H100 Worker：每 600 Updates 评测全部 EMA Checkpoint

## 任务结论

本任务是在冻结的 PXM-Legacy 1,093-target evaluation 输入上，对当前训练阶段的
EMA checkpoint 建立一条可持续更新的正式能力曲线。

固定合同如下：

| 项目 | 固定值 |
| --- | --- |
| checkpoint 类型 | 只评 `*_ema_0.999.pt` |
| 评测间隔 | 每 600 个已完成 optimizer updates |
| 正式集合 | `EVAL_SCOPE=full`, `EVAL_SUBSET=combined` |
| 正式 diffusion | `EVAL_MODE=formal`，5 samples × 200 steps |
| target 数 | 1,093 |
| Template | 独立 required cache，cutoff `2021-09-30`，最多 4 个 |
| MSA | 2,271 个 A3M 全量 hash-lock |
| 执行拓扑 | 每个 evaluator 单机单卡 |
| 随机性 | 固定 seed 42 |

这里的“600 steps”明确指 600 个 optimizer updates，不是梯度累计前的
micro-steps。当前训练 `iters_to_accumulate=4`，不要使用
`last_global_step` 决定评测点。

正式代码库：

```text
/mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90
```

当前训练目录：

```text
/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627
```

这里的“推理”指运行 5 个 diffusion samples 并立刻计算、保存 PXM 指标和逐 target
审计记录。当前 `eval_only` 路径不会把 1,093×5 份预测全部导出成 CIF。若后续需要
永久保存每个 sample 的结构文件，应另建 prediction-artifact 任务并先规划存储，
不能在本队列中静默改变输出合同。

唯一允许使用的单-checkpoint 入口：

```text
eval_af3_pxm_legacy_ema_checkpoint.sh
```

不要使用 `run_crop384_db48_ema_eval_queue.sh`。旧队列只接受 5,000-step
间隔，而且调用的是旧 `eval_crop384_db48_ema_checkpoint.sh`，不是本次冻结
MSA/Template 的 PXM evaluator。

## Checkpoint 编号规则

训练每 200 个 updates 保存一对 raw/EMA checkpoint。文件名中的数字是零基
`step`：

```text
完成 update 200  -> 199_ema_0.999.pt
完成 update 600  -> 599_ema_0.999.pt
完成 update 1200 -> 1199_ema_0.999.pt
```

因此选择公式必须是：

```text
completed_updates = filename_step + 1
completed_updates % 600 == 0
```

不能搜索 `600_ema_0.999.pt`，该文件代表的是完成 update 601。

如果训练跑满 50,000 updates：

- 整 600 间隔共有 83 个点：u600、u1200、……、u49800；
- 最后一个整间隔文件是 `49799_ema_0.999.pt`；
- 最终 `49999_ema_0.999.pt` 对应 u50000，不在 600 网格上；
- 为了保留最终模型，正式队列结束时应额外评测 u50000，总计 84 个点。

本阶段是从 v8 step 999 做 params-only warm-start 后重新从 update 0 开始。若要画
完整曲线，可以单独评测以下源 EMA，标记为 `target_v4_u0`，但不要把它伪装成
600-grid checkpoint：

```text
/mnt/project_abd/klp/proteinx_output/protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026/checkpoints/999_ema_0.999.pt
```

## 查看当前待评 checkpoint

checkpoint 先写临时文件，再通过 `os.replace` 原子发布正式 `.pt` 路径。因此队列
应扫描实际存在的正式 EMA 文件，避免训练恰好在文件发布后、账本追加前退出时漏掉
一个有效 checkpoint。`checkpoints/checkpoints.jsonl` 用作交叉审计，不应是唯一
发现来源。

下面的命令只枚举正式 EMA，并按 update 数字排序：

```bash
TRAIN_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627

python - "${TRAIN_RUN_DIR}/checkpoints" <<'PY'
import re
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
records = []
for path in checkpoint_dir.glob("*_ema_0.999.pt"):
    match = re.fullmatch(r"(\d+)_ema_0\.999\.pt", path.name)
    if match is None:
        continue
    step = int(match.group(1))
    update = step + 1
    if update % 600 == 0:
        records.append((update, path))
for update, path in sorted(records):
    print(f"{update}\t{path}")
PY
```

在 2026-07-29 10:14 UTC 的审计快照中，当前已经存在 13 个正式评测点：

```text
u600   599_ema_0.999.pt
u1200  1199_ema_0.999.pt
u1800  1799_ema_0.999.pt
u2400  2399_ema_0.999.pt
u3000  2999_ema_0.999.pt
u3600  3599_ema_0.999.pt
u4200  4199_ema_0.999.pt
u4800  4799_ema_0.999.pt
u5400  5399_ema_0.999.pt
u6000  5999_ema_0.999.pt
u6600  6599_ema_0.999.pt
u7200  7199_ema_0.999.pt
u7800  7799_ema_0.999.pt
```

实际执行时必须动态扫描正式 EMA 文件，并用账本交叉审计，不能把这 13 个点写死，
因为训练仍在继续。

## H100 上线前的强制检查

1. 使用不属于训练任务的独立 H100。
2. `CUDA_VISIBLE_DEVICES` 每个 evaluator 只能暴露一张卡。
3. 该卡原有显存占用默认必须不超过 2,048 MiB。
4. 第一次只启动一个 worker，让 CUDA extension 编译和缓存完成。
5. 不要修改或覆盖 `PXM_*` 路径；正式入口已有正确的冻结输入默认值。
6. MSA 全量预检会重新读取约 1.95 GiB A3M，通常需要数分钟；预检期间 GPU
   可能仍为空闲，这不是卡死。

先检查 H100：

```bash
nvidia-smi
```

然后选当前最新的 600-grid EMA 做 dry-run。下面以审计时的 u7800 为例；如果训练
已经继续，允许换成新的网格点：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
DRY_RUN=1 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7799_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=quick \
RUN_NAME=af3_pxm_preflight_quick_u7800 \
bash eval_af3_pxm_legacy_ema_checkpoint.sh
```

dry-run 必须同时看到：

- Template validator：`status=valid`、1,119 sequences；
- MSA validator：`status=valid`、2,271 files；
- `requested_target_count=12`；
- 最终命令包含 `N_sample=5`、`N_step=20`、PF64/cz256/cm128/DB48。

dry-run 只验证冻结输入和最终参数，不会打开 checkpoint 检查其内部架构与 payload。
PF64/cz256/cm128/DB48 的 checkpoint 合同会在下面第一次非 dry-run 的真实 quick
smoke 中由 validator 校验。

之后依次完成一次真实 quick smoke 和一次 formal smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7799_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=quick \
RUN_NAME=af3_pxm_smoke_quick_u7800 \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_smoke_quick_u7800.log 2>&1

CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/7799_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=formal \
RUN_NAME=af3_pxm_smoke_formal_u7800 \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_smoke_formal_u7800.log 2>&1
```

只有两次 smoke 都成功后，才能开启全量队列。quick 结果只用于排错，不能与
full/formal 正式曲线混合。

## 单个正式 checkpoint 的命令

下面是 u600 的完整示例：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29681 \
EMA_CHECKPOINT=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627/checkpoints/599_ema_0.999.pt \
EVAL_SCOPE=full \
EVAL_SUBSET=combined \
EVAL_MODE=formal \
EVAL_BASE_DIR=/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600 \
RUN_NAME=af3_pxm_full_combined_formal_u600 \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_full_combined_formal_u600.log 2>&1
```

正式输出目录会在 `RUN_NAME` 后自动追加时间戳：

```text
/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600/
  af3_pxm_full_combined_formal_u600_YYYYMMDD_HHMMSS/
```

## 持续队列必须实现的行为

H100 worker 应新建独立脚本：

```text
run_af3_pxm_ema_eval_queue.sh
```

不要修改单-checkpoint evaluator。新队列应借鉴旧 5k queue 的轮询框架，但必须
满足以下合同：

1. 每 60–300 秒扫描一次训练的 `checkpoints/*_ema_0.999.pt`。
2. 文件名必须完整匹配 `[0-9]+_ema_0.999.pt`；可再与
   `checkpoints/checkpoints.jsonl` 交叉审计，但不能只依赖账本发现文件。
3. 从文件名计算 `update=step+1`，只选择 `update % 600 == 0`。
4. 以 update 数字升序处理，不使用字符串排序。
5. state namespace 必须包含：
   `source_run/full/combined/formal/every600`。
6. 每个 update 使用独立 log、attempt 和结果目录。
7. 默认失败后等待 300 秒重试一次；第二次失败后停止整个正式队列，不能留下
   曲线空洞后继续。
8. 只有完整结果复审通过后，才原子写
   `update_<N>.done.json`；目录存在或 evaluator exit code 为 0 都不足以判成功。
9. 重启队列时只跳过已有有效 done marker 的 update。
10. 训练 `run_status.json` 为 `running` 时继续轮询；
    `completed` 时处理完剩余 600-grid 点，并额外处理最终 u50000；
    `failed` 时处理完已经正式发布的待评点后非零退出。

建议状态目录：

```text
/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600/
  queue_state/<source_run>/full_combined_formal_every600/
```

每个 done JSON 至少记录：

```text
completed_updates
checkpoint_path
checkpoint_step
scope
subset
mode
run_dir
log_path
sample_count
finished_at
```

推荐的队列环境：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29681 \
TRAIN_RUN_DIR=/mnt/project_abd/klp/proteinx_output/protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627 \
EVAL_EVERY_UPDATES=600 \
INCLUDE_FINAL=true \
EVAL_SCOPE=full \
EVAL_SUBSET=combined \
EVAL_MODE=formal \
EVAL_BASE_DIR=/mnt/project_abd/klp/protenix_eval_outputs/af3_pxm_every600 \
POLL_SECONDS=60 \
MAX_RETRIES=1 \
RETRY_DELAY_SECONDS=300 \
bash run_af3_pxm_ema_eval_queue.sh \
  > af3_pxm_every600_queue.log 2>&1
```

在队列脚本真正实现并通过 dry-run 前，不要直接执行上面的最后一条命令。

## 每个 checkpoint 的成功判据

从 evaluator log 中提取实际 `run_dir`。runner 会打印：

```text
Using run name: <timestamped-name>, run dir: <absolute-run-dir>, ...
```

必须同时满足：

1. `manifest/run_status.json`：

   ```text
   status == "completed"
   exit_reason == "eval_only_completed"
   ```

2. `logs/eval_metrics.jsonl` 存在，正式记录满足：

   ```text
   sample_count == 1093
   num_batches == 1093
   is_ema == true
   ```

3. `logs/eval_samples_rank0.jsonl` 恰有 1,093 条样本记录。
4. `eval_metrics.jsonl` 最后一条正式记录的 `metrics` 对象中，以下七类都存在，
   且 `top1_pct / best5_pct / top1_best5_pct` 的 `.avg` 值全为 finite：

   ```text
   intra_prot
   prot_prot
   ligand_prot
   intra_dna
   dna_prot
   intra_rna
   rna_prot
   ```

   完整 key 的格式是：

   ```text
   recentPDB_1536_sample384_0925/ema0.999_lddt/af3_proxy/<type>/<metric>.avg
   ```

   例如：

   ```text
   recentPDB_1536_sample384_0925/ema0.999_lddt/af3_proxy/intra_prot/top1_pct.avg
   ```

   evaluator 还会保存 `.median`、`.pct90`、`.min`、`.max`；它们也应执行 finite
   审计，但正式曲线默认读取 `.avg`。
5. 日志中没有以下错误（文本搜索应忽略大小写）：

   ```text
   Traceback
   CUDA out of memory
   NaN/Inf
   template/MSA parser error
   missing target
   random replacement
   ```

日志文本搜索只能作为补充；NaN/Inf 的最终判定必须解析结构化 JSON，并对上述数值
调用 `isfinite`，不能只靠 `grep`。

下面的命令只用于快速查看 lifecycle，不能代替完整指标复审：

```bash
jq '{status, exit_reason, run_name, run_dir}' \
  <actual-run-dir>/manifest/run_status.json

tail -n 1 <actual-run-dir>/logs/eval_metrics.jsonl | jq \
  '{sample_count, num_batches, is_ema, eval_seconds}'

wc -l <actual-run-dir>/logs/eval_samples_rank0.jsonl
```

## 多 GPU 并行

单个 evaluator 强制单卡，但一台 8×H100 可以运行多个彼此独立的 evaluator。
第一次真实 smoke 和第一次 extension 编译必须单进程完成；之后再并行。

并行时必须满足：

- GPU `i` 只暴露 `CUDA_VISIBLE_DEVICES=i`；
- GPU `i` 使用唯一 `MASTER_PORT=$((29681+i))`；
- 每个 update 使用唯一 `RUN_NAME`、log 和 state；
- 调度器必须用原子 claim，防止两个 GPU 领取同一 update；
- 每个 evaluator 默认会开 8 个 DataLoader worker，8 路并发即 64 个 worker；
- 2,271 个 MSA 的全量 hash 预检也会并发读取共享存储，建议先从 2 路并发开始，
  观察 CPU、共享 I/O 和 GPU 利用率后再增加到 4 或 8 路。

不能启动 8 份上面所述的单进程 queue 并共享同一 done 目录；正式多卡版本必须有
中央 dispatcher 或基于 `mkdir`/`flock` 的原子领取机制。

## 性能与优先级

full/formal 的工作量是：

```text
1,093 targets × 5 samples × 200 diffusion steps / checkpoint
```

单 H100 可能跟不上每 600 个训练 updates 产生一个新点，这是允许的：训练和评测
完全解耦，checkpoint 不会因队列积压而丢失。不要为了追进度把正式队列偷偷改成
quick。

若资源有限，优先级建议：

1. 当前最新 600-grid checkpoint 的 quick smoke；
2. 同一 checkpoint 的 formal smoke；
3. 当前最新 checkpoint 的 full/formal，尽快获得当前能力；
4. 再从 u600 开始回填完整 600-grid 曲线；
5. 训练结束后补最终 u50000；
6. 如需完整学习曲线，再单独补 target-v4 u0 warm-start 基线。

## Worker 最终交付

H100 worker 完成后应提供：

- 已成功 updates 列表；
- 尚未生成、待训练发布的 updates 列表；
- 失败及重试记录；
- 每个 update 的实际 run directory；
- 每个 update 的七类正式指标；
- 每个 update 的 `eval_seconds`；
- 明确说明是否包含 target-v4 u0 基线和最终 u50000。

主评测定义、冻结输入与时间切分限制分别见：

- [PXM 主评测方案](af3_testset_evaluation_cn.md)
- [冻结 MSA/Template 输入](af3_pxm_locked_msa_template_inputs_cn.md)
