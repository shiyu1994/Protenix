# Soft cluster anti-repeat（gamma=0.2）与 checkpoint 历史迁移

## 当前状态

本仓库是从 `Protenix_train2_templatecache_top60_uniform20_id90` 复制出的
独立实验副本。源仓库不受影响。本轮只实现、生成和验证离线采样资产，**不会启动
训练**。

## 采样定义

PDB、Short、Long、disordered 四个 child dataset 的顺序和 raw outer masses
保持原值：

```text
0.49 / 0.005 / 0.505 / 0.02
```

PyTorch 会按这些 raw masses 的总和归一化。PDB 和 disordered 仍使用原来的
weighted sampling with replacement。新策略只替换 runtime-index-backed 的
Short/Long cluster 选择。

对某个 Short 或 Long cluster `i`，令 `c_i` 为在迁移边界前后累计被选择的次数，
则下一次选择该 cluster 的未归一化权重为：

```text
w_i = gamma ** c_i, gamma = 0.2
```

因此曝光 0、1、2、3 次的 cluster 权重依次为 `1、0.2、0.04、0.008`。
被访问过的 cluster 不会被永久排除；每次被选中后只把 `c_i` 加一。实现按 exposure
bucket 计算 `N_c * gamma**c`，再在 bucket 内均匀选 cluster，避免为三千万个
cluster 每次重建完整权重。

同一 cluster 有多个 member 时，使用稳定随机起点后轮转 member；一个轮次内会先
覆盖所有 member，再回到起点。单 member cluster 自然只能重复该 member。

## 为什么训练时读取只读 schedule

训练不会在 128 个 rank 中各维护一份三千万 cluster 的可变 exposure 数组。否则：

- checkpoint 的 `all_gather_object` 会收集 128 份大状态；
- DataLoader worker prefetch 会使“sampler 已生成状态”领先于“trainer 已消费游标”；
- 中途保存很难保证恢复后与未中断流一致。

离线工具先生成不可变的 `global_rows.u32`。它按同步 microstep 的
`rank 0..world_size-1` 顺序存储最终 flattened dataset row。训练时每个 rank 只
mmap 此文件，并用自己的已消费 batch 游标定位：

```text
schedule_index =
    (absolute_consumed_batches - origin_consumed_batches) * world_size + rank
```

checkpoint 中仍只保存原有的小型 sampler epoch/offset/consumed-batches 状态和
schedule 指纹，不保存 exposure 大数组。

## 指定 checkpoint 作为历史边界

历史边界不写死为某个 step，也不由目录名或 wall clock 猜测。离线工具要求显式
提供：

- source run directory；
- source raw checkpoint；
- 未来绝对 target update；
- 必要时，发生过 sampler-policy 变更的更早阶段 coverage。

工具以 raw checkpoint 内的字段为权威，并验证：

- raw checkpoint 是 raw 而不是 EMA；
- `completed_updates = checkpoint.step + 1`；
- `global_step`、gradient accumulation 和每个 rank 的
  `consumed_global_batches` 一致；
- source run config、world size、epoch size、sampler seed、四库 masses、
  runtime-index 指纹和 checkpoint 中保存的 sampler signature 一致；
- raw/EMA pair id、source training/data/model contract 与 manifest 记录一致。

生成资产会绑定 source raw checkpoint 的 realpath、SHA-256、step、pair id、run id
和 data contract。换一个 checkpoint 时必须重新生成相应资产；过期资产不能用于
较新的 checkpoint。

这里的历史 exposure 是由 seed 和不可变 index 重放出的 **planned sampler
rows**。DataLoader 在坏样本时发生的 worker 内随机替换无法仅从 seed 还原。当前
Short/Long 配置不进行这种随机替换，因此这不会改变两份 soft-cluster exposure；
其他 child 的 replacement 也不参与 `c_i`。

若从 step 0 起 sampler signature 始终相同，即使训练跨越多个 strict-resume run
directory，也只应按绝对游标重放一次，不能重复合并旧前缀。若曾在非零 step 改过
sampler/data catalog，必须显式给出阶段起点和前一阶段 coverage；工具应 fail
closed，不会自动猜 lineage。

## 资产与恢复语义

发布目录至少包含：

```text
manifest.json
_SUCCESS.json
global_rows.u32
<Short>.initial_exposure.u32
<Short>.final_exposure.u32
<Long>.initial_exposure.u32
<Long>.final_exposure.u32
```

`manifest.json` 记录算法版本、`gamma=0.2`、member policy、source checkpoint
边界、拓扑、accumulation、dataset offsets/masses、runtime-index fingerprints、
schedule span 和每个资产的 SHA-256。`_SUCCESS.json` 只在所有文件写完并复核后原子
发布。

第一次从旧 IID sampler checkpoint 切换属于一次显式、单向的 data-policy
migration：model、EMA、optimizer、scheduler、GradScaler、step 和各 rank RNG
全部恢复，但 sampler policy 变为本 schedule。loader 只在 manifest 中保存的
source signature 和 origin cursor 完全匹配时接受这一次迁移。迁移后产生的
checkpoint 使用新 schedule signature，可按同一合同普通 strict resume。

## 选择任意 checkpoint 构建历史

下面的 `SOURCE_STEP` 只是使用者未来要选择的例子，不是代码中的固定值。raw
checkpoint 自己保存的 `step`、`next_global_step` 和各 rank dataloader cursor 才是
权威信息；命令给出的文件与这些字段不一致会直接失败。

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

SOURCE_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run_name>
SOURCE_STEP=<selected_zero_based_checkpoint_step>
SOURCE_RAW_CHECKPOINT="${SOURCE_RUN_DIR}/checkpoints/${SOURCE_STEP}.pt"
SOURCE_EMA_CHECKPOINT="${SOURCE_RUN_DIR}/checkpoints/${SOURCE_STEP}_ema_0.999.pt"
TARGET_UPDATES=50000
SCHEDULE_DIR=/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/<source_run_name>.step${SOURCE_STEP}.to${TARGET_UPDATES}.g02

/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/prepare_soft_cluster_schedule.py \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-checkpoint "${SOURCE_RAW_CHECKPOINT}" \
  --source-ema-checkpoint "${SOURCE_EMA_CHECKPOINT}" \
  --target-updates "${TARGET_UPDATES}" \
  --gamma 0.2 \
  --output-dir "${SCHEDULE_DIR}"
```

例如 checkpoint 文件内 `step=4799`、`accumulation=4`，工具会推导：

```text
completed_updates = 4800
origin_consumed_batches_per_rank = 4800 * 4 = 19200
```

未来 schedule 从每个 rank 的第 19,200 个已消费 batch 之后开始，绝不会把
checkpoint 所在 update 重复计算。换成 `step=6999` 时边界自然变成
`7000 * 4`；无需修改代码。

正常 strict-resume 链且 sampler/catalog 从 update 0 起没有变化时，不要传
`--history-start-update`。只有已知曾在某个非零 update 改过 sampler/catalog 时，
才使用：

```bash
  --history-start-update <stage_first_completed_update> \
  --initial-coverage /path/to/previous_stage_coverage
```

`history-start-update > 0` 却没有可验证的 prior coverage 会被拒绝，工具不会猜测
历史 lineage。当前高级分段入口还要求 prior prefix 与 source checkpoint 使用相同
的 world size 和 accumulation；若历史中这两项也变过，当前实现会明确拒绝，不能
把不同规模下的 update 数直接当成同一 exposure 前缀。

## 构建后独立校验

正式启动前可先在 CPU 节点执行以下只读校验；它不会启动训练：

```bash
/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python \
  scripts/runtime/validate_soft_cluster_migration_request.py \
  --schedule "${SCHEDULE_DIR}" \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --source-step "${SOURCE_STEP}" \
  --source-raw-checkpoint "${SOURCE_RAW_CHECKPOINT}" \
  --source-ema-checkpoint "${SOURCE_EMA_CHECKPOINT}" \
  --target-updates "${TARGET_UPDATES}"
```

校验覆盖 schedule 完整性、算法/拓扑、四库顺序与 masses、Short/Long 绑定、指定
checkpoint step/path 和 future span。训练 runner 还会在反序列化之前检查 raw/EMA
SHA-256、pair id、source/target data contract 与模型结构合同。

## 未来 AIStudio 迁移入口（本轮不要执行）

等我们决定迁移 checkpoint 后，16 个 AIStudio worker 使用同一组
`SOURCE_RUN_DIR / SOURCE_STEP / SCHEDULE_DIR`：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_crop384_db48_disordered_v4_softcluster_g02_v9_straggler_<tag> \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR="${SOURCE_RUN_DIR}" \
RESUME_STEP="${SOURCE_STEP}" \
RESUME_RAW_CHECKPOINT="${SOURCE_RAW_CHECKPOINT}" \
RESUME_EMA_CHECKPOINT="${SOURCE_EMA_CHECKPOINT}" \
SOFT_CLUSTER_SCHEDULE_DIR="${SCHEDULE_DIR}" \
bash migrate_crop384_db48_ddp_disordered_v4_to_softcluster_g02.sh \
  > "softcluster_g02.node${LOG_NODE_ID}.log" 2>&1
```

本轮只完成代码和测试，尚未运行上述正式 schedule 构建、校验或训练，也未占用
GPU。
