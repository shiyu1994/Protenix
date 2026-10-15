# crop384 / DB48 / DDP 5 万次更新训练的严格断点续训

## 审计结论

普通参数加载和完整续训不是同一件事。原有 `format_version=2` raw checkpoint 已经保存：

- raw 模型参数；
- optimizer state；
- scheduler state；
- 当前 `step`；
- GradScaler state（BF16 时为 disabled-state 空 mapping）；
- 每个 rank 的 Python、NumPy、torch CPU 和本地 CUDA RNG；
- 每个 rank 的 weighted sampler epoch、下一个 local offset、累计 batch 数及 sampler 配置指纹。

EMA checkpoint 是独立的 params-only snapshot。原实现仍有三个会造成“伪 resume”的缺口：

1. `load_params_only` 或各类 `skip_load_*` 可以绕过 optimizer、scheduler 或 step；
2. legacy raw 缺少 RNG/data state 时，只会重构 sampler cursor 并继续；
3. raw 和 EMA 只校验 checkpoint 类型与步数，不能证明来自同一次保存。

严格入口现在只接受带 `resume_contract.schema_version=3` 的新 checkpoint。raw
与 EMA 必须具有相同的 `checkpoint_pair_id`、源 run、step、global step、
world size、训练 profile ID、数据合同 hash、模型架构 hash 和 EMA decay。
数据合同的 canonical 内容包括有序 train sets、outer weights、epoch size、
sampler seed/replacement、命名 data profile、MSA 三分类概率/顺序/true-true
语义、template 概率、决策算法版本和 legacy template dropout；模型合同包括
Pairformer 深度与各模块全部
parameter-shaping 配置。任何状态缺失或不匹配都会在权重加载前失败。

当前 `weighted_pdb_afdb_mgnify_short_long_disordered_v3` 的 training-data contract 为 schema 7。
它额外记录全局配置与 dataset override 浅合并后实际送入 MSA/template featurizer 的配置、
完整 dataset 配置、`train_ref_pos_augment`、`limits`，以及
`data.train_sampler.seed<0` 时从全局 `seed` 解析得到的有效 sampler seed。这样修改继承的
MSA 数据库/template 路径或全局 seed，也会直接改变数据合同 hash。

v3 还锁定四个有序 dataset、raw weights
`0.5 / 0.005 / 0.495 / 0.02`、MSA categorical
`0.1 / 0.3 / 0.6`、多链 unpaired-shuffle requested
segment/whole-physical-chain=`0.5/0.5`，以及 Short/Long runtime 的 partition role、阈值和
共同 pair ID
`9c87d52dd446f622be689b7a09611f302d3791ac2448ecd86f8aee2488fb8734`。
精确 MSA mode policy 是
`categorical_crop_aware_partial_query_only_all_protein_multichain_segment_or_whole_shuffle_species_paired_v4`。
pair receipt SHA-256 为
`1b47dd0a7f15840b4a3f27ad67e1d821c954b263ee58c4c110c6c45a880a5476`，
Short/Long manifest SHA-256 分别为
`cab1953754d9c45302d8ab359006b21352d332628d767584471778433431ff1c` /
`b935c0d623d1fc2982a1449fab7ffed50b259653e11fb4c41c7bda131f535d9a`。
正式 schema-7 training-data contract SHA-256 为
`b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`。

历史 PDB-only schema 3 payload 不增加这些字段，旧合同 hash 保持不变；历史
`weighted_pdb_afdb_mgnify_disordered_v2` 也保持三源、`0.5/0.5/0.02`、
MSA `0.2/0.3/0.5` 和 segment-only 语义，其 schema-7 data-contract SHA-256 为
`c22ccd37db686dcad12d51f35bd51de23db9d9702c449f16dd74923fda124d59`。
v1/v2 checkpoint 都不能 strict-resume 到 v3。

## 多 worker 数据连续性的准确边界

主进程和 sampler 状态可以保存，但 PyTorch DataLoader 不提供正在运行的 worker Python/NumPy/torch RNG state。
所以这里必须区分两种保证：

- 完整训练状态恢复：模型、optimizer、scheduler、step、GradScaler、每 rank 主进程 RNG、精确 sampler
  cursor、sampler 配置及匹配 EMA 都恢复；常规每 1000 step checkpoint 均可使用。
- worker 预取逐位连续：`num_dl_workers=8` 时不承诺。续训创建的新 worker 会重新
  初始化内部 RNG，因此同一 target 的随机 crop/full-MSA 行子集等仍可能与不中断
  运行不同。当前 v3 MSA=`0.1/0.3/0.6` categorical gate 与 template-use=0.2 gate
  是例外：它们以 sampler occurrence 的稳定键计算，不依赖 worker RNG，resume
  后逐 occurrence 一致。partial query-only chain selection、row shuffle 和
  `multichain_msa_shuffle_variant` 使用彼此独立的 occurrence hash stream；给定同一 crop，
  segment/whole 请求和行排列可重现。历史 profile 的 paired-hit shuffle 边界不因此改变。
  crop 方法和 full-MSA 行抽样仍属于 worker RNG
  边界，不能据此声称整个样本逐位连续。

checkpoint 会明确记录：

```text
data_resume_scope=sampler_cursor_and_rank_main_process_rng
data_worker_rng_state_checkpointed=false
```

这不是把不完整状态冒充完整，而是将可恢复的训练状态与 PyTorch worker 内部预取随机流明确分开。如果实验必须追求数据
增强逐位连续，应从一开始使用 `num_dl_workers=0`，或者另行实现基于样本键的确定性 feature seed。

## 历史 weightedPDB-only 首次随机初始化

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k

RUN_NAME=crop384_db48_ddp_scratch_acc4_50k \
bash run_crop384_db48_ddp_scratch_50k.sh \
> crop384_db48_ddp_scratch_acc4_50k.log 2>&1
```

以下命令和合同仅保留用于复现历史 weightedPDB-only v1；不是当前 v3 入口。这个入口拒绝所有
checkpoint 来源，是真正的随机初始化，并为新 checkpoint 写入固定训练合约：

```text
crop384_db48_ddp_scratch_50k_ws128_acc4_data_weightedpdb_only_v1_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_q20_u30_p50_v8
```

该合同锁定命名数据 profile `weighted_pdb_only_v1`，即只使用
`weightedPDB_before210930_v20260101`、outer mass 1.0。未来加入任何蒸馏
dataset 都必须使用新的 profile 和合同，不能 strict-resume 本合同的 checkpoint。
它同时锁定新的 `lr=1e-3 / warmup=1000 / AF3 decay_every=25000` 轨迹。
旧 v7 checkpoint 带有旧 scheduler state，必须被 v8 strict resume 在加载参数前
拒绝，不能静默继承旧 `2000/50000` 曲线。

## 历史 weightedPDB-only 严格 resume

以下示例从源 run 的 `999.pt` 与 `999_ema_0.999.pt` 继续，目标仍是绝对
50,000 次 optimizer update，而不是再训练 50,000 次：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k

RUN_NAME=crop384_db48_ddp_scratch_acc4_50k_resume_1000 \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<source_run> \
RESUME_STEP=999 \
bash resume_crop384_db48_ddp_scratch_50k.sh \
> crop384_db48_ddp_scratch_acc4_50k_resume_1000.log 2>&1
```

以下参数只描述历史 weightedPDB-only 合同。入口锁定 DDP、16 节点 × 8 GPU、梯度累计 4、crop384、diffusion
batch/chunk 48/48、Pairformer 64 层、PF/MSA/Diff checkpoint 分组 8/4/3、
MSA query-only/unpaired-shuffle/species-paired=`0.2/0.3/0.5`、
template-use=0.2、MSA2048、GPU EMA 0.999、fused Adam、
DDP bucket view 及原始 optimizer/scheduler。Diffusion `c_s=384/c_token=768`
和 conditioning 保持不变。正式多机续训
从当前 AIStudio `MLP_*` 变量或显式 `RESUME_*` rendezvous 坐标解析 topology，
并在 `torchrun` 前要求恰好 16×8=128；runner child 再校验 world/local world
size，随后要求 checkpoint 合约与 128 份 rank-state 完全匹配。本合约不允许
resume 时改成 8×16、32×4 或任何小规模 smoke。
它显式传递：

```text
load_params_only=false
skip_load_optimizer=false
skip_load_scheduler=false
skip_load_step=false
load_step_for_scheduler=false
load_strict=true
```

## 当前 disordered-v3 随机初始化

16 个节点执行相同命令；平台提供 `MLP_WORKER_NUM`、`MLP_ROLE_INDEX` 和
`MLP_WORKER_0_HOST`：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v3_crop384_db48_disordered_v3_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v3.sh \
  > "disordered_v3.node${LOG_NODE_ID}.log" 2>&1
```

wrapper 锁定 v3 profile、16×8 topology、梯度累计 4、四源顺序/weights、
MSA `0.1/0.3/0.6`、多链 50/50 variant、protein-only disordered diffusion 和 8 个
DataLoader worker。`RUN_NAME` 必须包含 `disordered_v3` 与 `v9_straggler`，不接受 CLI override。

## 当前 disordered-v3 严格 resume

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/\
Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<v3_source_run> \
RESUME_STEP=999 \
RUN_NAME=protenix_v3_crop384_db48_disordered_v3_resume1000_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_disordered_v3.sh \
  > "disordered_v3_resume.node${LOG_NODE_ID}.log" 2>&1
```

源 raw/EMA checkpoint 必须来自同一个 v3 保存事件。v1/v2→v3 不存在 strict resume 路线；
profile、四项 outer weights、Short/Long pair identity、disordered artifacts、MSA policy 或 topology
任一不同都会在模型权重加载前失败。仅加载旧参数属于另一个需要单独命名和审计的初始化工作流，
不能伪装成 strict full-state resume。

## 一步生命周期 smoke

smoke 会完整读取并校验 checkpoint pair，恢复所有状态，只运行指定更新数，并禁止 checkpoint 写入：

```bash
RUN_NAME=protenix_v3_crop384_db48_disordered_v3_resume_smoke_v9_straggler \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<v3_source_run> \
RESUME_STEP=999 \
RESUME_SMOKE_STEPS=1 \
bash resume_crop384_db48_ddp_disordered_v3.sh
```

一步 resume smoke 也必须使用完整 16×8 拓扑；把 128 卡 checkpoint 缩到单节点
会在启动器阶段直接失败。一次“更新”包含四个 micro-step。DDP reducer 的初始化
状态不属于 checkpoint：每次 resume 创建新 DDP wrapper 后，第一个 micro-step
的 backward 都会强制同步一次，第 2–3 个 micro-step 使用 `no_sync()`，第 4 个
在更新边界同步。此后恢复稳态 `no_sync × 3 → sync/update` 累计。这只在每次
resume 增加一次 all-reduce，不改变恢复后的 optimizer、scheduler step、有效
batch 512 或梯度数值。

严格 checkpoint 的完整 resume 生命周期目前仍只做过 dry-run 和 CPU 单元级
round-trip，没有完成 128 卡真实 checkpoint smoke。DDP reducer 修复另已通过
真实两卡 NCCL 的首次启动、后续累计和非零 global step 新 wrapper 回归；它不能
替代本节的 128 卡 checkpoint smoke。

## 如何确认成功

日志应依次出现：

```text
Loading optimizer state
Loading checkpoint step
Loading scheduler state
Restored RNG and train sampler cursor
Finish loading checkpoint, current step: 1000
```

新 run 的 `manifest/run_manifest.json` 中应包含：

- `resume.strict_full_resume=true`；
- `resume.loaded_checkpoint_step=999`；
- `resume.next_training_step=1000`；
- 非空且匹配的 `resume.training_contract_id` 和 `resume.checkpoint_pair_id`；
- `resume.data_profile_id=weighted_pdb_afdb_mgnify_short_long_disordered_v3`；
- `resume.data_contract_sha256=b76b782fee58b6353992d2ebf8c4138173bf013f5dd7d05344abeaf739a8f55b`；
- `resume.model_architecture_sha256` 非空，且架构合同显示
  `pairformer.n_blocks=64`、Diffusion `c_s=384/c_token=768`；
- `resume.legacy_sampler_cursor_reconstructed=false`。

下列情况会直接失败，不会退化为只加载权重：

- legacy 或缺少 `resume_contract` 的 checkpoint；
- raw 没有非空 optimizer moments、scheduler、GradScaler、RNG 或 dataloader state；
- raw/EMA 不是同一次保存；
- EMA decay、data profile、train set 顺序/weights、epoch size、world size、
  worker 数、sampler 配置、augmentation 概率/算法或模型架构发生变化；
- Short/Long partition role、共同 pair ID、pair receipt、manifest hash 或 disordered
  immutable artifact identity 发生变化；
- 任一 `skip_load_*`、`load_params_only=true` 或 `load_strict=false`。
