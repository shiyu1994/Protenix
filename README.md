# 当前恢复入口：从 step47199 同合同 full-state resume

2026-08-13 的训练在 metric step `47324` 后中断；最后完整 checkpoint pair 为
`47199.pt` + `47199_ema_0.999.pt`。继续使用原来的 16×8 AIStudio 规格，但必须改用
`resume_step47199_msaindependent_only.sh`，不能再次执行锁死到 `27799` 的迁移入口。

16 个 AIStudio worker 使用同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_msaindependent_only/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_from47199_msaindependent_only_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_NNODES=16 \
bash resume_step47199_msaindependent_only.sh \
  > "from47199_msaindependent_only.node${LOG_NODE_ID}.log" 2>&1
```

该入口恢复 model、Adam optimizer、scheduler、step、BF16 scaler state、EMA、
128-rank 主进程 RNG 与 sampler cursor；下一 update 是 `47200`，继续到绝对终点
`50000`。完整 checkpoint 哈希、合同边界、日志判定和 dry-run 见
[step47199 同合同 full-state resume](docs/step47199_same_contract_full_resume_cn.md)。
本入口始终保持 gamma `0.2`，不包含 gamma `0` / hard-min；后者只能在 50k
checkpoint 之后建立新 schedule 和新合同。

# 实验起点：step27799 weightedPDB MSA 双源独立预算

本目录是从 `softcluster_g02` 父仓库复制的单变量实验副本。它从锁定的
step `27799` raw/EMA 完整恢复，并继续使用原
`step21999_residual_ge200_g02_v2` schedule；唯一训练语义变化是 weightedPDB
的 `pairing.a3m` 前 10,000 行和真正 `non_pairing.a3m` 前 10,000 行各有独立
source 预算，最终 MSA 组装仍不超过 16,384。LR 不 re-warm，PDB member
schedule、四库权重、sampler、optimizer 和其余训练设置不变。

完整合同、哈希、校验和回退方法见
[step27799 weightedPDB MSA 双源独立预算续训合同](docs/weightedpdb_msa_independent_step27799_resume_cn.md)。

16 个 AIStudio worker 使用同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_msaindependent_only/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_step27799_msaindependent_only_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_NNODES=16 \
bash migrate_step27799_to_weightedpdb_msa_independent.sh \
  > "step27799_msaindependent_only.node${LOG_NODE_ID}.log" 2>&1
```

该入口已经用于创建 2026-08-07 的生产 run；当前仅保留作迁移起点审计，不应用于
中断后的普通续训。

# 历史基线：Soft cluster anti-repeat，gamma=0.2

本目录是在原 template-cache 训练仓库之外创建的独立实验副本。当前正在实现
Short/Long 蒸馏 cluster 的 checkpoint-bound soft anti-repeat sampler：
`P(i) ∝ 0.2 ** historical_exposure_i`。历史边界由用户指定的 raw checkpoint
推导，不绑定固定 step；PDB/disordered 的原采样以及四库 outer masses 保持不变。

仓库包含只读 schedule 的生成/加载逻辑和恢复合同；step 21399 与 step 21999 的
正式 schedule 已分别完成验收，**本轮验收没有启动训练**。算法、历史迁移、
fail-closed 校验和资产布局见
[Soft cluster gamma=0.2 实现文档](docs/soft_cluster_sampling_gamma02_checkpoint_migration_cn.md)。

## Step 21999 residual/ge200 catalog replacement（已正式验收）

新增入口
`migrate_step21999_to_residual_ge200_softcluster_g02.sh`，独立锁定
`21999.pt`、`21999_ema_0.999.pt`、下一 update `22000`、16×8 / ws128、
accumulation=4、gamma=0.2 和绝对终点 50000；它复用已发布的 residual/ge200
runtime catalogs，并只接受新 schedule 目录：

```text
/mnt/project_abd/klp/proteinx_output/catalog_replacement_configs/step21999_residual_ge200_v1
/mnt/project_abd/klp/proteinx_output/soft_cluster_schedules/step21999_residual_ge200_g02_v2
```

两项资产均已发布并通过正式验收：checkpoint step `21999` 对应下一 update
`22000`，schedule 为 schema v2、共 `14,336,000` 行。raw/EMA/pair、128-rank
cursor、catalog crosswalk 和全部 artifact hash 均由 validator 验证通过；详细哈希见
[step 21999 中文合同](docs/catalog_replacement_step21999_resume_contract_cn.md)。本轮仅
完成准备和验收，尚未启动训练。真实启动时 launcher 和 runner 仍会重新核对
raw/EMA 与 schedule 的绑定，失败则不会应用训练状态。`SKIP_PATH_CHECK=1` 仅允许
dry-run，不能绕过真实训练校验。

正式启动时，16 个 AIStudio worker 使用同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_step21999_residual_ge200_softcluster_g02_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash migrate_step21999_to_residual_ge200_softcluster_g02.sh \
  > "step21999_residual_ge200_softcluster_g02.node${LOG_NODE_ID}.log" 2>&1
```

不要传旧的 step21399 schedule。历史入口
`migrate_step21399_to_residual_ge200_softcluster_g02.sh` 继续保留，语义未被新入口
覆盖。

## 2026-08-05 initial-stage 学习率评测决定

基于固定 1,315-target PXM-Legacy EMA evaluation 的 u10200-u21600 完整结果，
Micro、Macro-7 与 complex gPDE 的长期趋势仍然向上；u21600 是主要由小样本
RNA/DNA 接口类别驱动的高位单点回撤。由于 residual/ge200 catalog 与 gamma=0.2
soft-cluster 从 update 22000 才开始，而 u22200 以后尚无完整评测，本阶段决定不在
主线手工提高学习率。训练继续使用 `1e-3` 到 update 25000，并允许既有 scheduler
自然降至 `9.5e-4`。定量分析、confidence-ranking insight、后续观察点和重新调参
触发条件见
[initial-stage 学习率评测观察与决策](docs/initial_stage_lr_evaluation_decision_20260805_cn.md)。

# 原副本正式合同：Template cache、MSA 双预算与 checkpoint-200

本目录在原 `disordered-v4/v9-straggler` 训练实现上锁定三项生产变化：

- [按序列离线 Template cache](docs/sequence_template_cache_top60_uniform20_id90_cn.md)；
- [训练期 MSA 独立 raw-pool 预算](docs/msa_top16384_training_fast_path_cn.md)：
  默认 paired/unpaired 各最多 10,000 行；AFDB/MGnify Short/Long 的唯一
  direct-unpaired A3M 最多 16,384 行；最终合并池始终最多 16,384 行；
- raw/EMA checkpoint 每 200 次 optimizer update 成对保存。文件名使用 zero-based
  update 编号，因此第一组是 `199.pt` 与 `199_ema_0.999.pt`。

MSA 双预算同时覆盖蛋白与 RNA；两个 raw 池的未用预算不能互相转移，也不会
padding 到 10,000 或 16,384。`crop_size=384` 仍裁剪整个 target，
`max_nuc_tokens=128` 只限制 crop 中的核酸 token 子集。该 raw 前缀策略发生在
dedup 前，属于新的 data-policy；正式从旧 checkpoint 继续前必须建立对应的新
训练合同，不能把它描述成 trajectory-exact strict resume。

正式 stage-init 和 strict-resume launcher 会自动锁定并预检已完成的 template
cache，不要求用户手工传 cache 环境变量。训练请求真实 template 时，若某条
unique query sequence 的 cache 缺失、损坏或不符合 schema/policy，则该 target
中所有相同序列的 chain 使用固定 dummy template；其他 unique sequence 已加载的
真实 template 保留。它不会在线搜索，也不会让训练任务因此崩溃。全局 cache
根目录或两份完成 stats 不正确仍会在 `torchrun` 前失败。

# 当前正式阶段：disordered-v4 从 v8 step 999 参数初始化

当前正式入口是
`run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh`。它把 v8
`999.pt` 加载为 live/raw model，把配对的 `999_ema_0.999.pt` 加载为 EMA
shadow；optimizer、scheduler、step、GradScaler、RNG 和 sampler cursor 全部
重置，新阶段从 update 0 开始训练 50,000 updates。

四源 raw masses 固定为 PDB/Short/Long/disordered =
`0.49/0.005/0.505/0.02`，MSA 仍为 `q10/u30/p60`。16 个 AIStudio worker
执行相同命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh \
  > "disordered_v4_from_v8s999_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200.node${LOG_NODE_ID}.log" 2>&1
```

详细合同与恢复命令见
[disordered-v4 stage-init 文档](docs/crop384_db48_disordered_v4_from_v8_step999_cn.md)。

## AF3-like RecentPDB EMA 评测

AF3 Fig. 2d 使用的 1,220-target model-selection validation ID 没有公开。
本仓库已经准备好 PXM-Legacy RecentPDB/RNA/DNA 交集的 7-category proxy：12-target smoke
与 1,094-target full，固定 5 samples，支持 20-step quick 和 200-step
formal 两档。必须在不属于 128-GPU 训练任务的独立单卡上运行：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/path/to/7399_ema_0.999.pt \
EVAL_SCOPE=smoke \
EVAL_MODE=quick \
bash eval_af3_pxm_legacy_ema_checkpoint.sh \
  > af3_pxm_smoke_u7400.log 2>&1
```

集合定义、精确 checkpoint 命令、输出 key、正式 full 流程和泛化性限制见
[AF3 测试集与评测文档](docs/af3_testset_evaluation_cn.md)。

## 历史 PDB-only 随机初始化入口

本仓库当前的正式目标是从 **真正的随机初始化** 开始训练 Protenix-v2
架构：不读取官方 `protenix-v2.pt`，也不读取任何历史 raw/EMA checkpoint。
`MODEL_NAME=protenix-v2` 只选择网络架构；模型仍会保留架构定义中有意使用的
zero/fixed initializer，这不属于预训练权重。

当前入口保留的训练基线：

- Pairformer 从官方 v2 架构的 48 层扩为 **64 层**；`c_s=384`、
  Diffusion `c_token=768` 及 conditioning 均保持不变。总参数由
  464,442,431 增为 **529,902,655**，其中可训练参数 529,902,065；
- `train_crop_size=384` 是每个 target 的 crop 上限，
  `diffusion_batch_size=48`；
- 16 节点 × 8 GPU 的 DDP（总 world size 128）、BF16、每步 1–4 次
  recycle（上限 `N_cycle=4`）、20 个 diffusion step；
- 每个 rank 每个 micro-step 处理 1 个 target，梯度累计 4 次后更新，因此全局
  target batch 为 `128 × 4 = 512`；总目标为绝对
  `max_steps=50000` 次 optimizer update，即正常情况下每 rank 200,000 个
  micro-step、全局 25,600,000 次 target draw；
- 每次训练 target occurrence 使用一个互斥三分类 MSA gate：20% 真正的
  query-only、30% 旧 `true/true` unpaired-shuffle、50% 正常 Protenix
  species-paired。一个 target 的全部 chain 和 recycle 共用同一决定；
- query-only 对 protein/RNA 都不读取 homolog MSA，所有 chain 共同组成一条
  全局 query row，并从该行重算 `profile/deletion_mean`；旧 `true/true`
  分支关闭 species pairing，把 paired hits 并入各链 unpaired pool，并在保留
  query 的前提下 shuffle；50% normal 分支显式使用 `false/false`；
- paired 与 unpaired raw 候选池各自最多 10,000 行，最终 assembled pool
  最多 16,384 行；两个 raw 预算独立，不能借用且不足时不 padding。每个 recycle 先采
  `K ~ UniformInteger[1, min(actual_depth, 2048)]`，再从完整 pool 无放回抽
  K 行；当 `K<=1024` 时固定保留 index 0 的全局 query，并只抽取 `K-1` 个
  非 query index，因此最终高度仍严格为 K，不会增加成 1,025。`K>1024`
  保持原随机策略；该规则对训练和 inference 都生效。真实 feature/OPM 高度是 K，
  只有 MSAStack 的部分运算临时补到 2,048，`msa_chunk_size=2048`；
- 另一个相互独立的 occurrence gate 以 20% 概率请求真实 template、80% 使用官方
  固定 4-slot dummy-template 表示。请求真实 template 且存在有效候选时至少选
  1 个、最多 4 个；无候选时仍落到 dummy。MSA 三分类与 template gate 分别由
  sampler occurrence 的稳定 SHA256 stream 产生，不消耗 worker 的
  NumPy/Python/torch RNG，strict resume 后模式决定保持一致；
- 当前命名数据 profile 为 `weighted_pdb_only_v1`：只使用
  `weightedPDB_before210930_v20260101`，outer mass=`1.0`，
  `max_nuc_tokens=128`；
- Adam（不是 AdamW），`lr=1e-3`、warmup 1,000 update、
  `beta=(0.9, 0.95)`、`weight_decay=1e-8`，EMA decay 为 `0.999`。
  AF3 scheduler 每 25,000 update 乘 `0.95`：第一个边界后使用
  `9.5e-4`，第二个边界位于 50,000-update 训练结束处；
- `log_interval=25`，raw/EMA checkpoint 每 200 update 保存；第一组文件为
  `199.pt` 和 `199_ema_0.999.pt`。主训练明确保持
  `eval_interval=0 / eval_at_end=false`；每 5,000 update 的固定 EMA 验证由
  独立单卡任务读取 checkpoint 完成，不占用 128 张训练卡，也不消耗训练进程 RNG；
- 最终生产实现使用 PF64 + PF8 checkpoint grouping + c48：
  Pairformer/MSA/Diffusion activation checkpoint
  分组为 `8/4/3`，`diffusion_chunk_size=48`，不做 checkpoint input CPU
  offload，关闭 Transition chunk checkpoint；
- DDP 开启 `gradient_as_bucket_view`，使用 fused Adam、GPU-resident EMA shadow
  和 diffusion shared-variable cache。每次新建 DDP wrapper 后，第一次
  backward 会强制同步一次以初始化 PyTorch static-graph reducer；首个累计窗口
  的第 2–3 个 micro-step 使用 `no_sync()`，第 4 个在更新边界同步。后续稳态是
  `no_sync × 3 → sync/update`。这只在 scratch 启动或 resume 后各增加一次
  all-reduce，不改变有效 batch 512、梯度数值或稳态吞吐。
- v9 training DataLoader 锁定 `num_workers=8 / persistent_workers=true /
  prefetch_factor=2 / pin_memory=false`。worker 在 20-update sampler epoch 边界不再
  反复销毁和重建，避免每个节点每轮重启 64 个进程、全任务重启 1,024 个进程；
- v9 的 loss-health 检查仍在每个 micro-step 做本地 finite 判断，但 128-rank
  `MAX all_reduce` 延后到第 4 个 micro-step 的 optimizer-update 边界，每个
  update 从 4 次 health collective 降为 1 次；任一 rank/任一累计 micro-step
  非有限时，整个 update 仍会一致 skip；
- sampler `epoch_size=10240`，每 rank 每轮正好 80 个 micro-step/20 次更新，
  与 v8 保持不变，用于 matched sampler-stream 比较，并避免 128 卡下旧值
  10,000 导致不同 rank 的 sampler epoch 错位。

当前 weightedPDB-only 数据 profile 与 official initial-training loss 已锁定。
后续多个蒸馏数据集通过版本化 profile registry 接入，并使用新的阶段入口和
checkpoint 合同；现有入口不会接受环境变量切换到其他 profile。完整审计、未来
扩展步骤与 MSA 精确语义见
[crop384 / DB48 数据集与 loss 决策记录](docs/crop384_db48_dataset_loss_decision_cn.md)。

> **必须使用**
> `run_crop384_db48_ddp_scratch_50k.sh`。仓库目录名中的 `200k` 仅为历史兼容；
> 当前正式终点是 50,000 次 optimizer update。不要直接执行
> `finetune_demo.sh`：后者的默认逻辑会 fallback 到官方 checkpoint，因而不再是
> 真正的随机初始化训练。

AIStudio 多机多卡任务的每个 worker 都执行完全相同的命令；外层日志用节点编号
区分，不能让多个节点覆盖同一个日志文件：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k_v9_straggler/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_50k_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_scratch_50k.sh \
  > "protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_50k_v9_straggler.node${LOG_NODE_ID}.log" 2>&1
```

该命令假定 AIStudio 分配 16 个 worker、每 worker 8 张 H100；入口会在
`torchrun` 前锁定 `16×8=128`，并在 child 进程再次校验
`WORLD_SIZE=128 / LOCAL_WORLD_SIZE=8`。`diffusion_batch_size=48` 是每个
target 内部的 diffusion noise sample 数，不是全局 target batch。

严格恢复必须使用 `resume_crop384_db48_ddp_scratch_50k.sh`。它要求同一训练合约、
同一次保存的 raw/EMA pair、匹配的 step、EMA decay 和 world size，并恢复
model、optimizer、scheduler、step、GradScaler、各 rank 主进程 RNG、sampler
cursor 与 EMA；不完整或 legacy checkpoint 会直接失败。正式 resume 的目标仍是
绝对 50,000 次 optimizer update，而不是在 checkpoint 后再跑 50,000 次更新。

query-row 保留策略把正式合同升级为 `v11`。旧 `v10` checkpoint 的参数结构仍然
兼容，但因为随机采样语义已经改变，不能伪装成同一合同做普通 strict resume。
本仓库为当前 disordered-v4 run 提供一条固定、单向、fail-closed 的
`v10 → v11` full-state migration；除此以外的跨合同恢复仍会被拒绝。
train/test 阈值及“保留 query 但总深度仍为 K”的策略会写入所有数据 profile
共用的 data-contract hash；配置缺失按历史行为 `0/0`（禁用）处理，正式
launcher 显式锁定为 `1024/1024`。

当前 source run 已验证完成的首选迁移点是 `9999.pt` 与
`9999_ema_0.999.pt`。该文件名表示已经完成 10,000 次 optimizer update；
恢复后从 step 10000 / global micro-step 40000 开始。16 个 AIStudio worker
执行同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_STEP=9999 \
bash migrate_crop384_db48_ddp_disordered_v4_v10_to_v11.sh \
  > "disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999.node${LOG_NODE_ID}.log" 2>&1
```

该 strict-resume 入口已经锁定 `16×8`，因此某个 AIStudio worker 未注入
`MLP_WORKER_NUM` 时仍使用 16 个节点。node-rank 优先读取
`MLP_ROLE_INDEX`、`SLURM_NODEID` 或显式 `RESUME_NODE_RANK`；三者都缺失时，
只允许从精确的 `...-worker-N` hostname 后缀推导，并在日志中记录
`source=hostname_suffix`。非数字、越界或非精确 hostname 继续 fail-closed。

入口固定 source run、v10/v11 training contract、source/target data SHA、
模型架构 SHA、四源 profile、template cache 和 `16×8` 拓扑，并恢复 raw model、
匹配 EMA、optimizer、scheduler、step/global-step、GradScaler、128 个 rank
主进程 RNG 和 sampler cursor。8 个 persistent DataLoader worker 的内部 RNG
原本就不在 checkpoint 中，因此这是一条完整状态恢复的策略阶段切换，但不宣称
trajectory bitwise exact。若停训前已经生成更晚的完整 200-update raw/EMA pair，
可把 `RESUME_STEP` 改成对应的 zero-based 文件编号；`step+1` 必须能被 200
整除。第一个 v11 checkpoint 保存后，后续同合同中断恢复使用现有
`resume_crop384_db48_ddp_disordered_v4.sh`，不再使用 migration 入口。

当前 checkpoint 合同中的 MSA raw-budget 段随 profile 锁定：

- PDB-only 等默认 profile：
  `msaraw_p10000_u10000_final16384`；
- AFDB/MGnify Short/Long profile：
  `msaraw_p10000_u10000_distillu16384_final16384`。

完整合同其余部分保持
`crop384_db48_ddp_scratch_50k_ws128_acc4_data_<profile>_arch_pf64_cs384_dct768_pfckpt8_c48_msamix_<mix>_..._msaquerypinle1024_templatecache_top60u20id90_chaindummy_ckpt200_dlpersist1_prefetch2_deferredhealth_v11`。

`v9-straggler` 继承 v8 的 `lr=1e-3 / warmup=1000 /
decay_every=25000`，但 DataLoader 生命周期和 loss-health collective 合同已经
改变；v8 及更早 checkpoint 不能 strict-resume 到 v9。
50,000 update 是当前可执行的主评估点。表中 60,000–75,000 的“允许延长”
是看完 50k EMA 固定验证后的决策上限，不是本次任务自动多跑的步数；当前入口会
fail-closed 拒绝超过 50k。若决定延长，应从 49,999 raw/EMA pair 建立一个显式
continuation stage 和新合同，不能原地篡改 v8 的终点。

固定 EMA 验证必须提交为另一个单 GPU AIStudio 任务，而不是在 128 卡训练任务的
worker 上执行。把 `<actual_training_run_dir>` 替换成带时间戳的真实输出目录：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop384_db48_ddp_fast_200k/

CUDA_VISIBLE_DEVICES=0 \
TRAIN_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<actual_training_run_dir> \
bash run_crop384_db48_ema_eval_queue.sh \
  > crop384_db48_v8_ema_eval_queue.log 2>&1
```

队列依次等待 `4999/9999/.../49999_ema_0.999.pt`，即 5,000 到 50,000
update 的十个 EMA snapshot；验证固定使用单卡、PoseBusters≤512、seed 42、
`N_sample=5`、`N_step=20`、关闭 MC dropout 和 test reference-position
augmentation。单个 checkpoint 也可通过以下命令验证：

```bash
CUDA_VISIBLE_DEVICES=0 \
EMA_CHECKPOINT=/path/to/4999_ema_0.999.pt \
bash eval_crop384_db48_ema_checkpoint.sh
```

注意：当前复制进来的 `eval_crop384_db48_ema_checkpoint.sh` 仍只接受 v8 合同，
不能用于 v9 checkpoint；在首个 v9 固定 EMA 验证点前，应单独升级 evaluator
合同并通过对应 launcher test，不能绕过预检。

该三分类 gate 只接入训练 dataset；inference 继续使用当前独立的
`InferenceMSAFeaturizer` 和现有推理配置，不受这些概率影响。严格恢复能逐
occurrence 重放模式选择，但旧 `true/true` 内部的 paired-hit shuffle 仍使用
DataLoader worker NumPy RNG，不承诺中断前后逐行完全相同。

继承自 v8 的科学配置和严格恢复边界见
[crop384 / DB48 / DDP 随机初始化快速训练说明](docs/crop384_db48_ddp_scratch_fast_training_cn.md)；
其中旧路径、旧 contract 和命令仅作历史审计。当前 AIStudio 正式命令、v8
慢样本证据、v9 的唯一变量和 100/250-update matched benchmark 方案见
[v9 straggler 优化与 benchmark 说明](docs/crop384_db48_ddp_v9_straggler_benchmark_cn.md)。

PF8+c48 的实现参数曾在 **48 层 Pairformer** 上完成 matched benchmark：
240 个 exact-crop-384 样本，正式 20 步为 `11.694 s/step`、
`262.70 token/s`，后 10 步稳态为 `10.390 s/step`，峰值 allocated
`38.83 GiB`；无 OOM、非有限 loss/gradient 或 optimizer skip。扩为 64 层后
计算量、参数量和显存均已变化，因此这些数值只能证明 checkpoint grouping/chunk
策略的旧基线，**不能当作 PF64 的最终速度或显存结论**；正式 128 卡训练前仍需
跑 20-update PF64 smoke/benchmark。下方
crop1024/localattention 等章节是复制仓库时保留的历史内容，不代表本仓库当前
锁定入口的训练参数。

历史记录：2026-07-24 的首次 128 卡 acc2 启动在 step 0 暴露了 PyTorch 2.7.1
`DDP(static_graph=True)` 首次 backward 不能进入 `no_sync()` 的 reducer 断言。
当前修复已通过真实两卡 NCCL 回归：覆盖首次强制同步、后续
`no_sync × 3 → sync`、非零 global step 的新 DDP/resume 场景，并验证两 rank 梯度
严格等于全局 target 均值。该失败 run 没有 optimizer update 或 checkpoint；
修复后仍需重新执行正式 128 卡 smoke。

---

# Protenix: Protein + X

## Local-attention 实验副本

本目录于 2026-07-20 从
`Protenix_train2_fsdp2_crop1024_nuc128_db40` 原样复制，作为后续
local-attention 修改的独立实验代码库。

当前已经在 `protenix/model/modules/primitives.py` 实现 local-attention
rearrange 优化：直接构造局部索引和局部 mask/bias，避免旧实现的较大 dense
中间张量。该改动不新增模型参数，不改变 checkpoint 结构；其训练显存和速度收益
必须通过本仓库的严格 A/B benchmark 验证。

## 本仓库：crop1024 / nuc128 / diffusion batch 40 试验

本仓库从 db48 试验仓库独立复制，正式入口锁定：

- 总 `train_crop_size=1024`；
- 三个训练集均为 `max_nuc_tokens=128`；
- 每个 target 内部的 `diffusion_batch_size=40`；
- `diffusion_chunk_size=4`，因此每个 target 分成 10 个 diffusion chunk。
- 恢复正常 species-paired MSA：
  `pair_msa_as_unpaired_only=false`、`shuffle_pair_msa_when_unpaired_only=false`。

`max_nuc_tokens=128` 是 1024-token crop 中允许保留的核酸 token 上限，不会把
核酸 padding 到 128，也不把总 crop 改为 128。除上述 paired-MSA 语义外，
MSA 深度、checkpoint、FSDP2、CPU EMA 和 activation offload 设置保持不变。

AIStudio 正式训练命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh \
  > "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2.node${LOG_NODE_ID}.log" 2>&1
```

`localattention` 标识用于将本实验的输出和普通 DB40 paired-MSA 基线明确区分。
每个 AIStudio worker 使用独立外层日志，避免多个节点同时覆盖同一个共享文件。

正式提交前的一步 GPU smoke：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_smoke1 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
PROTENIX_SMOKE_TEST=1 \
bash run_immune_heavy_old_content_crop1024_nuc128_db40_fsdp2_finetune.sh \
  > "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_smoke1.node${LOG_NODE_ID}.log" 2>&1
```

## 从指定 checkpoint 完整恢复

原训练在完成 step 499 checkpoint 后继续运行到 step 534，随后因
`torch_shm_manager` 的 `file_system` IPC 超时退出。最后一组可持久恢复文件为：

- raw：
  `/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_20260722_032221/checkpoints/499.pt`
- EMA：
  `/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_20260722_032221/checkpoints/499_ema_0.999.pt`

续训入口会将 raw checkpoint 恢复到活动模型，并同时恢复 Adam optimizer、
LR scheduler、GradScaler 和 step；EMA 文件只恢复 EMA shadow。它不会把 EMA
文件误当成 raw，也不会从 step 0 重新开始。`max_steps=2000` 是绝对总步数：
加载 `499.pt` 后从 step 500 开始，最终只补足到 2000。

AIStudio 多机多卡正式续训命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_resume499 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_20260722_032221 \
RESUME_STEP=499 \
RESUME_TOTAL_MAX_STEPS=2000 \
RESUME_NNODES=4 \
bash resume_immune_heavy_localattention_fsdp2.sh \
  > "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_resume499.node${LOG_NODE_ID}.log" 2>&1
```

仅验证一次完整加载、forward、backward、optimizer、scheduler 和 EMA 更新时，
使用以下一步 resume smoke。它把绝对总步数设为 501，只执行 step 500，
并禁用周期及最终 checkpoint 写入，因而不会产生新的多 GB checkpoint：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_fsdp2_crop1024_nuc128_db40_localattention/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
RUN_NAME=immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_resume499_smoke1 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_20260722_032221 \
RESUME_STEP=499 \
RESUME_SMOKE_STEPS=1 \
RESUME_NNODES=4 \
bash resume_immune_heavy_localattention_fsdp2.sh \
  > "immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_resume499_smoke1.node${LOG_NODE_ID}.log" 2>&1
```

本地开发机若只有部分 GPU 完全空闲，可仅在单节点 smoke 中同时设置
`CUDA_VISIBLE_DEVICES` 和 `RESUME_NPROC_PER_NODE` 使用隔离子集。该模式用于验证
full-state 加载、FSDP 重分片和单步更新，不等价于正式 4×8 的数据流/性能验证；
正式 resume 仍拒绝非 8 卡/节点。

resume 默认使用 `PROTENIX_MP_SHARING_STRATEGY=file_descriptor`，不再依赖
`torch_shm_manager`，并将 DataLoader 从每 rank 16 个 worker 降为 8 个。
若确实需要做共享策略对照，使用专用的
`RESUME_MP_SHARING_STRATEGY=file_system` 显式覆盖；普通 shell 中残留的
`PROTENIX_MP_SHARING_STRATEGY` 不会改变 resume 默认值。
启动器会在各 AIStudio 节点自动选择实际存在的一组 GCC/G++（依次检查 13、12、
11），并在启动摘要中打印 `cuda_host_compiler`。如需固定版本，必须同时设置
`PROTENIX_CUDA_HOST_CC` 和 `PROTENIX_CUDA_HOST_CXX` 为可执行文件的绝对路径；
路径缺失或只设置其中一个会在分配模型显存前直接失败。
正式续训锁定源任务相同的 4 节点 × 8 GPU 拓扑，并优先使用当前 AIStudio
注入的 `MLP_WORKER_*` 坐标，避免复用 shell 中残留的通用 torchrun 变量覆盖。
后续新 raw checkpoint 还会保存每个 rank 的 Python、NumPy、Torch CPU/CUDA
RNG 以及 DataLoader sampler 的 epoch/offset。旧的 format-v2 `499.pt` 没有
这些新增字段，加载时会依据已完成的 500 个 global batch 重建采样器位置，并在
日志中明确标记 legacy fallback；模型、optimizer、scheduler 和 step 仍从原文件
完整恢复。

如需在首次恢复前重新校验两个大文件，可在命令中加入
`RESUME_VERIFY_SHA256=1`。默认 checkpoint 的已审计 SHA256 为：

- raw：`3681c271febfad4336e444b4ba00c275f754c557e75325f27dfefabdb9aa8416`
- EMA：`ded9c659d46bd859fbe6c0ab7fbbf8fafd0e81e5d97a07ed869336eaa4adafbd`

2026-07-23 已在本机空闲 GPU 0–4 上完成一次 5 卡生命周期 smoke：

- EMA 先加载，raw 随后恢复 model、optimizer、scheduler 和 step，训练从 step 500 开始；
- step 500 的 `did_update=true`、`optimizer_skipped=false`、`loss_nan=false`，
  聚合 loss 为 `1.0046039`，grad norm 为 `0.1835131`；
- EMA 更新成功，5 个 rank 都完成 forward、backward 和 optimizer step；
- 进程退出码为 0，`run_status=completed`、`last_global_step=501`；
- smoke 未写 raw/EMA checkpoint，符合 `checkpoint_interval=-1` 和
  `skip_final_checkpoint=true`；
- 本次随机抽到的 target 为 239–467 token，因此它验证的是完整恢复/更新生命周期，
  不是 exact-crop-1024 峰值显存或正式 4×8 性能测试。

验证产物位于
`/mnt/project_abd/klp/proteinx_output/immune_heavy_old_content_crop1024_nuc128_db40_pairedmsa_localattention_fsdp2_resume499_smoke1_5gpu_retry1_20260723_180844`。

完整的差异边界、checkpoint 来源和建议验证顺序见
[crop1024 / nuc128 / diffusion batch 40 试验说明](docs/crop1024_nuc128_diffusion_batch40_experiment_cn.md)。
与原始 immune-heavy 的逐项实现差异、效果来源优先级和 matched 实验建议见
[diff_vs_original_immune_heavy.md](diff_vs_original_immune_heavy.md)。
与 db40 基线仓进行严格 1-step smoke 和 10+20 matched A/B 的锁定设置、AIStudio
命令及结果比较方法见
[DB40 paired-MSA local-attention A/B benchmark](docs/db40_paired_localattention_benchmark_cn.md)。
原 db32、db48 仓库及其已经启动的训练不读取本目录中的文件，不受本试验影响。

<div align="center" style="margin: 20px 0;">
  <span style="margin: 0 10px;">⚡ <a href="https://protenix-server.com">Protenix Web Server</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="docs/PTX_V1_Technical_Report_202602042356.pdf">Protenix-v1</a></span>
  &bull; <span style="margin: 0 10px;">📄 <a href="docs/PX2.pdf">Protenix-v2</a></span>
</div>

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=x)](https://x.com/ai4s_protenix)
[![Slack](https://img.shields.io/badge/Slack-Join-yellow?logo=slack)](https://join.slack.com/t/protenixworkspace/shared_invite/zt-3drypwagk-zRnDF2VtOQhpWJqMrIveMw)
[![Wechat](https://img.shields.io/badge/Wechat-Join-brightgreen?logo=wechat)](https://github.com/bytedance/Protenix/issues/52)
[![Email](https://img.shields.io/badge/Email-Contact-lightgrey?logo=gmail)](#contact-us)
</div>

We’re excited to introduce **Protenix** — Toward High-Accuracy Open-Source Biomolecular Structure Prediction.

Protenix is built for high-accuracy structure prediction. It serves as an initial step in our journey toward advancing accessible and extensible research tools for the computational biology community.

<img src="assets/protenix_predictions.gif" style="width: 100%; height: auto;" alt="Protenix predictions">

## 🌟 Related Projects
- **[PXDesign](https://protenix.github.io/pxdesign/)** is a model suite for de novo protein-binder design built on the Protenix foundation model. PXDesign achieves 20–73% experimental success rates across multiple targets — 2–6× higher than prior SOTA methods such as AlphaProteo and RFdiffusion. The framework is freely accessible via the Protenix Server.

- **[PXMeter](https://github.com/bytedance/PXMeter/)** is an open-source toolkit designed for reproducible evaluation of structure prediction models, released with high-quality benchmark dataset that has been manually reviewed to remove experimental artifacts and non-biological interactions. The associated study presents an in-depth comparative analysis of state-of-the-art models, drawing insights from extensive metric data and detailed case studies. The evaluation of Protenix is based on PXMeter.

- **[Protenix-Dock](https://github.com/bytedance/Protenix-Dock)**: Our implementation of a classical protein-ligand docking framework that leverages empirical scoring functions. Without using deep neural networks, Protenix-Dock delivers competitive performance in rigid docking tasks.

## 🎉 Latest Updates
- **2026-04-08: Protenix-v2 Released** 💪💪 [[Protenix-v2 Technical Report](docs/PX2.pdf)]
  - Protenix-v2 shows clear gains on antibody-antigen structure prediction, together with an additional update in ligand-related plausibility.
- **2026-02-05: Protenix-v1 Released** 💪 [[Protenix-v1 Technical Report](docs/PTX_V1_Technical_Report_202602042356.pdf)]
  - Supported Template/RNA MSA features and improved training dynamics, along with further Inference-time model performance enhancements.
- **2025-11-05: Protenix-v0.7.0 Released** 🚀
  - Introduced advanced diffusion inference optimizations: Shared variable caching, efficient kernel fusion, and TF32 acceleration. See our [performance analysis](./assets/inference_time_vs_ntoken.png).
- **2025-07-17: Protenix-Mini & Constraint Features**
  - Released lightweight model variants ([Protenix-Mini](https://arxiv.org/abs/2507.11839)) that drastically reduce inference costs with minimal accuracy loss.
  - Added support for [atom-level contact and pocket constraints](docs/infer_json_format.md#constraint), enhancing prediction accuracy through physical priors.
- **2025-01-16: Pipeline Enhancements**
  - Open-sourced the full [training data pipeline](./docs/prepare_training_data.md) and [MSA pipeline](./docs/msa_template_pipeline.md).
  - Integrated local [ColabFold-compatible search](./docs/colabfold_compatible_msa.md) for streamlined MSA generation.


## 🚀 Getting Started

### 🛠 Quick Installation

```bash
pip install protenix
```

### 🧬 Quick Prediction

```bash
# Predict structure using a JSON input
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
```

#### Key Model Descriptions
| Model Name | MSA | RNA MSA | Template | Params | Training Data Cutoff | Model Release Date |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `protenix-v2` | ✅ | ✅ | ✅ | 464 M | 2021-09-30 | 2026-04-08 |
| `protenix_base_default_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2021-09-30 | 2026-02-05 |
| `protenix_base_20250630_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2025-06-30 | 2026-02-05 |
| `protenix_base_default_v0.5.0` | ✅ | ❌ | ❌ | 368 M | 2021-09-30 | 2025-05-30 |

- **protenix-v2**: An enhanced-capacity version of the base model, featuring increased representation dimensionality and expanded parameter space (~464M), along with substantial training and optimization improvements.
- **protenix_base_default_v1.0.0**: Base model, trained with a data cutoff aligned with AlphaFold3 (2021-09-30). The total parameter count of protenix_base_default_v1.0.0 is close to that of AlphaFold3.
- **protenix_base_20250630_v1.0.0**: Applied model, trained with an updated data cutoff (2025-06-30) for better practical performance. This model can be used for practical application scenarios.
- **protenix_base_default_v0.5.0**: Previous version of the model, maintained primarily for backward compatibility with users who developed based on v0.5.0.

For a complete list of supported models, please refer to [Supported Models](docs/supported_models.md).

For detailed instructions on installation, data preprocessing, inference, and training, please refer to the [Training and Inference Instructions](docs/training_inference_instructions.md). We recommend users refer to [inference_demo.sh](inference_demo.sh) for detailed inference methods and input explanations.

For the train2 full-model DDP/FSDP2 implementation and matched AIStudio benchmark workflow, see [the FSDP2 training benchmark guide](docs/fsdp2_full_model_implementation_and_aistudio_benchmark_cn.md).


### 📊 Benchmark

#### Protenix-v2

Protenix-v2 (refers to the `protenix-v2` model) shows clear gains on antibody-antigen structure prediction, together with an additional update in ligand-related plausibility. Compared to baselines and the earlier Protenix-v1, Protenix-v2 demonstrates a substantial improvement trend. At the DockQ > 0.23 threshold, Protenix-v2 achieves absolute success rate gains of 9 to 13 percentage points over Protenix-v1 across three collections. Remarkably, Protenix-v2 at only 5 seeds already exceeds the performance of Protenix-v1 at 1000 seeds, indicating a clear gain in efficiency.

<img src="./assets/protenix-v2.png" style="width: 100%; height: auto;" alt="Protenix-v2 model Metrics">


#### Protenix-v1

Protenix-v1 (refers to the `protenix_base_default_v1.0.0` model), the first fully open-source model that outperforms AlphaFold3 across diverse benchmark sets while adhering to the same training data cutoff, model scale, and inference budget as AlphaFold3. For challenging targets, such as antigen-antibody complexes, the prediction accuracy of Protenix-v1 can be further enhanced through inference-time scaling – increasing the sampling budget from several to hundreds of candidates leads to consistent log-linear gains.

<img src="./assets/protenix_base_default_v1.0.0_metrics.png" style="width: 100%; height: auto;" alt="protenix-v1 model Metrics">

<img src="./assets/protenix_base_default_v1.0.0_metrics2.png" style="width: 100%; height: auto;" alt="protenix-v1 model Metrics 2">

For detailed benchmark metrics on each dataset, please refer to [docs/model_1.0.0_benchmark.md](docs/model_1.0.0_benchmark.md).

## Citing Protenix

If you use Protenix in your research, please cite the following:

```
@article {Zhang2026.04.10.717613,
	author = {Zhang, Yuxuan and Gong, Chengyue and Sun, Jinyuan and Guan, Jiaqi and Ren, Milong and Xue, Song and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Xiao, Wenzhi},
	title = {Protenix-v2: Broadening the Reach of Structure Prediction and Biomolecular Design},
	elocation-id = {2026.04.10.717613},
	year = {2026},
	doi = {10.64898/2026.04.10.717613},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/04/11/2026.04.10.717613},
	eprint = {https://www.biorxiv.org/content/early/2026/04/11/2026.04.10.717613.full.pdf},
	journal = {bioRxiv}
}

@article {Zhang2026.02.05.703733,
	author = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Guan, Jiaqi and Wang, Lan and Yang, Yanping and Xia, Yu and Xiao, Wenzhi},
	title = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
	elocation-id = {2026.02.05.703733},
	year = {2026},
	doi = {10.64898/2026.02.05.703733},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1},
	eprint = {https://www.biorxiv.org/content/early/2026/02/22/2026.02.05.703733.1.full.pdf},
	journal = {bioRxiv}
}

@article {2025.01.08.631967,
	author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
	title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
	elocation-id = {2025.01.08.631967},
	year = {2025},
	doi = {10.1101/2025.01.08.631967},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967},
	eprint = {https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967.full.pdf},
	journal = {bioRxiv}
}
```

### 📚 Citing Related Work
Protenix is built upon and inspired by several influential projects. If you use Protenix in your research, we also encourage citing the following foundational works where appropriate:
```
@article{abramson2024accurate,
  title={Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick, Joshua and others},
  journal={Nature},
  volume={630},
  number={8016},
  pages={493--500},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@article{ahdritz2024openfold,
  title={OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
  author={Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O’Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccol{\`o} and others},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1514--1524},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  volume={19},
  number={6},
  pages={679--682},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
```

## Contributing to Protenix

We welcome contributions from the community to help improve Protenix!

📄 Check out the [Contributing Guide](CONTRIBUTING.md) to get started.

✅ Code Quality: 
We use `pre-commit` hooks to ensure consistency and code quality. Please install them before making commits:

```bash
pip install pre-commit
pre-commit install
```

🐞 Found a bug or have a feature request? [Open an issue](https://github.com/bytedance/Protenix/issues).



## Acknowledgements


The implementation of LayerNorm operators refers to both [OneFlow](https://github.com/Oneflow-Inc/oneflow) and [FastFold](https://github.com/hpcaitech/FastFold).
We also adopted several [module](protenix/openfold_local/) implementations from [OpenFold](https://github.com/aqlaboratory/openfold), except for [`LayerNorm`](protenix/model/layer_norm/), which is implemented independently.


## Code of Conduct

We are committed to fostering a welcoming and inclusive environment.
Please review our [Code of Conduct](CODE_OF_CONDUCT.md) for guidelines on how to participate respectfully.


## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

The Protenix project including both code and model parameters is released under the [Apache 2.0 License](./LICENSE). It is free for both academic research and commercial use.

## Contact Us

We welcome inquiries and collaboration opportunities for advanced applications of our model, such as developing new features, fine-tuning for specific use cases, and more. Please feel free to contact us at ai4s-bio@bytedance.com.
