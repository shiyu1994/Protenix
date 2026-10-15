# 47999 原代码的 crop640 / SAbDab 新数据阶段

本库复制自实际生成 47999 的代码库：
`Protenix_train2_templatecache_top60_uniform20_id90_softcluster_g02_msaindependent_only`。
不是当前运行的 lr5e4 分支，也不移植后来的 generic/pseudo 或 numeric-MSA cache 实现。

## 入口

在 AIStudio 分配 16 节点、每节点 8 卡；所有节点运行同一命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_crop640_antibody_stage3_16x8_acc4_nopeptide_v2

RUN_NAME=protenix_v4_crop640_stage3_16x8_acc4_nopeptide_v2_from67999
BASE_DIR=/mnt/project_abd/klp/output
LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"
NODE_LOG_DIR="${BASE_DIR}/_launcher_logs/${RUN_NAME}"
mkdir -p "${NODE_LOG_DIR}"

export RUN_NAME BASE_DIR
exec bash ./train.sh > "${NODE_LOG_DIR}/node${LOG_NODE_ID}.log" 2>&1
```

默认新任务输出：
`/mnt/project_abd/klp/output/protenix_v4_crop640_stage3_16x8_acc4_nopeptide_v2_from67999_<timestamp>/checkpoints/`。

未显式指定 checkpoint 时，入口优先选择本 RUN_NAME 下最大的完整 raw/EMA pair；
首次启动没有自己的 checkpoint，才使用以下两个文件：

- `/mnt/project_abd/klp/output/initial_checkpoints/stage3_nopeptide_v2_from47999/checkpoints/67999.pt`
- `/mnt/project_abd/klp/output/initial_checkpoints/stage3_nopeptide_v2_from47999/checkpoints/67999_ema_0.999.pt`

它们分别是源 `47999.pt`、`47999_ema_0.999.pt` 的字节复制，已逐字节比对；
没有用 torch.load/save 重写，没有新增 checkpoint 字段。原文件未修改。

## 状态与编号

- live model 使用原 raw 参数，EMA shadow 使用原 EMA 参数；两者不混用。
- Adam、scheduler、GradScaler、rank 主进程 RNG 继续恢复原状态，不清空、不新 warmup。
- 只有首次进入新数据计划时，将训练 step 一次性设为 `START_STEP=68000`，
  global step 设为 `68000 × 4 = 272000`。之后日志、文件名和新 checkpoint 内部 step 一致。
- Adam 的历史动量和更新次数、scheduler 的历史计数保持原值，不随训练编号重置。
  `START_STEP` 不写入 checkpoint，不增加 checkpoint 字段；本库不再使用编号偏移。
- 每 50 个 optimizer updates 保存一对 raw/EMA checkpoint；第一对是 68049，最终是 169999。
- `MAX_STEPS=170000` 为训练上限，从 68000 开始还运行 102000 updates。
  这与原终点 50000 延长 100000 后的剩余训练预算一致。
- 原 AF3 scheduler 保留：base LR 0.001，原 warmup 1000，之后每 25000 updates 乘 0.95。
  延长上限不重置或拉伸 scheduler，也不使用当前 lr5e4 分支的学习率。
  真实 checkpoint 预检确认恢复 LR 为 0.00095、scheduler.last_epoch=48000、_step_count=48001。

显式 `RESUME_RUN_DIR`、`RESUME_STEP`、`RESUME_RAW_CHECKPOINT`、
`RESUME_EMA_CHECKPOINT` 优先于自动选择；`RESUME_STEP` 指文件编号。
输入文件名不必等于 payload 内部 step；入口不再按文件名添加 expected-step 限制，
仍保留 raw/EMA、模型结构和 payload 内部状态的一致性校验。
`START_STEP` 由 Bash 控制，0 也是有效值；它只在首次新数据阶段生效。
同计划 checkpoint 恢复时忽略 `START_STEP`，继续保存的训练步数和数据游标。
若更换首次新数据阶段的起点，需要用相同起点生成对应的数据计划。
`RESUME_SMOKE_STEPS=N` 的长度由 runner 在实际恢复／rebase 后确定，
不使用文件名推算起点；smoke 沿用不保存 checkpoint 的原行为。

## 数据

保留原四库内部数据定义，追加生产使用的两套 SAbDab，不加入 peptide-to-ligand。

| 数据集 | 原始混合权重 | 归一化抽样比例 |
| --- | ---: | ---: |
| weightedPDB_before210930_v20260101 | 0.49 | 46.6667% |
| afdb_mgnify_legacy_residual_distillation_v20260804 | 0.005 | 0.4762% |
| all_sources_ge200_structural_distillation_v20260804 | 0.505 | 48.0952% |
| openfold3_disordered_v20260725 | 0.02 | 1.9048% |
| sabdab-train | 0.02 | 1.9048% |
| distill-sabdab | 0.01 | 0.9524% |

SAbDab train 使用已过滤 `release_date <= 2021-09-30` 的索引：
`/mnt/project_abd/klp/databases/protenix_sabdab_exact_antibody_interface_release_le_20210930_v2_20260824/sabdab-train_exact_antibody_interface_release_le_20210930_v2.csv`。

distill 使用：
`/mnt/project_abd/klp/databases/protenix_sabdab_exact_antibody_interface_v1_20260821/distill-sabdab_exact_antibody_interface_v1.csv`。

两 SAbDab 沿用生产的结构、MSA、template 路径和 max_nuc_tokens=256；
其余四库的 nuc 上限仍为原 128。全局 template 概率仍 0.2，cache 仍为原
`protenix_sequence_template_cache_top60_uniform20_id90_v1`、required / chain_dummy。
MSA 仍通过 47999 原 loader 读取同一序列数据，不移植后来 numeric cache 加速代码。

新离线采样计划：
`/mnt/project_abd/klp/output/soft_cluster_schedules/stage3_nopeptide_v2_fresh67999_to169999_g02`。

保留原 soft-exposure 算法 `gamma=0.2` 和 cluster member 轮换。
仅 residual、GE200 两个原本使用 soft-exposure 的库有这套历史次数；此次全部清零。
六库外层抽样从 epoch 0 的序列重新开始，新计划第一条对应本次恢复位置。
计划包含 52224000 个抽样位置，覆盖全部新增训练预算；训练只读取预计算行。
没有在 dataloader 热路径上新增曝光计算或 sampler 包装层。

首次恢复明确切换为 fresh 数据计划，loader 累计计数与新训练起点对齐；
原保存校验未删除。后续保存使用原 checkpoint 格式记录新计划与游标；
同计划恢复时，即使入口仍有 `PROTENIX_RESET_DATA_SAMPLER=1`，也不会再次清零。
新计划不绑定任何模型文件路径、模型文件 SHA 或 checkpoint pair ID。

## 保留项和验证边界

128 卡、accumulation=4、每 update 512 个结构；crop 从 384 改为 640。
其余模型、loss、permutation、MSA、Adam、EMA 0.999、diffusion batch/chunk 48/48、
普通 activation checkpoint、PF/MSA CPU offload 关闭、GPU EMA、8 workers、
prefetch=2、原有 ordered DataLoader 均沿用 47999 原实现。

2026-09-10 首次 crop640 训练在第一步 Pairformer 反向重计算时 OOM。
随后仅将以下四项配置对齐 68999 的实际运行设置，复用本库已有实现：

- Pairformer `blocks_per_ckpt=4`（原 8）。
- MSA `blocks_per_ckpt=1`（原 4）。
- Diffusion `blocks_per_ckpt=1`（原 3）。
- Pairformer 和 Diffusion 的 `train_pair_transition_chunk_size=128`，
  `train_pair_transition_checkpoint_chunks=true`（原不分块、不做分块重计算）。

这不包含 68999 的其他显存设置：EMA shadow 仍在 GPU，
`enable_diffusion_shared_vars_cache` 仍为 true，PF/MSA CPU offload 仍关闭。
权重、Adam/scheduler 恢复、数据计划、步数和每 50 updates 保存均不变；
AIStudio 入口保持不变。配置和局部算子验证不代表完整 crop640 训练已通过显存验证。
这四项的专项验证共 24 项通过，包含最终 launcher 参数展开、配置解析及 GPU 局部分块
前向/反向对比。两套旧 launcher 测试另有 23 项通过、49 个 subtests 通过、9 项失败；
失败涉及已有 MAX_STEPS、日志文本和 RUN_NAME 旧预期，未扩大修改这些无关断言。

2026-09-10 后续修复仅增加 LayerNorm 同步修复和中断编译恢复：

- CUDA kernel 与 68999 使用的文件相同：三个 gamma/beta 梯度规约 kernel
  改为所有线程参与 block 同步，仅有效列访问数据，覆盖实际 833 维 LayerNorm 的尾列。
  前向公式、epsilon、精度类型、模型参数及 checkpoint 格式不变。
- LayerNorm 初始化使用节点本地文件锁，覆盖缓存恢复、校验、编译和记录发布，
  不在每个训练 step 或 dataloader 样本上增加操作。
- 仅当启用原缓存校验且记录缺失时，清除该扩展的 `.so`、对象文件、Ninja 文件和
  残留 PyTorch 编译锁，再从源码重编译并原子生成记录；不会给未知二进制直接补签。
- 记录已存在时仍执行原校验，完整缓存直接复用，记录损坏或二进制被更改仍报错。
- 外层文件锁不删除；进程退出后操作系统自动释放锁。此次源码变化会使标准入口
  自动选择新的缓存目录。不要让仍在运行的旧版编译进程与新版显式共用自定义缓存目录。
- AIStudio 入口和其他训练设置均未改变；无需清理节点整个 `/tmp`，也不自动提交或停止任务。

LayerNorm 修复专项验证：生产 CUDA 12.6 / H100 下共 35 项测试通过。
包括 833 维 FP32/BF16 前向及梯度对照、无记录半成品与残留锁恢复、两个独立进程
同时导入只重编译一次、完整缓存再次导入不重编译，以及原有篡改拒绝和 launcher 参数验证。
测试全部使用独立临时缓存，没有清理训练节点缓存；完整 128 卡训练启动仍需实际确认。

测试覆盖真实 CPU Adam/scheduler 更新后的保存与二次恢复、raw/EMA 分离、
主进程 RNG、首启一次性 fresh、原游标保存校验、计划边界和 Bash 覆盖/最新 pair 选择。
输入 raw/EMA 预检包含 128 个 rank 状态、5082 条 Adam 状态、5086 个模型 tensor。
验证还包括实际训练 `get_datasets` 构建的六库 sampler 与完整新计划匹配、
initial exposure 为零、首尾边界和 loader 起始游标。
本轮一次性 rebase 方案的 136 项测试全部通过，真实 raw/EMA 严格恢复预检通过。
25 次实际 CPU optimizer 更新后，保存和再次恢复已验证：文件及内部 step 同为 68024，
Adam/scheduler 的历史更新进度仍为 48025；再次恢复从 68025 继续，不重复重标。
验证记录保存在起始 checkpoint 目录的上一层：`resume_preflight.json`、
`validation_tests.log`、`fresh_schedule_validation.log`、`fresh_schedule_rebase67999.log`。
这不等同于实际 128 卡 crop640 训练 smoke；没有自动提交或停止任何训练任务。
原有 worker 内部 RNG 未写入 checkpoint 的边界也未改变，不承诺 crop/MSA 随机细节逐位重放。
重标内部 step 会改变以 step 为种子的 N_cycle 序列，以及以 sampler occurrence 为种子的
部分数据增广序列；概率和算法不变，但不是此前仅改变输出编号方案的逐样本等价运行。

初始两个输入文件只是旧 checkpoint 的重命名副本，内部仍为 47999；
新训练入口允许这一点。训练第一次保存及以后的 checkpoint，
内部 step 均与文件编号一致；模型权重格式本身未变。
