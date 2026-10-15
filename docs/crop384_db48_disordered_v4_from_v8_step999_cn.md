# Disordered-v4：从 v8 step 999 参数初始化

## 训练语义

本阶段不是 strict resume。模型架构合同完全相同，SHA256 为
`cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c`，
因此使用严格 model-state 加载：

- live/raw model ← v8 `999.pt`；
- EMA shadow ← v8 `999_ema_0.999.pt`；
- optimizer、scheduler、step、GradScaler、rank RNG、sampler cursor 全部重置；
- 目标阶段从 update 0 开始，重新 warmup 1,000 updates；
- 正式目标为新的 50,000 optimizer updates。

源 checkpoint pair 固定为：

```text
run:
/mnt/project_abd/klp/proteinx_output/protenix_v2_scratch_crop384_db48_ddp_ws128_acc4_msamix_q20_u30_p50_50k_v8_20260724_155026

raw SHA256:
b92361c83de2bf3d300c082bb84a0b797f5b2d92837c8deead7a4f1b3da17ca8

EMA SHA256:
a4ecad60c39cad6072e0e1093d6f1394e66142ef1b24f03b931f048a48f7b5f7

pair ID:
1220d08b9f62db65364a5e936e2e7d22577c72e71fcca4ae2a261f0045fb9001
```

## v4 数据合同

v4 继承 v3 的四源 artifacts、crop-aware partial query-only、
segment/whole-physical-chain MSA shuffle、protein-only disordered diffusion
和 `q10/u30/p60`。唯一的数据采样变化是 raw masses：

```text
weightedPDB / Short / Long / disordered
0.49 / 0.005 / 0.505 / 0.02
```

raw mass 总和为 1.02；归一化概率约为
48.0392% / 0.4902% / 49.5098% / 1.9608%。

本代码库在该 v4 采样合同之上另外锁定默认 paired/unpaired raw MSA 各
10,000 行；AFDB/MGnify Short/Long 的唯一 direct-unpaired A3M 使用
16,384 行预算；最终 assembled pool 仍最多 16,384 行。同时锁定离线
sequence-template cache。
cache 中某条 unique query sequence 缺失或损坏时，同序列的所有 chain 使用
fixed dummy template，其他 unique sequence 的真实 template 保留。它们属于新的
data-policy，运行时 data-contract SHA256 会由最终配置生成，不能继续沿用旧
schema-7 的 SHA256。

## 16×8 AIStudio 正式启动

16 个 worker 执行相同命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash run_crop384_db48_ddp_disordered_v4_from_v8_step999.sh \
  > "disordered_v4_from_v8s999_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200.node${LOG_NODE_ID}.log" 2>&1
```

首次提交可增加以下两个命令级环境变量做 1–20 update smoke：

```bash
PROTENIX_SMOKE_TEST=1 \
PROTENIX_SMOKE_UPDATES=1 \
```

如需在启动节点重新计算两份源文件的完整 SHA256，再增加：

```bash
PROTENIX_WARMSTART_VERIFY_FULL_SHA256=1
```

常规启动会校验固定路径、文件名、大小、format/kind/step、pair ID、source
training/data/architecture contract 和 5,086 个 model keys；完整 SHA 已在发布本入口时
计算并固定。入口还会自动锁定并预检 template cache；AIStudio 命令无需额外传
cache 环境变量。

## 后续 strict resume

本阶段每 200 optimizer updates 保存一组 raw/EMA checkpoint；zero-based
文件名下第一组为 `199.pt` 与 `199_ema_0.999.pt`。首次保存后可使用：

```bash
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<v4_source_run> \
RESUME_STEP=199 \
RUN_NAME=protenix_v4_crop384_db48_disordered_v4_resume200_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200 \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_disordered_v4.sh
```

不能用 v8 PDB-only checkpoint 调用 v4 strict-resume；v8 只允许通过固定 stage-init
入口作为模型参数来源。

## 当前 v10 run 切换到 query-pin v11

当前正在训练的 disordered-v4 run 创建于 query-pin 策略合入之前，其 checkpoint
合同结尾为 `deferredhealth_v10`。新代码在 model-side MSA 抽样高度
`K <= 1024` 时固定保留 index 0 的全局 query，并保持最终高度仍为 K，因此
data-contract 已升级为 `msaquerypinle1024...v11`。这不是同合同的
trajectory-exact resume。

仓库提供专用入口
`migrate_crop384_db48_ddp_disordered_v4_v10_to_v11.sh`。它只允许以下单向边界：

- source run 固定为
  `protenix_v4_crop384_db48_disordered_v4_from_v8s999_50k_v9_straggler_templatecache_chaindummy_msabase10ku10k_distillu16384_ckpt200_20260726_061627`；
- source data SHA256 固定为
  `cffda1e6694a9f293c8ed213d1b33e188d828d10a4d0eebdf66b2f428e75756c`；
- target data SHA256 固定为
  `6c98c5cab75197b7e4d41b7ce5287af65ca3f750a707ec69a1bdaf29ed24b8de`；
- source/target architecture SHA256 都必须为
  `cd1cbb658e352f0d56a2dcd65c5e0e95ae2e29c321f4fd0da9f79c6145edc34c`；
- target data contract 删除 query-pin 字段后必须精确重建 source SHA，证明
  唯一允许的数据语义变化就是 query inclusion；
- raw/EMA 必须来自同一次保存，且 step、world size、EMA、optimizer、
  scheduler、GradScaler、128 份 rank RNG 与 sampler cursor 均通过原 strict
  validator。

`9999.pt` 与 `9999_ema_0.999.pt` 已完整生成并审计通过。它们记录
`step=9999 / global_step=39999 / next_step=10000 /
next_global_step=40000`，而且所有 rank 的 sampler cursor 都正好处于 epoch
边界。16 个 AIStudio worker 执行相同命令：

```bash
cd /mnt/project_abd/klp/workspace/proteinx-selected/Proteinx-v2/Protenix_train2_templatecache_top60_uniform20_id90/

LOG_NODE_ID="${MLP_ROLE_INDEX:-${NODE_RANK:-$(hostname)}}"

RUN_NAME=protenix_v4_crop384_db48_disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
RESUME_STEP=9999 \
bash migrate_crop384_db48_ddp_disordered_v4_v10_to_v11.sh \
  > "disordered_v4_v10_to_v11_msaquerypinle1024_from_s9999.node${LOG_NODE_ID}.log" 2>&1
```

入口自身锁定 `16×8`，不依赖每个 worker 都提供 `MLP_WORKER_NUM`。
node-rank 环境变量缺失时，只从 AIStudio 精确的 `...-worker-N` hostname
后缀提取 `N`；解析来源写入启动日志，非法或越界值仍会在 `torchrun` 前退出。

入口恢复：

```text
raw model + matched EMA + optimizer + scheduler + update/global step
+ BF16 GradScaler mapping + each-rank Python/NumPy/torch/CUDA RNG
+ train sampler cursor
```

8 个 persistent DataLoader worker 的内部 RNG 原本没有写入 checkpoint，所以
manifest 会明确记录 `data_worker_rng_state_checkpointed=false` 和
`trajectory_exact=false`。若停止旧任务前已经出现更晚的完整 checkpoint pair，
可以把 `RESUME_STEP` 改为相应编号，但 `step+1` 必须能被 200 整除。迁移后的
第一个 v11 raw/EMA pair 保存后，后续继续训练改回普通入口：

```bash
RESUME_RUN_DIR=/mnt/project_abd/klp/proteinx_output/<new_v11_run> \
RESUME_STEP=<complete_v11_checkpoint_step> \
RUN_NAME=protenix_v4_crop384_db48_disordered_v4_resume_v11_v9_straggler \
BASE_DIR=/mnt/project_abd/klp/proteinx_output \
bash resume_crop384_db48_ddp_disordered_v4.sh
```
