# finetune_demo.sh 运行流程分析

## 1. Shell 层（`finetune_demo.sh`）

### 环境变量
- `PYTHONPATH` 追加当前目录
- `PROTENIX_ROOT_DIR` = `/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix`
- `checkpoint_path` = `${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt`

### 启动命令
```
torchrun --nproc_per_node gpu ./runner/train.py [参数...]
```

### 关键 CLI 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `--model_name` | protenix-v2 | 模型名称 |
| `--run_name` | protenix_finetune_sabdist0.1_sabtrain0.3_pdb0.6_lr1e-5 | 运行名称 |
| `--base_dir` | /vepfs-mlp2/project-abd/zhujianwei/share/klp/proteinx_output | 输出根目录 |
| `--dtype` | bf16 | 训练精度 |
| `--diffusion_batch_size` | 48 | 扩散批大小 |
| `--train_crop_size` | 384 | 训练裁剪尺寸 |
| `--max_steps` | 100000 | 最大训练步数 |
| `--warmup_steps` | 2000 | 预热步数 |
| `--lr` | 0.00001 | 学习率 |
| `--ema_decay` | 0.999 | EMA 衰减率 |
| `--model.N_cycle` | 4 | 循环数（覆盖默认10） |
| `--sample_diffusion.N_step` | 20 | 推理时扩散步数 |
| `--iters_to_accumulate` | 1 | 梯度累积步数 |
| `--eval_interval` | 400 | 评估间隔 |
| `--log_interval` | 50 | 日志间隔 |
| `--checkpoint_interval` | 100 | 保存间隔 |
| `--data.train_sets` | distill-sabdab,sabdab-train,weightedPDB_before2109_wopb_nometalc_0925 | 训练集 |
| `--data.train_sampler.train_sample_weights` | 0.1,0.3,0.6 | 训练集采样权重 |
| `--data.test_sets` | posebusters_0925 | 测试集 |
| `--data.posebusters_0925.base_info.max_n_token` | 512 | 测试集最大token数 |

---

## 2. 训练入口（`runner/train.py:734 main()`）

```
main()
  ├── 解析命令行参数 (parse_sys_args)
  ├── 合并 configs_base + data_configs
  ├── 第一轮解析 → 获取 model_name = "protenix-v2"
  ├── 深度合并模型专属配置 (configs_model_type.py)
  ├── 第二轮解析 → CLI 参数覆盖默认值
  └── AF3Trainer(configs).run()
```

### 配置合并优先级（由低到高）
1. `configs_base.py` — 基础/模型/优化器/损失/置换配置
2. `configs_data.py` — 数据集路径和采样配置
3. `configs_model_type.py` — 模型专属覆盖（如 protenix-v2）
4. CLI 参数 — 最高优先级

---

## 3. AF3Trainer 初始化（`runner/train.py:62`）

```
AF3Trainer.__init__()
  ├── init_env()
  │     ├── CUDA 设备设置 (cuda:{local_rank})
  │     ├── NCCL 分布式初始化 (world_size > 1 时)
  │     └── 随机种子 (rank-specific 或 deterministic)
  │
  ├── init_basics()
  │     └── 创建输出目录: {base_dir}/{run_name}_{timestamp}/
  │           ├── checkpoints/
  │           ├── predictions/
  │           ├── structures/
  │           ├── dumps/
  │           ├── errors/
  │           └── config.yaml
  │
  ├── init_log()
  │     └── wandb 初始化 (本脚本 use_wandb=false)
  │
  ├── init_model()
  │     ├── Protenix(configs).to(device)   — 构建模型 (~464M 参数)
  │     ├── DDP 包装 (world_size > 1 时)
  │     ├── EMAWrapper (ema_decay=0.999)
  │     ├── get_optimizer()                — AdamW, 支持分层学习率
  │     └── init_scheduler()
  │           └── FinetuneLRScheduler (检测到 finetune_params_with_substring 非空)
  │
  ├── init_loss()
  │     ├── ProtenixLoss                   — 多任务损失函数
  │     ├── SymmetricPermutation           — 对称置换对齐
  │     └── LDDTMetrics                    — lDDT 评估指标
  │
  ├── init_data()
  │     └── get_dataloaders() → train_dl + test_dls
  │
  └── try_load_checkpoint()
        ├── 加载 EMA checkpoint (load_ema_checkpoint_path) → 仅加载参数
        └── 加载标准 checkpoint (load_checkpoint_path) → 加载参数 (不加载 optim/scheduler/step)
```

---

## 4. 数据管线

### 4.1 数据集构建（`protenix/data/pipeline/dataset.py:1162`）

#### 训练集

| 数据集名称 | 采样权重 | mmCIF 路径 | 索引文件 | 用途 |
|-----------|---------|-----------|---------|------|
| `distill-sabdab` | 0.1 | `distill-sabdab_cif/` | `indices/distill-sabdab.csv` | 抗体-抗原蒸馏数据 |
| `sabdab-train` | 0.3 | `sabdab-train_cif/` | `indices/sabdab-train.csv` | 抗体结构数据 |
| `weightedPDB_before2109_wopb_nometalc_0925` | 0.6 | `mmcif/` | `indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz` | 通用 PDB 结构 |

训练集组装为 `WeightedMultiDataset`：
- 跨数据集按 `train_sample_weights` (0.1, 0.3, 0.6) 加权
- 数据集内部每条数据按 cluster 大小和分子类型加权（`calc_weights_for_df`）
- `epoch_size=10000`，有放回采样

#### 测试集

| 数据集名称 | 配置 |
|-----------|------|
| `posebusters_0925` | `max_n_token=512`, find_pocket=True |

测试集使用 `KeySumBalancedSampler` 按 token 数均衡分配

### 4.2 单条数据处理流程（`BaseSingleDataset.process_one()`）

```
process_one(idx)
  ├── 1. 读取样本索引 (pdb_id, chain, entity, mol_type 等)
  ├── 2. 加载 bioassembly 数据 (*.pkl.gz)
  │     ├── atom_array (原子坐标、链ID、实体ID 等)
  │     └── token_array (token 级别的标注)
  │
  ├── 3. 可选预处理
  │     ├── use_reference_chains_only → 过滤非参考链
  │     ├── shuffle_mols → 随机打乱分子顺序
  │     └── shuffle_sym_ids → 随机化对称链 ID
  │
  ├── 4. Cropping (3种方法按权重随机选)
  │     ├── ContiguousCropping (权重 0.2)
  │     ├── SpatialCropping (权重 0.4)
  │     └── SpatialInterfaceCropping (权重 0.4)
  │     └── crop_size = 384
  │
  ├── 5. Featurizer 提取特征
  │     ├── 输入特征: token_index, asym_id, ref_pos, atom_to_token_idx,
  │     │              ref_atom_to_token_idx, is_protein/ligand/rna/dna 等
  │     └── 标签: gt_pos, gt_atom_to_token_idx, resolution 等
  │
  ├── 6. MSA Featurizer (蛋白质 + RNA MSA 搜索)
  │
  ├── 7. Template Featurizer (同源模板搜索)
  │
  └── 8. 返回 {input_feature_dict, label_dict, label_full_dict, basic}
```

### 4.3 DataLoader

- 训练: `DistributedWeightedSampler` + `DistributedDataLoader`, batch_size=1, num_workers=16
- 测试: `KeySumBalancedSampler` + `DataLoader`, batch_size=1, num_workers=16

---

## 5. 模型架构

### Protenix-v2（~464M 参数）

基于 AlphaFold3 架构，以下参数为 protenix-v2 特有覆盖：

```
Protenix
  ├── InputEmbedder
  │     c_atom=128, c_atompair=16, c_token=384
  │
  ├── RelativePositionEncoding
  │     r_max=32, s_max=2, c_z=256
  │
  ├── TemplateEmbedder (n_blocks=2, c=64, c_z=256, hidden_scale_up=True)
  │
  ├── MSAModule (n_blocks=4, c_m=128, c_z=256, hidden_scale_up=True)
  │
  ├── Pairformer (n_blocks=48, c_z=256, c_s=384, n_heads=16, hidden_scale_up=True)
  │
  ├── DiffusionModule (c_z=256)
  │     ├── AtomEncoder (n_blocks=3, n_heads=4)
  │     ├── Transformer (n_blocks=24, n_heads=16)
  │     └── AtomDecoder (n_blocks=3, n_heads=4)
  │
  ├── ConfidenceHead (n_blocks=4, c_z=256, hidden_scale_up=True)
  │
  └── DistogramHead (c_z=256, no_bins=64)
```

### 关键架构参数
- `c_s` = 384 (single representation)
- `c_z` = 256 (pair representation, protenix-v2 特有)
- `N_cycle` = 4 (命令行覆盖，默认10)
- `diffusion_batch_size` = 48 (命令行覆盖，默认64)
- `N_step` = 20 (推理时扩散步数，命令行覆盖，默认200)

---

## 6. 训练循环（`AF3Trainer.run()`）

```
while step < max_steps (100000):
    for batch in train_dl:
        │
        ├── [前向传播] train_step(batch)
        │     ├── autocast(bf16) 混合精度
        │     ├── model_forward()
        │     │     └── model(input_feature_dict, label_dict, label_full_dict, mode="train")
        │     │           返回 pred_dict, label_dict, log_dict
        │     │
        │     ├── get_loss()
        │     │     └── ProtenixLoss:
        │     │           ├── diffusion_loss (MSE + bond + smooth_lddt)
        │     │           ├── confidence_loss (pLDDT + pDE + resolved + PAE)
        │     │           └── distogram_loss
        │     │
        │     ├── loss.backward() (支持梯度累积)
        │     ├── grad_clip_norm = 10
        │     ├── optimizer.step()
        │     └── lr_scheduler.step()
        │
        ├── [日志] 每 50 步 → 打印 train metrics + LR
        │
        ├── [保存] 每 100 步 → 保存 checkpoint + EMA checkpoint
        │
        ├── [评估] 每 400 步 → evaluate()
        │     ├── 标准模型评估
        │     └── EMA 模型评估 (apply_shadow → evaluate → restore)
        │     └── 在每个 test_dl 上:
        │           ├── model_forward (mode="eval")
        │           ├── get_loss (mode="eval")
        │           └── LDDTMetrics (lDDT + 聚合指标)
        │
        └── [EMA] 每有效步更新 EMA shadow weights
```

---

## 7. 损失函数（`ProtenixLoss`）

| 损失组件 | 权重 | 说明 |
|---------|------|------|
| `diffusion_mse` | 4.0 (×1/3) | 扩散坐标 MSE，DNA ×5, RNA ×5, Ligand ×10 |
| `diffusion_bond` | 4.0 (×0.0) | 键长约束 (本脚本 alpha_bond=0, 关闭) |
| `smooth_lddt` | 4.0 (×1.0) | 平滑 LDDT 损失 |
| `confidence_plddt` | 1e-4 | pLDDT 预测 |
| `confidence_pde` | 1e-4 | 预测距离误差 |
| `confidence_resolved` | 1e-4 | 分辨率预测 |
| `confidence_pae` | 0.0 | PAE 预测 (本脚本关闭) |
| `distogram` | 3e-2 | 距离直方图 |

---

## 8. 输入输出总结

### 输入

| 类型 | 路径 | 格式 |
|------|------|------|
| 预训练权重 | `$PROTENIX_ROOT_DIR/checkpoint/protenix-v2.pt` | .pt |
| mmCIF 结构 | `$PROTENIX_ROOT_DIR/mmcif/` | .cif |
| Bioassembly | `$PROTENIX_ROOT_DIR/mmcif_bioassembly/` | .pkl.gz |
| 蒸馏结构 | `$PROTENIX_ROOT_DIR/distill-sabdab_cif/` | .cif |
| SAbDab 结构 | `$PROTENIX_ROOT_DIR/sabdab-train_cif/` | .cif |
| MSA 数据 | `$PROTENIX_ROOT_DIR/mmcif_msa_template/` | MSA 目录 |
| RNA MSA | `$PROTENIX_ROOT_DIR/rna_msa/` | MSA 目录 |
| 索引文件 | `$PROTENIX_ROOT_DIR/indices/*.csv` | CSV |
| 测试集 | `$PROTENIX_ROOT_DIR/posebusters_mmcif/` | .cif |
| 组件词典 | `$PROTENIX_ROOT_DIR/common/components.cif` | .cif |

### 输出

输出根目录: `/vepfs-mlp2/project-abd/zhujianwei/share/klp/proteinx_output/protenix_finetune_sabdist0.1_sabtrain0.3_pdb0.6_lr1e-5_YYYYMMDD_HHMMSS/`

| 子目录/文件 | 内容 | 频率 |
|------------|------|------|
| `config.yaml` | 完整运行配置 | 开始时 1 次 |
| `checkpoints/{step}.pt` | 模型+优化器+调度器状态 | 每 100 步 |
| `checkpoints/{step}_ema_0.999.pt` | EMA 模型权重 | 每 100 步 |
| `predictions/` | 评估时的预测结果 | 每 400 步 |
| `structures/` | 预测的 3D 结构文件 | 每 400 步 |
| `dumps/` | 调试输出 | 按需 |
| `errors/` | 数据处理错误日志 | 出错时 |

---

## 9. 完整调用链路图

```
finetune_demo.sh
  └── torchrun ./runner/train.py [args]
        └── runner/train.py:main()
              ├── parse_sys_args()          protenix/config/config.py
              ├── parse_configs() × 2        protenix/config/config.py
              └── AF3Trainer(configs)
                    ├── init_env()           CUDA + NCCL + seed
                    ├── init_basics()        输出目录
                    ├── init_model()
                    │     ├── Protenix()     protenix/model/protenix.py
                    │     ├── DDP()          torch.nn.parallel
                    │     ├── EMAWrapper()   runner/ema.py
                    │     ├── get_optimizer() protenix/utils/training.py
                    │     └── FinetuneLRScheduler() protenix/utils/lr_scheduler.py
                    ├── init_loss()
                    │     ├── ProtenixLoss()      protenix/model/loss.py
                    │     ├── SymmetricPermutation() protenix/utils/permutation/
                    │     └── LDDTMetrics()       protenix/metrics/lddt_metrics.py
                    ├── init_data()
                    │     └── get_dataloaders()   protenix/data/pipeline/dataloader.py
                    │           └── get_datasets() protenix/data/pipeline/dataset.py
                    │                 ├── BaseSingleDataset() × N
                    │                 │     └── process_one()
                    │                 │           ├── DataPipeline.crop()
                    │                 │           ├── Featurizer()
                    │                 │           ├── MSAFeaturizer()
                    │                 │           └── TemplateFeaturizer()
                    │                 └── WeightedMultiDataset()
                    ├── try_load_checkpoint()
                    │     └── torch.load(protenix-v2.pt)
                    └── run()
                          ├── train_step() × N
                          ├── evaluate() 每 400 步
                          ├── save_checkpoint() 每 100 步
                          └── EMA update 每步
```
