# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **2026-05-08**: 去除纯核酸（pure nuc）数据输入 — `remove_pure_nuc` 配置项
  - 在 `BaseSingleDataset.read_indices_list()` 中新增纯核酸过滤逻辑
  - 链类型 `mol_1_type=="nuc"` 和界面类型 `mol_1_type==mol_2_type=="nuc"` 的样本被移除
  - 配置: `configs/configs_data.py` 中所有训练集 `base_info.remove_pure_nuc: True`
  - 验证: weightedPDB 移除 41,052 条纯核酸行（11,758 PDB），nuc_prot 混合界面保留
- **2026-05-08**: 裁剪时限制 nuc token 数量上限 — `max_nuc_tokens` 配置项
  - `CropData` 新增 `_get_nuc_mask()` 和 `_limit_nuc_tokens()` 方法
  - Spatial cropping: 保留距离 reference 最近的 N 个 nuc token
  - Contiguous cropping: 保留链上顺序前 N 个 nuc token
  - 所有非 nuc token（蛋白、配体）完全保留
  - 配置: `default_weighted_pdb_configs.cropping_configs.max_nuc_tokens: 128`
  - 验证: nuc 256→128 精确截断，prot token 100% 保留，8 GPU 训练管线正常
- Test cases for installation and compatibility issues by Shad Nygren (@ShadNygren)
  - `test_installation.py` - Tests for DeepSpeed/Pydantic compatibility (issue #182)
  - `test_triton_compatibility.py` - Tests for Triton GPU compatibility (issue #185)
  - `test_esm_loading.py` - Tests for ESM weights loading with PyTorch 2.6+ (issue #176)
- CHANGELOG.md to track project changes (Apache 2.0 compliance)

### Fixed
- Issue #182: DeepSpeed and Pydantic v2.x compatibility by upgrading to DeepSpeed 0.17.5
- Issue #185: Triton kernel compatibility with consumer GPUs (RTX 3090/4090)
  - Added PyTorch fallback when Triton is unavailable or unsupported
  - Enables Protenix to run on consumer hardware without code changes
  - Modified `protenix/model/tri_attention/__init__.py` to provide transparent fallback
- Inference robustness:
  - Fixed invalid MSA success assertion in FASTA flow (`runner/batch_inference.py`).
  - Added explicit input JSON validation for non-empty top-level list (`runner/inference.py`).
  - Hardened model-name parsing to avoid crashes on unexpected naming (`runner/inference.py`, `runner/batch_inference.py`).

### Contributors
- Shad Nygren, Virtual Hipster Corporation (@ShadNygren)

## [v0.6.1] - 2025-08-20

### Fixed
- ESM models loading in PyTorch 2.6 and newer versions (commit dbbee14)

## [v0.6.0] - Previous releases

See commit history for earlier changes.
