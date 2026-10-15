# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import time
import traceback
from bisect import bisect_right
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from protenix.data.constants import EvaluationChainInterface
from protenix.data.constraint.constraint_featurizer import ConstraintFeatureGenerator
from protenix.data.core.featurizer import Featurizer
from protenix.data.msa.msa_featurizer import (
    DIRECT_PROTEIN_UNPAIRED_MSA_PATH,
    MSAFeaturizer,
)
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.pipeline.distillation_runtime_index import (
    DistillationRuntimeIndex,
    structure_basename,
)
from protenix.data.pipeline.sample_context import (
    AUGMENTATION_DECISION_VERSION,
    MSA_MODE_CONFIGURED,
    MSA_MODE_QUERY_ONLY,
    MSA_MODE_SPECIES_PAIRED,
    MSA_MODE_UNPAIRED_SHUFFLE,
    MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
    QUERY_ONLY_CHAIN_SELECTION_STREAM,
    SampleOccurrence,
    VIRTUAL_SEGMENT_MSA_SHUFFLE_STREAM,
    deterministic_occurrence_bernoulli,
    deterministic_occurrence_categorical,
    deterministic_occurrence_seed,
    split_sample_index,
)
from protenix.data.template.template_featurizer import (
    DummyTemplateFeaturizer,
    SEQUENCE_TEMPLATE_CACHE_ERROR_REASONS,
    SequenceTemplateCacheError,
    TemplateFeaturizer,
)
from protenix.data.tokenizer import TokenArray
from protenix.data.utils import (
    data_type_transform,
    get_antibody_clusters,
    make_dummy_feature,
)
from protenix.utils.cropping import CropData
from protenix.utils.file_io import read_indices_csv
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import dict_to_tensor

logger = get_logger(__name__)


def _strict_bool_option(
    options: dict[str, Any],
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean dataset option without accepting truthy strings."""
    value = options.get(name, default)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a boolean, got {type(value).__name__}: {value!r}"
        )
    return bool(value)


def _strict_nonnegative_int_option(
    options: dict[str, Any],
    name: str,
    default: int = 0,
) -> int:
    """Read a non-negative integer option without accepting bools/floats."""
    value = options.get(name, default)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(
            f"{name} must be a non-negative integer, got "
            f"{type(value).__name__}: {value!r}"
        )
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _diffusion_supervised_token_mask(
    token_array: TokenArray,
    atom_array: AtomArray,
    *,
    protein_only: bool,
) -> torch.Tensor:
    """Return tokens containing an atom used by diffusion coordinate losses."""
    eligible_atoms = np.asarray(atom_array.is_resolved, dtype=bool)
    if protein_only:
        eligible_atoms = np.logical_and(
            eligible_atoms,
            np.asarray(atom_array.is_protein, dtype=bool),
        )
    eligible_tokens = np.fromiter(
        (
            bool(
                eligible_atoms[
                    np.asarray(token.atom_indices, dtype=np.int64)
                ].any()
            )
            for token in token_array
        ),
        dtype=bool,
        count=len(token_array),
    )
    return torch.from_numpy(eligible_tokens)


def _resolved_protein_token_mask(
    token_array: TokenArray,
    atom_array: AtomArray,
) -> torch.Tensor:
    """Return tokens containing at least one resolved protein atom."""
    return _diffusion_supervised_token_mask(
        token_array=token_array,
        atom_array=atom_array,
        protein_only=True,
    )


def _count_diffusion_supervised_feature_tokens(
    features: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    *,
    protein_only: bool,
) -> int:
    """Count final model-input tokens carrying diffusion coordinate labels."""
    coordinate_mask = labels["coordinate_mask"].bool().flatten()
    atom_to_token_idx = features["atom_to_token_idx"].long().flatten()
    if coordinate_mask.shape != atom_to_token_idx.shape:
        raise ValueError(
            "coordinate_mask and atom_to_token_idx must have the same shape: "
            f"{tuple(coordinate_mask.shape)} versus "
            f"{tuple(atom_to_token_idx.shape)}"
        )
    if protein_only:
        is_protein = features["is_protein"].bool().flatten()
        if is_protein.shape != coordinate_mask.shape:
            raise ValueError(
                "is_protein and coordinate_mask must have the same shape: "
                f"{tuple(is_protein.shape)} versus "
                f"{tuple(coordinate_mask.shape)}"
            )
        coordinate_mask = torch.logical_and(
            coordinate_mask,
            is_protein,
        )
    if not coordinate_mask.any():
        return 0
    return int(torch.unique(atom_to_token_idx[coordinate_mask]).numel())


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels, the keys and the shape are defined in protenix.data.utils
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[TemplateFeaturizer] = None,
        name: str = None,
        source_roots_fpath: Optional[Union[str, Path]] = None,
        runtime_index_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.bioassembly_relpath_column = kwargs.get(
            "bioassembly_relpath_column"
        )
        if (
            self.bioassembly_relpath_column is not None
            and self.bioassembly_dict_dir is None
        ):
            raise ValueError(
                "bioassembly_relpath_column requires bioassembly_dict_dir"
            )
        self.indices_fpath = indices_fpath
        self.runtime_distillation_index = (
            DistillationRuntimeIndex(runtime_index_dir)
            if runtime_index_dir is not None
            else None
        )
        if self.runtime_distillation_index is not None:
            runtime_roots_path = self.runtime_distillation_index.roots_path
            if source_roots_fpath is not None:
                configured_roots_path = (
                    Path(source_roots_fpath).expanduser().resolve(strict=True)
                )
                if configured_roots_path != runtime_roots_path:
                    raise ValueError(
                        "runtime_index_dir cannot be combined with a different "
                        "source_roots_fpath: "
                        f"{configured_roots_path} != {runtime_roots_path}"
                    )
            self.source_roots_fpath = runtime_roots_path
            self.source_roots = self.runtime_distillation_index.source_roots
        else:
            self.source_roots_fpath = source_roots_fpath
            self.source_roots = self._load_source_roots(source_roots_fpath)
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)
        self.disable_non_protein_diffusion_weights = _strict_bool_option(
            kwargs,
            "disable_non_protein_diffusion_weights",
        )
        self.require_resolved_protein_crop = _strict_bool_option(
            kwargs,
            "require_resolved_protein_crop",
        )
        self.min_diffusion_supervised_tokens = (
            _strict_nonnegative_int_option(
                kwargs,
                "min_diffusion_supervised_tokens",
            )
        )
        configured_crop_size = int(
            self.cropping_configs.get("crop_size", -1)
        )
        if (
            configured_crop_size > 0
            and self.min_diffusion_supervised_tokens
            > configured_crop_size
        ):
            raise ValueError(
                "min_diffusion_supervised_tokens cannot exceed crop_size: "
                f"{self.min_diffusion_supervised_tokens} > "
                f"{configured_crop_size}"
            )
        if self.disable_non_protein_diffusion_weights and not self.is_distillation:
            raise ValueError(
                "disable_non_protein_diffusion_weights requires "
                "is_distillation=True"
            )
        self.query_only_msa_probability = float(
            kwargs.get("query_only_msa_probability", 0.0)
        )
        self.pair_msa_as_unpaired_probability = float(
            kwargs.get("pair_msa_as_unpaired_probability", 0.0)
        )
        self.template_use_probability = float(
            kwargs.get("template_use_probability", 1.0)
        )
        self.augmentation_decision_version = str(
            kwargs.get(
                "augmentation_decision_version",
                AUGMENTATION_DECISION_VERSION,
            )
        )
        for name, probability in (
            ("query_only_msa_probability", self.query_only_msa_probability),
            (
                "pair_msa_as_unpaired_probability",
                self.pair_msa_as_unpaired_probability,
            ),
            ("template_use_probability", self.template_use_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")
        msa_gate_probability = (
            self.query_only_msa_probability
            + self.pair_msa_as_unpaired_probability
        )
        if msa_gate_probability > 1.0:
            raise ValueError(
                "query_only_msa_probability + "
                "pair_msa_as_unpaired_probability must be <= 1, got "
                f"{msa_gate_probability}"
            )
        if not self.augmentation_decision_version.strip():
            raise ValueError("augmentation_decision_version must be non-empty")

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.min_n_token = kwargs.get("min_n_token", -1)
        self.pdb_list = kwargs.get("pdb_list", None)
        if self.pdb_list is not None and len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.remove_pure_nuc = kwargs.get("remove_pure_nuc", False)
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test
        # Configs for constraint
        self.constraint = kwargs.get("constraint", {})
        if self.constraint.get("enable", False):
            logger.info(f"[{self.name}] constraint config: {self.constraint}")
            # Do not rely on new files for users who do not use constraint feature
            self.ab_top2_clusters = get_antibody_clusters()
            self.constraint_generator = ConstraintFeatureGenerator(
                self.constraint, self.ab_top2_clusters
            )

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer

        # Read data
        if self.runtime_distillation_index is None:
            self.indices_list = self.read_indices_list(indices_fpath)
        else:
            if not self.is_distillation:
                raise ValueError(
                    "runtime_index_dir requires is_distillation=True"
                )
            if self.random_sample_if_failed:
                raise ValueError(
                    "runtime distillation must fail closed; "
                    "random_sample_if_failed must be False"
                )
            incompatible = {
                "group_by_pdb_id": self.group_by_pdb_id,
                "sort_by_n_token": self.sort_by_n_token,
                "find_eval_chain_interface": self.find_eval_chain_interface,
                "pdb_list": self.pdb_list is not None,
                "max_n_token": self.max_n_token > 0,
                "min_n_token": self.min_n_token > 0,
                "exclusion": bool(self.exclusion_dict),
                "limits": self.limits > 0,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "runtime distillation does not support dataframe filters: "
                    + ", ".join(enabled)
                )
            self.indices_list = self.runtime_distillation_index
            logger.info(
                "[%s] runtime distillation index: rows=%d clusters=%d dir=%s",
                self.name,
                len(self.runtime_distillation_index),
                self.runtime_distillation_index.cluster_count,
                self.runtime_distillation_index.runtime_dir,
            )

    @staticmethod
    def _load_source_roots(
        source_roots_fpath: Optional[Union[str, Path]],
    ) -> dict[str, Path]:
        """Load an optional ``root_id -> source directory`` mapping.

        The mapping is only used by row-addressed distillation datasets.  An
        empty mapping preserves the legacy flat ``bioassembly_dict_dir``
        lookup exactly.
        """
        if source_roots_fpath is None:
            return {}

        roots_path = Path(source_roots_fpath).expanduser()
        with roots_path.open("r", encoding="utf-8") as f:
            document = json.load(f)
        records = document.get("roots")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"{roots_path} must contain a non-empty 'roots' list"
            )

        roots: dict[str, Path] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(
                    f"{roots_path} contains a non-object roots entry"
                )
            root_id = str(record.get("root_id", "")).strip()
            root_value = record.get("path")
            if not root_id or not isinstance(root_value, str) or not root_value:
                raise ValueError(
                    f"{roots_path} roots entries require root_id and path"
                )
            if root_id in roots:
                raise ValueError(
                    f"{roots_path} contains duplicate root_id {root_id!r}"
                )
            root_path = Path(root_value).expanduser()
            if not root_path.is_absolute():
                root_path = roots_path.parent / root_path
            roots[root_id] = root_path
        return roots

    def _resolve_row_addressed_distillation_paths(
        self, sample_indice: pd.Series
    ) -> tuple[Path, Path]:
        """Resolve one catalog row to its nested bioassembly and direct A3M."""
        runtime_index = getattr(self, "runtime_distillation_index", None)
        if runtime_index is not None:
            if "row_id" not in sample_indice or pd.isna(sample_indice["row_id"]):
                raise ValueError("Runtime distillation sample is missing row_id")
            paths = runtime_index.resolve_paths(int(sample_indice["row_id"]))
            return paths.bioassembly_path, paths.a3m_path

        missing = [
            field
            for field in ("source_root_id", "cif_relpath")
            if field not in sample_indice or pd.isna(sample_indice[field])
        ]
        if missing:
            raise ValueError(
                "source_roots_fpath requires catalog columns: "
                + ", ".join(missing)
            )

        root_id = str(sample_indice["source_root_id"]).strip()
        if root_id not in self.source_roots:
            raise KeyError(
                f"Unknown source_root_id {root_id!r}; configured roots are "
                f"{sorted(self.source_roots)}"
            )

        cif_relpath = PurePosixPath(str(sample_indice["cif_relpath"]).strip())
        if (
            cif_relpath.is_absolute()
            or ".." in cif_relpath.parts
            or not (
                cif_relpath.name.lower().endswith(".cif")
                or cif_relpath.name.lower().endswith(".cif.gz")
            )
        ):
            raise ValueError(
                f"Unsafe or non-CIF cif_relpath: {str(cif_relpath)!r}"
            )

        cif_path = self.source_roots[root_id].joinpath(*cif_relpath.parts)
        basename = structure_basename(cif_relpath)
        bioassembly_path = cif_path.parent / "bioassembly" / f"{basename}.pkl.gz"
        return bioassembly_path, cif_path.parent / f"{basename}.a3m"

    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs if the input is valid, otherwise None.
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        self.check_indices_list(indices_list, "initial loading")
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")
            self.check_indices_list(indices_list, "pdb_list filtering")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"max_n_token ({self.max_n_token}) filtering"
            )

        if self.min_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) >= self.min_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by min_n_token ({self.min_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"min_n_token ({self.min_n_token}) filtering"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"exclusion_dict ({col_name}) filtering"
            )

        # Filter pure nucleic acid data: keep only samples with at least one protein or ligand
        if self.remove_pure_nuc:
            cond_chain_nuc = (indices_list["type"] == "chain") & (
                indices_list["mol_1_type"] == "nuc"
            )
            cond_interface_nuc_nuc = (
                (indices_list["type"] == "interface")
                & (indices_list["mol_1_type"] == "nuc")
                & (indices_list["mol_2_type"] == "nuc")
            )
            remove_mask = cond_chain_nuc | cond_interface_nuc_nuc
            removed_list = indices_list[remove_mask]
            indices_list = indices_list[~remove_mask].reset_index(drop=True)
            logger.info(
                f"[removed pure nuc] #Rows: {len(removed_list)}"
            )
            if len(removed_list) > 0:
                logger.info(
                    f"[removed pure nuc] #PDB: {removed_list['pdb_id'].nunique()}"
                )
            logger.info(
                f"[filtered by remove_pure_nuc] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(indices_list, "remove_pure_nuc filtering")

        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                # Vectorized equivalent of:
                #   indices_list[indices_list["eval_type"].apply(lambda x: x in EvaluationChainInterface)]
                # .isin() checks each element of the Series against the set, same semantics as the lambda.
                indices_list = indices_list[
                    indices_list["eval_type"].isin(EvaluationChainInterface)
                ]
            self.check_indices_list(indices_list, "find_eval_chain_interface filtering")
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
            self.check_indices_list(indices_list, "limits filtering")
        return indices_list

    def check_indices_list(self, indices_list, step_name: str = ""):
        if len(indices_list) == 0:
            msg = "After filtering, the dataset is empty."
            if step_name:
                msg = f"After {step_name}, the dataset is empty."
            raise ValueError(msg)

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        col1 = df["mol_1_type"].astype(str)
        col2 = df["mol_2_type"].astype(str).str.replace("nan", "intra", regex=False)
        lo = np.where(col1.values <= col2.values, col1.values, col2.values)
        hi = np.where(col1.values <= col2.values, col2.values, col1.values)
        lo = np.asarray(lo, dtype=str)
        hi = np.asarray(hi, dtype=str)
        df["mol_group_type"] = np.char.add(np.char.add(lo, "_"), hi)

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    def __getitem__(self, idx: Union[int, SampleOccurrence]):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """
        idx, occurrence = split_sample_index(idx)
        msa_mode = self._msa_mode_decision(occurrence)
        template_use_requested = self._augmentation_decision(
            occurrence,
            probability=self.template_use_probability,
            stream_name="template_use",
        )

        # Try at most 10 times. The occurrence-level augmentation decisions are
        # intentionally kept fixed if a broken row is replaced within this
        # dataset.
        last_error = None
        for _ in range(10):
            try:
                data = self.process_one(
                    idx,
                    msa_mode=msa_mode,
                    template_use_requested=template_use_requested,
                    sample_occurrence=occurrence,
                )
                return data
            except Exception as e:
                last_error = e
                error_message = f"{e} at idx {idx}:\n{traceback.format_exc()}"
                self.save_error_data(idx, error_message)

                # A required offline cache is a dataset-level contract, not a
                # broken-row condition. Never hide missing/corrupt cache files
                # by replacing the target with a random sample.
                if isinstance(e, SequenceTemplateCacheError):
                    logger.exception(
                        f"[fatal sequence template cache error] {error_message}"
                    )
                    raise
                if self.random_sample_if_failed:
                    logger.exception(f"[skip data {idx}] {error_message}")
                    # Random sample an index
                    idx = random.choice(range(len(self.indices_list)))
                    continue
                else:
                    raise Exception(e)
        raise RuntimeError(
            "failed to produce a valid sample after 10 replacement attempts"
        ) from last_error

    def _augmentation_decision(
        self,
        occurrence: Optional[SampleOccurrence],
        *,
        probability: float,
        stream_name: str,
    ) -> bool:
        return deterministic_occurrence_bernoulli(
            occurrence,
            probability=probability,
            stream_name=stream_name,
            algorithm_version=self.augmentation_decision_version,
        )

    def _msa_mode_decision(
        self,
        occurrence: Optional[SampleOccurrence],
    ) -> str:
        query_probability = self.query_only_msa_probability
        unpaired_probability = self.pair_msa_as_unpaired_probability
        if query_probability == 0.0 and unpaired_probability == 0.0:
            return MSA_MODE_CONFIGURED
        return deterministic_occurrence_categorical(
            occurrence,
            probabilities=(
                (MSA_MODE_QUERY_ONLY, query_probability),
                (MSA_MODE_UNPAIRED_SHUFFLE, unpaired_probability),
                (
                    MSA_MODE_SPECIES_PAIRED,
                    1.0 - query_probability - unpaired_probability,
                ),
            ),
            stream_name="msa_mode",
            algorithm_version=self.augmentation_decision_version,
        )

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sample_indice = self._get_sample_indice(idx=idx)
        direct_protein_unpaired_msa_fpath = None
        if self.source_roots:
            (
                bioassembly_dict_fpath,
                direct_protein_unpaired_msa_fpath,
            ) = self._resolve_row_addressed_distillation_paths(sample_indice)
        elif self.bioassembly_relpath_column is not None:
            column = str(self.bioassembly_relpath_column)
            if column not in sample_indice or pd.isna(sample_indice[column]):
                raise ValueError(
                    f"Missing bioassembly relative path column {column!r}"
                )
            relative_path = PurePosixPath(
                str(sample_indice[column]).strip()
            )
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or not str(relative_path).endswith(".pkl.gz")
            ):
                raise ValueError(
                    "Unsafe bioassembly relative path: "
                    f"{str(relative_path)!r}"
                )
            bioassembly_dict_fpath = Path(
                self.bioassembly_dict_dir
            ).joinpath(*relative_path.parts)
        elif self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        if direct_protein_unpaired_msa_fpath is not None:
            expected_root_id = str(sample_indice["source_root_id"])
            expected_cif_relpath = str(sample_indice["cif_relpath"])
            stored_root_id = bioassembly_dict.get("source_root_id")
            stored_cif_relpath = bioassembly_dict.get("source_cif_relpath")
            if (
                stored_root_id is not None
                and str(stored_root_id) != expected_root_id
            ):
                raise ValueError(
                    "Bioassembly source_root_id does not match runtime index: "
                    f"{stored_root_id!r} != {expected_root_id!r}"
                )
            if (
                stored_cif_relpath is not None
                and str(stored_cif_relpath) != expected_cif_relpath
            ):
                raise ValueError(
                    "Bioassembly source_cif_relpath does not match runtime "
                    f"index: {stored_cif_relpath!r} != "
                    f"{expected_cif_relpath!r}"
                )
            bioassembly_dict[DIRECT_PROTEIN_UNPAIRED_MSA_PATH] = str(
                direct_protein_unpaired_msa_fpath
            )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        entity_pair_to_chain_pairs = bioassembly_dict.get("entity_pair_to_chain_pairs")
        if sample_indice["type"] == "interface" and entity_pair_to_chain_pairs:
            entity1 = sample_indice.entity_1_id
            entity2 = sample_indice.entity_2_id
            chain_pairs = entity_pair_to_chain_pairs.get((entity1, entity2))
            chain1, chain2 = random.choice(chain_pairs)
            new_sample_indice = deepcopy(sample_indice)
            new_sample_indice["chain_1_id"] = chain1
            new_sample_indice["chain_2_id"] = chain2
            return new_sample_indice, bioassembly_dict, bioassembly_dict_fpath
        else:
            return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        In experiments conducted to observe overfitting effects using training sets,
        the pre-stored AtomArray in the training set may experience issues with discontinuous chain IDs due to filtering.
        Consequently, a temporary patch has been implemented to resolve this issue.

        e.g. 3x6u asym_id_int = [0, 1, 2, ... 18, 20] -> reassigned_asym_id_int [0, 1, 2, ..., 18, 19]
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            return np.searchsorted(array_uniq, array)

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array.
        Atoms/tokens with the same mol_id will be shuffled as a integrated component.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        (
            token_array,
            atom_array,
        ) = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random sym_id for chains of the same entity_id
        e.g.
        when entity_id = 0
            sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
        when entity_id = 1
            sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            indices = np.searchsorted(x_unique, x)
            return x_shuffled[indices].copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    def process_one(
        self,
        idx: int,
        return_atom_token_array: bool = False,
        *,
        query_only_msa: bool = False,
        msa_mode: Optional[str] = None,
        template_use_requested: bool = True,
        sample_occurrence: Optional[SampleOccurrence] = None,
    ) -> dict[str, dict]:
        """
        Processes a single data sample by retrieving bioassembly data, applying various transformations, and cropping the data.
        It then extracts features and labels, and optionally returns the processed atom and token arrays.

        Args:
            idx: The index of the data sample to process.
            return_atom_token_array: Whether to return the processed atom and token arrays.

        Returns:
            A dict containing the input features, labels, basic_info and optionally the processed atom and token arrays.
        """
        process_start = time.perf_counter()
        if msa_mode is None:
            msa_mode = (
                MSA_MODE_QUERY_ONLY if query_only_msa else MSA_MODE_CONFIGURED
            )
        valid_msa_modes = {
            MSA_MODE_CONFIGURED,
            MSA_MODE_QUERY_ONLY,
            MSA_MODE_UNPAIRED_SHUFFLE,
            MSA_MODE_SPECIES_PAIRED,
        }
        if msa_mode not in valid_msa_modes:
            raise ValueError(f"unsupported msa_mode: {msa_mode!r}")
        if query_only_msa and msa_mode != MSA_MODE_QUERY_ONLY:
            raise ValueError(
                "query_only_msa=True conflicts with "
                f"msa_mode={msa_mode!r}"
            )
        query_only_msa = msa_mode == MSA_MODE_QUERY_ONLY
        pair_msa_as_unpaired_only = (
            True if msa_mode == MSA_MODE_UNPAIRED_SHUFFLE else None
        )
        shuffle_pair_msa_when_unpaired_only = (
            True if msa_mode == MSA_MODE_UNPAIRED_SHUFFLE else None
        )
        if msa_mode == MSA_MODE_SPECIES_PAIRED:
            pair_msa_as_unpaired_only = False
            shuffle_pair_msa_when_unpaired_only = False
        elif msa_mode == MSA_MODE_QUERY_ONLY:
            pair_msa_as_unpaired_only = False
            shuffle_pair_msa_when_unpaired_only = False
        effective_pair_as_unpaired = (
            bool(getattr(self.msa_featurizer, "pair_msa_as_unpaired_only", False))
            if pair_msa_as_unpaired_only is None
            else pair_msa_as_unpaired_only
        )
        effective_shuffle = (
            bool(
                getattr(
                    self.msa_featurizer,
                    "shuffle_pair_msa_when_unpaired_only",
                    False,
                )
            )
            if shuffle_pair_msa_when_unpaired_only is None
            else shuffle_pair_msa_when_unpaired_only
        )
        if query_only_msa:
            effective_pair_as_unpaired = False
            effective_shuffle = False
        virtual_segment_shuffle_seed = (
            deterministic_occurrence_seed(
                sample_occurrence,
                stream_name=VIRTUAL_SEGMENT_MSA_SHUFFLE_STREAM,
                algorithm_version=self.augmentation_decision_version,
            )
            if (
                msa_mode == MSA_MODE_UNPAIRED_SHUFFLE
                and sample_occurrence is not None
            )
            else None
        )
        msa_shuffle_variant_seed = (
            deterministic_occurrence_seed(
                sample_occurrence,
                stream_name=MULTICHAIN_MSA_SHUFFLE_VARIANT_STREAM,
                algorithm_version=self.augmentation_decision_version,
            )
            if (
                effective_pair_as_unpaired
                and effective_shuffle
                and sample_occurrence is not None
                and bool(
                    getattr(
                        self.msa_featurizer,
                        "enable_multichain_msa_shuffle_variant_sampling",
                        False,
                    )
                )
            )
            else None
        )
        query_only_chain_selection_seed = (
            deterministic_occurrence_seed(
                sample_occurrence,
                stream_name=QUERY_ONLY_CHAIN_SELECTION_STREAM,
                algorithm_version=self.augmentation_decision_version,
            )
            if (
                msa_mode == MSA_MODE_QUERY_ONLY
                and sample_occurrence is not None
            )
            else None
        )

        bioassembly_start = time.perf_counter()
        (
            sample_indice,
            bioassembly_dict,
            bioassembly_dict_fpath,
        ) = self._get_bioassembly_data(idx=idx)
        bioassembly_load_seconds = time.perf_counter() - bioassembly_start

        if self.use_reference_chains_only:
            # Get the reference chains
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            # Remove other chains from the bioassembly_dict
            # Remove them safely using the crop method
            token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
            )
            token_chain_id = bioassembly_dict["atom_array"][
                token_centre_atom_indices
            ].chain_id
            is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
            (
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
            ) = CropData.select_by_token_indices(
                token_array=bioassembly_dict["token_array"],
                atom_array=bioassembly_dict["atom_array"],
                selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
            )

        if self.shuffle_mols:
            (
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
            ) = self._shuffle_array_based_on_mol_id(
                token_array=bioassembly_dict["token_array"],
                atom_array=bioassembly_dict["atom_array"],
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        max_entity_mol_id = bioassembly_dict["atom_array"].entity_mol_id.max()

        # Crop
        crop_start = time.perf_counter()
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            query_only_msa=query_only_msa,
            pair_msa_as_unpaired_only=pair_msa_as_unpaired_only,
            shuffle_pair_msa_when_unpaired_only=(
                shuffle_pair_msa_when_unpaired_only
            ),
            virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
            msa_shuffle_variant_seed=msa_shuffle_variant_seed,
            query_only_chain_selection_seed=(
                query_only_chain_selection_seed
            ),
            template_use_requested=template_use_requested,
            **self.cropping_configs,
        )
        crop_seconds = time.perf_counter() - crop_start

        feature_label_start = time.perf_counter()
        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
            max_entity_mol_id=max_entity_mol_id,
        )
        feature_label_seconds = time.perf_counter() - feature_label_start
        diffusion_supervised_token_count = (
            _count_diffusion_supervised_feature_tokens(
                features=feat,
                labels=label,
                protein_only=self.disable_non_protein_diffusion_weights,
            )
        )
        if (
            diffusion_supervised_token_count
            < self.min_diffusion_supervised_tokens
        ):
            raise RuntimeError(
                "final feature/label tensors have too few "
                "diffusion-supervised tokens: "
                f"{diffusion_supervised_token_count} < "
                f"{self.min_diffusion_supervised_tokens}"
            )

        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_diffusion_supervised_token": torch.tensor(
                [diffusion_supervised_token_count],
                dtype=torch.int64,
            ),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
            "msa_mode": msa_mode,
            "query_only_msa": torch.tensor([query_only_msa], dtype=torch.bool),
            "msa_query_only": torch.tensor([query_only_msa], dtype=torch.bool),
            "unpaired_msa": torch.tensor(
                [msa_mode == MSA_MODE_UNPAIRED_SHUFFLE],
                dtype=torch.bool,
            ),
            "pair_msa_as_unpaired_only": torch.tensor(
                [effective_pair_as_unpaired],
                dtype=torch.bool,
            ),
            "shuffle_pair_msa_when_unpaired_only": torch.tensor(
                [effective_shuffle],
                dtype=torch.bool,
            ),
            "template_use_requested": torch.tensor(
                [template_use_requested], dtype=torch.bool
            ),
            "template_dropout_applied": torch.tensor(
                [not template_use_requested], dtype=torch.bool
            ),
            "N_template_valid": torch.tensor(
                [self._count_valid_templates(cropped_template_features)],
                dtype=torch.int64,
            ),
            "bioassembly_load_seconds": bioassembly_load_seconds,
            "crop_seconds": crop_seconds,
            "feature_label_seconds": feature_label_seconds,
        }
        if self.msa_featurizer is not None:
            msa_profile = self.msa_featurizer.get_last_profile()
            basic_info["msa_featurizer_seconds"] = float(
                msa_profile.get("elapsed_seconds", 0.0)
            )
            basic_info["N_msa_virtual_segment"] = torch.tensor(
                [int(msa_profile.get("virtual_segment_count", 0))],
                dtype=torch.int64,
            )
            basic_info["msa_virtual_segment_shuffle_applied"] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "virtual_segment_shuffle_applied",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info["N_msa_shuffle_group"] = torch.tensor(
                [int(msa_profile.get("msa_shuffle_group_count", 0))],
                dtype=torch.int64,
            )
            basic_info["msa_shuffle_requested_variant_code"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "msa_shuffle_requested_variant_code",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["msa_shuffle_variant_eligible"] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "msa_shuffle_variant_eligible",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info["msa_shuffle_variant_applied"] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "msa_shuffle_variant_applied",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info[
                "msa_shuffle_whole_physical_chain_applied"
            ] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "msa_shuffle_whole_physical_chain_applied",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info["N_msa_crop_protein_chain"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "msa_shuffle_crop_protein_chain_count",
                            msa_profile.get(
                                "query_only_crop_protein_chain_count",
                                0,
                            ),
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["N_msa_query_only_protein_chain"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "query_only_selected_protein_chain_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["N_msa_retained_protein_chain"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "query_only_retained_protein_chain_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["msa_partial_query_only_applied"] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "query_only_partial_applied",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info["msa_global_query_only_applied"] = torch.tensor(
                [
                    bool(
                        msa_profile.get(
                            "query_only_global_applied",
                            False,
                        )
                    )
                ],
                dtype=torch.bool,
            )
            basic_info[
                "N_openfold3_representative_query_mismatch"
            ] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "openfold3_representative_query_mismatch_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["N_openfold3_representative_remap"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "openfold3_representative_remap_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info[
                "N_openfold3_representative_query_only_fallback"
            ] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "openfold3_representative_query_only_fallback_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
            basic_info["N_openfold3_explicit_query_only"] = torch.tensor(
                [
                    int(
                        msa_profile.get(
                            "openfold3_explicit_query_only_chain_count",
                            0,
                        )
                    )
                ],
                dtype=torch.int64,
            )
        if self.template_featurizer is not None:
            template_profile = self.template_featurizer.get_last_profile()
            basic_info["template_featurizer_seconds"] = float(
                template_profile.get("elapsed_seconds", 0.0)
            )
            template_phase_seconds = {
                phase: sum(
                    float(chain_profile.get("time_track", {}).get(phase, 0.0))
                    for chain_profile in template_profile.get("chain_profiles", [])
                )
                for phase in ("load", "align", "feature")
            }
            basic_info.update(
                {
                    "template_load_seconds": template_phase_seconds["load"],
                    "template_align_seconds": template_phase_seconds["align"],
                    "template_feature_seconds": template_phase_seconds["feature"],
                    "sequence_template_cache_load_seconds": float(
                        template_profile.get(
                            "sequence_template_cache_load_seconds",
                            0.0,
                        )
                    ),
                    "sequence_template_cache_files_loaded": int(
                        template_profile.get(
                            "sequence_template_cache_files_loaded",
                            0,
                        )
                    ),
                    "sequence_template_cache_candidates": int(
                        template_profile.get(
                            "sequence_template_cache_candidates",
                            0,
                        )
                    ),
                    "sequence_template_cache_selected": int(
                        template_profile.get(
                            "sequence_template_cache_selected",
                            0,
                        )
                    ),
                    "sequence_template_cache_reuses": int(
                        template_profile.get(
                            "sequence_template_cache_reuses",
                            0,
                        )
                    ),
                    "N_sequence_template_cache_chain_dummy_fallback_unique_sequence": torch.tensor(
                        [
                            int(
                                template_profile.get(
                                    "sequence_template_cache_chain_dummy_fallback_unique_sequence_count",
                                    0,
                                )
                            )
                        ],
                        dtype=torch.int64,
                    ),
                    "N_sequence_template_cache_chain_dummy_fallback_chain": torch.tensor(
                        [
                            int(
                                template_profile.get(
                                    "sequence_template_cache_chain_dummy_fallback_affected_chain_count",
                                    0,
                                )
                            )
                        ],
                        dtype=torch.int64,
                    ),
                }
            )
            fallback_reason_counts = template_profile.get(
                "sequence_template_cache_chain_dummy_fallback_reason_counts",
                {},
            )
            for reason in SEQUENCE_TEMPLATE_CACHE_ERROR_REASONS:
                basic_info[
                    "N_sequence_template_cache_chain_dummy_fallback_reason_"
                    f"{reason}"
                ] = torch.tensor(
                    [int(fallback_reason_counts.get(reason, 0))],
                    dtype=torch.int64,
                )
        if sample_occurrence is not None:
            basic_info.update(
                {
                    "sample_sampler_epoch": torch.tensor(
                        [sample_occurrence.sampler_epoch], dtype=torch.int64
                    ),
                    "sample_global_position": torch.tensor(
                        [sample_occurrence.global_position], dtype=torch.int64
                    ),
                }
            )

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }
        basic_info["data_process_seconds"] = time.perf_counter() - process_start

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array
        return data

    @staticmethod
    def _count_valid_templates(template_features: dict[str, Any]) -> int:
        atom_mask = template_features.get("template_atom_mask")
        if atom_mask is None:
            return 0
        atom_mask = np.asarray(atom_mask)
        if atom_mask.ndim < 2 or atom_mask.shape[0] == 0:
            return 0
        valid_axes = tuple(range(1, atom_mask.ndim))
        return int(np.any(atom_mask.astype(bool), axis=valid_axes).sum())

    def crop(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
        max_nuc_tokens: int = -1,
        query_only_msa: bool = False,
        pair_msa_as_unpaired_only: Optional[bool] = None,
        shuffle_pair_msa_when_unpaired_only: Optional[bool] = None,
        virtual_segment_shuffle_seed: Optional[int] = None,
        query_only_chain_selection_seed: Optional[int] = None,
        template_use_requested: bool = True,
        msa_shuffle_variant_seed: Optional[int] = None,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        required_token_mask = None
        min_required_tokens = 1
        diffusion_supervised_token_mask = None
        if self.min_diffusion_supervised_tokens > 0:
            diffusion_supervised_token_mask = (
                _diffusion_supervised_token_mask(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    protein_only=(
                        self.disable_non_protein_diffusion_weights
                    ),
                )
            )
        if self.require_resolved_protein_crop:
            required_token_mask = (
                diffusion_supervised_token_mask
                if (
                    self.disable_non_protein_diffusion_weights
                    and diffusion_supervised_token_mask is not None
                )
                else _resolved_protein_token_mask(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                )
            )
            # Keep the protein-crop constraint at least as strong as the
            # configured diffusion minimum even when a future run supervises
            # all molecule types. This avoids satisfying K with ligand/nucleic
            # tokens while retaining only one protein token.
            min_required_tokens = max(
                1,
                self.min_diffusion_supervised_tokens,
            )

        crop_result = DataPipeline.crop(
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
            max_nuc_tokens=max_nuc_tokens,
            query_only_msa=query_only_msa,
            pair_msa_as_unpaired_only=pair_msa_as_unpaired_only,
            shuffle_pair_msa_when_unpaired_only=(
                shuffle_pair_msa_when_unpaired_only
            ),
            virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
            msa_shuffle_variant_seed=msa_shuffle_variant_seed,
            query_only_chain_selection_seed=(
                query_only_chain_selection_seed
            ),
            template_use_requested=template_use_requested,
            required_token_mask=required_token_mask,
            min_required_tokens=min_required_tokens,
            supervision_token_mask=diffusion_supervised_token_mask,
            min_supervision_tokens=self.min_diffusion_supervised_tokens,
        )
        cropped_token_array = crop_result[1]
        cropped_atom_array = crop_result[2]
        if self.min_diffusion_supervised_tokens > 0:
            cropped_diffusion_token_count = int(
                _diffusion_supervised_token_mask(
                    token_array=cropped_token_array,
                    atom_array=cropped_atom_array,
                    protein_only=(
                        self.disable_non_protein_diffusion_weights
                    ),
                ).sum()
            )
            if (
                cropped_diffusion_token_count
                < self.min_diffusion_supervised_tokens
            ):
                raise RuntimeError(
                    "crop has too few diffusion-supervised tokens: "
                    f"{cropped_diffusion_token_count} < "
                    f"{self.min_diffusion_supervised_tokens}"
                )
        if self.require_resolved_protein_crop:
            resolved_protein_atom_count = int(
                np.logical_and(
                    cropped_atom_array.is_protein,
                    cropped_atom_array.is_resolved,
                ).sum()
            )
            if resolved_protein_atom_count == 0:
                raise RuntimeError(
                    "require_resolved_protein_crop requires at least one "
                    "resolved protein atom in every crop"
                )
        return crop_result

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        runtime_index = getattr(self, "runtime_distillation_index", None)
        if runtime_index is not None:
            root_id, cif_relpath = (
                runtime_index.get_address(idx)
            )
            cluster_id, cluster_size = (
                runtime_index.get_row_cluster(idx)
            )
            pdb_id = structure_basename(cif_relpath)
            resolved = runtime_index.resolve_paths(idx)
            return pd.Series(
                {
                    "entity_1_id": "1",
                    "chain_1_id": "A",
                    "mol_1_type": "prot",
                    "cluster_1_id": cluster_id,
                    "entity_2_id": np.nan,
                    "chain_2_id": np.nan,
                    "mol_2_type": np.nan,
                    "cluster_2_id": np.nan,
                    "cluster_id": cluster_id,
                    "pdb_id": pdb_id,
                    "release_date": "9999-12-31",
                    "num_tokens": -1,
                    "num_prot_chains": 1,
                    "resolution": -1.0,
                    "type": "chain",
                    "mol_type_group": "intra_prot",
                    "sub_mol_1_type": "prot",
                    "sub_mol_2_type": np.nan,
                    "eval_type": "intra_prot",
                    "row_id": idx,
                    "source_root_id": root_id,
                    "cif_relpath": cif_relpath,
                    "a3m_relpath": resolved.a3m_path.relative_to(
                        runtime_index.source_roots[root_id]
                    ).as_posix(),
                    "bioassembly_relpath": resolved.bioassembly_path.relative_to(
                        runtime_index.source_roots[root_id]
                    ).as_posix(),
                    "global_id": resolved.global_id,
                    "cluster_size": cluster_size,
                }
            )
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_pdb_indice(self, idx: int) -> pd.core.series.Series:
        if getattr(self, "runtime_distillation_index", None) is not None:
            return self._get_sample_indice(idx).to_frame().T
        if self.group_by_pdb_id:
            pdb_indice = self.indices_list[idx].copy()
        else:
            pdb_indice = self.indices_list.iloc[idx : idx + 1].copy()
        return pdb_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].isin(EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError("Cannot find a chain/interface for evaluation in the PDB.")

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                "Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def get_constraint_feature(
        self,
        idx,
        atom_array,
        token_array,
        msa_features,
        max_entity_mol_id,
        full_atom_array,
    ):
        sample_indice = self._get_sample_indice(idx=idx)
        pdb_indice = self._get_pdb_indice(idx=idx)
        features_dict = {}
        (
            token_array,
            atom_array,
            msa_features,
            constraint_feature_dict,
            feature_info,
            log_dict,
            full_atom_array,
        ) = self.constraint_generator.generate(
            atom_array,
            token_array,
            sample_indice,
            pdb_indice,
            msa_features,
            max_entity_mol_id,
            full_atom_array,
        )
        features_dict["constraint_feature"] = constraint_feature_dict
        features_dict.update(feature_info)
        features_dict["constraint_log_info"] = log_dict
        return token_array, atom_array, features_dict, msa_features, full_atom_array

    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
        max_entity_mol_id: int = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.
            max_entity_mol_id: Maximum entity mol ID in the full atom array.
        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        features_dict = {}
        if self.constraint.get("enable", False):
            (
                token_array,
                atom_array,
                features_dict,
                msa_features,
                full_atom_array,
            ) = self.get_constraint_feature(
                idx,
                atom_array,
                token_array,
                msa_features,
                max_entity_mol_id,
                full_atom_array,
            )

        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
        )
        features_dict.update(feat.get_all_input_features())
        labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Pocket Metrics
        if self.find_pocket:
            # Get entity_id of the interested ligand
            sample_indice = self._get_sample_indice(idx=idx)
            if sample_indice.mol_1_type == "ligand":
                lig_entity_id = str(sample_indice.entity_1_id)
                lig_chain_id = str(sample_indice.chain_1_id)
            elif sample_indice.mol_2_type == "ligand":
                lig_entity_id = str(sample_indice.entity_2_id)
                lig_chain_id = str(sample_indice.chain_2_id)
            else:
                raise ValueError("Cannot find ligand from this data point.")
            # Make sure the cropped array contains interested ligand
            assert lig_entity_id in set(atom_array.label_entity_id)
            assert lig_chain_id in set(atom_array.chain_id)

            # Get asym ID of the specific ligand in the `main` pocket
            lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
            assert len(np.unique(lig_asym_id)) == 1
            lig_asym_id = lig_asym_id[0]
            ligands = [lig_asym_id]

            if self.find_all_pockets:
                # Get asym ID of other ligands with the same entity_id
                all_lig_asym_ids = set(
                    full_atom_array[
                        full_atom_array.label_entity_id == lig_entity_id
                    ].label_asym_id
                )
                ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

            # Note: the `main` pocket is the 0-indexed one.
            # [N_pocket, N_atom], [N_pocket, N_atom].
            # If not find_all_pockets, then N_pocket = 1.
            interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
                atom_array=full_atom_array, lig_label_asym_id=ligands
            )

            label_full_dict["pocket_mask"] = pocket_mask
            label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            (
                eval_type,
                cluster_id,
                chain_1_mask,
                chain_2_mask,
            ) = self._get_eval_chain_interface_mask(
                idx=idx, atom_array_chain_id=full_atom_array.chain_id
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        features_dict["disable_non_protein_diffusion_weights"] = torch.tensor(
            [self.disable_non_protein_diffusion_weights],
            dtype=torch.bool,
        )
        features_dict["min_diffusion_supervised_tokens"] = torch.tensor(
            [self.min_diffusion_supervised_tokens],
            dtype=torch.int64,
        )
        if self.is_distillation is True:
            features_dict["resolution"] = torch.tensor([-1.0])
        return features_dict, labels_dict, label_full_dict


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    msa_info = configs["data"]["msa"]
    msa_args = deepcopy(msa_info)
    msa_args["dataset_name"] = dataset_name
    # If the dataset has special MSA settings, then overwrite the default MSA settings
    if "msa" in (dataset_config := configs["data"][dataset_name]):
        for k, v in dataset_config["msa"].items():
            msa_args[k] = v
    max_size_config = getattr(msa_args, "max_size", {})
    max_msa_size = max_size_config.get(stage, max_size_config.get("train", 16384))
    return MSAFeaturizer(
        dataset_name=msa_args.dataset_name,
        prot_seq_or_filename_to_msadir_jsons=msa_args.prot_seq_or_filename_to_msadir_jsons,
        prot_msadir_raw_paths=msa_args.prot_msadir_raw_paths,
        rna_seq_or_filename_to_msadir_jsons=msa_args.rna_seq_or_filename_to_msadir_jsons,
        rna_msadir_raw_paths=msa_args.rna_msadir_raw_paths,
        prot_pairing_dbs=msa_args.prot_pairing_dbs,
        prot_non_pairing_dbs=msa_args.prot_non_pairing_dbs,
        prot_indexing_methods=msa_args.prot_indexing_methods,
        rna_indexing_methods=msa_args.rna_indexing_methods,
        enable_prot_msa=msa_args.enable_prot_msa,
        enable_rna_msa=msa_args.enable_rna_msa,
        max_msa_size=max_msa_size,
        pair_msa_as_unpaired_only=msa_args.pair_msa_as_unpaired_only,
        shuffle_pair_msa_when_unpaired_only=getattr(
            msa_args, "shuffle_pair_msa_when_unpaired_only", False
        ),
        enable_virtual_segment_msa_shuffle=getattr(
            msa_args,
            "enable_virtual_segment_msa_shuffle",
            False,
        ),
        virtual_segment_residue_gap=getattr(
            msa_args,
            "virtual_segment_residue_gap",
            24,
        ),
        virtual_segment_min_length=getattr(
            msa_args,
            "virtual_segment_min_length",
            9,
        ),
        enable_multichain_msa_shuffle_variant_sampling=getattr(
            msa_args,
            "enable_multichain_msa_shuffle_variant_sampling",
            False,
        ),
        multichain_msa_shuffle_segment_probability=getattr(
            msa_args,
            "multichain_msa_shuffle_segment_probability",
            0.5,
        ),
        enable_crop_aware_partial_query_only=getattr(
            msa_args,
            "enable_crop_aware_partial_query_only",
            False,
        ),
        partial_query_only_max_chains=getattr(
            msa_args,
            "partial_query_only_max_chains",
            2,
        ),
        openfold3_alignment_npz_dirs=getattr(
            msa_args,
            "openfold3_alignment_npz_dirs",
            (),
        ),
        max_input_paired_msa_depth=(
            getattr(
                msa_args,
                "max_input_paired_msa_depth",
                max_msa_size,
            )
            if stage == "train"
            else None
        ),
        max_input_unpaired_msa_depth=(
            getattr(
                msa_args,
                "max_input_unpaired_msa_depth",
                max_msa_size,
            )
            if stage == "train"
            else None
        ),
        independent_pairing_non_pairing_budget=(
            getattr(
                msa_args,
                "independent_pairing_non_pairing_budget",
                False,
            )
            if stage == "train"
            else False
        ),
    )


def get_template_featurizer(
    configs: ConfigDict, dataset_name: str, stage: str
) -> Union[Callable, None]:
    template_info = configs["data"]["template"]
    template_args = deepcopy(template_info)
    template_args["dataset_name"] = dataset_name
    # If the dataset has special template settings, then overwrite the default template settings
    if "template" in (dataset_config := configs["data"][dataset_name]):
        for k, v in dataset_config["template"].items():
            template_args[k] = v
    emit_dummy_when_disabled = template_args.get(
        "emit_dummy_template_when_disabled",
        False,
    )
    if not isinstance(emit_dummy_when_disabled, (bool, np.bool_)):
        raise TypeError(
            "emit_dummy_template_when_disabled must be a boolean, "
            f"got {type(emit_dummy_when_disabled).__name__}: "
            f"{emit_dummy_when_disabled!r}"
        )
    if not template_args.get("enable_prot_template", True):
        if stage == "train" and emit_dummy_when_disabled:
            return DummyTemplateFeaturizer(max_templates=4)
        return None
    template_args.update(
        {
            "stage": stage,
        }
    )
    return TemplateFeaturizer(**template_args)


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.

    Args:
        datasets: A list of Dataset objects.
        dataset_names: A list of dataset names corresponding to the datasets.
        datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
        dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: Union[int, SampleOccurrence]) -> dict[str, dict]:
        index, occurrence = split_sample_index(index)
        dataset_index = self.dataset_indices[index]
        child_index = self.within_dataset_indices[index]
        if occurrence is not None:
            child_index = occurrence.with_index(child_index)
        data = self.datasets[dataset_index][child_index]
        data.setdefault("basic", {})["dataset_name"] = self.dataset_names[dataset_index]
        return data


class HierarchicalWeightedMultiDataset(Dataset):
    """Flatten datasets while retaining compact child sampling definitions.

    Unlike :class:`WeightedMultiDataset`, this container does not materialize
    one global weight tensor or Python row-to-dataset lists. The matching
    hierarchical sampler first selects a child using its outer mass and then
    asks that child's compact sampling definition for a local row.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        child_sampling_specs: list[Any],
        dataset_sample_weights: list[float],
    ) -> None:
        if not (
            len(datasets)
            == len(dataset_names)
            == len(child_sampling_specs)
            == len(dataset_sample_weights)
        ):
            raise ValueError(
                "hierarchical datasets, names, sampling specs, and masses "
                "must have equal lengths"
            )
        if not datasets:
            raise ValueError("hierarchical training dataset cannot be empty")

        self.datasets = datasets
        self.dataset_names = dataset_names
        self.hierarchical_child_sampling_specs = child_sampling_specs
        self.hierarchical_dataset_masses = [
            float(value) for value in dataset_sample_weights
        ]
        offsets = [0]
        for dataset in datasets:
            size = len(dataset)
            if size <= 0:
                raise ValueError("every hierarchical child must be non-empty")
            offsets.append(offsets[-1] + size)
        self.hierarchical_dataset_offsets = tuple(offsets)

    def __len__(self) -> int:
        return self.hierarchical_dataset_offsets[-1]

    def __getitem__(
        self, index: Union[int, SampleOccurrence]
    ) -> dict[str, dict]:
        index, occurrence = split_sample_index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = bisect_right(
            self.hierarchical_dataset_offsets, index
        ) - 1
        child_index = index - self.hierarchical_dataset_offsets[dataset_index]
        if occurrence is not None:
            child_index = occurrence.with_index(child_index)
        data = self.datasets[dataset_index][child_index]
        data.setdefault("basic", {})["dataset_name"] = self.dataset_names[
            dataset_index
        ]
        return data


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Get sample weight for each example in a weighted PDB dataset.

        data_type (str): Type of data, either 'chain' or 'interface'.
        cluster_size (int): Cluster size of this chain/interface.
        chain_count (dict): Count of each kind of chains, e.g., {"prot": int, "nuc": int, "ligand": int}.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-9.
        beta_dict (Optional[dict], optional): Dictionary containing beta values for 'chain' and 'interface'.
        alpha_dict (Optional[dict], optional): Dictionary containing alpha values for different chain types.

    Returns:
         float: Calculated weight for the given chain/interface.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame,
    beta_dict: dict[str, Any],
    alpha_dict: dict[str, Any],
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Calculate weights for each example in the dataframe.

    Args:
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.

    Returns:
        A pandas DataFrame with an column 'weights' containing the calculated weights.
    """
    # Specific to assembly, and entities (chain or interface)
    df = indices_df.copy()

    df[["entity_1_id", "entity_2_id"]] = (
        df[["entity_1_id", "entity_2_id"]].astype(object).fillna("None")
    )

    s = df["pdb_id"].astype(str)
    if "assembly_id" in df.columns:
        s = s.str.cat(df["assembly_id"].astype(str), sep="_")

    e1s = df["entity_1_id"].astype(str)
    e2s = df["entity_2_id"].astype(str)
    emin = e1s.where(e1s <= e2s, e2s)
    emax = e2s.where(e1s <= e2s, e1s)
    df["pdb_sorted_entity_id"] = s.str.cat(emin, sep="_").str.cat(emax, sep="_")

    df["pdb_sorted_entity_id_member_num"] = df.groupby("pdb_sorted_entity_id")[
        "pdb_id"
    ].transform("size")

    df["cluster_size"] = df.groupby("cluster_id")["pdb_sorted_entity_id"].transform(
        "nunique"
    )
    beta_dict_bytes = {k.encode(): v for k, v in beta_dict.items()}
    df["beta"] = df["type"].map({**beta_dict, **beta_dict_bytes}).astype(float)

    weighted_count = np.zeros(len(df), dtype=np.float64)
    mol1 = df["mol_1_type"]
    mol2 = df["mol_2_type"]
    for t, alpha in alpha_dict.items():
        c_t = mol1.eq(t).to_numpy(dtype=np.int8) + mol2.eq(t).to_numpy(dtype=np.int8)
        weighted_count += float(alpha) * c_t

    tmp_weights = (
        df["beta"].to_numpy() * weighted_count / (df["cluster_size"].to_numpy() + eps)
    )

    df["tmp_weights"] = tmp_weights  # do not use this
    df["weights"] = df["tmp_weights"] / df["pdb_sorted_entity_id_member_num"]
    return df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Computes sample weights based on the specified sampler type.

    Args:
        sampler_type: The type of sampler to use ('weighted' or 'uniform').
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.
        force_recompute_weight: Whether to force recomputation of weights even if they already exist.

    Returns:
        A list of sample weights.

    Raises:
        ValueError: If an unknown sampler type is provided.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[
    Union[WeightedMultiDataset, HierarchicalWeightedMultiDataset],
    dict[str, BaseSingleDataset],
]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        dataset_augmentation = config_dict.get("augmentation", {})
        if stage == "train":
            query_only_msa_probability = dataset_augmentation.get(
                "query_only_msa_probability",
                configs.data.get("query_only_msa_probability", 0.0),
            )
            pair_msa_as_unpaired_probability = dataset_augmentation.get(
                "pair_msa_as_unpaired_probability",
                configs.data.get("pair_msa_as_unpaired_probability", 0.0),
            )
            template_use_probability = dataset_augmentation.get(
                "template_use_probability",
                configs.data.get("template_use_probability", 1.0),
            )
        else:
            query_only_msa_probability = 0.0
            pair_msa_as_unpaired_probability = 0.0
            template_use_probability = 1.0
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "template_featurizer": get_template_featurizer(
                configs, dataset_name, stage
            ),
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "constraint": config_dict.get("constraint", {}),
            "query_only_msa_probability": query_only_msa_probability,
            "pair_msa_as_unpaired_probability": (
                pair_msa_as_unpaired_probability
            ),
            "template_use_probability": template_use_probability,
            "augmentation_decision_version": configs.data.get(
                "augmentation_decision_version",
                AUGMENTATION_DECISION_VERSION,
            ),
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        train_dataset = BaseSingleDataset(**dataset_param)
        train_datasets.append(train_dataset)
        if train_dataset.runtime_distillation_index is not None:
            sampler_type = data_config[train_name]["sampler_configs"].get(
                "sampler_type"
            )
            if sampler_type != "clustered":
                raise ValueError(
                    f"{train_name} runtime distillation requires "
                    "sampler_type='clustered'"
                )
            datapoint_weights.append(
                {
                    "kind": "clustered",
                    "source": train_dataset.runtime_distillation_index,
                }
            )
        else:
            datapoint_weights.append(
                get_sample_weights(
                    **data_config[train_name]["sampler_configs"],
                    indices_df=train_dataset.indices_list,
                )
            )

    if any(
        isinstance(spec, dict) and spec.get("kind") == "clustered"
        for spec in datapoint_weights
    ):
        child_sampling_specs = [
            spec
            if isinstance(spec, dict) and "kind" in spec
            else {"kind": "weighted", "weights": spec}
            for spec in datapoint_weights
        ]
        train_dataset = HierarchicalWeightedMultiDataset(
            datasets=train_datasets,
            dataset_names=list(data_config.train_sets),
            child_sampling_specs=child_sampling_specs,
            dataset_sample_weights=list(
                data_config.train_sampler.train_sample_weights
            ),
        )
    else:
        train_dataset = WeightedMultiDataset(
            datasets=train_datasets,
            dataset_names=data_config.train_sets,
            datapoint_weights=datapoint_weights,
            dataset_sample_weights=(
                data_config.train_sampler.train_sample_weights
            ),
        )

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        test_dataset = BaseSingleDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
