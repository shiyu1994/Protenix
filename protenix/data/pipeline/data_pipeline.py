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

import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

MP_SHARING_STRATEGY_ENV = "PROTENIX_MP_SHARING_STRATEGY"
SUPPORTED_MP_SHARING_STRATEGIES = frozenset({"file_descriptor", "file_system"})
_requested_mp_sharing_strategy = os.environ.get(
    MP_SHARING_STRATEGY_ENV, "file_system"
)
if _requested_mp_sharing_strategy not in SUPPORTED_MP_SHARING_STRATEGIES:
    supported = ", ".join(sorted(SUPPORTED_MP_SHARING_STRATEGIES))
    raise ValueError(
        f"{MP_SHARING_STRATEGY_ENV} must be one of {{{supported}}}, "
        f"got {_requested_mp_sharing_strategy!r}"
    )

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from protenix.data.core.parser import (
    DistillationMMCIFParser,
    MMCIFParser,
    RecentPDB_MMCIFParser,
)
from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.template.template_featurizer import (
    TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
    TEMPLATE_USAGE_USE_REAL_TEMPLATE,
    TemplateFeaturizer,
)
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.utils.cropping import CropData
from protenix.utils.file_io import load_gzip_pickle
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def _configure_mp_sharing_strategy(requested: str) -> str:
    """Configure and verify the process-wide PyTorch tensor sharing strategy."""
    try:
        torch.multiprocessing.set_sharing_strategy(requested)
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(
            f"PyTorch cannot enable {MP_SHARING_STRATEGY_ENV}={requested!r}; "
            "this platform or PyTorch build does not support it"
        ) from exc

    actual = torch.multiprocessing.get_sharing_strategy()
    if actual != requested:
        raise RuntimeError(
            "PyTorch multiprocessing sharing strategy mismatch: "
            f"requested={requested!r}, actual={actual!r}"
        )
    logger.info(
        "PyTorch multiprocessing sharing strategy configured: "
        "%s=%s, actual=%s",
        MP_SHARING_STRATEGY_ENV,
        requested,
        actual,
    )
    return actual


MP_SHARING_STRATEGY = _configure_mp_sharing_strategy(
    _requested_mp_sharing_strategy
)


class DataPipeline(object):
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """

    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
        distillation_fast_path: str = "off",
        raise_on_error: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".
            distillation_fast_path (str, optional): Predicted-protein fast path
                mode: ``"off"``, ``"auto"``, or ``"require"``.
            raise_on_error (bool, optional): Re-raise conversion failures instead
                of returning an empty result.
        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict(
                    predicted_protein_fast_path=distillation_fast_path
                )
            elif dataset == "RecentPDB":
                parser = RecentPDB_MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict,
                pdb_cluster_file=pdb_cluster_file,
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array
            return sample_indices_list, bioassembly_dict

        except Exception as e:
            if raise_on_error:
                raise
            logger.warning("Gen data failed for %s due to %s", mmcif, e)
            traceback.print_exc()
            return [], {}

    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id

    @staticmethod
    def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(
            bioassembly_dict_fpath
        ), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict

    @staticmethod
    def _map_ref_chain(
        one_sample: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> list[int]:
        """
        Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.

        Returns:
            list[int]: A list of asym_id_lnt of the chosen chain or interface, length 1 or 2.
        """
        atom_array = bioassembly_dict["atom_array"]
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            chain_id = one_sample[chain_id_field]
            assert np.isin(
                chain_id, np.unique(atom_array.chain_id)
            ), f"PDB {bioassembly_dict['pdb_id']} {chain_id_field}:{chain_id} not in atom_array"
            chain_asym_id = atom_array[atom_array.chain_id == chain_id].asym_id_int[0]
            ref_chain_indices.append(chain_asym_id)
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def get_msa_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: Optional[np.ndarray],
        msa_featurizer: Optional[MSAFeaturizer],
        query_only: bool = False,
        pair_msa_as_unpaired_only: Optional[bool] = None,
        shuffle_pair_msa_when_unpaired_only: Optional[bool] = None,
        virtual_segment_shuffle_seed: Optional[int] = None,
        query_only_chain_selection_seed: Optional[int] = None,
        msa_shuffle_variant_seed: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """
        Get tokenized MSA features of the bioassembly

        Args:
            bioassembly_dict (Mapping[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (torch.Tensor): Cropped token indices.
            msa_featurizer (MSAFeaturizer): MSAFeaturizer instance.
            query_only: Whether to retain only the target-sequence row.
            pair_msa_as_unpaired_only: Optional occurrence-level pairing mode.
            shuffle_pair_msa_when_unpaired_only: Optional occurrence-level
                paired-hit shuffle mode.
            virtual_segment_shuffle_seed: Optional deterministic seed for
                crop-segment MSA row permutations.
            query_only_chain_selection_seed: Optional deterministic seed for
                crop-visible physical protein chain selection in query-only
                mode.
            msa_shuffle_variant_seed: Independent deterministic seed for the
                multi-chain segment-versus-whole grouping decision.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized MSA features of the bioassembly.
        """
        if msa_featurizer is None:
            return {}

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        profile_start = time.perf_counter()
        msa_feats = msa_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
            query_only=query_only,
            pair_msa_as_unpaired_only=pair_msa_as_unpaired_only,
            shuffle_pair_msa_when_unpaired_only=(
                shuffle_pair_msa_when_unpaired_only
            ),
            virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
            msa_shuffle_variant_seed=msa_shuffle_variant_seed,
            query_only_chain_selection_seed=(
                query_only_chain_selection_seed
            ),
        )
        set_last_profile = getattr(msa_featurizer, "set_last_profile", None)
        if callable(set_last_profile):
            get_last_profile = getattr(
                msa_featurizer,
                "get_last_profile",
                None,
            )
            previous_profile = (
                get_last_profile() if callable(get_last_profile) else None
            )
            profile = dict(previous_profile or {})
            profile.update(
                {
                    "elapsed_seconds": time.perf_counter() - profile_start,
                    "query_only": bool(query_only),
                    "pair_msa_as_unpaired_only": pair_msa_as_unpaired_only,
                    "shuffle_pair_msa_when_unpaired_only": (
                        shuffle_pair_msa_when_unpaired_only
                    ),
                }
            )
            set_last_profile(profile)

        return msa_feats

    @staticmethod
    def get_template_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: Optional[np.ndarray],
        template_featurizer: Optional[TemplateFeaturizer],
        template_use_requested: Optional[bool] = None,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized template features of the bioassembly.

        Args:
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (np.ndarray): Cropped token indices.
            template_featurizer: Optional template featurizer.
            template_use_requested: Explicit occurrence decision. ``True``
                requests real templates, ``False`` forces the official dummy
                representation, and ``None`` preserves legacy behavior.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized template features of the bioassembly,
                or None if the template featurizer is not provided.
        """
        if template_featurizer is None:
            return {}

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        template_usage_decision = None
        if template_use_requested is not None:
            template_usage_decision = (
                TEMPLATE_USAGE_USE_REAL_TEMPLATE
                if template_use_requested
                else TEMPLATE_USAGE_FORCE_NO_TEMPLATE
            )

        template_feats = template_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
            template_usage_decision=template_usage_decision,
        )
        return template_feats

    @staticmethod
    def crop(
        one_sample: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        msa_featurizer: Optional[MSAFeaturizer],
        template_featurizer: Optional[TemplateFeaturizer],
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        max_nuc_tokens: int = -1,
        query_only_msa: bool = False,
        pair_msa_as_unpaired_only: Optional[bool] = None,
        shuffle_pair_msa_when_unpaired_only: Optional[bool] = None,
        virtual_segment_shuffle_seed: Optional[int] = None,
        query_only_chain_selection_seed: Optional[int] = None,
        template_use_requested: Optional[bool] = None,
        required_token_mask: Optional[torch.Tensor] = None,
        min_required_tokens: int = 1,
        supervision_token_mask: Optional[torch.Tensor] = None,
        min_supervision_tokens: int = 0,
        msa_shuffle_variant_seed: Optional[int] = None,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop data based on the crop size and reference chain indices.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): A dict of bioassembly dict with sequence, atom_array and token_array.
            crop_size (int): the crop size.
            msa_featurizer (MSAFeaturizer): Default to an empty replacement for msa featurizer.
            template_featurizer (None): Placeholder for the template featurizer.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig (bool): Whether to crop the complete ligand in ContiguousCropping method.
            spatial_crop_complete_lig (bool): Whether to crop the complete ligand in SpatialCropping method.
            drop_last (bool): Whether to drop the last fragment in ContiguousCropping.
            remove_metal (bool): Whether to remove metal atoms from the crop.
            query_only_msa: Whether to build one global target-sequence MSA row.
            pair_msa_as_unpaired_only: Per-occurrence override for species pairing.
            shuffle_pair_msa_when_unpaired_only: Per-occurrence paired-hit shuffle.
            virtual_segment_shuffle_seed: Stable per-occurrence seed for
                crop-segment MSA row permutations.
            query_only_chain_selection_seed: Stable per-occurrence seed for
                partial query-only physical-chain selection.
            msa_shuffle_variant_seed: Independent stable per-occurrence seed
                for the multi-chain grouping-variant decision.
            template_use_requested: Explicit target-occurrence template policy.
            required_token_mask: Optional token-level mask. When provided, the
                crop is guaranteed to contain ``min_required_tokens`` selected
                tokens.
            min_required_tokens: Minimum number of selected eligible tokens
                when ``required_token_mask`` is provided.
            supervision_token_mask: Optional token-level mask used only for a
                hard pre-MSA validation of the final crop plan.
            min_supervision_tokens: Minimum selected tokens required by
                ``supervision_token_mask``.

        Returns:
            tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
                crop_method (str): The crop method.
                cropped_token_array (TokenArray): TokenArray after cropping.
                cropped_atom_array (AtomArray): AtomArray after cropping.
                cropped_msa_features (dict[str, Any]): The cropped msa features.
                cropped_template_features (dict[str, Any]): The cropped template features.
        """
        if isinstance(min_supervision_tokens, bool) or not isinstance(
            min_supervision_tokens,
            int,
        ):
            raise TypeError(
                "min_supervision_tokens must be a non-negative integer"
            )
        if min_supervision_tokens < 0:
            raise ValueError("min_supervision_tokens must be >= 0")
        resolved_supervision_mask = None
        if supervision_token_mask is not None:
            resolved_supervision_mask = torch.as_tensor(
                supervision_token_mask,
                dtype=torch.bool,
            ).flatten()
            expected_shape = (len(bioassembly_dict["token_array"]),)
            if tuple(resolved_supervision_mask.shape) != expected_shape:
                raise ValueError(
                    "supervision_token_mask must have one value per token: "
                    f"{tuple(resolved_supervision_mask.shape)} versus "
                    f"{expected_shape}"
                )
        elif min_supervision_tokens > 0:
            raise ValueError(
                "min_supervision_tokens requires supervision_token_mask"
            )

        if crop_size <= 0:
            if required_token_mask is not None:
                eligible_count = int(
                    torch.as_tensor(
                        required_token_mask,
                        dtype=torch.bool,
                    ).sum()
                )
                if eligible_count < min_required_tokens:
                    raise ValueError(
                        "required_token_mask contains too few eligible tokens: "
                        f"{eligible_count} < {min_required_tokens}"
                    )
            if resolved_supervision_mask is not None:
                supervision_count = int(
                    resolved_supervision_mask.sum()
                )
                if supervision_count < min_supervision_tokens:
                    raise ValueError(
                        "structure has too few supervision tokens before MSA "
                        f"I/O: {supervision_count} < "
                        f"{min_supervision_tokens}"
                    )
            selected_indices = None
            # Prepare msa
            msa_features = DataPipeline.get_msa_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                msa_featurizer=msa_featurizer,
                query_only=query_only_msa,
                pair_msa_as_unpaired_only=pair_msa_as_unpaired_only,
                shuffle_pair_msa_when_unpaired_only=(
                    shuffle_pair_msa_when_unpaired_only
                ),
                virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
                msa_shuffle_variant_seed=msa_shuffle_variant_seed,
                query_only_chain_selection_seed=(
                    query_only_chain_selection_seed
                ),
            )
            # Prepare template
            template_features = DataPipeline.get_template_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                template_featurizer=template_featurizer,
                template_use_requested=template_use_requested,
            )
            return (
                "no_crop",
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
                msa_features or {},
                template_features or {},
                -1,
            )

        ref_chain_indices = DataPipeline._map_ref_chain(
            one_sample=one_sample, bioassembly_dict=bioassembly_dict
        )

        crop = CropData(
            crop_size=crop_size,
            ref_chain_indices=ref_chain_indices,
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
            max_nuc_tokens=max_nuc_tokens,
            required_token_mask=required_token_mask,
            min_required_tokens=min_required_tokens,
        )
        # Get crop method
        crop_method = crop.random_crop_method()
        # Get crop indices based crop method
        selected_indices, reference_token_index = crop.get_crop_indices(
            crop_method=crop_method
        )
        if required_token_mask is not None:
            selected_required_count = int(
                torch.as_tensor(
                    required_token_mask,
                    dtype=torch.bool,
                )[selected_indices].sum()
            )
            if selected_required_count < min_required_tokens:
                raise RuntimeError(
                    "crop contains too few required tokens: "
                    f"{selected_required_count} < {min_required_tokens}"
                )
        if resolved_supervision_mask is not None:
            selected_supervision_count = int(
                resolved_supervision_mask[selected_indices].sum()
            )
            if selected_supervision_count < min_supervision_tokens:
                raise RuntimeError(
                    "crop has too few supervision tokens before MSA I/O: "
                    f"{selected_supervision_count} < "
                    f"{min_supervision_tokens}"
                )
        # Prepare msa
        cropped_msa_features = DataPipeline.get_msa_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            msa_featurizer=msa_featurizer,
            query_only=query_only_msa,
            pair_msa_as_unpaired_only=pair_msa_as_unpaired_only,
            shuffle_pair_msa_when_unpaired_only=(
                shuffle_pair_msa_when_unpaired_only
            ),
            virtual_segment_shuffle_seed=virtual_segment_shuffle_seed,
            msa_shuffle_variant_seed=msa_shuffle_variant_seed,
            query_only_chain_selection_seed=(
                query_only_chain_selection_seed
            ),
        )
        # Prepare template
        cropped_template_features = DataPipeline.get_template_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            template_featurizer=template_featurizer,
            template_use_requested=template_use_requested,
        )
        (
            cropped_token_array,
            cropped_atom_array,
        ) = crop.crop_by_indices(
            selected_token_indices=selected_indices,
        )

        if crop_method == "ContiguousCropping":
            resovled_atom_num = cropped_atom_array.is_resolved.sum()
            # The criterion of “more than 4 atoms” is chosen arbitrarily.
            assert (
                resovled_atom_num > 4
            ), f"{resovled_atom_num=} <= 4 after ContiguousCropping"

        return (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
        )

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
