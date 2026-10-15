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

import copy
import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from biotite.structure import AtomArray
from scipy.spatial.distance import cdist

from protenix.data.tokenizer import TokenArray


def identify_mol_type(
    ref_space_uid: torch.Tensor,
    atom_sums: torch.Tensor,
    chain_id: torch.Tensor,
    chain_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Generate mol_type masks based on the given rules.

    Args:
        ref_space_uid (torch.Tensor): A tensor of unique ids, shape (N,).
        atom_sums (torch.Tensor): A tensor of atom sums corresponding to each unique id, shape (N,).
        chain_id (torch.Tensor): A tensor of chain IDs corresponding to each unique id, shape (N,).
        chain_lengths (torch.Tensor): A tensor of chain lengths, shape (num_chains,).

    Returns:
        is_metal (torch.Tensor): A mask indicating metals.
        first_indices (torch.Tensor): A tensor of first indices for each unique id, shape (N,).
        last_indices (torch.Tensor): A tensor of last indices for each unique id, shape (N,).
    """

    assert (
        ref_space_uid.shape == atom_sums.shape
    ), "ref_space_uid and atom_sums must have the same shape."
    # Initialize masks
    is_metal = torch.zeros_like(ref_space_uid, dtype=torch.bool)
    first_indices = torch.zeros_like(ref_space_uid, dtype=torch.long)
    last_indices = torch.zeros_like(ref_space_uid, dtype=torch.long)

    # Count occurrences of each ref_space_uid
    unique_ids, counts = torch.unique(ref_space_uid, return_counts=True)
    for unique_id, count in zip(unique_ids, counts):
        mask = ref_space_uid == unique_id
        first_index = mask.nonzero(as_tuple=False)[0].item()
        last_index = mask.nonzero(as_tuple=False)[-1].item()
        first_indices[mask] = first_index
        last_indices[mask] = last_index
        atom_sum = atom_sums[mask]

        if count == 1 and chain_lengths[chain_id[mask].long()] == 1:
            is_metal[mask] = atom_sum == 1

    return (
        is_metal,
        first_indices,
        last_indices,
    )


def get_interface_token(
    chain_id: torch.Tensor,
    reference_chain_id: torch.Tensor,
    token_distance: torch.Tensor,
    token_distance_mask: torch.Tensor,
    interface_minimal_distance: int = 15,
) -> torch.Tensor:
    """
    Get tokens in contact with the other chain.
    Args:
        chain_id:           [all_token_length, ], chain ID of each token
        reference_chain_id: [1] or [2], the reference atom is selected within the reference chains
        token_distance:     [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask:[chain/interface_token_length, all_token_length], indicates valid distance
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        interface_token_indices: indices of tokens of interface
    """
    # expand reference_chain_id to chain_id shape
    expand_reference_chain_id = torch.zeros(chain_id.size(), dtype=torch.int)
    for _chain_id in reference_chain_id:
        expand_reference_chain_id += chain_id == _chain_id

    # get distance mask, difference chain mask
    mask_distance = token_distance < interface_minimal_distance
    mask_diff_chain = (chain_id[None, :] != chain_id[:, None])[
        expand_reference_chain_id.nonzero(as_tuple=True)[0]
    ]

    mask = mask_distance * mask_diff_chain * token_distance_mask
    mask_interface = torch.sum(mask, dim=-1)
    interface_token_indices = torch.nonzero(mask_interface, as_tuple=True)[0]
    return interface_token_indices


def get_spatial_crop_index(
    tokens: torch.Tensor,
    chain_id: torch.Tensor,
    token_distance: torch.Tensor,
    token_distance_mask: torch.Tensor,
    reference_chain_id: torch.Tensor,
    ref_space_uid_token: torch.Tensor,
    crop_size: int,
    crop_complete_ligand_unstdRes: bool = False,
    interface_crop: bool = False,
    interface_minimal_distance: int = 15,
) -> torch.Tensor:
    """
    Crop sequences continuesly across chains.
    Args:
        tokens:   [all_token_length,], all tokens within an assembly
        chain_id: [all_token_length,], all tokens' chain ID within an assembly
        token_distance: [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask: [chain/interface_token_length, all_token_length], indicates valid distance
        reference_chain_id:  [1] or [2],the reference atom is selected within the reference_chains ID
        crop_size: total crop size of the whole assembly
        interface_crop: whether use interface tokens as referenced token
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        selected_token_indices: torch.Tensor, shape=(min(crop_size, tokens.shape[0]), )
    """

    # interface spatial cropping: select reference tokens with contact to the other
    if interface_crop and interface_minimal_distance is not None:
        reference_token_indices = get_interface_token(
            chain_id=chain_id,
            reference_chain_id=reference_chain_id,
            token_distance=token_distance,
            token_distance_mask=token_distance_mask,
            interface_minimal_distance=interface_minimal_distance,
        )
        if len(reference_token_indices) < 1 and len(reference_chain_id) == 1:
            # If a chain does not contain any interfacial atoms, use all resolved tokens.
            reference_token_indices = torch.nonzero(
                token_distance_mask.bool().any(-1), as_tuple=True
            )[0]
    else:
        # select reference tokens within the given chain or interface
        reference_token_indices = torch.nonzero(
            token_distance_mask.bool().any(-1), as_tuple=True
        )[0]

    # random select one token from reference_token_indices
    assert len(reference_token_indices) > 0, "No resolved atoms in reference tokens!"

    random_idx = torch.randint(0, reference_token_indices.shape[0], (1,)).item()
    reference_token_idx = reference_token_indices[random_idx].item()

    assert (
        token_distance_mask[reference_token_idx].bool().any()
    ), "Select a unresolved reference token"
    distance_to_reference = token_distance[reference_token_idx]
    # add noise to break tie
    noise_break_tie = torch.arange(0, distance_to_reference.shape[0]).float() * 1e-3

    distance_to_reference_mask = token_distance_mask[reference_token_idx]
    distance_to_reference = torch.where(
        distance_to_reference_mask.bool(), distance_to_reference, torch.inf
    )

    # find k nearest tokens
    nearest_k = min(crop_size, tokens.shape[0])
    selected_token_indices = (
        torch.topk(distance_to_reference + noise_break_tie, nearest_k, largest=False)
        .indices.sort()
        .values
    )

    def drop_uncompleted_mol(selected_token_indices):
        selected_uid = ref_space_uid_token[selected_token_indices]
        mask = torch.ones_like(ref_space_uid_token, dtype=torch.bool)
        mask[selected_token_indices] = False
        unselected_uid = ref_space_uid_token[mask]

        # Find overlap elements
        overlap_uid = torch.Tensor(np.intersect1d(selected_uid, unselected_uid))

        # Remove overlap elements from elements_B
        remain_indices = selected_token_indices[
            ~torch.isin(selected_uid, overlap_uid)
        ].long()
        return remain_indices

    selected_token_indices = torch.flatten(selected_token_indices)
    if crop_complete_ligand_unstdRes is True:
        selected_token_indices = drop_uncompleted_mol(selected_token_indices)
    assert (
        selected_token_indices.shape[0] <= crop_size
    ), f"Spatial cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices, reference_token_idx


def get_continues_crop_index(
    tokens: torch.Tensor,
    chain_id: torch.Tensor,
    ref_space_uid_token: torch.Tensor,
    atom_sums: torch.Tensor,
    crop_size: int,
    crop_complete_ligand_unstdRes: Optional[bool] = False,
    drop_last: Optional[bool] = False,
    remove_metal: Optional[bool] = False,
) -> torch.Tensor:
    """
    Crop sequences continuesly across chains. Reference: AF-multimer Algorithm 1.
    Args:
        tokens:    [all_token_length,], flatten tokens
        chain_id:  [all_token_length,], all tokens' chain ID within an assembly
        atom_sums: [all_token_length,] sum of atoms within one ref_space_uid
        ref_space_uid: [all_atom_length,] unique chain-residue id
        crop_size: total crop size of the whole assembly
        crop_complete_ligand_unstdRes: Whether to crop the complete ligand or unstandard residues.
                              If False, the ligand is usually fragmented during sequential cropping.
        drop_last: whether to ensure all ligands or unstandard residues to be cropped completely,
                    if not, we will ignore the completion of the last one to meet the crop_size quota.
        remove_metal: whether remove all metal/ions
    Returns:
        selected_token_indices: torch.Tensor, shape=(crop_size, )
    """
    # get chain counts info
    unique_chain_id = torch.unique(chain_id)
    chain_lengths = torch.bincount(chain_id.long())
    chain_offset_list = torch.tensor(
        [torch.where(chain_id == chain_idx)[0][0] for chain_idx in unique_chain_id],
    )

    # identify the mol type
    (
        is_metal,
        uid_first_indices,
        uid_last_indices,
    ) = identify_mol_type(ref_space_uid_token, atom_sums, chain_id, chain_lengths)

    def _qualify_crop_size(cur_crop_size, crop_size_min, N_added):
        if cur_crop_size < crop_size_min:
            return False
        if cur_crop_size + N_added > crop_size:
            return False
        return True

    def _determine_start_end_point(start_idx, end_idx, crop_size_min, N_added):
        if start_idx == end_idx:
            return start_idx, end_idx

        # determine the start_idx
        left_start_point = right_start_point = start_idx
        # if this is not the first time this uid occurants, then it must be a middle point
        if uid_first_indices[start_idx] != start_idx:
            start_in_middle = True
            left_start_point = uid_first_indices[start_idx]
            right_start_point = uid_last_indices[start_idx] + 1
        else:
            start_in_middle = False

        # determine the end_idx
        left_end_point = right_end_point = end_idx
        # if this is not the last time this uid occurants, then it must be a middle point
        if end_idx > 0 and uid_last_indices[end_idx - 1] != end_idx - 1:
            end_in_middle = True
            left_end_point = uid_first_indices[end_idx - 1]
            right_end_point = uid_last_indices[end_idx - 1] + 1
        else:
            end_in_middle = False

        if start_in_middle is False and end_in_middle is False:
            return start_idx, end_idx
        elif start_in_middle is True and end_in_middle is True:
            # alwalys use left edge
            start_in_middle = False
            start_idx = left_start_point

        if start_in_middle is False and end_in_middle is True:
            # need to determine: use left end or right end
            left_crop_size = left_end_point - start_idx
            right_crop_size = right_end_point - start_idx
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, N_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, N_added)
            if is_left_ok and is_right_ok:
                end_idx = (
                    left_end_point
                    if torch.randint(low=0, high=2, size=(1,)).item() == 0
                    else right_end_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return start_idx, left_end_point
            elif is_right_ok:
                return start_idx, right_end_point
            elif drop_last is True:
                end_point = left_end_point
                while end_point - start_idx + N_added > crop_size:
                    if end_point > start_idx:
                        end_point = uid_first_indices[end_point - 1]
                    else:
                        break
                return start_idx, end_point
            else:
                cur_crop_size = min(end_idx - start_idx, crop_size - N_added)
                return start_idx, start_idx + cur_crop_size
        elif start_in_middle is True and end_in_middle is False:
            # need to determine: use left start or right start
            left_crop_size = end_idx - left_start_point
            right_crop_size = end_idx - right_start_point
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, N_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, N_added)
            if is_left_ok and is_right_ok:
                start_idx = (
                    left_start_point
                    if torch.randint(low=0, high=2, size=(1,)).item() == 0
                    else right_start_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return left_start_point, end_idx
            elif is_right_ok:
                return right_start_point, end_idx
            elif drop_last is True:
                return right_start_point, end_idx
            else:
                return start_idx, end_idx

    # shuffle the list of chains
    chain_shuffle_index = torch.randperm(len(unique_chain_id))

    # crop over chains iteratively
    selected_token_indices = []
    N_added = 0  # number of tokens already selected
    N_remaining = len(tokens)  # number of tokens in remaining chains
    if remove_metal is True:
        N_remaining -= sum(is_metal).item()
    for idx in chain_shuffle_index:
        if N_added >= crop_size:
            break

        # get chain type: whether it is metal/ions
        curr_is_metal = is_metal[chain_offset_list[idx]]
        # whether remove metal chain
        if remove_metal is True and curr_is_metal:
            # skip if it is metal/ions
            continue

        chain_length = chain_lengths[unique_chain_id[idx].int()]
        N_remaining -= chain_length

        # determine the crop size
        crop_size_min = min(chain_length, max(0, crop_size - (N_added + N_remaining)))
        crop_size_max = min(crop_size - N_added, chain_length)
        if crop_size_min > crop_size_max:
            print(f"error crop_size: {crop_size_min} > {crop_size_max}")

        chain_crop_size = torch.randint(
            low=crop_size_min,
            high=crop_size_max + 1,
            size=(1,),
            device=tokens.device,
        ).item()

        chain_crop_start = torch.randint(
            low=0,
            high=chain_length - chain_crop_size + 1,
            size=(1,),
            device=tokens.device,
        ).item()

        chain_offset = chain_offset_list[idx]
        start_token_index = chain_offset + chain_crop_start
        end_token_index = chain_offset + chain_crop_start + chain_crop_size
        if crop_complete_ligand_unstdRes is True:
            start_token_index, end_token_index = _determine_start_end_point(
                start_token_index, end_token_index, crop_size_min, N_added
            )
            assert (
                end_token_index >= start_token_index
            ), f"invalid crop indices!! {start_token_index}, {end_token_index}"
            chain_crop_size = end_token_index - start_token_index

        selected_token_indices.append(
            torch.arange(
                start_token_index,
                end_token_index,
            )
        )
        N_added += chain_crop_size
        if crop_complete_ligand_unstdRes is True and drop_last is True:
            if start_token_index < end_token_index:
                assert uid_first_indices[start_token_index] == start_token_index
                assert uid_last_indices[end_token_index - 1] == end_token_index - 1

    selected_token_indices = torch.concat(selected_token_indices).sort().values
    selected_token_indices = torch.flatten(selected_token_indices)
    if drop_last is True:
        assert (
            selected_token_indices.shape[0] <= crop_size
        ), f"Continuous cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices


class CropData(object):
    """
    Crop the data based on the given crop size and reference chain indices (asym_id).

    Args:
        crop_size (int): The size of the crop to be sampled.
        ref_chain_indices (list[int]): The "asym_id_int" of the reference chains.
        token_array (TokenArray): The token array.
        atom_array (AtomArray): The atom array.
        method_weights (list[float], optional): The weights corresponding to these three cropping methods:
                                      ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"]. Defaults to [0.2, 0.4, 0.4].
        contiguous_crop_complete_lig (bool, optional): Whether to crop the complete ligand in ContiguousCropping method. Defaults to False.
        spatial_crop_complete_lig (bool, optional): Whether to crop the complete ligand in SpatialCropping method. Defaults to False.
        drop_last (bool, optional): whether to ensure all ligands or unstandard residues to be cropped completely. Defaults to False.
        remove_metal (bool, optional): whether remove all metal/ions. Defaults to False.
        max_nuc_tokens (int, optional): Maximum selected nucleic-acid tokens.
        required_token_mask (torch.Tensor, optional): Token-level eligibility
            mask. Every returned crop must contain ``min_required_tokens``
            eligible tokens.
        min_required_tokens (int, optional): Minimum number of eligible tokens
            required in every crop.
    """

    def __init__(
        self,
        crop_size: int,
        ref_chain_indices: list[int],
        token_array: TokenArray,
        atom_array: AtomArray,
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        max_nuc_tokens: int = -1,
        required_token_mask: Optional[torch.Tensor] = None,
        min_required_tokens: int = 1,
    ) -> None:
        self.crop_size = crop_size
        self.ref_chain_indices = ref_chain_indices
        self.token_array = token_array
        self.atom_array = atom_array
        self.method_weights = method_weights
        self.cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        self.contiguous_crop_complete_lig = contiguous_crop_complete_lig
        self.spatial_crop_complete_lig = spatial_crop_complete_lig
        self.drop_last = drop_last
        self.remove_metal = remove_metal
        self.max_nuc_tokens = max_nuc_tokens
        if isinstance(min_required_tokens, bool) or not isinstance(
            min_required_tokens, int
        ):
            raise TypeError(
                "min_required_tokens must be a positive integer, got "
                f"{type(min_required_tokens).__name__}: "
                f"{min_required_tokens!r}"
            )
        if min_required_tokens < 1:
            raise ValueError(
                "min_required_tokens must be >= 1, got "
                f"{min_required_tokens}"
            )
        if crop_size > 0 and min_required_tokens > crop_size:
            raise ValueError(
                "min_required_tokens cannot exceed crop_size: "
                f"{min_required_tokens} > {crop_size}"
            )
        self.min_required_tokens = min_required_tokens
        self.required_token_mask = (
            None
            if required_token_mask is None
            else torch.as_tensor(required_token_mask, dtype=torch.bool).flatten()
        )
        if self.required_token_mask is not None:
            if self.required_token_mask.shape != (len(self.token_array),):
                raise ValueError(
                    "required_token_mask must have one value per token: "
                    f"{self.required_token_mask.shape=} versus "
                    f"{len(self.token_array)=}"
                )
            eligible_count = int(self.required_token_mask.sum())
            if eligible_count < self.min_required_tokens:
                raise ValueError(
                    "required_token_mask contains too few eligible tokens: "
                    f"{eligible_count} < {self.min_required_tokens}"
                )

    def _get_nuc_mask(self) -> np.ndarray:
        centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[centre_atom_indices]
        return np.logical_or(centre_atoms.is_dna, centre_atoms.is_rna)

    def _limit_nuc_tokens(
        self,
        selected_token_indices: torch.Tensor,
        reference_token_index: Optional[int] = None,
    ) -> torch.Tensor:
        if self.max_nuc_tokens <= 0:
            return selected_token_indices

        nuc_mask = self._get_nuc_mask()
        selected_nuc = nuc_mask[selected_token_indices.numpy()]
        nuc_positions = np.where(selected_nuc)[0]

        if len(nuc_positions) <= self.max_nuc_tokens:
            return selected_token_indices

        if reference_token_index is not None:
            ref_centre_idx = int(
                self.token_array.get_annotation("centre_atom_index")[reference_token_index]
            )
            ref_coord = self.atom_array.coord[ref_centre_idx]
            nuc_abs = selected_token_indices[nuc_positions].numpy()
            nuc_centre = np.array(
                self.token_array.get_annotation("centre_atom_index")
            )[nuc_abs]
            nuc_coords = self.atom_array.coord[nuc_centre]
            dists = np.linalg.norm(nuc_coords - ref_coord, axis=1)
            keep_order = np.argsort(dists)[: self.max_nuc_tokens]
            keep_nuc_positions = nuc_positions[keep_order]
        else:
            keep_nuc_positions = nuc_positions[: self.max_nuc_tokens]

        keep_mask = np.ones(len(selected_token_indices), dtype=bool)
        keep_mask[nuc_positions] = False
        keep_mask[keep_nuc_positions] = True

        return selected_token_indices[torch.tensor(keep_mask)].sort().values

    def _ensure_required_tokens(
        self,
        selected_token_indices: torch.Tensor,
        reference_token_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Enforce the token-level supervision minimum before MSA I/O.

        The original crop remains the geometric anchor. Missing eligible
        tokens are chosen by distance to that anchor. If space is needed,
        complete ``ref_space_uid`` groups containing no already-selected
        eligible token are removed first, preserving both supervision and
        complete multi-token molecules whenever possible.
        """
        if self.required_token_mask is None:
            return selected_token_indices

        selected_token_indices = torch.unique(
            torch.as_tensor(
                selected_token_indices,
                dtype=torch.long,
            ).flatten(),
            sorted=True,
        )
        if selected_token_indices.numel() > self.crop_size:
            raise RuntimeError(
                "input crop exceeds crop_size before enforcing required tokens"
            )
        selected_required_count = int(
            self.required_token_mask[selected_token_indices].sum()
        )
        if selected_required_count >= self.min_required_tokens:
            return selected_token_indices

        centre_atom_indices = np.asarray(
            self.token_array.get_annotation("centre_atom_index")
        )
        centre_coordinates = np.asarray(
            self.atom_array.coord[centre_atom_indices],
            dtype=np.float64,
        )
        required_indices = torch.nonzero(
            self.required_token_mask,
            as_tuple=True,
        )[0]
        required_coordinates = centre_coordinates[
            required_indices.numpy()
        ].copy()

        # A token may have a missing centre atom while another supervised atom
        # in the token is finite. Use the latter only as its distance proxy.
        for required_position, required_index in enumerate(
            required_indices.tolist()
        ):
            atom_indices = np.asarray(
                self.token_array[required_index].atom_indices,
                dtype=np.int64,
            )
            centre_atom_index = int(centre_atom_indices[required_index])
            atom_indices = np.concatenate(
                (
                    np.asarray([centre_atom_index], dtype=np.int64),
                    atom_indices[atom_indices != centre_atom_index],
                )
            )
            atom_coordinates = np.asarray(
                self.atom_array.coord[atom_indices],
                dtype=np.float64,
            )
            preferred_atom_mask = np.isfinite(atom_coordinates).all(axis=-1)
            if hasattr(self.atom_array, "is_resolved"):
                preferred_atom_mask &= np.asarray(
                    self.atom_array.is_resolved[atom_indices],
                    dtype=bool,
                )
            if hasattr(self.atom_array, "is_protein"):
                token_protein_mask = np.asarray(
                    self.atom_array.is_protein[atom_indices],
                    dtype=bool,
                )
                if token_protein_mask.any():
                    preferred_atom_mask &= token_protein_mask
            preferred_atom_coordinates = atom_coordinates[
                preferred_atom_mask
            ]
            if preferred_atom_coordinates.shape[0] > 0:
                required_coordinates[required_position] = (
                    preferred_atom_coordinates[0]
                )
            elif not np.isfinite(
                required_coordinates[required_position]
            ).all():
                finite_atom_coordinates = atom_coordinates[
                    np.isfinite(atom_coordinates).all(axis=-1)
                ]
                if finite_atom_coordinates.shape[0] > 0:
                    required_coordinates[required_position] = (
                        finite_atom_coordinates[0]
                    )

        if reference_token_index is not None and 0 <= int(
            reference_token_index
        ) < len(self.token_array):
            anchor_indices = np.asarray(
                [int(reference_token_index)],
                dtype=np.int64,
            )
        else:
            anchor_indices = selected_token_indices.numpy()
        anchor_coordinates = centre_coordinates[anchor_indices]
        anchor_coordinates = anchor_coordinates[
            np.isfinite(anchor_coordinates).all(axis=-1)
        ]
        if (
            anchor_coordinates.shape[0] == 0
            and selected_token_indices.numel() > 0
        ):
            selected_coordinates = centre_coordinates[
                selected_token_indices.numpy()
            ]
            anchor_coordinates = selected_coordinates[
                np.isfinite(selected_coordinates).all(axis=-1)
            ]

        finite_required_mask = np.isfinite(required_coordinates).all(axis=-1)
        required_distance = np.full(
            len(required_indices),
            np.inf,
            dtype=np.float64,
        )
        if (
            anchor_coordinates.shape[0] > 0
            and finite_required_mask.any()
        ):
            required_distance[finite_required_mask] = cdist(
                required_coordinates[finite_required_mask],
                anchor_coordinates,
            ).min(axis=-1)
        required_index_array = required_indices.numpy()
        required_order = np.lexsort(
            (required_index_array, required_distance)
        )
        selected_set = set(selected_token_indices.tolist())
        ordered_candidates = [
            int(required_index_array[position])
            for position in required_order
            if int(required_index_array[position]) not in selected_set
        ]
        missing_count = (
            self.min_required_tokens - selected_required_count
        )
        if len(ordered_candidates) < missing_count:
            raise RuntimeError(
                "not enough unselected eligible tokens to satisfy crop "
                f"minimum {self.min_required_tokens}"
            )
        insertion_indices = ordered_candidates[:missing_count]
        required_position_by_index = {
            int(required_index): position
            for position, required_index in enumerate(required_index_array)
        }
        insertion_coordinates = required_coordinates[
            [
                required_position_by_index[index]
                for index in insertion_indices
            ]
        ]
        insertion_coordinates = insertion_coordinates[
            np.isfinite(insertion_coordinates).all(axis=-1)
        ]
        distance_targets = (
            insertion_coordinates
            if insertion_coordinates.shape[0] > 0
            else anchor_coordinates
        )

        slots_needed = max(
            0,
            selected_token_indices.numel()
            + missing_count
            - self.crop_size,
        )
        if slots_needed > 0:
            token_ref_space_uid = np.asarray(
                self.atom_array.ref_space_uid[centre_atom_indices]
            )
            selected_numpy = selected_token_indices.numpy()
            selected_uids = token_ref_space_uid[selected_numpy]
            selected_required = self.required_token_mask[
                selected_token_indices
            ].numpy()
            protected_required_uids = set(
                selected_uids[selected_required].tolist()
            )
            protected_reference_uid = None
            if (
                reference_token_index is not None
                and int(reference_token_index) in selected_set
            ):
                protected_reference_uid = token_ref_space_uid[
                    int(reference_token_index)
                ]

            def uid_distance(uid: Any) -> float:
                uid_token_indices = selected_numpy[selected_uids == uid]
                uid_coordinates = centre_coordinates[uid_token_indices]
                uid_coordinates = uid_coordinates[
                    np.isfinite(uid_coordinates).all(axis=-1)
                ]
                if (
                    uid_coordinates.shape[0] == 0
                    or distance_targets.shape[0] == 0
                ):
                    return float("inf")
                return float(
                    cdist(uid_coordinates, distance_targets)
                    .min(axis=-1)
                    .max()
                )

            candidate_uids = [
                uid
                for uid in np.unique(selected_uids)
                if uid not in protected_required_uids
            ]
            unprotected_uids = [
                uid
                for uid in candidate_uids
                if uid != protected_reference_uid
            ]
            protected_reference_candidates = [
                uid
                for uid in candidate_uids
                if uid == protected_reference_uid
            ]
            removable_uids = sorted(
                unprotected_uids,
                key=lambda uid: (uid_distance(uid), str(uid)),
                reverse=True,
            ) + protected_reference_candidates

            keep_mask = np.ones(
                selected_token_indices.numel(),
                dtype=bool,
            )
            removed_count = 0
            for uid in removable_uids:
                uid_positions = selected_uids == uid
                keep_mask[uid_positions] = False
                removed_count += int(uid_positions.sum())
                if removed_count >= slots_needed:
                    break

            if removed_count < slots_needed:
                # Pathological mixed-UID data may place supervised and
                # unsupervised atom tokens in one ref_space_uid. The hard
                # supervision minimum takes precedence; remove only
                # unsupervised tokens as a last resort.
                remaining_positions = np.flatnonzero(
                    np.logical_and(keep_mask, ~selected_required)
                )
                for position in remaining_positions[::-1]:
                    keep_mask[position] = False
                    removed_count += 1
                    if removed_count >= slots_needed:
                        break
            if removed_count < slots_needed:
                raise RuntimeError(
                    "failed to make room for required tokens without "
                    "removing already-supervised tokens"
                )
            selected_token_indices = selected_token_indices[
                torch.from_numpy(keep_mask)
            ]

        selected_token_indices = torch.unique(
            torch.cat(
                (
                    selected_token_indices,
                    torch.tensor(insertion_indices, dtype=torch.long),
                )
            ),
            sorted=True,
        )
        if selected_token_indices.numel() > self.crop_size:
            raise RuntimeError(
                "failed to enforce required_token_mask within crop size"
            )
        selected_required_count = int(
            self.required_token_mask[selected_token_indices].sum()
        )
        if selected_required_count < self.min_required_tokens:
            raise RuntimeError(
                "failed to insert enough required tokens into crop: "
                f"{selected_required_count} < {self.min_required_tokens}"
            )
        return selected_token_indices

    def _ensure_required_token(
        self,
        selected_token_indices: torch.Tensor,
        reference_token_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for the plural minimum-enforcing path."""
        return self._ensure_required_tokens(
            selected_token_indices=selected_token_indices,
            reference_token_index=reference_token_index,
        )

    def random_crop_method(self) -> str:
        """
        Choose a random cropping method based on the given weights.

        Returns:
            str: The name of the randomly selected cropping method.
        """
        return random.choices(self.cand_crop_methods, k=1, weights=self.method_weights)[
            0
        ]

    def get_token_dist_mat(self, token_indices_in_ref: np.ndarray) -> np.ndarray:
        """
        Get the distance matrix of the tokens in the reference chain.

        Args:
            token_indices_in_ref (list): The indices of the tokens in the reference chain.

        Returns:
            numpy.ndarray: The distance matrix of the tokens in the reference chain,
                           shape=(len(tokens_in_ref_chain), len(tokens)).
        """
        centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atom_coords = self.atom_array.coord[centre_atom_indices]

        partial_token_dist_matrix = cdist(
            centre_atom_coords[token_indices_in_ref],
            centre_atom_coords,
            "euclidean",
        )

        assert partial_token_dist_matrix.shape == (
            len(token_indices_in_ref),
            len(self.token_array),
        )
        return partial_token_dist_matrix

    def extract_info(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """
        Extract information from the token array and atom array.

        Returns:
            tuple: A tuple containing the following elements:
                - tokens (torch.Tensor): The token array.
                - chain_id (torch.Tensor): The chain IDs of the atoms.
                - token_dist_mask_1d (torch.Tensor): The distance mask of the tokens.
                - token_indices_in_ref (list[int]): The indices of the tokens in the reference chain.
                - is_ligand (torch.Tensor): Whether chain type is ligand.
        """
        tokens = self.token_array.get_values()
        chain_id = []
        token_dist_mask_1d = []
        token_indices_in_ref = []

        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]
        chain_id = centre_atoms.asym_id_int
        token_dist_mask_1d = centre_atoms.is_resolved
        token_indices_in_ref = np.where(
            np.isin(centre_atoms.asym_id_int, self.ref_chain_indices)
        )[0]
        is_ligand = centre_atoms.is_ligand

        tokens = torch.Tensor(tokens)
        chain_id = torch.Tensor(chain_id)
        token_dist_mask_1d = torch.Tensor(token_dist_mask_1d)
        is_ligand = torch.Tensor(is_ligand)
        return tokens, chain_id, token_dist_mask_1d, token_indices_in_ref, is_ligand

    def crop_by_indices(
        self,
        selected_token_indices: torch.Tensor,
    ) -> tuple[TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop the token array, atom array, msa features and template features based on the selected token indices.
        """
        return self.select_by_token_indices(
            token_array=self.token_array,
            atom_array=self.atom_array,
            selected_token_indices=selected_token_indices,
        )

    @staticmethod
    def select_by_token_indices(
        token_array: TokenArray,
        atom_array: AtomArray,
        selected_token_indices: torch.Tensor,
    ) -> tuple[TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop the token array, atom array, msa features and template features based on the selected token indices.

        Args:
            token_array (TokenArray): the input token array
            atom_array (AtomArray): the input atom array
            selected_token_indices (torch.Tensor): The indices of the tokens to be cropped.

        Returns:
            cropped_token_array (TokenArray): The cropped token array.
            cropped_atom_array (AtomArray): The cropped atom array.
        """
        cropped_token_array = copy.deepcopy(token_array[selected_token_indices])

        cropped_atom_indices = []
        totol_atom_num = 0
        for idx, token in enumerate(cropped_token_array):
            cropped_atom_indices.extend(token.atom_indices)
            centre_idx_in_token_atoms = token.atom_indices.index(
                token.centre_atom_index
            )
            token_atom_num = len(token.atom_indices)
            token.atom_indices = list(
                range(totol_atom_num, totol_atom_num + token_atom_num)
            )
            token.centre_atom_index = token.atom_indices[centre_idx_in_token_atoms]
            totol_atom_num += token_atom_num

        cropped_atom_array = copy.deepcopy(atom_array[cropped_atom_indices])
        assert len(cropped_token_array) == selected_token_indices.shape[0]

        return (
            cropped_token_array,
            cropped_atom_array,
        )

    def get_crop_indices(self, crop_method: str = None) -> torch.Tensor:
        """
        Get selected indices based on the selected crop method.

        Args:
            crop_method (str): The cropping method to be used. Default is None.
        Returns:
            selected_indices : torch.Tensor, shape=(N_selected, )
        """
        (
            tokens,
            chain_id,
            token_dist_mask_1d,
            token_indices_in_ref,
            is_ligand,
        ) = self.extract_info()

        assert (
            crop_method in self.cand_crop_methods
        ), f"Unknown crop method: {crop_method}"

        # add token level ref_space_uid
        ref_space_uid_token = self.atom_array.ref_space_uid[
            self.token_array.get_annotation("centre_atom_index")
        ]

        atom_num_in_tokens = []
        for token in self.token_array:
            atom_num_in_tokens.append(len(token.atom_indices))

        uid_num_dict = defaultdict(int)
        for idx, uid in enumerate(ref_space_uid_token):
            uid_num_dict[uid] += atom_num_in_tokens[idx]
        atom_sums = torch.tensor(
            [uid_num_dict[uid] for idx, uid in enumerate(ref_space_uid_token)]
        )
        assert (atom_sums > 0).all().item(), "zero atoms"

        ref_space_uid_token = torch.Tensor(ref_space_uid_token)

        if crop_method == "ContiguousCropping":
            selected_token_indices = get_continues_crop_index(
                tokens=tokens,
                chain_id=chain_id,
                ref_space_uid_token=ref_space_uid_token,
                atom_sums=atom_sums,
                crop_size=self.crop_size,
                crop_complete_ligand_unstdRes=self.contiguous_crop_complete_lig,
                drop_last=self.drop_last,
                remove_metal=self.remove_metal,
            )
            selected_token_indices = self._limit_nuc_tokens(
                selected_token_indices, reference_token_index=None
            )
            selected_token_indices = self._ensure_required_tokens(
                selected_token_indices,
                reference_token_index=None,
            )
            return (
                selected_token_indices,
                token_indices_in_ref[-1].item(),
            )

        else:
            interface_crop = (
                True if crop_method == "SpatialInterfaceCropping" else False
            )
            token_distance = self.get_token_dist_mat(
                token_indices_in_ref=token_indices_in_ref
            )
            token_distance_mask = (
                token_dist_mask_1d[token_indices_in_ref][:, None]
                * token_dist_mask_1d[None, :]
            )
            selected_token_indices, ref_idx_in_ref = get_spatial_crop_index(
                tokens=tokens,
                chain_id=chain_id,
                token_distance=torch.Tensor(token_distance),
                token_distance_mask=torch.Tensor(token_distance_mask),
                reference_chain_id=self.ref_chain_indices,
                ref_space_uid_token=ref_space_uid_token,
                crop_size=self.crop_size,
                crop_complete_ligand_unstdRes=self.spatial_crop_complete_lig,
                interface_crop=interface_crop,
            )
            abs_ref_idx = token_indices_in_ref[ref_idx_in_ref].item()
            selected_token_indices = self._limit_nuc_tokens(
                selected_token_indices, reference_token_index=abs_ref_idx
            )
            selected_token_indices = self._ensure_required_tokens(
                selected_token_indices,
                reference_token_index=abs_ref_idx,
            )
            return (
                selected_token_indices,
                abs_ref_idx,
            )
