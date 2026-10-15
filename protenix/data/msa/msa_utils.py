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
import re
import string
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from io import StringIO
from typing import Any, List, Optional, Set, Tuple

import numpy as np

from protenix.data.constants import (
    DNA_CHAIN,
    MSA_DNA_SEQ_TO_ID,
    MSA_PROTEIN_SEQ_TO_ID,
    MSA_RNA_SEQ_TO_ID,
    PROTEIN_CHAIN,
    RNA_CHAIN,
    STANDARD_POLYMER_CHAIN_TYPES,
    STD_RESIDUES_WITH_GAP,
)
from protenix.data.tools.common import parse_fasta
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

# Type Aliases
FeatureDict = Mapping[str, np.ndarray]
MutableFeatureDict = MutableMapping[str, np.ndarray]

# Regex for species identification
_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_UNIREF_REGEX = re.compile(r"^UniRef100_[^_]+_([^_/]+)")

MSA_GAP_IDX = STD_RESIDUES_WITH_GAP.get("-")
NUM_SEQ_NUM_RES_MSA_FEATURES = ("msa", "msa_mask", "deletion_matrix")
NUM_SEQ_MSA_FEATURES = ("msa_species_identifiers",)
MSA_PAD_VALUES = {"msa": MSA_GAP_IDX, "msa_mask": 1, "deletion_matrix": 0}


class MSACore:
    """Basic MSA parsing and numerical conversion operations."""

    @staticmethod
    def parse_fasta_lines(
        lines: Iterable[str],
        depth_limit: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Parse a FASTA/A3M record prefix without consuming the tail."""
        if depth_limit is not None:
            if isinstance(depth_limit, (bool, np.bool_)) or not isinstance(
                depth_limit,
                (int, np.integer),
            ):
                raise TypeError(
                    "depth_limit must be a positive integer or None, got "
                    f"{depth_limit!r}"
                )
            depth_limit = int(depth_limit)
            if depth_limit < 1:
                raise ValueError(
                    f"depth_limit must be >= 1, got {depth_limit}"
                )

        sequences: List[str] = []
        descriptions: List[str] = []
        current_description: Optional[str] = None
        current_sequence: List[str] = []

        def flush_current() -> bool:
            if current_description is None:
                return False
            descriptions.append(current_description)
            sequences.append("".join(current_sequence))
            return (
                depth_limit is not None
                and len(sequences) >= depth_limit
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if flush_current():
                    return sequences, descriptions
                current_description = line[1:]
                current_sequence = []
            elif current_description is not None:
                current_sequence.append(line)

        flush_current()
        return sequences, descriptions

    @staticmethod
    def parse_fasta(
        fasta_str: str,
        depth_limit: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Parses a FASTA/A3M format string into sequences and descriptions.

        Args:
            fasta_str: The input string in FASTA or A3M format.

        Returns:
            A tuple of (list of sequences, list of descriptions).
        """
        if depth_limit is None:
            return parse_fasta(fasta_str)
        return MSACore.parse_fasta_lines(
            StringIO(fasta_str),
            depth_limit=depth_limit,
        )

    @staticmethod
    def sequences_to_array(
        sequences: Sequence[str],
        chain_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts sequence strings into numerical arrays and deletion matrices.

        Args:
            sequences: A list of aligned sequence strings.
            chain_type: Type of the chain (PROTEIN_CHAIN, DNA_CHAIN, RNA_CHAIN).

        Returns:
            A tuple of (msa_array, deletion_array).

        Raises:
            ValueError: If the chain type is unsupported.
        """
        char_maps = {
            PROTEIN_CHAIN: MSA_PROTEIN_SEQ_TO_ID,
            DNA_CHAIN: MSA_DNA_SEQ_TO_ID,
            RNA_CHAIN: MSA_RNA_SEQ_TO_ID,
        }
        if chain_type not in char_maps:
            raise ValueError(f"Unsupported chain type: {chain_type}")

        cmap = char_maps[chain_type]
        if not sequences:
            return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

        # Build lookup table: ASCII ordinal -> encoded value (-1 for insertion chars)
        lut = np.full(128, -1, dtype=np.int32)
        for char, val in cmap.items():
            lut[ord(char)] = val

        rows, cols = len(sequences), sum(1 for c in sequences[0] if c in cmap)
        msa_arr = np.zeros((rows, cols), dtype=np.int32)
        del_arr = np.zeros((rows, cols), dtype=np.int32)

        for i, seq in enumerate(sequences):
            char_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            mapped = lut[char_arr]
            aligned_mask = mapped >= 0
            aligned_vals = mapped[aligned_mask]
            # Deletion counts: number of insertion chars before each aligned position
            cum_inserts = np.cumsum(~aligned_mask)
            insert_before_aligned = cum_inserts[aligned_mask]
            del_counts = np.diff(insert_before_aligned, prepend=0)
            n = min(len(aligned_vals), cols)
            msa_arr[i, :n] = aligned_vals[:n]
            del_arr[i, :n] = del_counts[:n]
        return msa_arr, del_arr


class RawMsa:
    """
    Represents a single MSA and handles deduplication and featurization logic.

    Args:
        query: The query sequence string.
        chain_type: Type of the chain.
        sequences: List of sequence strings in the alignment.
        descriptions: List of description strings for the sequences.
        deduplicate: Whether to deduplicate sequences.
    """

    def __init__(
        self,
        query: str,
        chain_type: str,
        sequences: Sequence[str],
        descriptions: Sequence[str],
        deduplicate: bool = True,
        precomputed_deletion_matrix: Optional[np.ndarray] = None,
        depth_limit: Optional[int] = None,
    ) -> None:
        self.query = query
        self.chain_type = chain_type
        input_sequences = list(sequences)
        input_descriptions = list(descriptions)
        if len(input_sequences) != len(input_descriptions):
            raise ValueError(
                "MSA sequences and descriptions must have the same length: "
                f"{len(input_sequences)} != {len(input_descriptions)}"
            )

        deletion_matrix = None
        if precomputed_deletion_matrix is not None:
            deletion_matrix = np.asarray(precomputed_deletion_matrix)
            if deletion_matrix.ndim != 2:
                raise ValueError(
                    "precomputed_deletion_matrix must be rank 2, got "
                    f"shape {deletion_matrix.shape}"
                )
            if deletion_matrix.shape[0] != len(input_sequences):
                raise ValueError(
                    "precomputed_deletion_matrix depth does not match MSA "
                    f"sequences: {deletion_matrix.shape[0]} != "
                    f"{len(input_sequences)}"
                )
            if not np.issubdtype(deletion_matrix.dtype, np.integer):
                raise TypeError(
                    "precomputed_deletion_matrix must have integer dtype, got "
                    f"{deletion_matrix.dtype}"
                )
            if np.any(deletion_matrix < 0):
                raise ValueError(
                    "precomputed_deletion_matrix contains negative values"
                )

        if depth_limit is not None:
            if isinstance(depth_limit, (bool, np.bool_)) or not isinstance(
                depth_limit,
                (int, np.integer),
            ):
                raise TypeError(
                    "depth_limit must be a positive integer or None, got "
                    f"{depth_limit!r}"
                )
            depth_limit = int(depth_limit)
            if depth_limit < 1:
                raise ValueError(
                    f"depth_limit must be >= 1, got {depth_limit}"
                )
            input_sequences = input_sequences[:depth_limit]
            input_descriptions = input_descriptions[:depth_limit]
            if deletion_matrix is not None:
                deletion_matrix = deletion_matrix[:depth_limit]

        if deduplicate:
            keep_indices = self._deduplicate_indices(input_sequences)
            self.seqs = [input_sequences[i] for i in keep_indices]
            self.descs = [input_descriptions[i] for i in keep_indices]
            if deletion_matrix is not None:
                deletion_matrix = deletion_matrix[keep_indices]
        else:
            self.seqs, self.descs = input_sequences, input_descriptions

        # Make sure the MSA always has at least the query.
        if not self.seqs:
            self.seqs = [query]
            self.descs = ["Original query"]
            if deletion_matrix is not None:
                deletion_matrix = np.zeros(
                    (1, len(query)),
                    dtype=deletion_matrix.dtype,
                )

        self._precomputed_deletion_matrix = deletion_matrix
        self._deduplicated = bool(deduplicate)

        if not self._verify_query():
            logger.warning(
                "MSA first row does not match the canonical query for %s; "
                "falling back to query only.",
                chain_type,
            )
            self.seqs = [query]
            self.descs = ["Original query"]
            if self._precomputed_deletion_matrix is not None:
                self._precomputed_deletion_matrix = np.zeros(
                    (1, len(query)),
                    dtype=self._precomputed_deletion_matrix.dtype,
                )

    @staticmethod
    def _sequence_feature_ids(
        sequence: str,
        chain_type: str,
    ) -> Optional[List[int]]:
        """Encode one A3M row, ignoring only lowercase insertion residues."""
        char_maps = {
            PROTEIN_CHAIN: MSA_PROTEIN_SEQ_TO_ID,
            DNA_CHAIN: MSA_DNA_SEQ_TO_ID,
            RNA_CHAIN: MSA_RNA_SEQ_TO_ID,
        }
        char_map = char_maps.get(chain_type)
        if char_map is None:
            return None

        feature_ids = []
        for residue in sequence:
            if residue in char_map:
                feature_ids.append(char_map[residue])
            elif residue not in string.ascii_lowercase:
                return None
        return feature_ids

    def _verify_query(self) -> bool:
        """Ensure row zero is feature-equivalent to the canonical query."""
        if self.chain_type not in STANDARD_POLYMER_CHAIN_TYPES:
            return self.seqs[0] == self.query
        query_ids = self._sequence_feature_ids(self.query, self.chain_type)
        row_zero_ids = self._sequence_feature_ids(
            self.seqs[0],
            self.chain_type,
        )
        return query_ids is not None and query_ids == row_zero_ids

    @staticmethod
    def _deduplicate_indices(seqs: Sequence[str]) -> List[int]:
        """Return first-occurrence indices, ignoring insertion characters."""
        keep_indices, seen = [], set()
        table = str.maketrans("", "", string.ascii_lowercase)
        for index, sequence in enumerate(seqs):
            stripped = sequence.translate(table)
            if stripped not in seen:
                seen.add(stripped)
                keep_indices.append(index)
        return keep_indices

    @staticmethod
    def _deduplicate_sequences(
        seqs: Sequence[str], descs: Sequence[str]
    ) -> Tuple[List[str], List[str]]:
        """Removes duplicate sequences, ignoring insertions (lowercase characters)."""
        keep_indices = RawMsa._deduplicate_indices(seqs)
        return (
            [seqs[i] for i in keep_indices],
            [descs[i] for i in keep_indices],
        )

    @classmethod
    def from_aligned_arrays(
        cls,
        query: str,
        ctype: str,
        msa: np.ndarray,
        deletion_matrix: np.ndarray,
        descriptions: Sequence[str],
        dedup: bool = True,
        depth_limit: Optional[int] = None,
    ) -> "RawMsa":
        """Construct an MSA while preserving an existing deletion matrix.

        OpenFold3 alignment archives store an aligned ``N x L`` character
        matrix and the corresponding deletion counts separately.  Converting
        these arrays through A3M would discard insertion counts, so this
        constructor keeps them as authoritative numerical input.
        """
        msa = np.asarray(msa)
        deletion_matrix = np.asarray(deletion_matrix)
        if msa.ndim != 2:
            raise ValueError(f"msa must be rank 2, got shape {msa.shape}")
        if deletion_matrix.shape != msa.shape:
            raise ValueError(
                "deletion_matrix must match msa shape: "
                f"{deletion_matrix.shape} != {msa.shape}"
            )
        if msa.shape[0] == 0:
            raise ValueError("msa must contain at least the query row")
        if msa.dtype.kind not in {"S", "U"}:
            raise TypeError(
                f"msa must contain byte or Unicode characters, got {msa.dtype}"
            )

        if depth_limit is not None:
            if isinstance(depth_limit, (bool, np.bool_)) or not isinstance(
                depth_limit,
                (int, np.integer),
            ):
                raise TypeError(
                    "depth_limit must be a positive integer or None, got "
                    f"{depth_limit!r}"
                )
            depth_limit = int(depth_limit)
            if depth_limit < 1:
                raise ValueError(
                    f"depth_limit must be >= 1, got {depth_limit}"
                )
            msa = msa[:depth_limit]
            # Keep only a compact prefix. A view would retain the full
            # decompressed OpenFold payload through its ``base`` reference,
            # defeating the memory benefit of the training depth limit.
            deletion_matrix = deletion_matrix[:depth_limit].copy()
            descriptions = descriptions[:depth_limit]

        if msa.dtype.kind == "S" and msa.dtype.itemsize != 1:
            raise ValueError(
                f"byte-string msa entries must have width 1, got {msa.dtype}"
            )
        if (
            msa.dtype.kind == "U"
            and msa.dtype.itemsize != np.dtype("<U1").itemsize
        ):
            raise ValueError(
                f"Unicode msa entries must have width 1, got {msa.dtype}"
            )
        empty_value = b"" if msa.dtype.kind == "S" else ""
        if np.any(msa == empty_value):
            raise ValueError("msa entries must each contain exactly one character")

        sequences = [
            "".join(row.tolist())
            for row in msa.astype("<U1", copy=False)
        ]
        query_encoded, _ = MSACore.sequences_to_array([query], ctype)
        row_zero_encoded, _ = MSACore.sequences_to_array(
            [sequences[0]],
            ctype,
        )
        if not np.array_equal(query_encoded, row_zero_encoded):
            raise ValueError(
                "alignment query row does not match the canonical sequence"
            )
        if deletion_matrix.shape[1] != query_encoded.shape[1]:
            raise ValueError(
                "alignment width does not match the canonical sequence: "
                f"{deletion_matrix.shape[1]} != {query_encoded.shape[1]}"
            )

        return cls(
            query,
            ctype,
            sequences,
            [str(description) for description in descriptions],
            deduplicate=dedup,
            precomputed_deletion_matrix=deletion_matrix,
        )

    @classmethod
    def from_a3m(
        cls,
        query: str,
        ctype: str,
        a3m: str,
        depth_limit: Optional[int] = None,
        dedup: bool = True,
    ) -> "RawMsa":
        """
        Constructs a RawMsa instance from A3M content.

        Args:
            query: The query sequence.
            ctype: The chain type.
            a3m: The A3M format string.
            depth_limit: Optional maximum number of sequences to keep.
            dedup: Whether to deduplicate sequences.

        Returns:
            An instance of RawMsa.
        """
        s, d = MSACore.parse_fasta(
            a3m,
            depth_limit=depth_limit,
        )
        return cls(query, ctype, s, d, dedup)

    @classmethod
    def from_a3m_file(
        cls,
        query: str,
        ctype: str,
        path: Any,
        depth_limit: Optional[int] = None,
        dedup: bool = True,
    ) -> "RawMsa":
        """Construct an MSA without reading records beyond the top-N prefix."""
        with open(path, "r", encoding="utf-8") as handle:
            sequences, descriptions = MSACore.parse_fasta_lines(
                handle,
                depth_limit=depth_limit,
            )
        return cls(
            query,
            ctype,
            sequences,
            descriptions,
            deduplicate=dedup,
        )

    @classmethod
    def merge(
        cls,
        msas: Sequence["RawMsa"],
        deduplicate: bool = True,
        depth_limit: Optional[int] = None,
    ) -> "RawMsa":
        """
        Merges multiple RawMsa objects into one.

        Args:
            msas: A sequence of RawMsa objects.
            deduplicate: Whether to deduplicate the resulting merged MSA.

        Returns:
            A new RawMsa object representing the merged alignment.

        Raises:
            ValueError: If no MSAs are provided.
        """
        if not msas:
            raise ValueError("No MSAs provided for merging.")
        if depth_limit is not None:
            if isinstance(depth_limit, (bool, np.bool_)) or not isinstance(
                depth_limit,
                (int, np.integer),
            ):
                raise TypeError(
                    "depth_limit must be a positive integer or None, got "
                    f"{depth_limit!r}"
                )
            depth_limit = int(depth_limit)
            if depth_limit < 1:
                raise ValueError(
                    f"depth_limit must be >= 1, got {depth_limit}"
                )
        query = msas[0].query
        chain_type = msas[0].chain_type
        if any(m.query != query or m.chain_type != chain_type for m in msas):
            raise ValueError("Cannot merge MSAs with different queries or chain types")

        sequences = [sequence for msa in msas for sequence in msa.seqs]
        descriptions = [description for msa in msas for description in msa.descs]
        deletion_matrix = None
        if any(msa._precomputed_deletion_matrix is not None for msa in msas):
            deletion_matrices = [
                (
                    msa._precomputed_deletion_matrix
                    if msa._precomputed_deletion_matrix is not None
                    else msa.featurize()["deletion_matrix"]
                )
                for msa in msas
            ]
            deletion_matrix = np.concatenate(
                deletion_matrices,
                axis=0,
            )
        return cls(
            query,
            chain_type,
            sequences,
            descriptions,
            deduplicate,
            precomputed_deletion_matrix=deletion_matrix,
            depth_limit=depth_limit,
        )

    def shuffled_hits_preserving_query(self) -> "RawMsa":
        """Return a copy with hit rows shuffled and query row fixed."""
        if len(self.seqs) <= 2:
            return self
        hit_indices = np.arange(1, len(self.seqs))
        np.random.shuffle(hit_indices)
        order = np.concatenate(
            [np.zeros(1, dtype=np.int64), hit_indices]
        )
        deletion_matrix = self.featurize()["deletion_matrix"][order]
        return RawMsa(
            self.query,
            self.chain_type,
            [self.seqs[i] for i in order],
            [self.descs[i] for i in order],
            deduplicate=False,
            precomputed_deletion_matrix=deletion_matrix,
        )

    @property
    def depth(self) -> int:
        """Returns the number of sequences in the MSA."""
        return len(self.seqs)

    def featurize(self) -> MutableFeatureDict:
        """
        Extracts numerical features from the MSA.

        Returns:
            A mutable dictionary of MSA features.
        """
        msa_arr, del_arr = MSACore.sequences_to_array(self.seqs, self.chain_type)
        if self._precomputed_deletion_matrix is not None:
            if self._precomputed_deletion_matrix.shape != msa_arr.shape:
                raise ValueError(
                    "precomputed deletion matrix no longer matches encoded MSA: "
                    f"{self._precomputed_deletion_matrix.shape} != "
                    f"{msa_arr.shape}"
                )
            del_arr = self._precomputed_deletion_matrix.astype(
                np.int32,
                copy=False,
            )
        return {
            "msa": msa_arr,
            "deletion_matrix": del_arr,
            "msa_species_identifiers": np.array(
                MSAPairingEngine.get_species_ids(self.descs), dtype=object
            ),
            "num_alignments": np.array(len(self.seqs), dtype=np.int32),
        }

    def to_a3m(self) -> str:
        """Serializes the MSA back into A3M string format."""
        return "\n".join([f">{d}\n{s}" for d, s in zip(self.descs, self.seqs)]) + "\n"


class MSAPairingEngine:
    """Manages MSA pairing by species and post-pairing processing."""

    @staticmethod
    def get_species_ids(descriptions: Sequence[str]) -> List[str]:
        """Extracts species identifiers from UniProt or UniRef description lines."""
        ids = []
        for d in descriptions:
            d = d.strip()
            m = _UNIPROT_REGEX.match(d) or _UNIREF_REGEX.match(d)
            if m:
                ids.append(
                    m.group("SpeciesId") if "SpeciesId" in m.groupdict() else m.group(1)
                )
            else:
                ids.append("")
        return ids

    @staticmethod
    def _align_species(
        all_species: Sequence[str],
        chain_species_map: Sequence[Mapping[str, np.ndarray]],
        species_min_hits: Mapping[str, int],
    ) -> np.ndarray:
        """Aligns MSA row indices based on species."""
        species_blocks = []
        for species in all_species:
            chain_row_indices = []
            for species_to_rows in chain_species_map:
                min_msa_size = species_min_hits[species]
                if species not in species_to_rows:
                    row_indices = np.full(min_msa_size, fill_value=-1, dtype=np.int32)
                else:
                    row_indices = species_to_rows[species][:min_msa_size]
                chain_row_indices.append(row_indices)
            species_block = np.stack(chain_row_indices, axis=1)
            species_blocks.append(species_block)
        return np.concatenate(species_blocks, axis=0)

    @staticmethod
    def pair_chains_by_species(
        chains: Sequence[MutableFeatureDict],
        max_paired: int,
        active_chains: Set[str],
        max_per_species: int,
    ) -> List[MutableFeatureDict]:
        """
        Aligns MSA rows across different chains based on species identifiers.

        Args:
            chains: List of chain feature dictionaries.
            max_paired: Maximum number of paired rows.
            active_chains: Set of chain IDs that should be paired.
            max_per_species: Maximum number of rows allowed per species.

        Returns:
            A list of feature dictionaries with paired MSA information.
        """
        chain_species_map = []
        all_species_counts = {}
        species_min_hits = {}

        for c in chains:
            ids = c.get("msa_species_identifiers_all_seq", np.array([]))
            if (
                ids.size == 0
                or (ids.size == 1 and not ids[0])
                or c["chain_id"] not in active_chains
            ):
                chain_species_map.append({})
                continue

            row_indices = np.arange(len(ids))
            sort_idxs = ids.argsort()
            ids = ids[sort_idxs]
            row_indices = row_indices[sort_idxs]

            species, unique_row_indices = np.unique(ids, return_index=True)
            grouped_row_indices = np.split(row_indices, unique_row_indices[1:])
            mapping = dict(zip(species, grouped_row_indices))
            chain_species_map.append(mapping)

            for s in species:
                all_species_counts[s] = all_species_counts.get(s, 0) + 1

            for s, idxs in mapping.items():
                species_min_hits[s] = min(
                    species_min_hits.get(s, max_per_species), len(idxs)
                )

        ranked_species = {}
        for s, count in all_species_counts.items():
            if not s or count <= 1:
                continue
            ranked_species.setdefault(count, []).append(s)

        pair_idxs = [np.zeros((1, len(chains)), dtype=np.int32)]  # Always keep Query
        current_total = 0
        for count in sorted(ranked_species.keys(), reverse=True):
            all_species = ranked_species[count]
            rows = MSAPairingEngine._align_species(
                all_species, chain_species_map, species_min_hits
            )
            rank = np.sum(np.log(np.abs(rows.astype(np.float32)) + 1e-10), axis=1)
            pair_idxs.append(rows[np.argsort(rank)])
            current_total += rows.shape[0]
            if current_total >= max_paired:
                break

        final_idxs = np.concatenate(pair_idxs, axis=0)[:max_paired]

        new_chains = []
        for i, c in enumerate(chains):
            nc = {k: v for k, v in c.items() if "all_seq" not in k}
            sel = final_idxs[:, i]
            for f in ["msa", "deletion_matrix"]:
                src = c[f"{f}_all_seq"]
                padded = np.concatenate(
                    [src, np.full((1, src.shape[1]), MSA_PAD_VALUES[f], src.dtype)],
                    axis=0,
                )
                nc[f"{f}_all_seq"] = padded[sel]
            nc["num_alignments_all_seq"] = np.array(nc["msa_all_seq"].shape[0])
            new_chains.append(nc)
        return new_chains

    @staticmethod
    def cleanup_unpaired_features(
        chains: List[MutableFeatureDict],
    ) -> List[MutableFeatureDict]:
        """Removes sequences from unpaired MSA that are already present in paired MSA."""
        for c in chains:
            paired_bytes = {row.tobytes() for row in c["msa_all_seq"].astype(np.int8)}
            keep = [
                i
                for i, row in enumerate(c["msa"].astype(np.int8))
                if row.tobytes() not in paired_bytes
            ]
            for k in ["msa", "deletion_matrix", "msa_species_identifiers"]:
                c[k] = c[k][keep]
            c["num_alignments"] = np.array(c["msa"].shape[0], dtype=np.int32)
        return chains

    @staticmethod
    def filter_all_gapped_rows(
        chains: List[MutableFeatureDict], active_asyms: Sequence[int]
    ) -> List[MutableFeatureDict]:
        """Removes rows from the paired MSA that consist entirely of gaps."""
        subset = [c["msa_all_seq"] for c in chains if c["asym_id"][0] in active_asyms]
        if not subset:
            return chains
        non_gap_mask = np.any(np.concatenate(subset, axis=1) != MSA_GAP_IDX, axis=1)
        for c in chains:
            for k in ["msa_all_seq", "deletion_matrix_all_seq"]:
                if k in c:
                    c[k] = c[k][non_gap_mask]
            c["num_alignments_all_seq"] = np.array(np.sum(non_gap_mask))
        return chains

    @staticmethod
    def merge_chain_features(chains: List[FeatureDict], key: str) -> np.ndarray:
        """Merges MSA features across different chains, padding unpaired sequences where necessary."""
        if "_all_seq" in key:
            return np.concatenate([c[key] for c in chains], axis=1)
        max_d = max(c[key].shape[0] for c in chains)
        pads = [
            np.pad(
                c[key],
                ((0, max_d - c[key].shape[0]), (0, 0)),
                constant_values=MSA_PAD_VALUES.get(key, 0),
            )
            for c in chains
        ]
        return np.concatenate(pads, axis=1)


def map_to_standard(
    asym_ids: np.ndarray, res_ids: np.ndarray, meta: Mapping[int, Mapping[str, Any]]
) -> np.ndarray:
    """
    Maps residue indices to a standardized MSA coordinate system.

    Args:
        asym_ids: Array of asymmetric IDs.
        res_ids: Array of residue IDs.
        meta: Metadata dictionary containing chain info and sequences.

    Returns:
        Array of standardized indices.
    """
    uids = [f"{a}-{b}" for a, b in zip(asym_ids, res_ids)]
    std_uids = []
    for aid, info in meta.items():
        std_uids.extend([f"{aid}-{x}" for x in range(1, len(info["sequence"]) + 1)])

    lookup = {uid: i for i, uid in enumerate(std_uids)}
    return np.array(
        [lookup.get(u, lookup.get(f"{u.split('-')[0]}-1", -1)) for u in uids],
        dtype=np.int32,
    )
