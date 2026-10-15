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

"""Replay a completed prefix of the training sampler and audit ID coverage.

This utility reconstructs the *planned sampler rows* from the saved run
configuration.  It does not open structures, bioassembly pickles, MSAs, or
template features, and it never initializes CUDA.

The replay is exact when it is run with the same CPU PyTorch sampler
implementation and immutable dataset indices as training.  A dataset configured
with ``random_sample_if_failed=True`` can replace a broken planned row inside a
DataLoader worker; historical replacement rows cannot be recovered from the
sampler seed alone.  The compact bitmap therefore deliberately represents
planned sampler coverage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protenix.data.pipeline.dataloader import (  # noqa: E402
    DistributedHierarchicalWeightedSampler,
)
from protenix.data.pipeline.dataset import (  # noqa: E402
    BaseSingleDataset,
    WeightedMultiDataset,
    get_sample_weights,
)
from protenix.data.pipeline.distillation_runtime_index import (  # noqa: E402
    DistillationRuntimeIndex,
)


LOGGER = logging.getLogger("audit_train_sampler_coverage")
SCHEMA_VERSION = 1
BIT_ORDER = "little"


@dataclass
class DatasetCatalog:
    """Sampling definition and stable row-address lookup for one child."""

    name: str
    dataset: BaseSingleDataset
    sampling_spec: Mapping[str, Any] | Sequence[float]
    runtime_index: DistillationRuntimeIndex | None
    random_sample_if_failed: bool
    source_identity: Mapping[str, Any]

    @property
    def size(self) -> int:
        return len(self.dataset)


class _SizedDataset:
    """Minimal Dataset protocol required by PyTorch's sampler base class."""

    def __init__(self, size: int) -> None:
        self._size = int(size)

    def __len__(self) -> int:
        return self._size


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return document


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _effective_sampler_seed(config: Mapping[str, Any]) -> int:
    data = config["data"]
    configured = data["train_sampler"].get("seed", -1)
    if configured is None or int(configured) < 0:
        return int(config["seed"])
    return int(configured)


def _world_size(run_dir: Path, override: int | None) -> int:
    if override is not None:
        value = int(override)
    else:
        manifest_path = run_dir / "manifest" / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Cannot infer world size without manifest/run_manifest.json; "
                "pass --world-size"
            )
        value = int(_load_json(manifest_path)["distributed"]["world_size"])
    if value <= 0:
        raise ValueError(f"world size must be positive, got {value}")
    return value


def _build_single_dataset(
    *,
    name: str,
    dataset_config: Mapping[str, Any],
    global_data_config: Mapping[str, Any],
) -> BaseSingleDataset:
    """Build the index-only form of the production dataset."""

    base_info = dict(dataset_config["base_info"])
    return BaseSingleDataset(
        name=name,
        cropping_configs=dict(dataset_config["cropping_configs"]),
        msa_featurizer=None,
        template_featurizer=None,
        lig_atom_rename=dataset_config.get("lig_atom_rename", False),
        shuffle_mols=dataset_config.get("shuffle_mols", False),
        shuffle_sym_ids=dataset_config.get("shuffle_sym_ids", False),
        constraint={"enable": False},
        ref_pos_augment=global_data_config.get("train_ref_pos_augment", True),
        limits=global_data_config.get("limits", -1),
        **base_info,
    )


def _source_identity(
    dataset: BaseSingleDataset,
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_index = dataset.runtime_distillation_index
    if runtime_index is not None:
        return {
            "kind": "runtime_distillation_index",
            "runtime_dir": str(runtime_index.runtime_dir),
            "row_count": len(runtime_index),
            "cluster_count": runtime_index.cluster_count,
            "sampling_fingerprint": runtime_index.sampling_fingerprint,
        }

    index_path = Path(dataset_config["base_info"]["indices_fpath"]).resolve()
    filter_contract = {
        key: dataset_config["base_info"].get(key)
        for key in (
            "pdb_list",
            "max_n_token",
            "min_n_token",
            "exclusion",
            "remove_pure_nuc",
            "group_by_pdb_id",
            "sort_by_n_token",
            "find_eval_chain_interface",
        )
    }
    return {
        "kind": "csv_index",
        "indices_path": str(index_path),
        "indices_sha256": _sha256_file(index_path),
        "filtered_row_count": len(dataset),
        "filter_contract_sha256": _sha256_json(filter_contract),
    }


def build_catalogs(config: Mapping[str, Any]) -> list[DatasetCatalog]:
    """Recreate production child indices and child sampling definitions."""

    data_config = config["data"]
    train_sets = list(data_config["train_sets"])
    masses = list(data_config["train_sampler"]["train_sample_weights"])
    if len(train_sets) != len(masses):
        raise ValueError("train_sets and train_sample_weights have different lengths")

    catalogs: list[DatasetCatalog] = []
    for name in train_sets:
        dataset_config = data_config[name]
        dataset = _build_single_dataset(
            name=name,
            dataset_config=dataset_config,
            global_data_config=data_config,
        )
        runtime_index = dataset.runtime_distillation_index
        sampler_config = dict(dataset_config["sampler_configs"])
        if runtime_index is not None:
            if sampler_config.get("sampler_type") != "clustered":
                raise ValueError(
                    f"{name}: runtime index requires sampler_type='clustered'"
                )
            sampling_spec: Mapping[str, Any] | Sequence[float] = {
                "kind": "clustered",
                "source": runtime_index,
            }
        else:
            sampling_spec = get_sample_weights(
                **sampler_config,
                indices_df=dataset.indices_list,
            )
        catalogs.append(
            DatasetCatalog(
                name=name,
                dataset=dataset,
                sampling_spec=sampling_spec,
                runtime_index=runtime_index,
                random_sample_if_failed=bool(
                    dataset_config["base_info"].get(
                        "random_sample_if_failed",
                        False,
                    )
                ),
                source_identity=_source_identity(dataset, dataset_config),
            )
        )
    return catalogs


def _hierarchical_sampler(
    *,
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
    epoch_size: int,
    world_size: int,
    seed: int,
) -> DistributedHierarchicalWeightedSampler | None:
    if not any(
        isinstance(catalog.sampling_spec, Mapping)
        and catalog.sampling_spec.get("kind") == "clustered"
        for catalog in catalogs
    ):
        return None

    child_specs = [
        catalog.sampling_spec
        if isinstance(catalog.sampling_spec, Mapping)
        and "kind" in catalog.sampling_spec
        else {"kind": "weighted", "weights": catalog.sampling_spec}
        for catalog in catalogs
    ]
    offsets = [0]
    for catalog in catalogs:
        offsets.append(offsets[-1] + catalog.size)
    return DistributedHierarchicalWeightedSampler(
        _SizedDataset(offsets[-1]),
        dataset_offsets=offsets,
        dataset_masses=masses,
        child_sampling_specs=child_specs,
        num_samples=epoch_size,
        num_replicas=world_size,
        rank=0,
        replacement=True,
        seed=seed,
        emit_sample_occurrence=True,
    )


def _flat_weighted_weights(
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
) -> torch.Tensor:
    """Use the production container to preserve its float conversion order."""

    child_weights: list[Sequence[float]] = []
    for catalog in catalogs:
        if isinstance(catalog.sampling_spec, Mapping):
            raise ValueError("A clustered child requires hierarchical sampling")
        child_weights.append(catalog.sampling_spec)
    combined = WeightedMultiDataset(
        datasets=[catalog.dataset for catalog in catalogs],
        dataset_names=[catalog.name for catalog in catalogs],
        datapoint_weights=child_weights,
        dataset_sample_weights=list(masses),
    )
    return combined.merged_datapoint_weights


def replay_sampler_counts(
    *,
    catalogs: Sequence[DatasetCatalog],
    masses: Sequence[float],
    updates: int,
    accumulation: int,
    world_size: int,
    epoch_size: int,
    seed: int,
    replacement: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Replay a global sampler prefix and count occurrences per child row."""

    if updates < 0:
        raise ValueError("updates must be non-negative")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    if epoch_size <= 0:
        raise ValueError("epoch_size must be positive")
    if epoch_size % world_size != 0:
        raise ValueError(
            "This coverage audit currently requires epoch_size to be divisible "
            f"by world_size, got {epoch_size} and {world_size}"
        )
    if not replacement:
        raise ValueError("Only the production replacement=True profile is supported")

    total_occurrences = int(updates) * int(accumulation) * int(world_size)
    counts = [np.zeros(catalog.size, dtype=np.uint32) for catalog in catalogs]
    offsets = np.cumsum(
        np.asarray([0] + [catalog.size for catalog in catalogs], dtype=np.int64)
    )
    hierarchical = _hierarchical_sampler(
        catalogs=catalogs,
        masses=masses,
        epoch_size=epoch_size,
        world_size=world_size,
        seed=seed,
    )
    flat_weights = (
        None
        if hierarchical is not None
        else _flat_weighted_weights(catalogs, masses)
    )

    remaining = total_occurrences
    epoch = 0
    while remaining:
        if hierarchical is not None:
            hierarchical.set_epoch(epoch)
            flat_indices = np.asarray(
                hierarchical._global_indices(),
                dtype=np.int64,
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(seed + epoch)
            flat_indices = (
                torch.multinomial(
                    flat_weights,
                    epoch_size,
                    replacement=True,
                    generator=generator,
                )
                .numpy()
                .astype(np.int64, copy=False)
            )

        take = min(remaining, epoch_size)
        flat_indices = flat_indices[:take]
        child_ordinals = np.searchsorted(offsets[1:], flat_indices, side="right")
        for child_ordinal, child_counts in enumerate(counts):
            selected = child_ordinals == child_ordinal
            if not bool(selected.any()):
                continue
            local_rows = flat_indices[selected] - offsets[child_ordinal]
            np.add.at(child_counts, local_rows, 1)
        remaining -= take
        epoch += 1

    replay = {
        "total_occurrences": total_occurrences,
        "consumed_global_batches": int(updates) * int(accumulation),
        "full_epochs": total_occurrences // epoch_size,
        "partial_epoch_occurrences": total_occurrences % epoch_size,
        "next_sampler_epoch": total_occurrences // epoch_size,
        "next_global_epoch_offset": total_occurrences % epoch_size,
        "next_local_epoch_offset": (
            total_occurrences % epoch_size
        )
        // world_size,
        "sampler_kind": (
            "DistributedHierarchicalWeightedSampler"
            if hierarchical is not None
            else "DistributedWeightedSampler"
        ),
        "sampler_signature_rank0": (
            hierarchical.resume_signature()
            if hierarchical is not None
            else {
                "sampler_type": "DistributedWeightedSampler",
                "seed": seed,
                "num_samples": epoch_size,
                "replacement": replacement,
                "num_replicas": world_size,
            }
        ),
    }
    return counts, replay


def _pack_seen_bitmap(counts: np.ndarray, path: Path) -> int:
    seen = counts > 0
    packed = np.packbits(seen, bitorder=BIT_ORDER)
    path.write_bytes(packed.tobytes())
    return int(seen.sum())


def _write_seen_rows_npz(counts: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    row_ids = np.flatnonzero(counts).astype(np.int64, copy=False)
    sample_counts = counts[row_ids]
    np.savez_compressed(path, row_id=row_ids, sample_count=sample_counts)
    return row_ids, sample_counts


def _runtime_data_id(cif_relpath: str) -> str:
    return PurePosixPath(cif_relpath).stem


def _write_runtime_seen_ids(
    *,
    runtime_index: DistillationRuntimeIndex,
    row_ids: np.ndarray,
    sample_counts: np.ndarray,
    path: Path,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.write("data_id\tsampler_row_id\tsample_count\troot_id\tcif_relpath\n")
        for row_id, count in zip(row_ids.tolist(), sample_counts.tolist()):
            root_id, cif_relpath = runtime_index.get_address(row_id)
            handle.write(
                f"{_runtime_data_id(cif_relpath)}\t{row_id}\t{count}\t"
                f"{root_id}\t{cif_relpath}\n"
            )


def _dataframe_data_id_counts(
    dataset: BaseSingleDataset,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset.group_by_pdb_id:
        data_ids = np.asarray(
            [str(frame.iloc[0]["pdb_id"]) for frame in dataset.indices_list],
            dtype=object,
        )
    else:
        data_ids = dataset.indices_list["pdb_id"].astype(str).to_numpy()
    unique_ids, inverse = np.unique(data_ids, return_inverse=True)
    occurrence_counts = np.bincount(
        inverse,
        weights=counts.astype(np.float64, copy=False),
        minlength=len(unique_ids),
    ).astype(np.uint64)
    return unique_ids, occurrence_counts


def _write_dataframe_seen_rows(
    *,
    dataset: BaseSingleDataset,
    row_ids: np.ndarray,
    sample_counts: np.ndarray,
    path: Path,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.write(
            "data_id\tsampler_row_id\tsample_count\ttype\tchain_1_id\tchain_2_id\n"
        )
        for row_id, count in zip(row_ids.tolist(), sample_counts.tolist()):
            row = dataset._get_sample_indice(row_id)
            chain_2 = "" if row.get("chain_2_id") is None else str(row.get("chain_2_id"))
            if chain_2.lower() == "nan":
                chain_2 = ""
            handle.write(
                f"{row['pdb_id']}\t{row_id}\t{count}\t{row.get('type', '')}\t"
                f"{row.get('chain_1_id', '')}\t{chain_2}\n"
            )


def _write_id_counts(
    *,
    ids: np.ndarray,
    counts: np.ndarray,
    path: Path,
    seen: bool,
) -> None:
    selected = counts > 0 if seen else counts == 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
        if seen:
            handle.write("data_id\tsample_count\n")
            for data_id, count in zip(ids[selected], counts[selected]):
                handle.write(f"{data_id}\t{int(count)}\n")
        else:
            handle.write("data_id\n")
            for data_id in ids[selected]:
                handle.write(f"{data_id}\n")


def _write_runtime_unseen_ids(
    *,
    runtime_index: DistillationRuntimeIndex,
    counts: np.ndarray,
    path: Path,
    chunk_rows: int = 1_000_000,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.write("data_id\n")
        for start in range(0, len(counts), chunk_rows):
            stop = min(start + chunk_rows, len(counts))
            unseen_rows = np.flatnonzero(counts[start:stop] == 0) + start
            for row_id in unseen_rows.tolist():
                _, cif_relpath = runtime_index.get_address(row_id)
                handle.write(f"{_runtime_data_id(cif_relpath)}\n")


def merge_initial_coverages(
    *,
    catalogs: Sequence[DatasetCatalog],
    counts: Sequence[np.ndarray],
    coverage_dirs: Sequence[Path],
) -> list[dict[str, Any]]:
    """Union prior sampler stages after proving that row catalogs match."""

    merged: list[dict[str, Any]] = []
    catalog_by_name = {
        catalog.name: (catalog, child_counts)
        for catalog, child_counts in zip(catalogs, counts)
    }
    for raw_dir in coverage_dirs:
        coverage_dir = raw_dir.expanduser().resolve(strict=True)
        summary_path = coverage_dir / "summary.json"
        summary = _load_json(summary_path)
        if summary.get("event") != "planned_train_sampler_coverage":
            raise ValueError(
                f"Not a sampler coverage summary: {summary_path}"
            )
        prior_datasets = summary.get("datasets")
        if not isinstance(prior_datasets, dict):
            raise ValueError(f"Coverage summary has no datasets: {summary_path}")

        merged_occurrences = 0
        merged_datasets: list[str] = []
        for name, prior_record in prior_datasets.items():
            if name not in catalog_by_name:
                raise ValueError(
                    f"Prior coverage dataset {name!r} is absent from the "
                    "current run; merge it against a superset run config"
                )
            catalog, child_counts = catalog_by_name[name]
            if int(prior_record["catalog_rows"]) != catalog.size:
                raise ValueError(
                    f"{name}: prior/current catalog row counts differ"
                )
            if prior_record.get("source_identity") != dict(
                catalog.source_identity
            ):
                raise ValueError(
                    f"{name}: prior/current immutable source identities differ"
                )
            rows_name = prior_record["artifacts"]["seen_rows_npz"]
            rows_path = coverage_dir / rows_name
            with np.load(rows_path) as arrays:
                row_ids = arrays["row_id"].astype(np.int64, copy=False)
                sample_counts = arrays["sample_count"].astype(
                    np.uint64,
                    copy=False,
                )
            if len(row_ids) != len(sample_counts):
                raise ValueError(f"{rows_path}: row/count lengths differ")
            if len(row_ids) and (
                int(row_ids[0]) < 0 or int(row_ids[-1]) >= catalog.size
            ):
                raise ValueError(f"{rows_path}: row ID outside current catalog")
            updated = child_counts[row_ids].astype(np.uint64) + sample_counts
            if len(updated) and int(updated.max()) > np.iinfo(np.uint32).max:
                raise OverflowError(f"{name}: uint32 occurrence count overflow")
            child_counts[row_ids] = updated.astype(np.uint32)
            merged_occurrences += int(sample_counts.sum())
            merged_datasets.append(name)

        merged.append(
            {
                "coverage_dir": str(coverage_dir),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256_file(summary_path),
                "run_dir": summary.get("run_dir"),
                "updates": summary.get("updates"),
                "datasets": sorted(merged_datasets),
                "sampled_occurrences_added": merged_occurrences,
            }
        )
    return merged


def write_coverage(
    *,
    output_dir: Path,
    run_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    catalogs: Sequence[DatasetCatalog],
    counts: Sequence[np.ndarray],
    replay: Mapping[str, Any],
    updates: int,
    accumulation: int,
    world_size: int,
    epoch_size: int,
    seed: int,
    masses: Sequence[float],
    write_unseen_ids: bool,
    initial_coverages: Sequence[Mapping[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_summaries: dict[str, Any] = {}

    for catalog, child_counts in zip(catalogs, counts):
        bitmap_path = output_dir / f"{catalog.name}.seen_bitmap.bin"
        rows_path = output_dir / f"{catalog.name}.seen_rows.npz"
        seen_ids_path = output_dir / f"{catalog.name}.seen_ids.tsv.gz"
        seen_rows, sample_counts = _write_seen_rows_npz(child_counts, rows_path)
        unique_seen_rows = _pack_seen_bitmap(child_counts, bitmap_path)
        if unique_seen_rows != len(seen_rows):
            raise RuntimeError(f"{catalog.name}: bitmap and row list disagree")

        if catalog.runtime_index is not None:
            _write_runtime_seen_ids(
                runtime_index=catalog.runtime_index,
                row_ids=seen_rows,
                sample_counts=sample_counts,
                path=seen_ids_path,
            )
            unique_data_ids = catalog.size
            unique_seen_data_ids = unique_seen_rows
            if write_unseen_ids:
                _write_runtime_unseen_ids(
                    runtime_index=catalog.runtime_index,
                    counts=child_counts,
                    path=output_dir / f"{catalog.name}.unseen_ids.tsv.gz",
                )
        else:
            _write_dataframe_seen_rows(
                dataset=catalog.dataset,
                row_ids=seen_rows,
                sample_counts=sample_counts,
                path=output_dir / f"{catalog.name}.seen_rows.tsv.gz",
            )
            data_ids, data_id_counts = _dataframe_data_id_counts(
                catalog.dataset,
                child_counts,
            )
            unique_data_ids = len(data_ids)
            unique_seen_data_ids = int((data_id_counts > 0).sum())
            _write_id_counts(
                ids=data_ids,
                counts=data_id_counts,
                path=seen_ids_path,
                seen=True,
            )
            if write_unseen_ids:
                _write_id_counts(
                    ids=data_ids,
                    counts=data_id_counts,
                    path=output_dir / f"{catalog.name}.unseen_ids.tsv.gz",
                    seen=False,
                )

        sampled_occurrences = int(child_counts.astype(np.uint64).sum())
        dataset_summaries[catalog.name] = {
            "catalog_rows": catalog.size,
            "sampled_occurrences": sampled_occurrences,
            "unique_seen_rows": unique_seen_rows,
            "unseen_rows": catalog.size - unique_seen_rows,
            "row_coverage_fraction": unique_seen_rows / catalog.size,
            "unique_data_ids": unique_data_ids,
            "unique_seen_data_ids": unique_seen_data_ids,
            "unseen_data_ids": unique_data_ids - unique_seen_data_ids,
            "random_sample_if_failed": catalog.random_sample_if_failed,
            "source_identity": dict(catalog.source_identity),
            "artifacts": {
                "seen_bitmap": bitmap_path.name,
                "seen_bitmap_bit_order": BIT_ORDER,
                "seen_bitmap_valid_bits": catalog.size,
                "seen_rows_npz": rows_path.name,
                "seen_ids": seen_ids_path.name,
                "unseen_ids": (
                    f"{catalog.name}.unseen_ids.tsv.gz"
                    if write_unseen_ids
                    else None
                ),
            },
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "event": "planned_train_sampler_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "planned sampler target rows; data-worker replacements are excluded",
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "torch_version": torch.__version__,
        "updates": updates,
        "last_zero_based_update": updates - 1 if updates else None,
        "world_size": world_size,
        "accumulation": accumulation,
        "global_targets_per_update": world_size * accumulation,
        "epoch_size": epoch_size,
        "effective_sampler_seed": seed,
        "replacement": bool(config["data"]["train_sampler"].get("replacement", True)),
        "train_sets": [catalog.name for catalog in catalogs],
        "train_sample_weights_raw": [float(value) for value in masses],
        "train_sample_weights_normalized": [
            float(value) / float(sum(masses)) for value in masses
        ],
        "write_unseen_ids": write_unseen_ids,
        "initial_coverages": [dict(value) for value in initial_coverages],
        "coverage_total_occurrences": sum(
            value["sampled_occurrences"]
            for value in dataset_summaries.values()
        ),
        "replay": dict(replay),
        "datasets": dataset_summaries,
        "caveats": [
            "This is sampler-planned row coverage, not a replay of processed crop/MSA/template tensors.",
            "Rows replaced after random_sample_if_failed are not recoverable from the sampler seed.",
            (
                "Exact replay requires unchanged indices, weights, cluster "
                "arrays, CPU PyTorch sampler implementation, and cursor."
            ),
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a training sampler prefix and report planned seen/unseen IDs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.yaml and run manifest.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        required=True,
        help="Number of completed optimizer updates (zero-based step U means U+1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory in which to publish the coverage artifacts.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Override manifest world size (normally unnecessary).",
    )
    parser.add_argument(
        "--write-unseen-ids",
        action="store_true",
        help=(
            "Stream complete unseen-ID gzip files. For the 33M distillation "
            "catalog this is intentionally off by default and takes minutes."
        ),
    )
    parser.add_argument(
        "--initial-coverage",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prior coverage directory to union into this stage. Repeat for "
            "multi-stage model history; immutable row identities must match."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    start = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )

    config = _load_yaml(config_path)
    world_size = _world_size(run_dir, args.world_size)
    accumulation = int(config.get("iters_to_accumulate", 1))
    epoch_size = int(config["data"]["epoch_size"])
    seed = _effective_sampler_seed(config)
    replacement = bool(config["data"]["train_sampler"].get("replacement", True))
    masses = list(config["data"]["train_sampler"]["train_sample_weights"])
    LOGGER.info(
        "building index-only catalogs: run=%s updates=%d world_size=%d "
        "accumulation=%d epoch_size=%d seed=%d",
        run_dir,
        args.updates,
        world_size,
        accumulation,
        epoch_size,
        seed,
    )
    catalogs = build_catalogs(config)
    LOGGER.info(
        "catalogs: %s",
        ", ".join(f"{catalog.name}={catalog.size:,}" for catalog in catalogs),
    )
    counts, replay = replay_sampler_counts(
        catalogs=catalogs,
        masses=masses,
        updates=args.updates,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
        replacement=replacement,
    )
    if sum(int(child.astype(np.uint64).sum()) for child in counts) != int(
        replay["total_occurrences"]
    ):
        raise RuntimeError("Replay occurrence accounting mismatch")
    initial_coverages = merge_initial_coverages(
        catalogs=catalogs,
        counts=counts,
        coverage_dirs=args.initial_coverage,
    )
    summary_path = write_coverage(
        output_dir=output_dir,
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        catalogs=catalogs,
        counts=counts,
        replay=replay,
        updates=args.updates,
        accumulation=accumulation,
        world_size=world_size,
        epoch_size=epoch_size,
        seed=seed,
        masses=masses,
        write_unseen_ids=args.write_unseen_ids,
        initial_coverages=initial_coverages,
    )
    elapsed = time.perf_counter() - start
    LOGGER.info("coverage audit complete in %.2fs: %s", elapsed, summary_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
