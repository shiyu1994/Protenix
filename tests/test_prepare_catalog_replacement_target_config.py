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

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.runtime.prepare_catalog_replacement_target_config import (
    SOURCE_TRAIN_SETS,
    TARGET_TRAIN_SETS,
    TargetConfigError,
    prepare_target_config,
)


def _json_bytes(document) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_config() -> dict:
    child = {
        "augmentation": {"query_only_msa_probability": 0.1},
        "base_info": {
            "runtime_index_dir": "/legacy/runtime",
            "source_roots_fpath": "/legacy/runtime/roots.json",
            "mmcif_dir": "/legacy/runtime",
            "bioassembly_dict_dir": "/legacy/runtime",
            "indices_fpath": "/shared/provenance.csv.gz",
            "require_resolved_protein_crop": True,
        },
        "msa": {"max_input_unpaired_msa_depth": 16384},
    }
    return {
        "seed": 42,
        "iters_to_accumulate": 4,
        "max_steps": 50000,
        "lr": 0.001,
        "data": {
            "epoch_size": 10240,
            "train_sampler": {
                "replacement": True,
                "seed": -1,
                "sampler_type": "weighted",
                "train_sample_weights": [0.49, 0.005, 0.505, 0.02],
            },
            "train_sets": list(SOURCE_TRAIN_SETS),
            SOURCE_TRAIN_SETS[0]: {"fixture": "pdb"},
            SOURCE_TRAIN_SETS[1]: copy.deepcopy(child),
            SOURCE_TRAIN_SETS[2]: copy.deepcopy(child),
            SOURCE_TRAIN_SETS[3]: {"fixture": "disorder"},
            "global_policy": {"msa": "unchanged"},
        },
    }


def _write_source_run(tmp_path: Path, config: dict | None = None) -> Path:
    run = tmp_path / "source_run"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump(config or _source_config(), sort_keys=False),
        encoding="utf-8",
    )
    return run


def _write_runtime_pair(tmp_path: Path, *, ge200_marker: str = "same") -> tuple[Path, Path]:
    pair_id = "a" * 64

    def write_one(name: str, role: str, marker: str) -> Path:
        runtime = tmp_path / name
        runtime.mkdir()
        pair = {
            "format": "protenix.target_distillation_catalog_pair.v1",
            "pair_id": pair_id,
            "marker": marker,
            "children": {
                "legacy_residual": {"row_count": 3, "cluster_count": 2},
                "ge200_structural": {"row_count": 5, "cluster_count": 4},
            },
        }
        pair_payload = _json_bytes(pair)
        (runtime / "target_catalog_pair.json").write_bytes(pair_payload)
        (runtime / "roots.json").write_text('{"roots": []}\n', encoding="utf-8")
        row_count = 3 if role == "legacy_residual" else 5
        cluster_count = 2 if role == "legacy_residual" else 4
        manifest = {
            "schema_version": 1,
            "row_count": row_count,
            "cluster_count": cluster_count,
            "target_catalog": {
                "schema_version": 1,
                "role": role,
                "pair_id": pair_id,
                "pair_receipt": {
                    "path": "target_catalog_pair.json",
                    "size_bytes": len(pair_payload),
                    "sha256": hashlib.sha256(pair_payload).hexdigest(),
                },
            },
        }
        manifest_payload = _json_bytes(manifest)
        (runtime / "manifest.json").write_bytes(manifest_payload)
        success = {
            "schema_version": 1,
            "status": "complete",
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        }
        (runtime / "_SUCCESS.json").write_bytes(_json_bytes(success))
        return runtime

    residual = write_one("residual", "legacy_residual", "same")
    ge200 = write_one("ge200", "ge200_structural", ge200_marker)
    return residual, ge200


def test_atomically_derives_only_locked_catalog_delta(tmp_path: Path):
    source = _source_config()
    run = _write_source_run(tmp_path, source)
    residual, ge200 = _write_runtime_pair(tmp_path)
    output = tmp_path / "published_target_config"

    target_path = prepare_target_config(
        source_run_dir=run,
        residual_runtime_index_dir=residual,
        ge200_runtime_index_dir=ge200,
        output_dir=output,
        enforce_production_source_hash=False,
    )

    assert target_path == output / "target_config.yaml"
    assert (output / "receipt.json").is_file()
    assert (output / "_SUCCESS.json").is_file()
    target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert target["data"]["train_sets"] == TARGET_TRAIN_SETS
    assert SOURCE_TRAIN_SETS[1] not in target["data"]
    assert SOURCE_TRAIN_SETS[2] not in target["data"]
    for name, runtime in zip(TARGET_TRAIN_SETS[1:3], (residual, ge200)):
        base = target["data"][name]["base_info"]
        assert base["runtime_index_dir"] == str(runtime.resolve())
        assert base["source_roots_fpath"] == str(runtime.resolve() / "roots.json")
        assert base["mmcif_dir"] == str(runtime.resolve())
        assert base["bioassembly_dict_dir"] == str(runtime.resolve())
        assert base["indices_fpath"] == "/shared/provenance.csv.gz"
    assert target["data"][TARGET_TRAIN_SETS[0]] == source["data"][SOURCE_TRAIN_SETS[0]]
    assert target["data"][TARGET_TRAIN_SETS[3]] == source["data"][SOURCE_TRAIN_SETS[3]]
    assert target["data"]["train_sampler"] == source["data"]["train_sampler"]
    assert target["data"]["epoch_size"] == 10240
    assert target["seed"] == 42
    assert target["iters_to_accumulate"] == 4
    assert target["max_steps"] == 50000

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["checkpoint_step"] == 21399
    assert receipt["source"]["next_optimizer_update"] == 21400
    assert receipt["target"]["train_sample_weights"] == [0.49, 0.005, 0.505, 0.02]
    assert receipt["target_runtime_bindings"][TARGET_TRAIN_SETS[1]]["pair_id"] == "a" * 64


def test_records_explicit_source_checkpoint_boundary(tmp_path: Path):
    run = _write_source_run(tmp_path)
    residual, ge200 = _write_runtime_pair(tmp_path)
    output = tmp_path / "step21999_target_config"

    prepare_target_config(
        source_run_dir=run,
        source_checkpoint_step=21999,
        residual_runtime_index_dir=residual,
        ge200_runtime_index_dir=ge200,
        output_dir=output,
        enforce_production_source_hash=False,
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["checkpoint_step"] == 21999
    assert receipt["source"]["next_optimizer_update"] == 22000


@pytest.mark.parametrize("source_checkpoint_step", [-1, 49999])
def test_rejects_invalid_source_checkpoint_boundary(
    tmp_path: Path, source_checkpoint_step: int
):
    run = _write_source_run(tmp_path)
    residual, ge200 = _write_runtime_pair(tmp_path)
    output = tmp_path / "invalid_boundary"

    with pytest.raises(TargetConfigError):
        prepare_target_config(
            source_run_dir=run,
            source_checkpoint_step=source_checkpoint_step,
            residual_runtime_index_dir=residual,
            ge200_runtime_index_dir=ge200,
            output_dir=output,
            enforce_production_source_hash=False,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", "train_sampler", "train_sample_weights"), [0.48, 0.005, 0.515, 0.0]),
        (("data", "epoch_size"), 10241),
        (("seed",), 43),
        (("iters_to_accumulate",), 2),
        (("max_steps",), 60000),
    ],
)
def test_rejects_locked_source_contract_drift(tmp_path: Path, path, value):
    source = _source_config()
    node = source
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    run = _write_source_run(tmp_path, source)
    residual, ge200 = _write_runtime_pair(tmp_path)

    with pytest.raises(TargetConfigError, match="Locked source contract mismatch"):
        prepare_target_config(
            source_run_dir=run,
            residual_runtime_index_dir=residual,
            ge200_runtime_index_dir=ge200,
            output_dir=tmp_path / "output",
            enforce_production_source_hash=False,
        )
    assert not (tmp_path / "output").exists()


def test_rejects_nonidentical_target_pair_receipts(tmp_path: Path):
    run = _write_source_run(tmp_path)
    residual, ge200 = _write_runtime_pair(tmp_path, ge200_marker="different")

    with pytest.raises(TargetConfigError, match="byte-identical pair receipt"):
        prepare_target_config(
            source_run_dir=run,
            residual_runtime_index_dir=residual,
            ge200_runtime_index_dir=ge200,
            output_dir=tmp_path / "output",
            enforce_production_source_hash=False,
        )
    assert not (tmp_path / "output").exists()


def test_refuses_to_overwrite_published_output(tmp_path: Path):
    run = _write_source_run(tmp_path)
    residual, ge200 = _write_runtime_pair(tmp_path)
    output = tmp_path / "output"
    prepare_target_config(
        source_run_dir=run,
        residual_runtime_index_dir=residual,
        ge200_runtime_index_dir=ge200,
        output_dir=output,
        enforce_production_source_hash=False,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_target_config(
            source_run_dir=run,
            residual_runtime_index_dir=residual,
            ge200_runtime_index_dir=ge200,
            output_dir=output,
            enforce_production_source_hash=False,
        )
