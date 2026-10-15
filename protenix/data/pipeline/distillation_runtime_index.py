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

"""Lightweight random access to a compact distillation runtime index."""

from __future__ import annotations

import hashlib
import json
import mmap
import operator
import os
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, Mapping

import numpy as np


RUNTIME_SCHEMA_VERSION = 1
_FORMAT_NAME = "protenix_runtime_distillation_index"
_ARRAY_SPECS = {
    "row_offsets": (8, "artifacts"),
    "source_root_ids": (1, "artifacts"),
    "cif_relpaths": (1, "artifacts"),
    "cluster_ids": (4, "sampling"),
    "first_row_ids": (8, "sampling"),
    "cluster_sizes": (4, "sampling"),
}
_OPTIONAL_ARRAY_SPECS = {
    "a3m_row_offsets": (8, "artifacts"),
    "a3m_relpaths": (1, "artifacts"),
    "bioassembly_row_offsets": (8, "artifacts"),
    "bioassembly_relpaths": (1, "artifacts"),
    "global_id_offsets": (8, "artifacts"),
    "global_ids": (1, "artifacts"),
    "source_member_row_ids": (8, "artifacts"),
}


@dataclass(frozen=True, slots=True)
class DistillationPaths:
    """Resolved files for one row-addressed predicted protein."""

    root_id: str
    cif_relpath: str
    cif_path: Path
    a3m_path: Path
    bioassembly_path: Path
    global_id: str | None = None
    source_member_row_id: int | None = None


def _safe_relative(value: object) -> PurePosixPath:
    raw = str(value).strip()
    if not raw or "\x00" in raw or "\\" in raw:
        raise ValueError(f"Unsafe relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe relative path: {raw!r}")
    return path


def _is_cif_path(path: PurePosixPath) -> bool:
    """Accept normal mmCIF and the gzipped AFU mmCIF representation."""

    lower = path.name.lower()
    return lower.endswith(".cif") or lower.endswith(".cif.gz")


def structure_basename(value: str | PurePosixPath) -> str:
    """Return the protein ID without ``.cif``/``.cif.gz``."""

    name = PurePosixPath(value).name
    lower = name.lower()
    for suffix in (".cif.gz", ".cif"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported distillation structure path: {value!r}")


def _manifest_int(document: Mapping[str, Any], key: str) -> int:
    try:
        value = int(document[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Runtime manifest has invalid {key}") from exc
    return value


class DistillationRuntimeIndex:
    """Read-only O(1) row and cluster lookup backed by lazy mmaps.

    Only the small JSON metadata is parsed during construction.  Array files
    are opened and mapped on first use.  Inherited or pickled instances detect
    their current process and reopen mappings, so a single dataset instance is
    safe with both ``fork`` and ``spawn`` DataLoader workers.
    """

    def __init__(self, runtime_dir: str | Path):
        self.runtime_dir = Path(runtime_dir).expanduser().resolve(strict=True)
        if not self.runtime_dir.is_dir():
            raise NotADirectoryError(f"Runtime-index path is not a directory: {self.runtime_dir}")
        self._manifest = self._load_manifest()
        self.row_count = _manifest_int(self._manifest, "row_count")
        self.cluster_count = _manifest_int(self._manifest, "cluster_count")
        if self.row_count <= 0 or self.cluster_count <= 0:
            raise ValueError("Runtime row and cluster counts must be positive")
        if self.row_count > np.iinfo(np.int64).max:
            raise ValueError(
                "Runtime row count exceeds the signed int64 IDs required by "
                "PyTorch samplers"
            )
        self._root_ids, self._source_roots = self._load_roots()
        self._array_paths = self._load_array_paths()
        self._sampling_fingerprint = self._make_sampling_fingerprint()
        self._files: dict[str, IO[bytes]] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._owner_pid: int | None = None

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.runtime_dir / "manifest.json"
        try:
            payload = manifest_path.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read runtime manifest {manifest_path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"Runtime manifest must be an object: {manifest_path}")
        if document.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise ValueError(f"Unsupported runtime schema version: {document.get('schema_version')!r}")
        if document.get("format") != _FORMAT_NAME:
            raise ValueError(f"Unsupported runtime-index format: {document.get('format')!r}")
        self._manifest_sha256 = hashlib.sha256(payload).hexdigest()
        return document

    def _load_roots(self) -> tuple[tuple[str, ...], dict[str, Path]]:
        manifest_root_ids = self._manifest.get("root_ids")
        if (
            not isinstance(manifest_root_ids, list)
            or not manifest_root_ids
            or len(manifest_root_ids) > 256
            or not all(isinstance(root_id, str) and root_id for root_id in manifest_root_ids)
            or len(set(manifest_root_ids)) != len(manifest_root_ids)
        ):
            raise ValueError("Runtime manifest has an invalid root_ids table")

        artifacts = self._manifest.get("artifacts")
        roots_record = artifacts.get("roots") if isinstance(artifacts, dict) else None
        if not isinstance(roots_record, dict):
            raise ValueError("Runtime manifest has no roots artifact")
        roots_path = self._resolve_artifact_path(roots_record)
        try:
            payload = roots_path.read_bytes()
            document = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read roots document {roots_path}: {exc}") from exc
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        expected_sha256 = self._artifact_sha256(roots_record, "roots")
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Runtime roots SHA-256 mismatch: {actual_sha256} != "
                f"{expected_sha256}"
            )
        self._roots_path = roots_path
        records = document.get("roots") if isinstance(document, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError(f"{roots_path} must contain a non-empty roots list")

        root_ids: list[str] = []
        roots: dict[str, Path] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{roots_path} contains a non-object roots entry")
            root_id = str(record.get("root_id", "")).strip()
            root_value = record.get("path")
            if not root_id or not isinstance(root_value, str) or not root_value or "\x00" in root_value:
                raise ValueError(f"{roots_path} roots entries require root_id and path")
            if root_id in roots:
                raise ValueError(f"{roots_path} contains duplicate root_id {root_id!r}")
            root_path = Path(root_value).expanduser()
            if not root_path.is_absolute():
                root_path = roots_path.parent / root_path
            root_ids.append(root_id)
            roots[root_id] = root_path.resolve(strict=False)

        if root_ids != manifest_root_ids:
            raise ValueError("roots.json order does not match manifest root_ids")
        return tuple(root_ids), roots

    def _resolve_artifact_path(self, record: Mapping[str, Any]) -> Path:
        relative = _safe_relative(record.get("path", ""))
        path = self.runtime_dir.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"Missing runtime artifact: {path}")
        try:
            expected_size = int(record["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Runtime artifact has invalid size_bytes: {path}") from exc
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Runtime artifact size mismatch for {path}: {actual_size} != {expected_size}")
        return path

    def _load_array_paths(self) -> dict[str, Path]:
        artifacts = self._manifest.get("artifacts")
        sampling = self._manifest.get("sampling")
        arrays = sampling.get("arrays") if isinstance(sampling, dict) else None
        if not isinstance(artifacts, dict) or not isinstance(arrays, dict):
            raise ValueError("Runtime manifest is missing artifacts or sampling arrays")

        records = {
            "row_offsets": artifacts.get("row_offsets"),
            "source_root_ids": artifacts.get("source_root_ids"),
            "cif_relpaths": artifacts.get("cif_relpaths"),
            "cluster_ids": arrays.get("cluster_ids"),
            "first_row_ids": arrays.get("first_row_ids"),
            "cluster_sizes": arrays.get("cluster_sizes"),
        }
        expected_sizes = {
            "row_offsets": (self.row_count + 1) * 8,
            "source_root_ids": self.row_count,
            "cluster_ids": self.cluster_count * 4,
            "first_row_ids": self.cluster_count * 8,
            "cluster_sizes": self.cluster_count * 4,
        }
        expected_counts = {
            "row_offsets": self.row_count + 1,
            "source_root_ids": self.row_count,
            "cluster_ids": self.cluster_count,
            "first_row_ids": self.cluster_count,
            "cluster_sizes": self.cluster_count,
        }
        expected_dtypes = {
            "row_offsets": "uint64",
            "source_root_ids": "uint8",
            "cif_relpaths": "bytes",
            "cluster_ids": "uint32",
            "first_row_ids": "uint64",
            "cluster_sizes": "uint32",
        }
        paths: dict[str, Path] = {}
        for name, record in records.items():
            if not isinstance(record, dict):
                raise ValueError(f"Runtime manifest has no {name} artifact")
            path = self._resolve_artifact_path(record)
            expected_size = expected_sizes.get(name)
            if expected_size is not None and path.stat().st_size != expected_size:
                raise ValueError(
                    f"Runtime artifact format size mismatch for {path}: "
                    f"{path.stat().st_size} != {expected_size}"
                )
            expected_count = expected_counts.get(name, path.stat().st_size)
            try:
                manifest_count = int(record["count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Runtime artifact has an invalid count: {path}"
                ) from exc
            if manifest_count != expected_count:
                raise ValueError(
                    f"Runtime artifact count mismatch for {path}: "
                    f"{manifest_count} != {expected_count}"
                )
            expected_dtype = expected_dtypes[name]
            if record.get("dtype") != expected_dtype:
                raise ValueError(
                    f"Runtime artifact dtype mismatch for {path}: "
                    f"{record.get('dtype')!r} != {expected_dtype!r}"
                )
            expected_endian = (
                "not_applicable" if name == "cif_relpaths" else "little"
            )
            if record.get("endian") != expected_endian:
                raise ValueError(
                    f"Runtime artifact endian mismatch for {path}: "
                    f"{record.get('endian')!r} != {expected_endian!r}"
                )
            paths[name] = path

        optional_records = {
            name: artifacts.get(name) for name in _OPTIONAL_ARRAY_SPECS
        }
        present = {name for name, record in optional_records.items() if record is not None}
        required_groups = (
            {"a3m_row_offsets", "a3m_relpaths"},
            {"bioassembly_row_offsets", "bioassembly_relpaths"},
            {"global_id_offsets", "global_ids"},
        )
        for group in required_groups:
            overlap = present & group
            if overlap and overlap != group:
                raise ValueError(
                    "Runtime manifest has an incomplete optional artifact group: "
                    f"{sorted(group)}"
                )

        optional_expected = {
            "a3m_row_offsets": (self.row_count + 1, "uint64", (self.row_count + 1) * 8),
            "a3m_relpaths": (None, "bytes", None),
            "bioassembly_row_offsets": (
                self.row_count + 1,
                "uint64",
                (self.row_count + 1) * 8,
            ),
            "bioassembly_relpaths": (None, "bytes", None),
            "global_id_offsets": (
                self.row_count + 1,
                "uint64",
                (self.row_count + 1) * 8,
            ),
            "global_ids": (None, "bytes", None),
            "source_member_row_ids": (
                self.row_count,
                "uint64",
                self.row_count * 8,
            ),
        }
        for name in sorted(present):
            record = optional_records[name]
            if not isinstance(record, dict):
                raise ValueError(f"Runtime manifest has an invalid {name} artifact")
            path = self._resolve_artifact_path(record)
            expected_count, expected_dtype, expected_size = optional_expected[name]
            if expected_size is not None and path.stat().st_size != expected_size:
                raise ValueError(
                    f"Runtime artifact format size mismatch for {path}: "
                    f"{path.stat().st_size} != {expected_size}"
                )
            try:
                manifest_count = int(record["count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Runtime artifact has an invalid count: {path}") from exc
            if expected_count is None:
                expected_count = path.stat().st_size
            if manifest_count != expected_count:
                raise ValueError(
                    f"Runtime artifact count mismatch for {path}: "
                    f"{manifest_count} != {expected_count}"
                )
            if record.get("dtype") != expected_dtype:
                raise ValueError(
                    f"Runtime artifact dtype mismatch for {path}: "
                    f"{record.get('dtype')!r} != {expected_dtype!r}"
                )
            expected_endian = "not_applicable" if expected_dtype == "bytes" else "little"
            if record.get("endian") != expected_endian:
                raise ValueError(
                    f"Runtime artifact endian mismatch for {path}: "
                    f"{record.get('endian')!r} != {expected_endian!r}"
                )
            paths[name] = path
        return paths

    @staticmethod
    def _artifact_sha256(record: object, name: str) -> str:
        if not isinstance(record, dict):
            raise ValueError(f"Runtime manifest has no {name} artifact")
        sha256 = record.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256.lower())
        ):
            raise ValueError(f"Runtime manifest has an invalid SHA-256 for {name}")
        return sha256.lower()

    def _make_sampling_fingerprint(self) -> str:
        artifacts = self._manifest["artifacts"]
        sampling_arrays = self._manifest["sampling"]["arrays"]
        material = {
            "format": _FORMAT_NAME,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "manifest_sha256": self._manifest_sha256,
            "row_count": self.row_count,
            "cluster_count": self.cluster_count,
            "root_ids": list(self._root_ids),
            "artifacts": {
                name: self._artifact_sha256(artifacts[name], name)
                for name in ("row_offsets", "source_root_ids", "cif_relpaths", "roots")
            },
            "sampling": {
                name: self._artifact_sha256(sampling_arrays[name], name)
                for name in ("cluster_ids", "first_row_ids", "cluster_sizes")
            },
        }
        for name in _OPTIONAL_ARRAY_SPECS:
            if name in artifacts:
                material["artifacts"][name] = self._artifact_sha256(
                    artifacts[name], name
                )
        payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def __len__(self) -> int:
        return self.row_count

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._root_ids

    @property
    def roots_path(self) -> Path:
        """Return the roots document whose hash is bound by the manifest."""

        return self._roots_path

    @property
    def source_roots(self) -> dict[str, Path]:
        """Return a copy of the verified ``root_id -> directory`` mapping."""

        return dict(self._source_roots)

    @property
    def sampling_fingerprint(self) -> str:
        """Compact identity used to reject an incompatible strict resume."""

        return self._sampling_fingerprint

    def resume_signature(self) -> dict[str, Any]:
        """Return stable runtime-index identity for checkpoint contracts."""

        return {
            "type": type(self).__name__,
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "row_count": self.row_count,
            "cluster_count": self.cluster_count,
            "root_ids": list(self._root_ids),
            "manifest_sha256": self._manifest_sha256,
            "sampling_fingerprint": self._sampling_fingerprint,
        }

    def _reset_if_process_changed(self) -> None:
        current_pid = os.getpid()
        if self._owner_pid is not None and self._owner_pid != current_pid:
            for mapped in self._maps.values():
                mapped.close()
            for file_obj in self._files.values():
                file_obj.close()
            self._maps = {}
            self._files = {}
        self._owner_pid = current_pid

    def _map(self, name: str) -> mmap.mmap:
        self._reset_if_process_changed()
        mapped = self._maps.get(name)
        if mapped is not None:
            return mapped
        file_obj = self._array_paths[name].open("rb")
        try:
            mapped = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            file_obj.close()
            raise
        self._files[name] = file_obj
        self._maps[name] = mapped
        return mapped

    @staticmethod
    def _checked_index(value: int, *, count: int, label: str) -> int:
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{label} must be an integer, got {type(value).__name__}") from exc
        if not 0 <= index < count:
            raise IndexError(f"{label} {index} is outside [0, {count})")
        return index

    def get_address(self, row_id: int) -> tuple[str, str]:
        """Return ``(root_id, cif_relpath)`` for one stable catalog row."""

        row = self._checked_index(row_id, count=self.row_count, label="row_id")
        offsets = self._map("row_offsets")
        root_ordinals = self._map("source_root_ids")
        relpaths = self._map("cif_relpaths")
        start = struct.unpack_from("<Q", offsets, row * 8)[0]
        end = struct.unpack_from("<Q", offsets, (row + 1) * 8)[0]
        if not start < end <= len(relpaths):
            raise ValueError(f"Corrupt CIF-path offsets for row {row}: [{start}, {end})")
        ordinal = root_ordinals[row]
        if ordinal >= len(self._root_ids):
            raise ValueError(f"Corrupt root ordinal {ordinal} for row {row}")
        try:
            cif_relpath = relpaths[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupt UTF-8 CIF path for row {row}") from exc
        path = _safe_relative(cif_relpath)
        if not _is_cif_path(path):
            raise ValueError(f"Runtime row {row} has a non-CIF path: {cif_relpath!r}")
        return self._root_ids[ordinal], path.as_posix()

    def _get_string_artifact(
        self,
        row_id: int,
        *,
        offsets_name: str,
        payload_name: str,
        label: str,
    ) -> str:
        row = self._checked_index(row_id, count=self.row_count, label="row_id")
        if offsets_name not in self._array_paths or payload_name not in self._array_paths:
            raise KeyError(f"Runtime index has no explicit {label} artifact")
        offsets = self._map(offsets_name)
        payload = self._map(payload_name)
        start = struct.unpack_from("<Q", offsets, row * 8)[0]
        end = struct.unpack_from("<Q", offsets, (row + 1) * 8)[0]
        if not start < end <= len(payload):
            raise ValueError(
                f"Corrupt {label} offsets for row {row}: [{start}, {end})"
            )
        try:
            return payload[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Corrupt UTF-8 {label} for row {row}") from exc

    @property
    def has_explicit_paths(self) -> bool:
        return all(
            name in self._array_paths
            for name in (
                "a3m_row_offsets",
                "a3m_relpaths",
                "bioassembly_row_offsets",
                "bioassembly_relpaths",
            )
        )

    @property
    def has_global_ids(self) -> bool:
        return all(
            name in self._array_paths
            for name in ("global_id_offsets", "global_ids")
        )

    def get_global_id(self, row_id: int) -> str:
        value = self._get_string_artifact(
            row_id,
            offsets_name="global_id_offsets",
            payload_name="global_ids",
            label="global ID",
        )
        if "|" not in value or any(char in value for char in "\t\r\n"):
            raise ValueError(f"Invalid source-qualified global ID: {value!r}")
        return value

    def get_source_member_row_id(self, row_id: int) -> int:
        row = self._checked_index(row_id, count=self.row_count, label="row_id")
        if "source_member_row_ids" not in self._array_paths:
            raise KeyError("Runtime index has no source-member provenance artifact")
        return struct.unpack_from(
            "<Q", self._map("source_member_row_ids"), row * 8
        )[0]

    def resolve_paths(self, row_id: int) -> DistillationPaths:
        """Resolve CIF, sibling A3M and nested bioassembly paths for one row."""

        root_id, cif_relpath = self.get_address(row_id)
        relative = _safe_relative(cif_relpath)
        root = self._source_roots[root_id]
        cif_path = root.joinpath(*relative.parts)
        # The lexical checks above make these joins stay beneath ``root``
        # without a per-sample filesystem resolve/stat call.
        if self.has_explicit_paths:
            a3m_relative = _safe_relative(
                self._get_string_artifact(
                    row_id,
                    offsets_name="a3m_row_offsets",
                    payload_name="a3m_relpaths",
                    label="A3M path",
                )
            )
            bioassembly_relative = _safe_relative(
                self._get_string_artifact(
                    row_id,
                    offsets_name="bioassembly_row_offsets",
                    payload_name="bioassembly_relpaths",
                    label="bioassembly path",
                )
            )
            if a3m_relative.suffix.lower() != ".a3m":
                raise ValueError(f"Runtime row {row_id} has a non-A3M path")
            if not bioassembly_relative.name.lower().endswith(".pkl.gz"):
                raise ValueError(f"Runtime row {row_id} has a non-PKL path")
            a3m_path = root.joinpath(*a3m_relative.parts)
            bioassembly_path = root.joinpath(*bioassembly_relative.parts)
        else:
            basename = structure_basename(relative)
            a3m_path = cif_path.parent / f"{basename}.a3m"
            bioassembly_path = (
                cif_path.parent / "bioassembly" / f"{basename}.pkl.gz"
            )
        return DistillationPaths(
            root_id=root_id,
            cif_relpath=relative.as_posix(),
            cif_path=cif_path,
            a3m_path=a3m_path,
            bioassembly_path=bioassembly_path,
            global_id=self.get_global_id(row_id) if self.has_global_ids else None,
            source_member_row_id=(
                self.get_source_member_row_id(row_id)
                if "source_member_row_ids" in self._array_paths
                else None
            ),
        )

    def get_cluster_id(self, cluster_ordinal: int) -> int:
        """Return the stable source cluster ID for a compact cluster ordinal."""

        ordinal = self._checked_index(
            cluster_ordinal,
            count=self.cluster_count,
            label="cluster_ordinal",
        )
        return struct.unpack_from("<I", self._map("cluster_ids"), ordinal * 4)[0]

    def get_cluster_size(self, cluster_ordinal: int) -> int:
        """Return the number of member rows in one compact cluster ordinal."""

        ordinal = self._checked_index(
            cluster_ordinal,
            count=self.cluster_count,
            label="cluster_ordinal",
        )
        size = struct.unpack_from("<I", self._map("cluster_sizes"), ordinal * 4)[0]
        if size == 0:
            raise ValueError(f"Corrupt zero-sized cluster at ordinal {ordinal}")
        return size

    def get_member_row(self, cluster_ordinal: int, member_offset: int) -> int:
        """Map a cluster ordinal/member offset to its stable catalog row ID."""

        ordinal = self._checked_index(
            cluster_ordinal,
            count=self.cluster_count,
            label="cluster_ordinal",
        )
        cluster_size = self.get_cluster_size(ordinal)
        member = self._checked_index(
            member_offset,
            count=cluster_size,
            label="member_offset",
        )
        first_row = struct.unpack_from("<Q", self._map("first_row_ids"), ordinal * 8)[0]
        row_id = first_row + member
        if row_id >= self.row_count:
            raise ValueError(
                f"Corrupt cluster interval at ordinal {ordinal}: row {row_id} >= {self.row_count}"
            )
        return row_id

    def sample_member_rows(
        self,
        cluster_ordinals: np.ndarray,
        uniforms: np.ndarray,
    ) -> np.ndarray:
        """Vectorize uniform member sampling for an array of cluster ordinals."""

        ordinals = np.asarray(cluster_ordinals)
        random_values = np.asarray(uniforms)
        if ordinals.shape != random_values.shape:
            raise ValueError(
                "cluster_ordinals and uniforms must have the same shape, got "
                f"{ordinals.shape} and {random_values.shape}"
            )
        if not np.issubdtype(ordinals.dtype, np.integer):
            raise TypeError(f"cluster_ordinals must have an integer dtype, got {ordinals.dtype}")
        if not np.issubdtype(random_values.dtype, np.floating):
            raise TypeError(f"uniforms must have a floating dtype, got {random_values.dtype}")
        if ordinals.size == 0:
            return np.empty(ordinals.shape, dtype=np.int64)
        if np.any(ordinals < 0) or np.any(ordinals >= self.cluster_count):
            raise IndexError(f"cluster_ordinals must be inside [0, {self.cluster_count})")
        if not np.all(np.isfinite(random_values)) or np.any(random_values < 0.0) or np.any(
            random_values >= 1.0
        ):
            raise ValueError("uniforms must contain finite values inside [0, 1)")

        first_rows = np.frombuffer(
            self._map("first_row_ids"),
            dtype="<u8",
            count=self.cluster_count,
        )[ordinals]
        cluster_sizes = np.frombuffer(
            self._map("cluster_sizes"),
            dtype="<u4",
            count=self.cluster_count,
        )[ordinals]
        if np.any(cluster_sizes == 0):
            raise ValueError("sampled cluster has no members")
        member_offsets = np.floor(
            random_values * cluster_sizes
        ).astype(np.uint64)
        # Defend against a floating-point product rounding up to exactly the
        # cluster size for a value infinitesimally below 1.0.
        member_offsets = np.minimum(
            member_offsets,
            cluster_sizes.astype(np.uint64) - 1,
        )
        rows = first_rows + member_offsets
        if np.any(rows >= self.row_count):
            raise ValueError("sampled cluster interval extends beyond the runtime row count")
        return rows.astype(np.int64, copy=False)

    def get_row_cluster(self, row_id: int) -> tuple[int, int]:
        """Return ``(stable_cluster_id, cluster_size)`` for one row."""

        row = self._checked_index(row_id, count=self.row_count, label="row_id")
        first_rows = np.frombuffer(
            self._map("first_row_ids"),
            dtype="<u8",
            count=self.cluster_count,
        )
        ordinal = int(np.searchsorted(first_rows, row, side="right") - 1)
        if ordinal < 0:
            raise ValueError(f"Runtime sampling intervals do not cover row {row}")
        cluster_size = self.get_cluster_size(ordinal)
        first_row = int(first_rows[ordinal])
        if row >= first_row + cluster_size:
            raise ValueError(f"Runtime sampling intervals do not cover row {row}")
        return self.get_cluster_id(ordinal), cluster_size

    def close(self) -> None:
        """Close all mappings owned by the current process."""

        for mapped in self._maps.values():
            try:
                mapped.close()
            except (BufferError, OSError):
                pass
        for file_obj in self._files.values():
            try:
                file_obj.close()
            except OSError:
                pass
        self._maps = {}
        self._files = {}
        self._owner_pid = None

    def __enter__(self) -> "DistillationRuntimeIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_maps"] = {}
        state["_files"] = {}
        state["_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._maps = {}
        self._files = {}
        self._owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
