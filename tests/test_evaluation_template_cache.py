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

"""Focused tests for the locked PXM evaluation template-cache builder."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from protenix.data.template.evaluation_template_cache import (
    EVALUATION_CACHE_CUTOFF,
    EVALUATION_CACHE_CUTOFF_TEXT,
    EVALUATION_CACHE_MAX_TEMPLATES,
    EVALUATION_CACHE_POLICY_VERSION,
    EvaluationTemplateCacheBuilder,
    EvaluationTemplateCacheConfig,
    evaluation_cache_path,
    load_evaluation_template_cache,
    load_sequence_sources,
    load_template_hits,
    sequence_sha256,
    validate_evaluation_cache_payload,
)
from protenix.data.template.template_featurizer import (
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION,
    TemplateFeaturizer,
)
from protenix.data.template.template_parser import TemplateHit
from scripts.evaluation.build_af3_pxm_template_cache import main as build_main
from scripts.evaluation.validate_af3_pxm_template_cache import (
    main as validate_main,
)


_QUERY = "ACDEFGHIKL"


def _template_hit(name: str = "1abc_A") -> TemplateHit:
    return TemplateHit(
        index=1,
        name=name,
        aligned_cols=len(_QUERY),
        sum_probs=0.9,
        query=_QUERY,
        hit_sequence=_QUERY,
        indices_query=list(range(len(_QUERY))),
        indices_hit=list(range(len(_QUERY))),
    )


def _template_features(
    *,
    release_date: str = EVALUATION_CACHE_CUTOFF_TEXT,
) -> dict[str, np.ndarray | bytes]:
    mask = np.zeros((len(_QUERY), 37), dtype=np.float32)
    mask[0, 0] = 1.0
    return {
        "template_aatype": np.zeros((len(_QUERY),), dtype=np.int32),
        "template_all_atom_masks": mask,
        "template_all_atom_positions": np.zeros(
            (len(_QUERY), 37, 3),
            dtype=np.float32,
        ),
        "template_domain_names": np.array(b"1abc_A", dtype=object),
        "template_release_date": np.array(
            release_date.encode("utf-8"),
            dtype=object,
        ),
        "template_sequence": _QUERY.encode("utf-8"),
    }


class _FakeOnlineFeaturizer:
    def __init__(self, *, features=(), hits=()):
        self.features = list(features)
        self.hits = list(hits)
        self.calls = []

    def get_templates(self, **kwargs):
        self.calls.append(kwargs)
        return (
            SimpleNamespace(
                features=self.features,
                hits=self.hits,
                errors=[],
                warnings=[],
            ),
            {"fake_seconds": 0.0},
        )


class EvaluationTemplateSourceTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_nonempty_hhr_takes_precedence_over_a3m(self):
        source_dir = self.root / "source"
        source_dir.mkdir()
        (source_dir / "hmmsearch.a3m").write_text(
            "this A3M must not be parsed",
            encoding="utf-8",
        )
        (source_dir / "concat.hhr").write_text(
            "nonempty HHR",
            encoding="utf-8",
        )
        hit = _template_hit()

        with mock.patch(
            "protenix.data.template.evaluation_template_cache.HHRParser.parse",
            return_value=[hit],
        ) as hhr_parse, mock.patch(
            "protenix.data.template.evaluation_template_cache."
            "HmmsearchA3MParser.parse",
        ) as a3m_parse:
            loaded = load_template_hits(
                query_sequence=_QUERY,
                source_hit_path=source_dir,
            )

        self.assertEqual(loaded.hits, (hit,))
        self.assertEqual(loaded.source["selected_format"], "hhr")
        hhr_parse.assert_called_once_with(hhr_string="nonempty HHR")
        a3m_parse.assert_not_called()

    def test_empty_artifact_is_zero_hits_but_missing_source_fails(self):
        source_dir = self.root / "source"
        source_dir.mkdir()
        (source_dir / "hmmsearch.a3m").write_bytes(b"")

        loaded = load_template_hits(
            query_sequence=_QUERY,
            source_hit_path=source_dir,
        )
        self.assertEqual(loaded.hits, ())
        self.assertEqual(loaded.source["artifact_exists_count"], 1)
        self.assertEqual(loaded.source["selected_format"], "a3m")

        with self.assertRaises(FileNotFoundError):
            load_template_hits(
                query_sequence=_QUERY,
                source_hit_path=self.root / "missing",
            )

    def test_json_and_jsonl_sequence_source_schemas(self):
        source_path = self.root / "hits" / "hmmsearch.a3m"
        json_path = self.root / "sources.json"
        json_path.write_text(
            json.dumps({_QUERY: str(source_path)}),
            encoding="utf-8",
        )
        expected = {_QUERY: str(source_path.resolve(strict=False))}
        self.assertEqual(load_sequence_sources(json_path), expected)

        jsonl_path = self.root / "sources.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "sequence": _QUERY,
                    "source_hit_path": str(source_path),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(load_sequence_sources(jsonl_path), expected)


class EvaluationTemplateBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_dir = self.root / "source"
        self.source_dir.mkdir()
        (self.source_dir / "hmmsearch.a3m").write_text(
            "synthetic",
            encoding="utf-8",
        )

    def _config(self) -> EvaluationTemplateCacheConfig:
        return EvaluationTemplateCacheConfig(
            cache_root=str(self.root / "cache"),
            mmcif_dir=str(self.root / "mmcif"),
            kalign_binary_path=str(self.root / "kalign"),
            release_dates_path=str(self.root / "release_dates.json"),
            obsolete_pdbs_path=str(self.root / "obsolete.json"),
        )

    def test_builder_uses_online_flow_and_writes_content_addressed_cache(self):
        hit = _template_hit()
        online = _FakeOnlineFeaturizer(
            features=[_template_features()],
            hits=[hit],
        )
        builder = EvaluationTemplateCacheBuilder(
            self._config(),
            online_featurizer=online,
        )

        with mock.patch(
            "protenix.data.template.evaluation_template_cache."
            "HmmsearchA3MParser.parse",
            return_value=[hit],
        ):
            result = builder.build(
                sequence=_QUERY,
                source_hit_path=str(self.source_dir),
            )

        expected_path = evaluation_cache_path(
            self.root / "cache",
            _QUERY,
        )
        self.assertEqual(Path(result["output_path"]), expected_path)
        self.assertEqual(expected_path.parent.name, sequence_sha256(_QUERY))
        self.assertEqual(expected_path.parent.parent.name, sequence_sha256(_QUERY)[:2])
        payload = load_evaluation_template_cache(expected_path)
        validate_evaluation_cache_payload(
            payload,
            expected_sequence=_QUERY,
        )
        self.assertEqual(payload["policy_version"], EVALUATION_CACHE_POLICY_VERSION)
        self.assertEqual(payload["cutoff_date"], EVALUATION_CACHE_CUTOFF_TEXT)
        self.assertEqual(payload["max_templates"], EVALUATION_CACHE_MAX_TEMPLATES)
        self.assertEqual(len(payload["candidates"]), 1)
        self.assertEqual(len(online.calls), 1)
        self.assertEqual(
            online.calls[0]["max_template_date"],
            EVALUATION_CACHE_CUTOFF,
        )
        self.assertEqual(online.calls[0]["hits"], (hit,))

    def test_payload_rejects_template_after_locked_cutoff(self):
        hit = _template_hit()
        online = _FakeOnlineFeaturizer(
            features=[_template_features(release_date="2021-10-01")],
            hits=[hit],
        )
        builder = EvaluationTemplateCacheBuilder(
            self._config(),
            online_featurizer=online,
        )
        with mock.patch(
            "protenix.data.template.evaluation_template_cache."
            "HmmsearchA3MParser.parse",
            return_value=[hit],
        ), self.assertRaisesRegex(ValueError, "exceeds"):
            builder.build(
                sequence=_QUERY,
                source_hit_path=str(self.source_dir),
            )

    def test_builder_payload_is_consumed_by_runtime_loader_without_translation(self):
        self.assertEqual(
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION,
            EVALUATION_CACHE_POLICY_VERSION,
        )
        self.assertEqual(
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE,
            EVALUATION_CACHE_CUTOFF_TEXT,
        )
        self.assertEqual(
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES,
            EVALUATION_CACHE_MAX_TEMPLATES,
        )

        hit = _template_hit()
        online = _FakeOnlineFeaturizer(
            features=[_template_features()],
            hits=[hit],
        )
        builder = EvaluationTemplateCacheBuilder(
            self._config(),
            online_featurizer=online,
        )
        with mock.patch(
            "protenix.data.template.evaluation_template_cache."
            "HmmsearchA3MParser.parse",
            return_value=[hit],
        ):
            builder.build(
                sequence=_QUERY,
                source_hit_path=str(self.source_dir),
            )

        runtime = TemplateFeaturizer.__new__(TemplateFeaturizer)
        runtime.evaluation_sequence_template_cache_root = str(
            (self.root / "cache").resolve()
        )
        runtime.evaluation_sequence_template_cache_filename = (
            "template_features.pkl.gz"
        )
        runtime.evaluation_sequence_template_cache_policy = (
            EVALUATION_CACHE_POLICY_VERSION
        )
        runtime.sequence_template_cache_mode = (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE
        )
        candidates, raw_count, profile = (
            runtime._load_evaluation_template_candidates(_QUERY)
        )
        self.assertEqual(raw_count, 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["template_domain_names"].item(),
            b"1abc_A",
        )
        self.assertEqual(
            profile["sequence_template_cache_policy"],
            EVALUATION_CACHE_POLICY_VERSION,
        )

    def test_empty_source_cli_emits_success_and_full_validator_passes(self):
        source_dir = self.root / "empty_source"
        source_dir.mkdir()
        (source_dir / "hmmsearch.a3m").write_bytes(b"")
        mmcif_dir = self.root / "mmcif"
        mmcif_dir.mkdir()
        kalign = self.root / "kalign"
        kalign.write_text("#!/bin/sh\n", encoding="utf-8")
        release_dates = self.root / "release_dates.json"
        release_dates.write_text("{}\n", encoding="utf-8")
        obsolete = self.root / "obsolete.json"
        obsolete.write_text("{}\n", encoding="utf-8")
        sequence_sources = self.root / "sources.jsonl"
        sequence_sources.write_text(
            json.dumps(
                {
                    "sequence": _QUERY,
                    "source_hit_path": str(source_dir),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cache_root = self.root / "cache_cli"

        with contextlib.redirect_stdout(io.StringIO()):
            build_status = build_main(
                [
                    "--sequence-sources",
                    str(sequence_sources),
                    "--cache-root",
                    str(cache_root),
                    "--mmcif-dir",
                    str(mmcif_dir),
                    "--kalign-binary-path",
                    str(kalign),
                    "--release-dates-path",
                    str(release_dates),
                    "--obsolete-pdbs-path",
                    str(obsolete),
                    "--workers",
                    "1",
                ]
            )
            validate_status = validate_main(
                [
                    "--sequence-sources",
                    str(sequence_sources),
                    "--cache-root",
                    str(cache_root),
                ]
            )

        self.assertEqual(build_status, 0)
        self.assertEqual(validate_status, 0)
        self.assertTrue((cache_root / "_SUCCESS.json").is_file())
        payload = load_evaluation_template_cache(
            evaluation_cache_path(cache_root, _QUERY)
        )
        self.assertEqual(payload["status"], "complete_empty")
        self.assertEqual(payload["candidates"], [])

        evaluation_cache_path(cache_root, _QUERY).unlink()
        with self.assertRaisesRegex(ValueError, "missing artifacts"):
            with contextlib.redirect_stdout(io.StringIO()):
                validate_main(
                    [
                        "--sequence-sources",
                        str(sequence_sources),
                        "--cache-root",
                        str(cache_root),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
