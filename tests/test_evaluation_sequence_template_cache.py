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

"""Tests for the immutable, date-filtered evaluation template cache."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from protenix.data.template.template_featurizer import (
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION,
    EVALUATION_SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION,
    SequenceTemplateCacheError,
    TemplateFeaturizer,
)


def _template_features(
    sequence: str,
    *,
    name: str,
    release_date: str = "2020-01-01",
    position_value: float = 0.0,
) -> dict[str, object]:
    num_res = len(sequence)
    return {
        "template_aatype": np.zeros(num_res, dtype=np.int32),
        "template_all_atom_masks": np.ones(
            (num_res, 37),
            dtype=np.float32,
        ),
        "template_all_atom_positions": np.full(
            (num_res, 37, 3),
            position_value,
            dtype=np.float32,
        ),
        "template_domain_names": np.array(name.encode("utf-8"), dtype=object),
        "template_release_date": np.array(
            release_date.encode("utf-8"),
            dtype=object,
        ),
        "template_sequence": sequence.encode("utf-8"),
    }


def _payload(
    sequence: str,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION,
        "policy_version": EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION,
        "cutoff_date": EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_DATE,
        "max_templates": EVALUATION_SEQUENCE_TEMPLATE_CACHE_MAX_TEMPLATES,
        "status": "complete" if candidates else "complete_empty",
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "counts": {"raw_hit_count": len(candidates)},
        "candidates": [{"features": candidate} for candidate in candidates],
    }


class EvaluationSequenceTemplateCacheTest(unittest.TestCase):
    def _bare_featurizer(
        self,
        cache_root: str,
        *,
        filename: str = "template_features.pkl.gz",
    ) -> TemplateFeaturizer:
        featurizer = TemplateFeaturizer.__new__(TemplateFeaturizer)
        featurizer.evaluation_sequence_template_cache_root = os.path.abspath(
            cache_root
        )
        featurizer.evaluation_sequence_template_cache_filename = filename
        featurizer.evaluation_sequence_template_cache_policy = (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
        )
        featurizer.sequence_template_cache_mode = (
            EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE
        )
        return featurizer

    @staticmethod
    def _write_payload(
        featurizer: TemplateFeaturizer,
        sequence: str,
        payload: object,
    ) -> str:
        path = featurizer._evaluation_sequence_template_cache_path(sequence)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def _constructor_inputs(tmpdir: str) -> dict[str, object]:
        paths = {}
        for name in ("mapping", "release_dates", "obsolete"):
            path = os.path.join(tmpdir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            paths[name] = path
        return {
            "prot_template_mmcif_dir": tmpdir,
            "prot_template_raw_paths": [tmpdir],
            "prot_seq_or_filename_to_templatedir_jsons": [paths["mapping"]],
            "prot_indexing_methods": ["sequence"],
            "kalign_binary_path": "/bin/true",
            "release_dates_path": paths["release_dates"],
            "obsolete_pdbs_path": paths["obsolete"],
        }

    def test_sha256_layout_and_ordered_full_load_do_not_consume_rng(self):
        sequence = "ACDEFG"
        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = self._bare_featurizer(tmpdir)
            digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            expected_path = os.path.join(
                os.path.abspath(tmpdir),
                digest[:2],
                digest,
                "template_features.pkl.gz",
            )
            self.assertEqual(
                featurizer._evaluation_sequence_template_cache_path(sequence),
                expected_path,
            )

            candidates = [
                _template_features(
                    sequence,
                    name="first_A",
                    position_value=1.0,
                ),
                _template_features(
                    sequence,
                    name="second_B",
                    position_value=2.0,
                ),
            ]
            self._write_payload(
                featurizer,
                sequence,
                _payload(sequence, candidates),
            )

            random.seed(20260729)
            np.random.seed(20260729)
            expected_python_value = random.random()
            expected_numpy_value = np.random.random()
            random.seed(20260729)
            np.random.seed(20260729)

            loaded, raw_count, profile = (
                featurizer._load_evaluation_template_candidates(sequence)
            )

            self.assertEqual(random.random(), expected_python_value)
            self.assertEqual(np.random.random(), expected_numpy_value)
            self.assertEqual(raw_count, 2)
            self.assertEqual(
                [item["template_domain_names"].item() for item in loaded],
                [b"first_A", b"second_B"],
            )
            self.assertEqual(
                profile["template_source"],
                "evaluation_sequence_cache",
            )
            self.assertEqual(profile["sequence_template_cache_selected"], 2)

    def test_complete_empty_is_valid_and_missing_or_corrupt_is_fail_fast(self):
        sequence = "ACDEFG"
        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = self._bare_featurizer(tmpdir)
            with self.assertRaises(SequenceTemplateCacheError) as missing:
                featurizer._load_evaluation_template_candidates(sequence)
            self.assertEqual(missing.exception.reason, "missing")

            path = self._write_payload(
                featurizer,
                sequence,
                _payload(sequence, []),
            )
            candidates, raw_count, profile = (
                featurizer._load_evaluation_template_candidates(sequence)
            )
            self.assertEqual(candidates, [])
            self.assertEqual(raw_count, 0)
            self.assertEqual(
                profile["sequence_template_cache_status"],
                "complete_empty",
            )

            with open(path, "wb") as handle:
                handle.write(b"not-a-gzip-pickle")
            with self.assertRaises(SequenceTemplateCacheError) as corrupt:
                featurizer._load_evaluation_template_candidates(sequence)
            self.assertEqual(corrupt.exception.reason, "unreadable")

    def test_policy_cutoff_hash_count_and_shape_are_strictly_validated(self):
        sequence = "ACDEFG"
        valid_candidate = _template_features(sequence, name="valid_A")
        valid_payload = _payload(sequence, [valid_candidate])

        invalid_payloads = {}
        wrong_policy = copy.deepcopy(valid_payload)
        wrong_policy["policy_version"] = "top60_uniform20_identity_le_0.90_v1"
        invalid_payloads["training_policy"] = wrong_policy

        wrong_cutoff = copy.deepcopy(valid_payload)
        wrong_cutoff["cutoff_date"] = "3000-01-01"
        invalid_payloads["wrong_cutoff"] = wrong_cutoff

        wrong_hash = copy.deepcopy(valid_payload)
        wrong_hash["sequence_sha256"] = "0" * 64
        invalid_payloads["wrong_hash"] = wrong_hash

        too_many = _payload(
            sequence,
            [
                _template_features(sequence, name=f"candidate_{index}")
                for index in range(5)
            ],
        )
        invalid_payloads["more_than_four"] = too_many

        post_cutoff = _payload(
            sequence,
            [
                _template_features(
                    sequence,
                    name="future_A",
                    release_date="2021-10-01",
                )
            ],
        )
        invalid_payloads["post_cutoff"] = post_cutoff

        wrong_shape = copy.deepcopy(valid_payload)
        wrong_shape["candidates"][0]["features"][
            "template_all_atom_masks"
        ] = np.ones((len(sequence), 1), dtype=np.float32)
        invalid_payloads["wrong_shape"] = wrong_shape

        for name, payload in invalid_payloads.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                featurizer = self._bare_featurizer(tmpdir)
                self._write_payload(featurizer, sequence, payload)
                with self.assertRaises(SequenceTemplateCacheError) as error:
                    featurizer._load_evaluation_template_candidates(sequence)
                self.assertEqual(error.exception.reason, "invalid_payload")

    def test_training_contract_is_unchanged_and_test_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._constructor_inputs(tmpdir)

            # Evaluation-only values are deliberately invalid here. Training
            # ignores them and retains the existing required-cache behavior.
            train = TemplateFeaturizer(
                **kwargs,
                stage="train",
                sequence_template_cache_mode="required",
                sequence_template_cache_root=tmpdir,
                evaluation_sequence_template_cache_mode=object(),
                evaluation_sequence_template_cache_root=object(),
                evaluation_sequence_template_cache_filename=object(),
                evaluation_sequence_template_cache_policy=object(),
            )
            self.assertEqual(train.sequence_template_cache_mode, "required")
            self.assertIsNone(train.online_featurizer)

            # Inheriting a training required-cache setting is not enough to
            # activate the evaluation cache.
            online_test = TemplateFeaturizer(
                **kwargs,
                stage="test",
                sequence_template_cache_mode="required",
                sequence_template_cache_root=tmpdir,
                evaluation_sequence_template_cache_mode="online",
            )
            self.assertEqual(online_test.sequence_template_cache_mode, "online")
            self.assertIsNotNone(online_test.online_featurizer)

            # Explicit evaluation opt-in must avoid initializing Kalign/mmCIF
            # online machinery.
            with mock.patch(
                "protenix.data.template.template_featurizer.TemplateHitFeaturizer",
                side_effect=AssertionError("online featurizer was initialized"),
            ):
                cached_test = TemplateFeaturizer(
                    **kwargs,
                    stage="test",
                    evaluation_sequence_template_cache_mode="required",
                    evaluation_sequence_template_cache_root=tmpdir,
                    evaluation_sequence_template_cache_policy=(
                        EVALUATION_SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
                    ),
                )
            self.assertEqual(
                cached_test.sequence_template_cache_mode,
                EVALUATION_SEQUENCE_TEMPLATE_CACHE_EFFECTIVE_MODE,
            )
            self.assertIsNone(cached_test.online_featurizer)

    def test_required_evaluation_mode_rejects_empty_root_and_wrong_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = self._constructor_inputs(tmpdir)
            with self.assertRaisesRegex(
                ValueError,
                "evaluation_sequence_template_cache_root is required",
            ):
                TemplateFeaturizer(
                    **kwargs,
                    stage="test",
                    evaluation_sequence_template_cache_mode="required",
                )
            with self.assertRaisesRegex(
                ValueError,
                "evaluation_sequence_template_cache_policy",
            ):
                TemplateFeaturizer(
                    **kwargs,
                    stage="test",
                    evaluation_sequence_template_cache_mode="required",
                    evaluation_sequence_template_cache_root=tmpdir,
                    evaluation_sequence_template_cache_policy="wrong",
                )


if __name__ == "__main__":
    unittest.main()
