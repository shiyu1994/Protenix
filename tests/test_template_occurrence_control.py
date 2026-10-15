"""Tests for caller-controlled per-occurrence template usage."""

import gzip
import json
import os
import pickle
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ml_collections.config_dict import ConfigDict

from protenix.data.constants import STD_RESIDUES_WITH_GAP
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.pipeline.dataset import get_template_featurizer
from protenix.data.template.template_featurizer import (
    DummyTemplateFeaturizer,
    SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION,
    SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION,
    SequenceTemplateCacheError,
    TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
    TEMPLATE_USAGE_USE_REAL_TEMPLATE,
    TemplateFeaturizer,
)
from protenix.data.template.template_utils import (
    TemplateHitFeaturizer,
    _sample_template_count,
)


class _FakeTokenArray:
    def __init__(self, centre_atom_indices):
        self._centre_atom_indices = np.asarray(centre_atom_indices)

    def __getitem__(self, indices):
        return _FakeTokenArray(self._centre_atom_indices[indices])

    def __len__(self):
        return len(self._centre_atom_indices)

    def get_annotation(self, name):
        if name != "centre_atom_index":
            raise KeyError(name)
        return self._centre_atom_indices


class _FakeAtomArray:
    def __init__(self, asym_id_int, res_id, chain_id):
        self.asym_id_int = np.asarray(asym_id_int)
        self.res_id = np.asarray(res_id)
        self.chain_id = np.asarray(chain_id)
        self.is_ligand = np.zeros(len(self.asym_id_int), dtype=bool)

    def __getitem__(self, indices):
        return _FakeAtomArray(
            self.asym_id_int[indices],
            self.res_id[indices],
            self.chain_id[indices],
        )


def _single_chain_example(sequence="ACDEF"):
    num_tokens = len(sequence)
    return (
        {
            "pdb_id": "test",
            "atom_array": _FakeAtomArray(
                asym_id_int=np.zeros(num_tokens, dtype=np.int32),
                res_id=np.arange(1, num_tokens + 1, dtype=np.int32),
                chain_id=np.array(["A"] * num_tokens),
            ),
            "token_array": _FakeTokenArray(np.arange(num_tokens)),
            "sequences": {"1": sequence},
            "entity_poly_type": {"1": "polypeptide(L)"},
        },
        {"1": [0]},
    )


def _two_chain_example(sequence_a="ACDEF", sequence_b="GHIKL"):
    lengths = (len(sequence_a), len(sequence_b))
    num_tokens = sum(lengths)
    return (
        {
            "pdb_id": "two-chain-test",
            "atom_array": _FakeAtomArray(
                asym_id_int=np.concatenate(
                    [
                        np.zeros(lengths[0], dtype=np.int32),
                        np.ones(lengths[1], dtype=np.int32),
                    ]
                ),
                res_id=np.concatenate(
                    [
                        np.arange(1, lengths[0] + 1, dtype=np.int32),
                        np.arange(1, lengths[1] + 1, dtype=np.int32),
                    ]
                ),
                chain_id=np.array(
                    ["A"] * lengths[0] + ["B"] * lengths[1]
                ),
            ),
            "token_array": _FakeTokenArray(np.arange(num_tokens)),
            "sequences": {"1": sequence_a, "2": sequence_b},
            "entity_poly_type": {
                "1": "polypeptide(L)",
                "2": "polypeptide(L)",
            },
        },
        {"1": [0], "2": [1]},
    )


def _bare_featurizer(*, stage="train", dropout_rate=0.0):
    featurizer = TemplateFeaturizer.__new__(TemplateFeaturizer)
    featurizer.stage = stage
    featurizer.template_dropout_rate = dropout_rate
    featurizer._last_profile = {}
    return featurizer


def _template_hit(sequence):
    num_tokens = len(sequence)
    return {
        "template_aatype": np.zeros(num_tokens, dtype=np.int32),
        "template_all_atom_masks": np.ones((num_tokens, 37), dtype=np.float32),
        "template_all_atom_positions": np.zeros(
            (num_tokens, 37, 3), dtype=np.float32
        ),
        "template_domain_names": np.array(b"test_A", dtype=object),
        "template_release_date": np.array(b"2020-01-01", dtype=object),
        "template_sequence": np.array(sequence, dtype=object),
    }


class TestTemplateDropoutRateValidation(unittest.TestCase):
    def test_invalid_dropout_rates_fail_before_source_initialization(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TemplateFeaturizer(template_dropout_rate=value)

    def test_boundary_dropout_rates_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_path = f"{tmpdir}/mapping.json"
            with open(mapping_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            kwargs = {
                "prot_template_mmcif_dir": tmpdir,
                "prot_template_raw_paths": [tmpdir],
                "prot_seq_or_filename_to_templatedir_jsons": [mapping_path],
                "prot_indexing_methods": ["sequence"],
                "kalign_binary_path": "/bin/true",
                "release_dates_path": f"{tmpdir}/release_dates.json",
                "obsolete_pdbs_path": f"{tmpdir}/obsolete.json",
            }
            for value in (0.0, 1.0):
                with self.subTest(value=value):
                    featurizer = TemplateFeaturizer(
                        template_dropout_rate=value, **kwargs
                    )
                    self.assertEqual(featurizer.template_dropout_rate, value)

    def test_invalid_cache_error_policies_fail_before_source_initialization(self):
        for value in ("", "fallback", 1, None):
            with self.subTest(value=value):
                expected_error = TypeError if not isinstance(value, str) else ValueError
                with self.assertRaises(expected_error):
                    TemplateFeaturizer(
                        sequence_template_cache_error_policy=value
                    )


class TestTemplateUsageDecision(unittest.TestCase):
    def test_data_pipeline_maps_boolean_gate_to_explicit_template_policy(self):
        calls = []

        def fake_featurizer(**kwargs):
            calls.append(kwargs["template_usage_decision"])
            return {}

        bioassembly = {"atom_array": object()}
        with mock.patch.object(
            DataPipeline,
            "get_label_entity_id_to_asym_id_int",
            return_value={"1": {0}},
        ):
            for requested in (False, True, None):
                DataPipeline.get_template_raw_features(
                    bioassembly_dict=bioassembly,
                    selected_indices=np.array([0]),
                    template_featurizer=fake_featurizer,
                    template_use_requested=requested,
                )

        self.assertEqual(
            calls,
            [
                TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
                TEMPLATE_USAGE_USE_REAL_TEMPLATE,
                None,
            ],
        )

    def test_explicit_decisions_do_not_consume_global_numpy_rng(self):
        featurizer = _bare_featurizer(dropout_rate=0.8)
        with mock.patch(
            "protenix.data.template.template_featurizer.np.random.uniform",
            side_effect=AssertionError("explicit path consumed global RNG"),
        ):
            self.assertEqual(
                featurizer._resolve_template_usage_decision(
                    TEMPLATE_USAGE_FORCE_NO_TEMPLATE
                ),
                (TEMPLATE_USAGE_FORCE_NO_TEMPLATE, "explicit"),
            )
            self.assertEqual(
                featurizer._resolve_template_usage_decision(
                    TEMPLATE_USAGE_USE_REAL_TEMPLATE
                ),
                (TEMPLATE_USAGE_USE_REAL_TEMPLATE, "explicit"),
            )

    def test_invalid_explicit_decision_is_rejected(self):
        featurizer = _bare_featurizer()
        for decision in ("auto", "", 1, []):
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    featurizer._resolve_template_usage_decision(decision)

    def test_omitted_decision_preserves_legacy_dropout(self):
        featurizer = _bare_featurizer(dropout_rate=0.8)
        with mock.patch(
            "protenix.data.template.template_featurizer.np.random.uniform",
            return_value=0.2,
        ) as uniform:
            self.assertEqual(
                featurizer._resolve_template_usage_decision(None),
                (TEMPLATE_USAGE_FORCE_NO_TEMPLATE, "legacy_dropout_rate"),
            )
        uniform.assert_called_once_with(0, 1)

    def test_force_no_template_keeps_fixed_shape_dummy_semantics(self):
        featurizer = _bare_featurizer(dropout_rate=0.8)
        featurizer.get_template = mock.Mock(
            side_effect=AssertionError("force-no-template fetched templates")
        )
        bioassembly, entity_to_asym = _single_chain_example()

        features = featurizer.make_template_features(
            bioassembly,
            selected_token_indices=np.arange(5),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
        )

        self.assertEqual(features["template_aatype"].shape, (4, 5))
        self.assertTrue(
            np.all(
                features["template_aatype"][0]
                == STD_RESIDUES_WITH_GAP["-"]
            )
        )
        self.assertFalse(features["template_atom_mask"].any())
        self.assertFalse(features["template_atom_positions"].any())
        self.assertFalse(features["template_distogram"].any())
        profile = featurizer.get_last_profile()
        self.assertTrue(profile["drop_template_applied"])
        self.assertEqual(
            profile["template_usage_decision"],
            TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
        )
        self.assertEqual(profile["template_usage_decision_source"], "explicit")
        self.assertEqual(profile["valid_template_count"], 0)
        self.assertFalse(profile["has_real_template"])

    def test_explicit_real_template_requests_at_least_one_and_records_count(self):
        sequence = "ACDEF"
        featurizer = _bare_featurizer(dropout_rate=1.0)
        featurizer.get_template = mock.Mock(
            return_value=([_template_hit(sequence)], 7, {})
        )
        bioassembly, entity_to_asym = _single_chain_example(sequence)

        with mock.patch(
            "protenix.data.template.template_featurizer.np.random.uniform",
            side_effect=AssertionError("explicit path consumed global RNG"),
        ):
            features = featurizer.make_template_features(
                bioassembly,
                selected_token_indices=np.arange(len(sequence)),
                entity_to_asym_id_int=entity_to_asym,
                template_usage_decision=TEMPLATE_USAGE_USE_REAL_TEMPLATE,
            )

        self.assertEqual(features["template_aatype"].shape, (4, len(sequence)))
        self.assertEqual(
            featurizer.get_template.call_args.kwargs[
                "minimum_template_count"
            ],
            1,
        )
        profile = featurizer.get_last_profile()
        self.assertFalse(profile["drop_template_applied"])
        self.assertEqual(profile["raw_templates"], 7)
        self.assertEqual(profile["selected_templates"], 1)
        self.assertEqual(profile["valid_template_count"], 1)
        self.assertTrue(profile["has_real_template"])

    def test_required_cache_error_policy_raise_preserves_fail_fast(self):
        featurizer = _bare_featurizer()
        featurizer.sequence_template_cache_mode = "required"
        featurizer.sequence_template_cache_error_policy = "raise"
        featurizer.get_template = mock.Mock(
            side_effect=SequenceTemplateCacheError(
                "missing cache",
                reason="missing",
                cache_path="/cache/missing/template_features.pkl.gz",
            )
        )
        bioassembly, entity_to_asym = _single_chain_example()

        with self.assertRaisesRegex(
            SequenceTemplateCacheError,
            "missing cache",
        ):
            featurizer.make_template_features(
                bioassembly,
                selected_token_indices=np.arange(5),
                entity_to_asym_id_int=entity_to_asym,
                template_usage_decision=TEMPLATE_USAGE_USE_REAL_TEMPLATE,
            )

    def test_chain_dummy_preserves_other_sequence_and_rolls_back_failed_load_rng(
        self,
    ):
        sequence_a = "ACDEF"
        sequence_b = "GHIKL"
        featurizer = _bare_featurizer()
        featurizer.sequence_template_cache_mode = "required"
        featurizer.sequence_template_cache_error_policy = "chain_dummy"
        call_count = 0
        expected_python_rng_state = None
        expected_numpy_rng_state = None

        def get_template(**kwargs):
            nonlocal call_count
            nonlocal expected_python_rng_state
            nonlocal expected_numpy_rng_state
            call_count += 1
            # Model the two random streams consumed by count and candidate
            # selection in the real cache loader.
            random.random()
            np.random.random()
            if call_count == 1:
                expected_python_rng_state = random.getstate()
                expected_numpy_rng_state = np.random.get_state()
                return (
                    [_template_hit(kwargs["query_sequence"])],
                    7,
                    {
                        "template_source": "sequence_cache",
                        "sequence_template_cache_load_seconds": 0.1,
                        "sequence_template_cache_candidates": 3,
                        "sequence_template_cache_selected": 1,
                    },
                )
            raise SequenceTemplateCacheError(
                "missing second-chain cache",
                reason="missing",
                cache_path="/cache/missing/template_features.pkl.gz",
            )

        featurizer.get_template = mock.Mock(side_effect=get_template)
        bioassembly, entity_to_asym = _two_chain_example(
            sequence_a,
            sequence_b,
        )

        random.seed(20260726)
        np.random.seed(20260726)
        features = featurizer.make_template_features(
            bioassembly,
            selected_token_indices=np.arange(
                len(sequence_a) + len(sequence_b)
            ),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_USE_REAL_TEMPLATE,
        )

        self.assertEqual(call_count, 2)
        self.assertEqual(random.getstate(), expected_python_rng_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], expected_numpy_rng_state[0])
        np.testing.assert_array_equal(
            restored_numpy_state[1],
            expected_numpy_rng_state[1],
        )
        self.assertEqual(
            restored_numpy_state[2:],
            expected_numpy_rng_state[2:],
        )
        self.assertEqual(features["template_aatype"].shape, (4, 10))
        self.assertTrue(features["template_atom_mask"][:, :5].any())
        self.assertFalse(features["template_atom_mask"][:, 5:].any())

        profile = featurizer.get_last_profile()
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_unique_sequence_count"
            ],
            1,
        )
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_affected_chain_count"
            ],
            1,
        )
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_reason_counts"
            ],
            {"missing": 1},
        )
        self.assertEqual(
            profile["sequence_template_cache_chain_dummy_fallbacks"],
            [
                {
                    "query_sha256": featurizer._sequence_sha256(sequence_b),
                    "reason": "missing",
                    "cache_path": (
                        "/cache/missing/template_features.pkl.gz"
                    ),
                    "affected_chain_count": 1,
                }
            ],
        )
        self.assertEqual(profile["raw_templates"], 7)
        self.assertEqual(profile["selected_templates"], 1)
        self.assertEqual(profile["valid_template_count"], 1)
        self.assertTrue(profile["has_real_template"])
        self.assertFalse(profile["drop_template_applied"])
        self.assertEqual(
            profile["template_usage_decision"],
            TEMPLATE_USAGE_USE_REAL_TEMPLATE,
        )

    def test_chain_dummy_memoizes_failed_unique_sequence_for_all_matching_chains(
        self,
    ):
        sequence = "ACDEF"
        featurizer = _bare_featurizer()
        featurizer.sequence_template_cache_mode = "required"
        featurizer.sequence_template_cache_error_policy = "chain_dummy"

        def get_template(**kwargs):
            del kwargs
            random.random()
            np.random.random()
            raise SequenceTemplateCacheError(
                "corrupt shared cache",
                reason="invalid_payload",
                cache_path="/cache/corrupt/template_features.pkl.gz",
            )

        featurizer.get_template = mock.Mock(side_effect=get_template)
        bioassembly, entity_to_asym = _two_chain_example(
            sequence,
            sequence,
        )

        random.seed(20260726)
        np.random.seed(20260726)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        features = featurizer.make_template_features(
            bioassembly,
            selected_token_indices=np.arange(2 * len(sequence)),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_USE_REAL_TEMPLATE,
        )

        self.assertEqual(featurizer.get_template.call_count, 1)
        self.assertEqual(random.getstate(), python_rng_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_rng_state[0])
        np.testing.assert_array_equal(
            restored_numpy_state[1],
            numpy_rng_state[1],
        )
        self.assertEqual(restored_numpy_state[2:], numpy_rng_state[2:])
        self.assertFalse(features["template_atom_mask"].any())

        profile = featurizer.get_last_profile()
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_unique_sequence_count"
            ],
            1,
        )
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_affected_chain_count"
            ],
            2,
        )
        self.assertEqual(
            profile[
                "sequence_template_cache_chain_dummy_fallback_reason_counts"
            ],
            {"invalid_payload": 1},
        )
        self.assertEqual(
            profile["sequence_template_cache_chain_dummy_fallbacks"][0][
                "affected_chain_count"
            ],
            2,
        )

    def test_complete_empty_cache_is_legal_dummy_not_error_fallback(self):
        featurizer = _bare_featurizer()
        featurizer.sequence_template_cache_mode = "required"
        featurizer.sequence_template_cache_error_policy = "chain_dummy"
        featurizer.get_template = mock.Mock(
            return_value=(
                [],
                0,
                {
                    "template_source": "sequence_cache",
                    "sequence_template_cache_status": "complete_empty",
                    "sequence_template_cache_load_seconds": 0.1,
                    "sequence_template_cache_candidates": 0,
                    "sequence_template_cache_selected": 0,
                },
            )
        )
        bioassembly, entity_to_asym = _single_chain_example()

        features = featurizer.make_template_features(
            bioassembly,
            selected_token_indices=np.arange(5),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_USE_REAL_TEMPLATE,
        )

        self.assertFalse(features["template_atom_mask"].any())
        self.assertEqual(
            featurizer.get_last_profile()[
                "sequence_template_cache_chain_dummy_fallback_unique_sequence_count"
            ],
            0,
        )

    def test_required_cache_loader_classifies_missing_and_invalid_payloads(self):
        sequence = "ACDEF"
        with tempfile.TemporaryDirectory() as tmpdir:
            featurizer = _bare_featurizer()
            featurizer.sequence_template_cache_root = tmpdir
            featurizer.sequence_template_cache_filename = (
                "template_features.pkl.gz"
            )
            featurizer.source_mgr = SimpleNamespace(
                fetch_sequence_relative_dir=lambda _: "sequence-1"
            )

            with self.assertRaises(SequenceTemplateCacheError) as missing_ctx:
                featurizer._load_cached_template_candidates(
                    sequence,
                    minimum_template_count=1,
                )
            self.assertEqual(missing_ctx.exception.reason, "missing")

            cache_dir = os.path.join(tmpdir, "sequence-1")
            os.makedirs(cache_dir)
            cache_path = os.path.join(
                cache_dir,
                "template_features.pkl.gz",
            )
            with gzip.open(cache_path, "wb") as handle:
                pickle.dump(
                    {
                        "schema_version": -1,
                        "policy_version": "wrong",
                    },
                    handle,
                )

            with self.assertRaises(SequenceTemplateCacheError) as invalid_ctx:
                featurizer._load_cached_template_candidates(
                    sequence,
                    minimum_template_count=1,
                )
            self.assertEqual(
                invalid_ctx.exception.reason,
                "invalid_payload",
            )

            malformed_features = _template_hit(sequence)
            malformed_features["template_all_atom_masks"] = np.ones(
                (len(sequence), 1),
                dtype=np.float32,
            )
            with gzip.open(cache_path, "wb") as handle:
                pickle.dump(
                    {
                        "schema_version": (
                            SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION
                        ),
                        "policy_version": (
                            SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
                        ),
                        "sequence": sequence,
                        "sequence_sha256": featurizer._sequence_sha256(
                            sequence
                        ),
                        "status": "complete",
                        "candidates": [
                            {
                                "alignment_identity": 0.5,
                                "features": malformed_features,
                            }
                        ],
                    },
                    handle,
                )

            with self.assertRaises(SequenceTemplateCacheError) as shape_ctx:
                featurizer._load_cached_template_candidates(
                    sequence,
                    minimum_template_count=1,
                )
            self.assertEqual(shape_ctx.exception.reason, "invalid_payload")

            nonfinite_features = _template_hit(sequence)
            nonfinite_features["template_all_atom_positions"][0, 0, 0] = (
                np.nan
            )
            with gzip.open(cache_path, "wb") as handle:
                pickle.dump(
                    {
                        "schema_version": (
                            SEQUENCE_TEMPLATE_CACHE_SCHEMA_VERSION
                        ),
                        "policy_version": (
                            SEQUENCE_TEMPLATE_CACHE_POLICY_VERSION
                        ),
                        "sequence": sequence,
                        "sequence_sha256": featurizer._sequence_sha256(
                            sequence
                        ),
                        "status": "complete",
                        "candidates": [
                            {
                                "alignment_identity": 0.5,
                                "features": nonfinite_features,
                            }
                        ],
                    },
                    handle,
                )

            with self.assertRaises(SequenceTemplateCacheError) as finite_ctx:
                featurizer._load_cached_template_candidates(
                    sequence,
                    minimum_template_count=1,
                )
            self.assertEqual(
                finite_ctx.exception.reason,
                "invalid_payload",
            )


class TestDisabledTemplateDummyFeatures(unittest.TestCase):
    @staticmethod
    def _configs(*, emit_dummy):
        return ConfigDict(
            {
                "data": {
                    "template": {
                        "enable_prot_template": True,
                        "prot_seq_or_filename_to_templatedir_jsons": [
                            "/must/not/be/loaded.json"
                        ],
                    },
                    "distillation": {
                        "template": {
                            "enable_prot_template": False,
                            "emit_dummy_template_when_disabled": emit_dummy,
                        }
                    },
                }
            }
        )

    def test_training_can_emit_dummy_without_loading_template_database(self):
        featurizer = get_template_featurizer(
            self._configs(emit_dummy=True),
            dataset_name="distillation",
            stage="train",
        )
        self.assertIsInstance(featurizer, DummyTemplateFeaturizer)

        bioassembly, entity_to_asym = _single_chain_example()
        features = featurizer(
            bioassembly_dict=bioassembly,
            selected_indices=np.arange(5),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
        )

        expected_shapes = {
            "template_aatype": (4, 5),
            "template_atom_positions": (4, 5, 24, 3),
            "template_atom_mask": (4, 5, 24),
            "template_pseudo_beta_mask": (4, 5, 5),
            "template_distogram": (4, 5, 5, 39),
            "template_unit_vector": (4, 5, 5, 3),
            "template_backbone_frame_mask": (4, 5, 5),
        }
        self.assertEqual(set(features), set(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(features[name].shape, shape)
        self.assertTrue(
            np.all(
                features["template_aatype"][0]
                == STD_RESIDUES_WITH_GAP["-"]
            )
        )
        for name in expected_shapes:
            if name != "template_aatype":
                with self.subTest(masked_feature=name):
                    self.assertFalse(features[name].any())

        profile = featurizer.get_last_profile()
        self.assertEqual(profile["valid_template_count"], 0)
        self.assertFalse(profile["has_real_template"])
        self.assertTrue(profile["drop_template_applied"])
        self.assertEqual(
            profile["template_usage_decision_source"],
            "template_loading_disabled",
        )

    def test_disabled_source_dummy_matches_force_no_template_representation(self):
        regular = _bare_featurizer(dropout_rate=0.0)
        regular.get_template = mock.Mock(
            side_effect=AssertionError("force-no-template fetched templates")
        )
        bioassembly, entity_to_asym = _single_chain_example()
        expected = regular.make_template_features(
            bioassembly,
            selected_token_indices=np.arange(5),
            entity_to_asym_id_int=entity_to_asym,
            template_usage_decision=TEMPLATE_USAGE_FORCE_NO_TEMPLATE,
        )

        dummy = DummyTemplateFeaturizer(max_templates=4)
        actual = dummy(
            bioassembly_dict=bioassembly,
            selected_indices=np.arange(5),
            entity_to_asym_id_int=entity_to_asym,
        )

        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(name=name):
                np.testing.assert_array_equal(actual[name], expected[name])

    def test_dummy_mode_is_explicit_and_training_only(self):
        self.assertIsNone(
            get_template_featurizer(
                self._configs(emit_dummy=False),
                dataset_name="distillation",
                stage="train",
            )
        )
        self.assertIsNone(
            get_template_featurizer(
                self._configs(emit_dummy=True),
                dataset_name="distillation",
                stage="test",
            )
        )

    def test_dummy_uses_selected_token_count_for_none_and_boolean_index(self):
        dummy = DummyTemplateFeaturizer(max_templates=4)
        bioassembly, entity_to_asym = _single_chain_example()

        full = dummy(
            bioassembly_dict=bioassembly,
            selected_indices=None,
            entity_to_asym_id_int=entity_to_asym,
        )
        selected = dummy(
            bioassembly_dict=bioassembly,
            selected_indices=np.array([True, False, True, False, True]),
            entity_to_asym_id_int=entity_to_asym,
        )

        self.assertEqual(full["template_aatype"].shape, (4, 5))
        self.assertEqual(selected["template_aatype"].shape, (4, 3))

    def test_dummy_mode_rejects_truthy_strings(self):
        with self.assertRaisesRegex(
            TypeError,
            "emit_dummy_template_when_disabled must be a boolean",
        ):
            get_template_featurizer(
                self._configs(emit_dummy="true"),
                dataset_name="distillation",
                stage="train",
            )


class TestTemplateCountSampling(unittest.TestCase):
    @staticmethod
    def _hit_featurizer_with_one_processable_candidate():
        featurizer = TemplateHitFeaturizer.__new__(TemplateHitFeaturizer)
        featurizer._max_template_date = None
        featurizer._max_hits = None
        featurizer._shuffle_top_k_prefiltered = None
        featurizer._max_template_candidates_num = None
        featurizer._release_dates = {}
        featurizer._obsolete_pdbs = {}
        featurizer._strict_error_check = False
        featurizer._hit_filter = mock.Mock()
        featurizer._hit_filter.prefilter.return_value = SimpleNamespace(
            valid=True,
            error=None,
            warning=None,
        )
        hit = SimpleNamespace(sum_probs=1.0, hit_sequence="GGGGG")
        featurizer._hit_processor = mock.Mock()
        featurizer._hit_processor.process.return_value = (
            SimpleNamespace(
                error=None,
                warning=None,
                features={"template_sequence": "GGGGG"},
                hit=hit,
            ),
            {},
        )
        return featurizer, hit

    def test_explicit_minimum_processes_a_valid_candidate(self):
        featurizer, hit = self._hit_featurizer_with_one_processable_candidate()
        with mock.patch(
            "protenix.data.template.template_utils.random.randint",
            return_value=1,
        ) as randint:
            result, _ = featurizer.get_templates(
                sequence_uid="sample",
                query_sequence="ACDEF",
                hits=[hit],
                minimum_template_count=1,
            )
        randint.assert_called_once_with(1, 1)
        self.assertEqual(len(result.features), 1)
        featurizer._hit_processor.process.assert_called_once()

    def test_legacy_count_sampling_remains_zero_inclusive(self):
        with mock.patch(
            "protenix.data.template.template_utils.random.randint",
            return_value=0,
        ) as randint:
            count = _sample_template_count(
                num_candidates=10,
                max_hits=None,
                minimum_template_count=0,
            )
        self.assertEqual(count, 0)
        randint.assert_called_once_with(0, 10)

    def test_explicit_real_count_is_one_to_four(self):
        with mock.patch(
            "protenix.data.template.template_utils.random.randint",
            return_value=1,
        ) as randint:
            self.assertEqual(
                _sample_template_count(
                    num_candidates=10,
                    max_hits=None,
                    minimum_template_count=1,
                ),
                1,
            )
        randint.assert_called_once_with(1, 10)

        with mock.patch(
            "protenix.data.template.template_utils.random.randint",
            return_value=20,
        ):
            self.assertEqual(
                _sample_template_count(
                    num_candidates=20,
                    max_hits=None,
                    minimum_template_count=1,
                ),
                4,
            )

    def test_no_candidate_still_returns_zero(self):
        self.assertEqual(
            _sample_template_count(
                num_candidates=0,
                max_hits=None,
                minimum_template_count=1,
            ),
            0,
        )

    def test_invalid_minimum_is_rejected(self):
        for minimum in (-1, 5):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    _sample_template_count(
                        num_candidates=3,
                        max_hits=None,
                        minimum_template_count=minimum,
                    )
        for minimum in (True, 1.5):
            with self.subTest(minimum=minimum):
                with self.assertRaises(TypeError):
                    _sample_template_count(
                        num_candidates=3,
                        max_hits=None,
                        minimum_template_count=minimum,
                    )


if __name__ == "__main__":
    unittest.main()
