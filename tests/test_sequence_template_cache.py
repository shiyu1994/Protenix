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

"""Focused tests for the offline sequence-template cache contract."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from protenix.data.template.sequence_template_cache import (
    MAX_ALIGNMENT_IDENTITY,
    POLICY_VERSION,
    ParsedHit,
    SequenceTemplateCacheBuilder,
    SequenceTemplateCacheConfig,
    _identity_from_aligned_strings,
    deterministic_permutation,
    deterministic_seed,
    load_sequence_template_cache,
    select_top_quality_hits,
    validate_cache_payload,
)
from protenix.data.template.template_parser import TemplateHit


_QUERY = "A" * 100
_AMINO_ACIDS = "CDEFGHIKLMNPQRSTVWY"


def _encoded_template_sequence(index: int, length: int = 12) -> str:
    """Encode an integer as a unique, non-query amino-acid sequence."""

    residues = []
    value = index
    for _ in range(length):
        residues.append(_AMINO_ACIDS[value % len(_AMINO_ACIDS)])
        value //= len(_AMINO_ACIDS)
    return "".join(residues)


def _parsed_hit(
    index: int,
    *,
    sum_probs: float | None = None,
    identity: float = 0.5,
) -> ParsedHit:
    sequence = _encoded_template_sequence(index)
    hit = TemplateHit(
        index=index + 1,
        name=f"{index % 0x10000:04x}_A",
        aligned_cols=len(sequence),
        sum_probs=float(index) if sum_probs is None else sum_probs,
        query=_QUERY[: len(sequence)],
        hit_sequence=sequence,
        indices_query=list(range(len(sequence))),
        indices_hit=list(range(len(sequence))),
    )
    return ParsedHit(hit=hit, alignment_identity=identity)


class _FakeHitProcessor:
    """Return synthetic template features and optionally fail selected calls."""

    def __init__(self, fail_calls=()):
        self.fail_calls = set(fail_calls)
        self.calls = []

    def process(self, query_sequence, hit, *args):
        del query_sequence, args
        call_number = len(self.calls) + 1
        self.calls.append(hit.name)
        if call_number in self.fail_calls:
            result = SimpleNamespace(
                features=None,
                error=f"synthetic failure {call_number}",
                warning=None,
                hit=hit,
            )
        else:
            result = SimpleNamespace(
                features={
                    "template_sequence": np.asarray(
                        _encoded_template_sequence(
                            call_number,
                            length=len(_QUERY),
                        ).encode("utf-8")
                    )
                },
                error=None,
                warning=None,
                hit=hit,
            )
        return result, {"synthetic_seconds": 0.0}


class SequenceTemplateCacheSelectionTest(unittest.TestCase):
    def test_identity_uses_only_columns_where_both_residues_are_non_gap(self):
        identity = _identity_from_aligned_strings("ACD-EF", "ACXGEF")
        self.assertAlmostEqual(identity, 4 / 5)

        with self.assertRaisesRegex(ValueError, "lengths differ"):
            _identity_from_aligned_strings("AC", "A")
        with self.assertRaisesRegex(ValueError, "no columns"):
            _identity_from_aligned_strings("---", "---")

    def test_identity_threshold_is_inclusive_at_ninety_percent(self):
        at_threshold = _parsed_hit(1, identity=MAX_ALIGNMENT_IDENTITY)
        over_threshold = _parsed_hit(
            2,
            identity=MAX_ALIGNMENT_IDENTITY + 1e-8,
        )

        selected, counts, rejections = select_top_quality_hits(
            query_sequence=_QUERY,
            parsed_hits=[at_threshold, over_threshold],
        )

        self.assertEqual([item.hit.name for item in selected], ["0001_A"])
        self.assertEqual(counts["identity_rejected_count"], 1)
        self.assertEqual(rejections[-1]["stage"], "identity")

    def test_top_sixty_and_uniform_order_are_deterministic(self):
        parsed_hits = [_parsed_hit(index) for index in range(75)]

        selected, counts, _ = select_top_quality_hits(
            query_sequence=_QUERY,
            parsed_hits=parsed_hits,
        )
        self.assertEqual(counts["top_quality_count"], 60)
        self.assertEqual(
            [item.hit.sum_probs for item in selected],
            [float(index) for index in range(74, 14, -1)],
        )

        first = deterministic_permutation(_QUERY, selected)
        second = deterministic_permutation(_QUERY, selected)
        self.assertEqual(
            [item.hit.name for item in first],
            [item.hit.name for item in second],
        )
        expected_indices = list(range(60))
        random.Random(deterministic_seed(_QUERY)).shuffle(expected_indices)
        self.assertEqual(first, [selected[index] for index in expected_indices])
        self.assertCountEqual(
            [item.hit.name for item in first],
            [item.hit.name for item in selected],
        )


class SequenceTemplateCacheBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_root = self.root / "source"
        self.cache_root = self.root / "cache"
        self.relative_directory = "000/sequence-1"
        source_directory = self.source_root / self.relative_directory
        source_directory.mkdir(parents=True)
        (source_directory / "hmmsearch.a3m").write_text(
            ">placeholder\nAAAA\n",
            encoding="utf-8",
        )

    def _builder(
        self,
        processor,
        *,
        top_k_quality=60,
        max_valid_templates=20,
    ):
        config = SequenceTemplateCacheConfig(
            source_root=str(self.source_root),
            cache_root=str(self.cache_root),
            mmcif_dir=str(self.root / "mmcif"),
            kalign_binary_path="/unused/kalign",
            top_k_quality=top_k_quality,
            max_valid_templates=max_valid_templates,
        )
        return SequenceTemplateCacheBuilder(
            config,
            hit_processor=processor,
        )

    def _build_with_hits(self, hits, processor, **builder_kwargs):
        builder = self._builder(processor, **builder_kwargs)
        with mock.patch(
            "protenix.data.template.sequence_template_cache."
            "parse_hits_with_identities",
            return_value=hits,
        ):
            manifest_record = builder.build(
                sequence=_QUERY,
                relative_directory=self.relative_directory,
            )
        payload = load_sequence_template_cache(
            builder.output_path(self.relative_directory)
        )
        return manifest_record, payload

    def test_failed_candidates_are_replaced_from_remaining_top_set(self):
        processor = _FakeHitProcessor(fail_calls={1, 2})
        manifest_record, payload = self._build_with_hits(
            [_parsed_hit(index) for index in range(8)],
            processor,
            top_k_quality=8,
            max_valid_templates=4,
        )

        self.assertEqual(manifest_record["status"], "complete")
        self.assertEqual(payload["counts"]["attempted_top60_count"], 6)
        self.assertEqual(payload["counts"]["process_failed_count"], 2)
        self.assertEqual(payload["counts"]["valid_cached_count"], 4)
        self.assertEqual(len(processor.calls), 6)
        self.assertEqual(len(payload["candidates"]), 4)

    def test_no_hits_writes_valid_complete_empty_cache(self):
        manifest_record, payload = self._build_with_hits(
            [],
            _FakeHitProcessor(),
        )

        self.assertEqual(manifest_record["status"], "complete_empty")
        self.assertEqual(payload["status"], "complete_empty")
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["counts"]["attempted_top60_count"], 0)
        validate_cache_payload(payload, expected_sequence=_QUERY)

    def test_fewer_than_twenty_hits_preserves_every_valid_hit(self):
        processor = _FakeHitProcessor()
        _, payload = self._build_with_hits(
            [_parsed_hit(index) for index in range(3)],
            processor,
        )

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["counts"]["top_quality_count"], 3)
        self.assertEqual(payload["counts"]["attempted_top60_count"], 3)
        self.assertEqual(payload["counts"]["valid_cached_count"], 3)
        self.assertEqual(len(payload["candidates"]), 3)

    def test_schema_reader_and_validator_reject_policy_mismatch(self):
        _, payload = self._build_with_hits(
            [_parsed_hit(1)],
            _FakeHitProcessor(),
        )
        self.assertEqual(payload["policy_version"], POLICY_VERSION)
        validate_cache_payload(payload, expected_sequence=_QUERY)

        corrupt_payload = dict(payload)
        corrupt_payload["policy_version"] = "wrong-policy"
        with self.assertRaisesRegex(ValueError, "policy_version mismatch"):
            validate_cache_payload(corrupt_payload)


if __name__ == "__main__":
    unittest.main()
