# Copyright 2025 ByteDance and/or its affiliates.
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

import pytest

from scripts.evaluation.summarize_pxm22to25_abag_training_msa_budget import (
    simulate_depth_only_budget,
    summarize_rows,
    summarize_targets,
)


def _row(
    pdb_id: str,
    pairing: int,
    nonpairing: int,
    role: str = "H",
) -> dict[str, str]:
    return {
        "pdb_id": pdb_id,
        "roles": role,
        "pairing_raw_depth": str(pairing),
        "pairing_unique_depth": str(pairing),
        "non_pairing_raw_depth": str(nonpairing),
        "non_pairing_unique_depth": str(nonpairing),
    }


def test_pairing_first_starves_genuine_nonpairing_at_cap() -> None:
    result = simulate_depth_only_budget(_row("a", 12000, 1500), cap=10000)

    assert result["paired_pool_pairing_rows_read_including_query"] == 10000
    assert result["unpaired_pool_pairing_rows_read_including_query"] == 10000
    assert result["nonpairing_source_rows_read_including_query"] == 0
    assert result["nonpairing_homolog_hits_read"] == 0
    assert result["nonpairing_homolog_hits_not_read"] == 1499
    assert result["nonpairing_homologs_starved"] == 1


def test_nonpairing_uses_only_the_pairing_pool_remainder() -> None:
    result = simulate_depth_only_budget(_row("a", 8000, 3000), cap=10000)

    assert result["unpaired_pool_remaining_after_pairing"] == 2000
    assert result["nonpairing_source_rows_read_including_query"] == 2000
    assert result["nonpairing_source_rows_not_read_including_query"] == 1000
    assert result["nonpairing_homolog_hits_read"] == 1999
    assert result["nonpairing_homolog_hits_not_read"] == 1000
    assert result["nonpairing_homolog_survival_fraction"] == 1999 / 2999


def test_one_remaining_row_reads_only_nonpairing_query() -> None:
    result = simulate_depth_only_budget(_row("a", 9999, 3000), cap=10000)

    assert result["nonpairing_source_rows_read_including_query"] == 1
    assert result["nonpairing_homolog_hits_read"] == 0
    assert result["nonpairing_homologs_starved"] == 1


def test_rejects_unique_depth_greater_than_raw_depth() -> None:
    row = _row("a", 100, 200)
    row["pairing_unique_depth"] = "101"

    with pytest.raises(ValueError, match="pairing_unique_depth"):
        simulate_depth_only_budget(row, cap=10000)


def test_role_and_target_summaries() -> None:
    rows = [
        simulate_depth_only_budget(_row("a", 12000, 1500), cap=10000),
        simulate_depth_only_budget(_row("a", 5000, 1000, "L"), cap=10000),
        simulate_depth_only_budget(_row("b", 9000, 1000), cap=10000),
    ]

    entity_summary = summarize_rows(rows)
    target_summary = summarize_targets(rows)

    assert entity_summary["entity_count"] == 3
    assert entity_summary["nonpairing_homologs_starved_count"] == 1
    assert entity_summary["nonpairing_homologs_fully_read_count"] == 2
    assert target_summary["target_count"] == 2
    assert target_summary["any_antibody_entity_starved_count"] == 1
    assert target_summary["all_antibody_entities_starved_count"] == 0
    assert target_summary["all_antibody_entities_fully_read_count"] == 1
