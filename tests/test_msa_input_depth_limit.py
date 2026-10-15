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

from unittest.mock import patch

import numpy as np
from ml_collections.config_dict import ConfigDict

from protenix.data.constants import PROTEIN_CHAIN, RNA_CHAIN
from protenix.data.msa.msa_featurizer import (
    FeatureAssemblyLine,
    MSAFeaturizer,
    MSASourceManager,
    compute_msa_profile,
)
from protenix.data.msa.msa_utils import (
    MSACore,
    RawMsa,
    map_to_standard,
)
from protenix.data.pipeline import dataset as dataset_module


def _a3m(query: str, *hits: str) -> str:
    records = [("query", query)]
    records.extend(
        (f"hit-{index}", hit)
        for index, hit in enumerate(hits, start=1)
    )
    return "".join(
        f">{description}\n{sequence}\n"
        for description, sequence in records
    )


def test_independent_source_budget_is_train_only_at_dataset_boundary():
    configs = ConfigDict(
        {
            "data": {
                "msa": {
                    "prot_seq_or_filename_to_msadir_jsons": [""],
                    "prot_msadir_raw_paths": [""],
                    "rna_seq_or_filename_to_msadir_jsons": [""],
                    "rna_msadir_raw_paths": [""],
                    "prot_pairing_dbs": ["pairing"],
                    "prot_non_pairing_dbs": ["pairing-non_pairing"],
                    "prot_indexing_methods": ["sequence"],
                    "rna_indexing_methods": ["sequence"],
                    "enable_prot_msa": True,
                    "enable_rna_msa": True,
                    "max_size": {"train": 16384, "test": 16384},
                    "pair_msa_as_unpaired_only": False,
                    "independent_pairing_non_pairing_budget": True,
                },
                "weightedPDB": {},
            }
        }
    )

    with patch.object(dataset_module, "MSAFeaturizer") as constructor:
        dataset_module.get_msa_featurizer(
            configs,
            dataset_name="weightedPDB",
            stage="train",
        )
        assert constructor.call_args.kwargs[
            "independent_pairing_non_pairing_budget"
        ] is True

        constructor.reset_mock()
        dataset_module.get_msa_featurizer(
            configs,
            dataset_name="weightedPDB",
            stage="test",
        )
        assert constructor.call_args.kwargs[
            "independent_pairing_non_pairing_budget"
        ] is False
def test_streaming_parser_stops_before_tail_sequence_is_consumed():
    consumed = []

    def lines():
        for line in (
            ">query\n",
            "AAAA\n",
            ">hit-1\n",
            "CCCC\n",
            ">tail\n",
            "GGGG\n",
        ):
            consumed.append(line)
            yield line

    sequences, descriptions = MSACore.parse_fasta_lines(
        lines(),
        depth_limit=2,
    )

    assert descriptions == ["query", "hit-1"]
    assert sequences == ["AAAA", "CCCC"]
    assert "GGGG\n" not in consumed


def test_a3m_top_n_is_applied_before_deduplication():
    raw = RawMsa.from_a3m(
        "AAAA",
        PROTEIN_CHAIN,
        _a3m("AAAA", "AAAA", "CCCC", "DDDD"),
        depth_limit=2,
        dedup=True,
    )

    assert raw.seqs == ["AAAA"]
    assert raw.descs == ["query"]


def test_plain_a3m_merge_does_not_numerically_featurize_inputs():
    first = RawMsa.from_a3m(
        "AAAA",
        PROTEIN_CHAIN,
        _a3m("AAAA", "CCCC"),
        dedup=False,
    )
    second = RawMsa.from_a3m(
        "AAAA",
        PROTEIN_CHAIN,
        _a3m("AAAA", "DDDD"),
        dedup=False,
    )

    with patch.object(
        RawMsa,
        "featurize",
        side_effect=AssertionError("unexpected numerical conversion"),
    ):
        merged = RawMsa.merge(
            [first, second],
            deduplicate=False,
            depth_limit=3,
        )

    assert merged.seqs == ["AAAA", "CCCC", "AAAA"]


def test_single_input_merge_returns_an_independent_raw_msa():
    source = RawMsa.from_a3m(
        "AAAA",
        PROTEIN_CHAIN,
        _a3m("AAAA", "CCCC"),
        dedup=False,
    )

    merged = RawMsa.merge(
        [source],
        deduplicate=False,
        depth_limit=2,
    )

    assert merged is not source
    assert merged.seqs == source.seqs
    assert merged.descs == source.descs


def test_openfold_aligned_arrays_are_sliced_before_conversion():
    msa = np.asarray(
        [list("AAAA"), list("CCCC"), list("GGGG")],
        dtype="S1",
    )
    deletion = np.arange(12, dtype=np.int32).reshape(3, 4)
    raw = RawMsa.from_aligned_arrays(
        "AAAA",
        PROTEIN_CHAIN,
        msa,
        deletion,
        ["query", "hit-1", "tail"],
        dedup=False,
        depth_limit=2,
    )

    assert raw.seqs == ["AAAA", "CCCC"]
    np.testing.assert_array_equal(
        raw.featurize()["deletion_matrix"],
        deletion[:2],
    )
    assert not np.shares_memory(
        raw._precomputed_deletion_matrix,
        deletion,
    )


def test_openfold_sources_use_independent_top_n_pools(tmp_path):
    query = "AAAA"

    def payload(sequences, deletion_value):
        return {
            "msa": np.asarray(
                [list(sequence) for sequence in sequences],
                dtype="S1",
            ),
            "deletion_matrix": np.full(
                (len(sequences), len(query)),
                deletion_value,
                dtype=np.int32,
            ),
            "metadata": np.asarray(
                [f"row-{index}" for index in range(len(sequences))],
                dtype=object,
            ),
        }

    np.savez_compressed(
        tmp_path / "1abc_A.npz",
        uniref90_hits=np.asarray(
            payload([query, "CCCC", "GGGG"], 1),
            dtype=object,
        ),
        cfdb_hits=np.asarray(
            payload([query, "DDDD", "EEEE"], 2),
            dtype=object,
        ),
        mgnify_hits=np.asarray(
            payload([query, "FFFF", "HHHH"], 3),
            dtype=object,
        ),
        uniprot_hits=np.asarray(
            payload([query, "IIII", "KKKK", "LLLL", "MMMM"], 4),
            dtype=object,
        ),
    )
    featurizer = MSAFeaturizer(
        prot_seq_or_filename_to_msadir_jsons=[""],
        prot_msadir_raw_paths=[""],
        rna_seq_or_filename_to_msadir_jsons=[""],
        rna_msadir_raw_paths=[""],
        enable_prot_msa=True,
        enable_rna_msa=False,
        openfold3_alignment_npz_dirs=[str(tmp_path)],
    )

    unpaired, paired = featurizer._load_openfold3_chain_msas(
        pdb_id="1abc",
        label_asym_id="A",
        query=query,
        chain_type=PROTEIN_CHAIN,
        depth_limit=4,
    )

    assert [msa.depth for msa in unpaired] == [3, 1]
    assert [msa.depth for msa in paired] == [4]
    np.testing.assert_array_equal(
        unpaired[1].featurize()["deletion_matrix"],
        np.full((1, len(query)), 2, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        paired[0].featurize()["deletion_matrix"],
        np.full((4, len(query)), 4, dtype=np.int32),
    )

    full_unpaired, full_paired = featurizer._load_openfold3_chain_msas(
        pdb_id="1abc",
        label_asym_id="A",
        query=query,
        chain_type=PROTEIN_CHAIN,
        depth_limit=None,
    )
    assert [msa.depth for msa in full_unpaired] == [3, 3, 3]
    assert [msa.depth for msa in full_paired] == [5]

    bounded_unpaired, bounded_paired = (
        featurizer._load_openfold3_chain_msas(
            pdb_id="1abc",
            label_asym_id="A",
            query=query,
            chain_type=PROTEIN_CHAIN,
            paired_depth_limit=2,
            unpaired_depth_limit=4,
        )
    )
    assert [msa.depth for msa in bounded_unpaired] == [3, 1]
    assert [msa.depth for msa in bounded_paired] == [2]

    recovery = featurizer._load_openfold3_recovery_candidate(
        representative_id="1abc_A",
        chain_type=PROTEIN_CHAIN,
        paired_depth_limit=2,
        unpaired_depth_limit=4,
    )
    assert [msa.depth for msa in recovery.unpaired_msas] == [3, 1]
    assert [msa.depth for msa in recovery.paired_msas] == [2]


def test_profile_chunking_matches_reference_one_hot_reduction():
    msa = np.asarray(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 2, 2, 3],
        ],
        dtype=np.int32,
    )
    expected = (
        msa[..., None] == np.arange(8)[None, None, :]
    ).sum(axis=0) / msa.shape[0]

    actual = compute_msa_profile(
        msa,
        num_classes=8,
        max_chunk_cells=4,
    )

    np.testing.assert_array_equal(
        actual,
        expected.astype(np.float32),
    )


def test_source_manager_uses_one_top_n_budget_across_sources(tmp_path):
    query = "AAAA"
    paths = [tmp_path / "source-1", tmp_path / "source-2"]
    mappings = {}
    for index, path in enumerate(paths):
        msa_dir = path / f"entry-{index}"
        msa_dir.mkdir(parents=True)
        (msa_dir / "unpaired.a3m").write_text(
            _a3m(query, "CCCC", "GGGG"),
            encoding="utf-8",
        )
        mappings[index] = {query: f"entry-{index}"}

    manager = MSASourceManager(
        raw_paths=[str(path) for path in paths],
        methods=["sequence", "sequence"],
        mappings=mappings,
        enabled=True,
    )
    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["", ""],
        np_dbs=["unpaired", "unpaired"],
        depth_limit=4,
    )

    assert paired == []
    assert [msa.depth for msa in unpaired] == [3, 1]
    assert sum(msa.depth for msa in unpaired) == 4


def test_empty_a3m_does_not_consume_the_next_source_budget(tmp_path):
    query = "AAAA"
    paths = [tmp_path / "empty-source", tmp_path / "real-source"]
    mappings = {}
    for index, path in enumerate(paths):
        msa_dir = path / f"entry-{index}"
        msa_dir.mkdir(parents=True)
        mappings[index] = {query: f"entry-{index}"}
    (paths[0] / "entry-0" / "unpaired.a3m").write_text(
        "",
        encoding="utf-8",
    )
    (paths[1] / "entry-1" / "unpaired.a3m").write_text(
        _a3m(query, "CCCC"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(path) for path in paths],
        methods=["sequence", "sequence"],
        mappings=mappings,
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["", ""],
        np_dbs=["unpaired", "unpaired"],
        depth_limit=1,
    )

    assert paired == []
    assert len(unpaired) == 1
    assert unpaired[0].seqs == [query]


def test_protein_paired_and_unpaired_pools_have_independent_budgets(tmp_path):
    query = "AAAA"
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "paired.a3m").write_text(
        _a3m(query, "CCCC", "GGGG"),
        encoding="utf-8",
    )
    (entry / "unpaired.a3m").write_text(
        _a3m(query, "DDDD", "EEEE"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(tmp_path)],
        methods=["sequence"],
        mappings={0: {query: "entry"}},
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["paired"],
        np_dbs=["unpaired"],
        paired_depth_limit=2,
        unpaired_depth_limit=3,
    )

    assert [msa.depth for msa in paired] == [2]
    assert [msa.depth for msa in unpaired] == [3]
    assert paired[0].seqs == [query, "CCCC"]
    assert unpaired[0].seqs == [query, "DDDD", "EEEE"]


def test_pairing_copy_and_genuine_non_pairing_have_independent_budgets(
    tmp_path,
):
    query = "AAAAA"
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "pairing.a3m").write_text(
        _a3m(query, "CCCCC", "CCCCC", "GGGGG"),
        encoding="utf-8",
    )
    (entry / "non_pairing.a3m").write_text(
        _a3m(query, "DDDDD", "EEEEE", "FFFFF"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(tmp_path)],
        methods=["sequence"],
        mappings={0: {query: "entry"}},
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["pairing"],
        np_dbs=["pairing-non_pairing"],
        paired_depth_limit=4,
        unpaired_depth_limit=4,
        independent_pairing_non_pairing_budget=True,
    )

    assert [msa.depth for msa in paired] == [4]
    assert [msa.depth for msa in unpaired] == [4, 4]
    assert unpaired[0] is paired[0]
    assert unpaired[1].seqs == [query, "DDDDD", "EEEEE", "FFFFF"]
    assert paired[0].seqs == [query, "CCCCC", "CCCCC", "GGGGG"]
    combined = RawMsa.merge(unpaired, deduplicate=True)
    assert combined.seqs == [
        query,
        "CCCCC",
        "GGGGG",
        "DDDDD",
        "EEEEE",
        "FFFFF",
    ]

    features = FeatureAssemblyLine(
        max_msa_size=5,
        max_input_paired_msa_depth=4,
        max_input_unpaired_msa_depth=4,
    ).assemble(
        {
            1: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": query,
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": "",
                "unpaired_msa": combined,
                "unpaired_msa_contains_paired_source_rows": True,
            }
        },
        np.arange(len(query), dtype=np.int32),
    )
    assert features.rows.shape == (5, len(query))


def test_independent_budget_is_inactive_without_both_source_classes():
    direct_only = MSAFeaturizer(
        prot_pairing_dbs=[""],
        prot_non_pairing_dbs=[""],
        independent_pairing_non_pairing_budget=True,
    )
    pairing_only = MSAFeaturizer(
        prot_pairing_dbs=["pairing"],
        prot_non_pairing_dbs=["pairing"],
        independent_pairing_non_pairing_budget=True,
    )
    mixed = MSAFeaturizer(
        prot_pairing_dbs=["pairing"],
        prot_non_pairing_dbs=["pairing-non_pairing"],
        independent_pairing_non_pairing_budget=True,
    )

    assert not direct_only.independent_pairing_non_pairing_budget
    assert not pairing_only.independent_pairing_non_pairing_budget
    assert mixed.independent_pairing_non_pairing_budget


def test_independent_budget_still_finds_late_pairing_copy(tmp_path):
    query = "AAAA"
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "pairing.a3m").write_text(
        _a3m(query, "CCCC"),
        encoding="utf-8",
    )
    (entry / "non_pairing.a3m").write_text(
        _a3m(query, "DDDD"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(tmp_path)],
        methods=["sequence"],
        mappings={0: {query: "entry"}},
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["pairing"],
        np_dbs=["non_pairing-pairing"],
        paired_depth_limit=2,
        unpaired_depth_limit=2,
        independent_pairing_non_pairing_budget=True,
    )

    assert [msa.seqs for msa in unpaired] == [
        [query, "DDDD"],
        [query, "CCCC"],
    ]
    assert unpaired[1] is paired[0]


def test_unused_paired_budget_is_not_transferred_to_unpaired(tmp_path):
    query = "AAAA"
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "paired.a3m").write_text(
        _a3m(query),
        encoding="utf-8",
    )
    (entry / "unpaired.a3m").write_text(
        _a3m(query, "CCCA", "CCCG", "CCCT", "CCCC"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(tmp_path)],
        methods=["sequence"],
        mappings={0: {query: "entry"}},
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        PROTEIN_CHAIN,
        p_dbs=["paired"],
        np_dbs=["unpaired"],
        paired_depth_limit=4,
        unpaired_depth_limit=3,
    )

    assert sum(msa.depth for msa in paired) == 1
    assert sum(msa.depth for msa in unpaired) == 3


def test_pair_as_unpaired_uses_final_budget_after_independent_pool_caps():
    query = "AAAAA"
    paired = RawMsa(
        query=query,
        chain_type=PROTEIN_CHAIN,
        sequences=[query, "CCCCC", "DDDDD", "EEEEE", "IIIII"],
        descriptions=["query", "p1", "p2", "p3", "p4"],
        deduplicate=False,
    )
    unpaired = RawMsa(
        query=query,
        chain_type=PROTEIN_CHAIN,
        sequences=[query, "FFFFF", "GGGGG", "HHHHH", "KKKKK"],
        descriptions=["query", "u1", "u2", "u3", "u4"],
        deduplicate=False,
    )
    features = FeatureAssemblyLine(
        max_msa_size=6,
        pair_msa_as_unpaired_only=True,
        max_input_paired_msa_depth=4,
        max_input_unpaired_msa_depth=4,
    ).assemble(
        {
            1: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": query,
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": paired,
                "unpaired_msa": unpaired,
            }
        },
        np.arange(len(query), dtype=np.int32),
    )

    # Each raw side contributes up to four rows. The duplicate query is
    # removed, then the combined pool is cropped to the final budget of six;
    # it is not incorrectly re-cropped to the unpaired raw cap of four.
    assert features.rows.shape == (6, len(query))


def test_pair_as_unpaired_deduplicates_pairing_copy_before_final_cap():
    query = "AAAAA"
    paired = RawMsa(
        query=query,
        chain_type=PROTEIN_CHAIN,
        sequences=[query, "CCCCC", "DDDDD", "EEEEE"],
        descriptions=["query", "p1", "p2", "p3"],
        deduplicate=False,
    )
    precombined_unpaired = RawMsa(
        query=query,
        chain_type=PROTEIN_CHAIN,
        sequences=[
            query,
            "CCCCC",
            "DDDDD",
            "EEEEE",
            "FFFFF",
            "GGGGG",
        ],
        descriptions=["query", "p1", "p2", "p3", "u1", "u2"],
        deduplicate=True,
    )

    features = FeatureAssemblyLine(
        max_msa_size=6,
        pair_msa_as_unpaired_only=True,
        max_input_paired_msa_depth=4,
        max_input_unpaired_msa_depth=4,
    ).assemble(
        {
            1: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": query,
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": paired,
                "unpaired_msa": precombined_unpaired,
                "unpaired_msa_contains_paired_source_rows": True,
                "unpaired_msa_uses_independent_pairing_source_budget": True,
            }
        },
        np.arange(len(query), dtype=np.int32),
    )

    expected_genuine_hit, _ = MSACore.sequences_to_array(
        ["FFFFF"],
        PROTEIN_CHAIN,
    )
    assert features.rows.shape == (6, len(query))
    assert any(
        np.array_equal(row, expected_genuine_hit[0])
        for row in features.rows
    )


def test_precombined_partial_query_only_pool_uses_final_budget():
    query = "AAAAA"
    combined = RawMsa(
        query=query,
        chain_type=PROTEIN_CHAIN,
        sequences=[
            query,
            "CCCCC",
            "DDDDD",
            "EEEEE",
            "FFFFF",
            "GGGGG",
            "HHHHH",
        ],
        descriptions=["query", "p1", "p2", "p3", "u1", "u2", "u3"],
        deduplicate=False,
    )
    features = FeatureAssemblyLine(
        max_msa_size=6,
        max_input_paired_msa_depth=4,
        max_input_unpaired_msa_depth=4,
    ).assemble(
        {
            1: {
                "entity_id": "1",
                "chain_id": "A",
                "sequence": query,
                "chain_entity_type": PROTEIN_CHAIN,
                "paired_msa": "",
                "unpaired_msa": combined,
                "unpaired_msa_contains_paired_source_rows": True,
            }
        },
        np.arange(len(query), dtype=np.int32),
    )

    assert features.rows.shape == (6, len(query))


def test_training_config_locks_default_pools_and_distillation_override():
    from configs.configs_data import data_configs

    msa_config = data_configs["msa"]
    assert msa_config["max_input_paired_msa_depth"] == 10000
    assert msa_config["max_input_unpaired_msa_depth"] == 10000
    assert msa_config["max_size"]["train"] == 16384
    for dataset_name in (
        "afdb_mgnify_short_distillation_v20260725",
        "afdb_mgnify_long_distillation_v20260725",
    ):
        dataset_msa = data_configs[dataset_name]["msa"]
        assert dataset_msa["max_input_unpaired_msa_depth"] == 16384
        assert "max_input_paired_msa_depth" not in dataset_msa


def test_nonstandard_rna_tokens_repeat_the_same_canonical_msa_column():
    meta = {
        7: {
            "sequence": "ACGU",
        }
    }
    mapped = map_to_standard(
        np.asarray([7, 7, 7, 7, 7, 7]),
        np.asarray([1, 2, 2, 2, 3, 4]),
        meta,
    )

    np.testing.assert_array_equal(
        mapped,
        np.asarray([0, 1, 1, 1, 2, 3], dtype=np.int32),
    )


def test_rna_source_uses_training_depth_limit(tmp_path):
    query = "ACGU"
    entry = tmp_path / "rna-entry"
    entry.mkdir()
    (entry / "rna-entry_all.a3m").write_text(
        _a3m(query, "ACGA", "ACGC", "ACGG"),
        encoding="utf-8",
    )
    manager = MSASourceManager(
        raw_paths=[str(tmp_path)],
        methods=["sequence"],
        mappings={0: {query: ["rna-entry"]}},
        enabled=True,
    )

    unpaired, paired = manager.fetch_msas(
        query,
        "",
        RNA_CHAIN,
        unpaired_depth_limit=2,
    )

    assert paired == []
    assert len(unpaired) == 1
    assert unpaired[0].depth == 2
    assert unpaired[0].seqs[0] == query
