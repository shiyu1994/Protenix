import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_sab23h2_dockq import (
    INPUT_JSON,
    NATIVE_DIR,
    map_roles_to_structure,
)


PRED_ROOT = Path("/mnt/project_abd/klp/benchmark/SAb23H2/protenix-v2")


def input_entry(target):
    data = json.loads(INPUT_JSON.read_text())
    return next(entry for entry in data if entry["name"] == target)


class Sab23H2DockQMappingTest(unittest.TestCase):
    def test_native_mapping_uses_sequence_when_target_chain_ids_are_not_label_ids(self):
        mapping = map_roles_to_structure(
            "7trh_H_B_G",
            input_entry("7trh_H_B_G"),
            NATIVE_DIR / "7trh_H_B_G.cif",
            source="native",
        )

        self.assertEqual(mapping["immune_chains"], ["H", "L"])
        self.assertEqual(mapping["other_chains"], ["A"])
        self.assertEqual([a["target_chain"] for a in mapping["assignments"]], ["H", "B", "G"])
        self.assertTrue(all(a["score"] >= 0.90 for a in mapping["assignments"]))

    def test_native_fragment_mapping_is_high_confidence(self):
        mapping = map_roles_to_structure(
            "8aix_D_NA_K",
            input_entry("8aix_D_NA_K"),
            NATIVE_DIR / "8aix_D_NA_K.cif",
            source="native",
        )

        other = [a for a in mapping["assignments"] if a["role"] == "other"]
        self.assertEqual(len(other), 1)
        self.assertEqual(other[0]["chain"], "A")
        self.assertGreaterEqual(other[0]["score"], 0.90)
        self.assertTrue(other[0]["is_subsequence_match"])

    def test_model_mapping_matches_input_chain_order_by_sequence(self):
        model_path = (
            PRED_ROOT
            / "px_output_original_protenix_v2_template_5seeds"
            / "7trh_H_B_G"
            / "seed_31415"
            / "predictions"
            / "7trh_H_B_G_sample_0.cif"
        )
        mapping = map_roles_to_structure(
            "7trh_H_B_G",
            input_entry("7trh_H_B_G"),
            model_path,
            source="model",
        )

        self.assertEqual(mapping["immune_chains"], ["A", "B"])
        self.assertEqual(mapping["other_chains"], ["C"])
        self.assertTrue(all(a["score"] == 1.0 for a in mapping["assignments"]))


if __name__ == "__main__":
    unittest.main()
