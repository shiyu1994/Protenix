import csv
import gzip
import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark.audit_msa_depth import (
    audit_msa_depth,
    normalize_entity_id,
    write_audit_outputs,
)


def _write_a3m(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">{description}\n{sequence}\n" for description, sequence in records),
        encoding="utf-8",
    )


def _write_bioassembly(
    path: Path, sequences: dict[str, str], entity_types: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump({"sequences": sequences, "entity_poly_type": entity_types}, handle)


class MsaDepthAuditTest(unittest.TestCase):
    def test_normalize_entity_id_handles_csv_float_and_missing_values(self):
        self.assertEqual(normalize_entity_id("2.0"), "2")
        self.assertEqual(normalize_entity_id(3), "3")
        self.assertIsNone(normalize_entity_id("nan"))

    def test_audit_combines_and_deduplicates_pairing_and_non_pairing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / "indices.csv"
            mapping_path = root / "seq_to_pdb_index.json"
            bioassembly_dir = root / "bioassembly"
            msa_root = root / "msa"
            output_json = root / "out" / "audit.json"
            output_csv = root / "out" / "audit.csv"

            with index_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "pdb_id",
                        "num_tokens",
                        "entity_1_id",
                        "mol_1_type",
                        "entity_2_id",
                        "mol_2_type",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pdb_id": "p001",
                        "num_tokens": 1200,
                        "entity_1_id": "1.0",
                        "mol_1_type": "prot",
                        "entity_2_id": "2.0",
                        "mol_2_type": "prot",
                    }
                )
                writer.writerow(
                    {
                        "pdb_id": "p002",
                        "num_tokens": 1300,
                        "entity_1_id": "1.0",
                        "mol_1_type": "prot",
                        "entity_2_id": "",
                        "mol_2_type": "nan",
                    }
                )

            mapping_path.write_text(json.dumps({"AAAA": "0/deep"}), encoding="utf-8")
            _write_bioassembly(
                bioassembly_dir / "p001.pkl.gz",
                {"1": "AAAA", "2": "BBBB"},
                {"1": "polypeptide(L)", "2": "polypeptide(L)"},
            )
            _write_bioassembly(
                bioassembly_dir / "p002.pkl.gz",
                {"1": "CCCC"},
                {"1": "polypeptide(L)"},
            )

            # pairing contributes query + hit A.  non_pairing repeats the
            # query and hit A, then adds hit B; combined effective depth is 3.
            _write_a3m(
                msa_root / "0/deep/pairing.a3m",
                [("query", "AAAA"), ("hit-a", "A-AA")],
            )
            _write_a3m(
                msa_root / "0/deep/non_pairing.a3m",
                [("query", "AAAA"), ("hit-a-duplicate", "A-AA"), ("hit-b", "AA-A")],
            )

            report = audit_msa_depth(
                index_path=index_path,
                mapping_path=mapping_path,
                bioassembly_dir=bioassembly_dir,
                msa_root=msa_root,
                target_depth=3,
            )
            write_audit_outputs(report, output_json, output_csv)

            summary = report["summary"]
            self.assertEqual(summary["processed_rows"], 2)
            self.assertEqual(summary["protein_entities"], 3)
            self.assertEqual(summary["protein_entities_reaching_target"], 1)
            self.assertEqual(summary["rows_any_referenced_entity_reaching_target"], 1)
            self.assertEqual(summary["missing_mapping_entities"], 2)

            first = report["records"][0]
            deep = first["entities"][0]
            self.assertEqual(deep["pairing"]["raw_depth"], 2)
            self.assertEqual(deep["non_pairing"]["raw_depth"], 3)
            self.assertEqual(deep["combined_unique_depth"], 3)
            self.assertEqual(deep["potential_sampled_depth"], 3)
            self.assertTrue(deep["reaches_target"])
            self.assertEqual(first["max_effective_depth"], 3)
            self.assertEqual(first["min_effective_depth"], 1)
            self.assertFalse(first["all_referenced_entities_reach_target"])

            saved = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"], summary)
            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["referenced_protein_entity_ids"], "1;2")
            self.assertEqual(csv_rows[0]["max_effective_depth"], "3")

    def test_limit_audits_only_the_first_rows(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / "indices.csv"
            mapping_path = root / "mapping.json"
            bioassembly_dir = root / "bio"
            msa_root = root / "msa"
            index_path.write_text(
                "pdb_id,num_tokens,entity_1_id,mol_1_type\n"
                "p001,1200,1,prot\n"
                "p002,1300,1,prot\n",
                encoding="utf-8",
            )
            mapping_path.write_text("{}", encoding="utf-8")
            _write_bioassembly(
                bioassembly_dir / "p001.pkl.gz",
                {"1": "AAAA"},
                {"1": "polypeptide(L)"},
            )

            report = audit_msa_depth(
                index_path=index_path,
                mapping_path=mapping_path,
                bioassembly_dir=bioassembly_dir,
                msa_root=msa_root,
                target_depth=2048,
                limit=1,
            )

            self.assertEqual(report["summary"]["processed_rows"], 1)
            self.assertEqual(report["records"][0]["pdb_id"], "p001")
            self.assertEqual(report["inputs"]["limit"], 1)


if __name__ == "__main__":
    unittest.main()
