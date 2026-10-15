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

"""End-to-end golden tests for the predicted-protein distillation fast path."""

from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from biotite.structure import AtomArray

from configs.configs_data import data_configs
from protenix.data.core.predicted_protein_fast_path import (
    FastPathNotApplicable,
)
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.tokenizer import TokenArray


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HAS_CCD = Path(data_configs["ccd_components_file"]).is_file()
_AFDB_CIF_ENV = "PROTENIX_AFDB_FAST_PATH_CIF"
_DEFAULT_AFDB_CIF_CANDIDATES = (
    Path(
        "/mnt/project_abd_hs/klp/databases/afdb/msas/"
        "MSA_AB384BL512_0/0029/"
        "AF-A0A165SMH6-F1-model_v4.cif"
    ),
)
_MGNIFY_CIF_ENV = "PROTENIX_MGNIFY_FAST_PATH_CIF"
_DEFAULT_MGNIFY_CIF_CANDIDATES = (
    Path(
        "/mnt/project_abd_hs/biran/data/mgnify/msas/p011/"
        "MGYP000098938132.cif"
    ),
    Path(
        "/tmp/mgnify-cif-test.IgYmjb/batch/"
        "MGYP000000001020.cif"
    ),
)


def _find_fixture(
    environment_variable: str,
    default_candidates: tuple[Path, ...],
) -> Path | None:
    configured_path = os.environ.get(environment_variable)
    candidates = (
        (Path(configured_path).expanduser(),)
        if configured_path
        else default_candidates
    )
    return next(
        (path.resolve() for path in candidates if path.is_file()),
        None,
    )


_AFDB_CIF = _find_fixture(
    _AFDB_CIF_ENV,
    _DEFAULT_AFDB_CIF_CANDIDATES,
)
_MGNIFY_CIF = _find_fixture(
    _MGNIFY_CIF_ENV,
    _DEFAULT_MGNIFY_CIF_CANDIDATES,
)


def _canonical_bond_array(
    atom_array: AtomArray,
) -> np.ndarray | None:
    """Normalize bond endpoint direction and row order."""
    if atom_array.bonds is None:
        return None
    bonds = atom_array.bonds.as_array().copy()
    if bonds.size == 0:
        return bonds
    lower = np.minimum(bonds[:, 0], bonds[:, 1])
    upper = np.maximum(bonds[:, 0], bonds[:, 1])
    bonds[:, 0] = lower
    bonds[:, 1] = upper
    order = np.lexsort((bonds[:, 2], bonds[:, 1], bonds[:, 0]))
    return bonds[order]


class _PipelineEqualityMixin:
    """Field-by-field comparisons shared by the fixtures."""

    def assert_atom_arrays_equal(
        self,
        expected: AtomArray,
        actual: AtomArray,
    ) -> None:
        self.assertIs(type(actual), type(expected))
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(
            set(actual.get_annotation_categories()),
            set(expected.get_annotation_categories()),
        )
        for name in expected.get_annotation_categories():
            with self.subTest(
                component="atom_annotation",
                annotation=name,
            ):
                expected_values = expected.get_annotation(name)
                actual_values = actual.get_annotation(name)
                self.assertEqual(
                    actual_values.dtype,
                    expected_values.dtype,
                )
                np.testing.assert_array_equal(
                    actual_values,
                    expected_values,
                )

        self.assertEqual(actual.coord.dtype, expected.coord.dtype)
        np.testing.assert_array_equal(actual.coord, expected.coord)
        if expected.box is None:
            self.assertIsNone(actual.box)
        else:
            self.assertIsNotNone(actual.box)
            self.assertEqual(actual.box.dtype, expected.box.dtype)
            np.testing.assert_array_equal(actual.box, expected.box)

        expected_bonds = _canonical_bond_array(expected)
        actual_bonds = _canonical_bond_array(actual)
        if expected_bonds is None:
            self.assertIsNone(actual_bonds)
        else:
            self.assertIsNotNone(actual_bonds)
            self.assertEqual(
                actual_bonds.dtype,
                expected_bonds.dtype,
            )
            np.testing.assert_array_equal(
                actual_bonds,
                expected_bonds,
            )

    def assert_token_arrays_equal(
        self,
        expected: TokenArray,
        actual: TokenArray,
    ) -> None:
        self.assertIs(type(actual), type(expected))
        self.assertEqual(len(actual), len(expected))
        for token_index, (
            expected_token,
            actual_token,
        ) in enumerate(zip(expected.tokens, actual.tokens)):
            with self.subTest(
                component="token",
                token_index=token_index,
            ):
                self.assertEqual(
                    actual_token.value,
                    expected_token.value,
                )
                self.assertEqual(
                    set(actual_token._annot),
                    set(expected_token._annot),
                )
                for name, expected_value in (
                    expected_token._annot.items()
                ):
                    actual_value = actual_token._annot[name]
                    if isinstance(expected_value, np.ndarray):
                        self.assertIsInstance(
                            actual_value,
                            np.ndarray,
                        )
                        self.assertEqual(
                            actual_value.dtype,
                            expected_value.dtype,
                        )
                        np.testing.assert_array_equal(
                            actual_value,
                            expected_value,
                        )
                    else:
                        self.assertEqual(
                            actual_value,
                            expected_value,
                        )

    def assert_structure_dicts_equal(
        self,
        expected: Mapping[str, Any],
        actual: Mapping[str, Any],
    ) -> None:
        expected_keys = set(expected)
        actual_keys = set(actual)
        metadata_keys = expected_keys ^ actual_keys
        self.assertTrue(
            all("fast_path" in key for key in metadata_keys),
            "Unexpected top-level structure fields: "
            f"{sorted(metadata_keys)}",
        )

        for key in sorted(expected_keys & actual_keys):
            if "fast_path" in key:
                continue
            with self.subTest(
                component="structure_dict",
                field=key,
            ):
                expected_value = expected[key]
                actual_value = actual[key]
                if isinstance(expected_value, AtomArray):
                    self.assertIsInstance(actual_value, AtomArray)
                    self.assert_atom_arrays_equal(
                        expected_value,
                        actual_value,
                    )
                elif isinstance(expected_value, TokenArray):
                    self.assertIsInstance(actual_value, TokenArray)
                    self.assert_token_arrays_equal(
                        expected_value,
                        actual_value,
                    )
                elif isinstance(expected_value, np.ndarray):
                    self.assertIsInstance(actual_value, np.ndarray)
                    self.assertEqual(
                        actual_value.dtype,
                        expected_value.dtype,
                    )
                    np.testing.assert_array_equal(
                        actual_value,
                        expected_value,
                    )
                else:
                    self.assertIs(
                        type(actual_value),
                        type(expected_value),
                    )
                    self.assertEqual(
                        actual_value,
                        expected_value,
                    )

    def assert_indices_equal(
        self,
        expected: list[dict[str, Any]],
        actual: list[dict[str, Any]],
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for row_index, (
            expected_row,
            actual_row,
        ) in enumerate(zip(expected, actual)):
            with self.subTest(
                component="index",
                row_index=row_index,
            ):
                self.assertEqual(set(actual_row), set(expected_row))
                for key, expected_value in expected_row.items():
                    actual_value = actual_row[key]
                    self.assertIs(
                        type(actual_value),
                        type(expected_value),
                    )
                    self.assertEqual(
                        actual_value,
                        expected_value,
                    )

    def assert_full_pipeline_equal(
        self,
        cif_path: Path,
        fast_path_mode: str = "require",
    ) -> dict[str, Any]:
        expected_indices, expected_structure = (
            DataPipeline.get_data_from_mmcif(
                cif_path,
                dataset="Distillation",
                distillation_fast_path="off",
                raise_on_error=True,
            )
        )
        actual_indices, actual_structure = (
            DataPipeline.get_data_from_mmcif(
                cif_path,
                dataset="Distillation",
                distillation_fast_path=fast_path_mode,
                raise_on_error=True,
            )
        )

        self.assertTrue(expected_indices)
        self.assertTrue(actual_indices)
        self.assert_structure_dicts_equal(
            expected_structure,
            actual_structure,
        )
        self.assert_indices_equal(expected_indices, actual_indices)
        return actual_structure


@unittest.skipUnless(
    _HAS_CCD,
    "standard CCD components file is unavailable",
)
class PredictedProteinFastPathIntegrationTest(
    _PipelineEqualityMixin,
    unittest.TestCase,
):
    def test_2lwu_full_pipeline_matches_generic_output(self):
        structure = self.assert_full_pipeline_equal(
            _REPOSITORY_ROOT / "examples" / "2lwu.cif",
            fast_path_mode="auto",
        )
        fast_path_info = structure["predicted_protein_fast_path"]
        self.assertEqual(fast_path_info["mode"], "auto")
        self.assertTrue(fast_path_info["used"])
        self.assertIsNone(fast_path_info["fallback_reason"])
        self.assertFalse(fast_path_info["cif_reader_used"])
        self.assertIsNotNone(
            fast_path_info["cif_reader_fallback_reason"]
        )

    def test_auto_falls_back_only_for_an_ineligible_input(self):
        with mock.patch(
            "protenix.data.core.parser."
            "prepare_predicted_protein_fast_path",
            side_effect=FastPathNotApplicable("forced_ineligible"),
        ):
            indices, structure = DataPipeline.get_data_from_mmcif(
                _REPOSITORY_ROOT / "examples" / "2lwu.cif",
                dataset="Distillation",
                distillation_fast_path="auto",
                raise_on_error=True,
            )

        self.assertTrue(indices)
        fast_path_info = structure["predicted_protein_fast_path"]
        self.assertEqual(fast_path_info["mode"], "auto")
        self.assertFalse(fast_path_info["used"])
        self.assertEqual(
            fast_path_info["fallback_reason"],
            "forced_ineligible",
        )
        self.assertFalse(fast_path_info["cif_reader_used"])
        self.assertIsNotNone(
            fast_path_info["cif_reader_fallback_reason"]
        )

    def test_auto_does_not_hide_a_fast_path_implementation_error(
        self,
    ):
        with mock.patch(
            "protenix.data.core.parser."
            "complete_predicted_protein_atom_array",
            side_effect=RuntimeError(
                "forced implementation failure"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "forced implementation failure",
            ):
                DataPipeline.get_data_from_mmcif(
                    _REPOSITORY_ROOT / "examples" / "2lwu.cif",
                    dataset="Distillation",
                    distillation_fast_path="auto",
                    raise_on_error=True,
                )

    def test_auto_falls_back_for_ineligible_fast_cif_reader(self):
        with mock.patch(
            "protenix.data.core.parser."
            "read_predicted_protein_atom_array_fast",
            side_effect=FastPathNotApplicable(
                "forced_reader_ineligible"
            ),
        ):
            indices, structure = DataPipeline.get_data_from_mmcif(
                _REPOSITORY_ROOT / "examples" / "2lwu.cif",
                dataset="Distillation",
                distillation_fast_path="auto",
                raise_on_error=True,
            )

        self.assertTrue(indices)
        fast_path_info = structure["predicted_protein_fast_path"]
        self.assertTrue(fast_path_info["used"])
        self.assertFalse(fast_path_info["cif_reader_used"])
        self.assertEqual(
            fast_path_info["cif_reader_fallback_reason"],
            "forced_reader_ineligible",
        )

    def test_auto_does_not_hide_a_fast_cif_reader_error(self):
        with mock.patch(
            "protenix.data.core.parser."
            "read_predicted_protein_atom_array_fast",
            side_effect=RuntimeError(
                "forced reader implementation failure"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "forced reader implementation failure",
            ):
                DataPipeline.get_data_from_mmcif(
                    _REPOSITORY_ROOT / "examples" / "2lwu.cif",
                    dataset="Distillation",
                    distillation_fast_path="auto",
                    raise_on_error=True,
                )

    def test_require_rejects_an_ineligible_fast_cif_reader(self):
        with mock.patch(
            "protenix.data.core.parser."
            "read_predicted_protein_atom_array_fast",
            side_effect=FastPathNotApplicable(
                "forced_reader_ineligible"
            ),
        ):
            with self.assertRaisesRegex(
                FastPathNotApplicable,
                "forced_reader_ineligible",
            ):
                DataPipeline.get_data_from_mmcif(
                    _MGNIFY_CIF
                    or _REPOSITORY_ROOT / "examples" / "2lwu.cif",
                    dataset="Distillation",
                    distillation_fast_path="require",
                    raise_on_error=True,
                )

    @unittest.skipUnless(
        _AFDB_CIF is not None,
        f"set {_AFDB_CIF_ENV} to an AFDB predicted-protein CIF",
    )
    def test_afdb_full_pipeline_matches_generic_output(self):
        structure = self.assert_full_pipeline_equal(_AFDB_CIF)

        fast_path_info = structure["predicted_protein_fast_path"]
        self.assertEqual(fast_path_info["mode"], "require")
        self.assertTrue(fast_path_info["used"])
        self.assertIsNone(fast_path_info["fallback_reason"])
        self.assertTrue(fast_path_info["cif_reader_used"])
        self.assertIsNone(
            fast_path_info["cif_reader_fallback_reason"]
        )

    @unittest.skipUnless(
        _MGNIFY_CIF is not None,
        f"set {_MGNIFY_CIF_ENV} to an MGnify CIF "
        "that lacks terminal OXT",
    )
    def test_mgnify_full_pipeline_matches_generic_output(self):
        structure = self.assert_full_pipeline_equal(_MGNIFY_CIF)

        fast_path_info = structure["predicted_protein_fast_path"]
        self.assertEqual(fast_path_info["mode"], "require")
        self.assertTrue(fast_path_info["used"])
        self.assertIsNone(fast_path_info["fallback_reason"])
        self.assertTrue(fast_path_info["cif_reader_used"])
        self.assertIsNone(
            fast_path_info["cif_reader_fallback_reason"]
        )
        atom_array = structure["atom_array"]
        unresolved_indices = np.flatnonzero(
            ~atom_array.is_resolved
        )
        np.testing.assert_array_equal(
            unresolved_indices,
            [len(atom_array) - 1],
        )
        terminal_index = int(unresolved_indices[0])
        self.assertEqual(
            atom_array.atom_name[terminal_index],
            "OXT",
        )
        np.testing.assert_array_equal(
            atom_array.coord[terminal_index],
            np.zeros(3),
        )


if __name__ == "__main__":
    unittest.main()
