from __future__ import annotations

import unittest
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import torch

from runner.train import CudaPhaseMemoryRecorder, build_phase_cuda_memory_summary


@contextmanager
def mocked_cuda_allocator(
    state: dict[str, int],
) -> Iterator[dict[str, list[torch.device]]]:
    calls: dict[str, list[torch.device]] = {"synchronize": [], "reset": []}

    def synchronize(device: torch.device) -> None:
        calls["synchronize"].append(device)

    def reset(device: torch.device) -> None:
        calls["reset"].append(device)
        state["max_allocated_bytes"] = state["allocated_bytes"]
        state["max_reserved_bytes"] = state["reserved_bytes"]

    with ExitStack() as stack:
        stack.enter_context(patch("runner.train.torch.cuda.is_available", return_value=True))
        stack.enter_context(
            patch("runner.train.torch.cuda.synchronize", side_effect=synchronize)
        )
        stack.enter_context(
            patch("runner.train.torch.cuda.reset_peak_memory_stats", side_effect=reset)
        )
        stack.enter_context(
            patch(
                "runner.train.torch.cuda.memory_allocated",
                side_effect=lambda _device: state["allocated_bytes"],
            )
        )
        stack.enter_context(
            patch(
                "runner.train.torch.cuda.memory_reserved",
                side_effect=lambda _device: state["reserved_bytes"],
            )
        )
        stack.enter_context(
            patch(
                "runner.train.torch.cuda.max_memory_allocated",
                side_effect=lambda _device: state["max_allocated_bytes"],
            )
        )
        stack.enter_context(
            patch(
                "runner.train.torch.cuda.max_memory_reserved",
                side_effect=lambda _device: state["max_reserved_bytes"],
            )
        )
        yield calls


def phase(
    name: str,
    *,
    peak_allocated: int,
    peak_reserved: int,
    end_allocated: int,
    peak_increase: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "end_allocated_bytes": end_allocated,
        "peak_increase_bytes": peak_increase,
    }


class TestCudaPhaseMemoryRecorder(unittest.TestCase):
    def test_lifecycle_preserves_phase_and_inter_step_overall_peaks(self) -> None:
        device = torch.device("cuda:0")
        state = {
            "allocated_bytes": 100,
            "reserved_bytes": 200,
            # This peak belongs to work after the benchmark window opened but
            # before the first phase reset.
            "max_allocated_bytes": 900,
            "max_reserved_bytes": 1000,
        }

        with mocked_cuda_allocator(state) as calls:
            recorder = CudaPhaseMemoryRecorder(enabled=True, device=device)
            recorder.start_window()
            with self.assertRaises(RuntimeError):
                recorder.start_window()

            recorder.start_step(step=4, global_step=4, is_update_step=True)
            recorder.set_sample({"pdb_id": "sample-a", "n_token": 1024})
            state.update(
                allocated_bytes=180,
                reserved_bytes=260,
                max_allocated_bytes=300,
                max_reserved_bytes=350,
            )
            recorder.transition("trunk")
            state.update(
                allocated_bytes=240,
                reserved_bytes=310,
                max_allocated_bytes=700,
                max_reserved_bytes=800,
            )
            recorder.end_step()

            # Simulate an unattributed inter-step allocation. start_step must
            # fold this peak into the software whole-window maximum before it
            # resets the allocator counters for the next to_device phase.
            state.update(
                allocated_bytes=120,
                reserved_bytes=220,
                max_allocated_bytes=950,
                max_reserved_bytes=1050,
            )
            recorder.start_step(step=5, global_step=5, is_update_step=True)
            state.update(
                allocated_bytes=160,
                reserved_bytes=240,
                max_allocated_bytes=400,
                max_reserved_bytes=450,
            )
            recorder.end_step()
            recorder.finish_window()

            result = recorder.result()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(recorder.window_active)
        self.assertEqual(result["overall_max_allocated_bytes"], 950)
        self.assertEqual(result["overall_max_reserved_bytes"], 1050)
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][0]["sample"]["pdb_id"], "sample-a")
        self.assertEqual(
            [item["name"] for item in result["steps"][0]["phases"]],
            ["to_device", "trunk"],
        )
        self.assertEqual(
            result["steps"][0]["phases"][1]["peak_allocated_bytes"], 700
        )
        self.assertEqual(len(calls["reset"]), 3)
        self.assertEqual(len(calls["synchronize"]), 6)

    def test_disabled_recorder_is_a_noop(self) -> None:
        with patch("runner.train.torch.cuda.is_available", return_value=False):
            recorder = CudaPhaseMemoryRecorder(
                enabled=True,
                device=torch.device("cuda:0"),
            )
        recorder.start_window()
        recorder.start_step(step=0, global_step=0, is_update_step=True)
        recorder.transition("trunk")
        recorder.end_step()
        recorder.finish_window()
        self.assertFalse(recorder.enabled)
        self.assertIsNone(recorder.result())


class TestPhaseCudaMemorySummary(unittest.TestCase):
    def test_aggregates_phase_and_whole_window_maxima_across_ranks(self) -> None:
        rank_records = [
            {
                "rank": 0,
                "phase_cuda_memory": {
                    "schema_version": 1,
                    "mode": "synchronized_phase_peaks",
                    "throughput_comparable": False,
                    "overall_max_allocated_bytes": 700,
                    "overall_max_reserved_bytes": 900,
                    "steps": [
                        {
                            "step": 10,
                            "phases": [
                                phase(
                                    "trunk",
                                    peak_allocated=500,
                                    peak_reserved=800,
                                    end_allocated=300,
                                    peak_increase=250,
                                ),
                                phase(
                                    "loss",
                                    peak_allocated=400,
                                    peak_reserved=700,
                                    end_allocated=350,
                                    peak_increase=100,
                                ),
                            ],
                        }
                    ],
                },
            },
            {
                "rank": 1,
                "phase_cuda_memory": {
                    "schema_version": 1,
                    "mode": "synchronized_phase_peaks",
                    "throughput_comparable": False,
                    "overall_max_allocated_bytes": 950,
                    "overall_max_reserved_bytes": 1000,
                    "steps": [
                        {
                            "step": 11,
                            "phases": [
                                phase(
                                    "trunk",
                                    peak_allocated=850,
                                    peak_reserved=920,
                                    end_allocated=450,
                                    peak_increase=600,
                                )
                            ],
                        }
                    ],
                },
            },
            {"rank": 2, "phase_cuda_memory": None},
        ]

        summary = build_phase_cuda_memory_summary(rank_records)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["rank_record_count"], 2)
        self.assertEqual(summary["overall_max_allocated_bytes"], 950)
        self.assertEqual(summary["overall_max_reserved_bytes"], 1000)
        self.assertEqual(summary["overall_max_allocated_rank"], 1)
        self.assertEqual(summary["phase_order"], ["trunk", "loss"])
        trunk = summary["aggregate_by_phase"]["trunk"]
        self.assertEqual(trunk["observation_count"], 2)
        self.assertEqual(trunk["max_peak_allocated_bytes"], 850)
        self.assertEqual(trunk["max_peak_reserved_bytes"], 920)
        self.assertEqual(trunk["max_end_allocated_bytes"], 450)
        self.assertEqual(trunk["max_peak_increase_bytes"], 600)
        self.assertEqual(trunk["max_rank"], 1)
        self.assertEqual(trunk["max_step"], 11)
        self.assertEqual(
            summary["aggregate_by_phase"]["loss"]["observation_count"], 1
        )
        self.assertEqual(len(summary["rank_records"]), 2)

    def test_returns_none_without_rank_payloads(self) -> None:
        self.assertIsNone(
            build_phase_cuda_memory_summary(
                [{"rank": 0, "phase_cuda_memory": None}]
            )
        )


if __name__ == "__main__":
    unittest.main()
