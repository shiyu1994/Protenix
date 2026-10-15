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

"""Independent GPU workers for case-parallel peptide evaluation."""

import contextlib
import logging
import multiprocessing
import os
import signal
import traceback


def partition_cases(cases, workers):
    """Assign each case once, greedily balancing estimated quadratic token cost."""
    if workers < 1:
        raise ValueError("At least one worker is required")
    shards = [[] for _ in range(min(workers, len(cases)))]
    loads = [0] * len(shards)
    ordered = sorted(cases, key=lambda c: (-c["validation"]["input_tokens"], c["name"]))
    for case in ordered:
        slot = min(range(len(shards)), key=lambda i: (loads[i], i))
        shards[slot].append(case)
        loads[slot] += max(1, case["validation"]["input_tokens"]) ** 2
    return shards


def _worker_entry(target, common_args, job):
    # Spawn starts a fresh interpreter. Set visibility before importing torch
    # or the inference runner, so each process uses cuda:0 on its assigned GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = job["gpu"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(key, None)
    with open(job["log"], "a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            logging.basicConfig(level=logging.INFO, force=True)
            try:
                success = target(
                    *common_args, job["cases"], job["progress"], job["input"]
                )
            except BaseException:
                traceback.print_exc()
                raise
            if not success:
                raise SystemExit(1)


def launch_workers(target, common_args, jobs, on_update):
    """Run spawned workers and reap them before releasing the coordinator lock."""
    context = multiprocessing.get_context("spawn")
    children = []
    previous_handlers = {}

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Evaluation interrupted by signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop)
        for job in jobs:
            child = context.Process(
                target=_worker_entry, args=(target, common_args, job), daemon=True
            )
            child.start()
            children.append(child)
            print(
                f"GPU {job['gpu']}: {len(job['cases'])} complexes; log: {job['log']}",
                flush=True,
            )
        reported = set()
        while len(reported) < len(children):
            for index, child in enumerate(children):
                if index in reported:
                    continue
                child.join(timeout=0.2)
                if child.exitcode is not None:
                    print(
                        f"GPU {jobs[index]['gpu']}: worker exited with status {child.exitcode}",
                        flush=True,
                    )
                    reported.add(index)
            on_update()
        return [child.exitcode for child in children]
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        on_update()
