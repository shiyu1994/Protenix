#!/usr/bin/env python3
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

"""Serial training-aligned evaluation queue; requires idle GPUs before launching."""

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import time
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def ready_checkpoints(directory, after_step, settle_seconds=60):
    """Ignore temporary/EMA files and unfinished torch ZIP archives."""
    ready = []
    for path in directory.glob("*.pt"):
        if not path.stem.isdecimal() or int(path.stem) <= after_step:
            continue
        try:
            if time.time() - path.stat().st_mtime < settle_seconds:
                continue
            # Training saves to .tmp.<pid>, then atomically renames to .pt.
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                    continue
            ready.append(path)
        except (OSError, zipfile.BadZipFile):
            continue
    return sorted(ready, key=lambda path: int(path.stem))


def gpu_idle_status(gpus):
    """Fail closed unless every host GPU is unused and requested GPUs exist."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    rows = [tuple(int(value.strip()) for value in line.split(",")) for line in result.stdout.splitlines()]
    if not rows or not set(gpus).issubset({row[0] for row in rows}):
        return False, "Requested GPUs missing from nvidia-smi"
    busy = [f"GPU {index}: util={util}%, memory={memory} MiB" for index, util, memory in rows
            if util != 0 or memory > 64]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    ).stdout.strip()
    if processes:
        busy.append("GPU compute processes: " + processes.replace("\n", ", "))
    return not busy, "; ".join(busy) or "All GPUs idle"


def existing_evaluations():
    """Also wait for an existing evaluator's CPU preparation/scoring stages."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(arg == "scripts.evaluation.evaluate_peptide_pose"
               or Path(arg).name in {"evaluate_peptide_training_aligned.sh", "evaluate_peptide_pose.py"}
               for arg in argv if arg):
            matches.append(int(entry.name))
    return matches


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-suffix", default="20261022")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--after-step", default=7999, type=int)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", default=60, type=int)
    parser.add_argument("--retry-seconds", default=600, type=int)
    parser.add_argument("--max-attempts", default=3, type=int)
    args = parser.parse_args()
    if min(args.poll_seconds, args.retry_seconds, args.max_attempts) < 1:
        parser.error("Polling, retry interval, and attempts must be positive")
    args.checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # A common lock across queue state directories; inherited by the evaluator
    # so even an abruptly killed monitor cannot overlap its surviving child.
    lock = (REPO / "log" / "peptide_checkpoint_monitor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another checkpoint monitor/evaluation holds the queue lock")
    state_path = args.state_dir / "state.json"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    state = json.loads(state_path.read_text()) if state_path.exists() else {"config": config, "checkpoints": {}}
    if state["config"] != config:
        raise SystemExit("Queue configuration changed; use a separate state directory")
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        logging.info("Stop requested; any current evaluation will finish before exit")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state.update(pid=os.getpid(), status="starting", active_step=None)
    (args.state_dir / "monitor.pid").write_text(f"{os.getpid()}\n")
    save_state(state_path, state)
    logging.info("Monitor started: checkpoint_dir=%s, after=%s, GPUs=%s, poll=%ss",
                 args.checkpoint_dir, args.after_step, args.gpus, args.poll_seconds)
    idle_checks = 0
    while not stop:
        try:
            pending = [path for path in ready_checkpoints(args.checkpoint_dir, args.after_step)
                       if state["checkpoints"].get(path.stem, {}).get("status") not in {"complete", "failed"}]
            state.update(queued_steps=[int(path.stem) for path in pending], updated_at=time.time())
            idle, reason = gpu_idle_status([int(gpu) for gpu in args.gpus.split(",")])
            running = existing_evaluations()
            if running:
                idle, reason = False, f"Existing evaluator PIDs: {running}"
            idle_checks = idle_checks + 1 if idle else 0
            state.update(status="waiting", wait_reason=reason)
            if pending and idle_checks >= 2:
                checkpoint = pending[0]
                record = state["checkpoints"].setdefault(checkpoint.stem, {})
                if time.time() >= record.get("retry_after", 0):
                    output = f"{args.output_prefix}_{checkpoint.stem}_{args.output_suffix}"
                    log_path = args.state_dir / f"evaluation_{checkpoint.stem}.log"
                    command = ["bash", str(REPO / "evaluate_peptide_training_aligned.sh"), "all",
                               "--checkpoint", str(checkpoint), "--output-dir", output, "--gpus", args.gpus]
                    record.update(status="running", attempts=record.get("attempts", 0) + 1,
                                  output_dir=output, log=str(log_path), started_at=time.time(), command=command)
                    state.update(status="evaluating", active_step=int(checkpoint.stem))
                    save_state(state_path, state)
                    logging.info("Starting step %s, attempt %s; log=%s", checkpoint.stem, record["attempts"], log_path)
                    with log_path.open("a") as log:
                        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True,
                                                pass_fds=(lock.fileno(),))
                    success = result.returncode == 0
                    record.update(returncode=result.returncode, finished_at=time.time(),
                                  status="complete" if success else (
                                      "failed" if record["attempts"] >= args.max_attempts else "retry"),
                                  retry_after=time.time() + args.retry_seconds)
                    state.update(active_step=None, status="waiting")
                    save_state(state_path, state)
                    logging.info("Step %s: %s (exit %s)", checkpoint.stem, record["status"], result.returncode)
                    idle_checks = 0
            else:
                logging.info("Waiting: queue=%s; %s; idle observations=%s/2",
                             state["queued_steps"], reason, idle_checks)
            save_state(state_path, state)
        except (OSError, ValueError, subprocess.SubprocessError):
            idle_checks = 0
            logging.exception("Monitor check failed; retrying without starting another evaluation")
        for _ in range(args.poll_seconds):
            if stop:
                break
            time.sleep(1)
    state.update(status="stopped", active_step=None)
    save_state(state_path, state)
    logging.info("Monitor stopped")


if __name__ == "__main__":
    main()
