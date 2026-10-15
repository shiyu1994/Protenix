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

"""Launch one directory-sharded distillation rank on each SSH host.

``launch`` is dry-run by default.  Add ``--execute`` only after reviewing the
printed host-to-rank mapping and shell-quoted commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILDER = REPO_ROOT / "scripts" / "build_sharded_distillation_bioassembly.py"
_SAFE_HOST = re.compile(r"^[A-Za-z0-9_.:@-]+$")
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
    "PYTHONUNBUFFERED": "1",
}


@dataclass(frozen=True, slots=True)
class RankLaunch:
    rank: int
    host: str
    log_path: str
    pid_path: str
    command: tuple[str, ...]


def read_hosts(path: Path) -> list[str]:
    hosts = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        host = raw_line.split("#", maxsplit=1)[0].strip()
        if not host:
            continue
        if (
            host.startswith("-")
            or not _SAFE_HOST.fullmatch(host)
            or any(character.isspace() for character in host)
        ):
            raise ValueError(
                f"Unsafe SSH destination at {path}:{line_number}: {host!r}"
            )
        hosts.append(host)
    if not hosts:
        raise ValueError(f"Hosts file is empty: {path}")
    duplicates = sorted({host for host in hosts if hosts.count(host) > 1})
    if duplicates:
        raise ValueError(f"Hosts file contains duplicates: {duplicates}")
    return hosts


def _load_plan(plan_dir: Path) -> dict[str, Any]:
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if int(plan.get("version", -1)) != 3:
        raise ValueError(
            f"Expected plan version 3 in {plan_path}, "
            f"found {plan.get('version')!r}"
        )
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"]:
        raise ValueError(f"Missing plan_id in {plan_path}")
    return plan


def _shell_assignment(name: str, value: str | Path) -> str:
    return f"{name}={shlex.quote(str(value))}"


def build_remote_script(
    *,
    rank: int,
    plan_dir: Path,
    builder: Path,
    work_dir: Path,
    protenix_root_dir: Path,
    python_executable: str,
    workers: int,
    compresslevel: int,
    log_path: Path,
    pid_path: Path,
    resume: bool,
    fast_path_mode: str = "auto",
    durable_output: bool = False,
    tail_max_lanes_per_leaf: int = 0,
) -> str:
    if fast_path_mode not in {"off", "auto", "require"}:
        raise ValueError(f"Unsupported fast path mode: {fast_path_mode}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if not 0 <= compresslevel <= 9:
        raise ValueError(
            f"compresslevel must be in [0, 9], got {compresslevel}"
        )
    if tail_max_lanes_per_leaf < 0:
        raise ValueError(
            "tail_max_lanes_per_leaf must be non-negative, "
            f"got {tail_max_lanes_per_leaf}"
        )

    builder_arguments = [
        python_executable,
        str(builder),
        "run",
        "--plan-dir",
        str(plan_dir),
        "--rank",
        str(rank),
        "--workers",
        str(workers),
        "--compresslevel",
        str(compresslevel),
        "--protenix-root-dir",
        str(protenix_root_dir),
        "--tail-max-lanes-per-leaf",
        str(tail_max_lanes_per_leaf),
    ]
    if fast_path_mode == "off":
        builder_arguments.append("--disable-fast-path")
    elif fast_path_mode == "require":
        builder_arguments.append("--require-fast-path")
    if durable_output:
        builder_arguments.append("--durable-output")

    lock_path = log_path.parent / f"rank-{rank:03d}.lock"
    status_path = plan_dir / "status" / f"rank-{rank:03d}.json"
    rank_plan_path = plan_dir / f"rank-{rank:03d}.jsonl"
    environment = {
        **_THREAD_ENV,
        "PROTENIX_ROOT_DIR": str(protenix_root_dir),
    }
    env_prefix = " ".join(
        _shell_assignment(name, value)
        for name, value in environment.items()
    )
    builder_command = (
        f"export PYTHONPATH={shlex.quote(str(work_dir))}"
        '"${PYTHONPATH:+:${PYTHONPATH}}"; '
        f"exec env {env_prefix} {shlex.join(builder_arguments)}"
    )
    lease_action = (
        'echo "rank lease is already held: $lock_file"; exit 0'
        if resume
        else 'echo "rank lease is already held: $lock_file" >&2; exit 5'
    )
    completed_status_check = (
        "import json,sys;"
        "plan=json.load(open(sys.argv[1],encoding='utf-8'));"
        "status=json.load(open(sys.argv[2],encoding='utf-8'));"
        "raise SystemExit(0 if "
        "status.get('plan_id')==plan.get('plan_id') "
        "and status.get('state')=='complete' else 1)"
    )

    lines = [
        "set -euo pipefail",
        "umask 0022",
        "if ! command -v flock >/dev/null 2>&1; then",
        '  echo "flock is required on every worker" >&2',
        "  exit 2",
        "fi",
        f"mkdir -p {shlex.quote(str(log_path.parent))}",
        f"pid_file={shlex.quote(str(pid_path))}",
        f"lock_file={shlex.quote(str(lock_path))}",
        f"log_file={shlex.quote(str(log_path))}",
        f"status_file={shlex.quote(str(status_path))}",
        f"python_bin={shlex.quote(str(python_executable))}",
        f"builder_file={shlex.quote(str(builder))}",
        f"plan_file={shlex.quote(str(plan_dir / 'plan.json'))}",
        f"rank_plan_file={shlex.quote(str(rank_plan_path))}",
        f"work_directory={shlex.quote(str(work_dir))}",
        f"ccd_file={shlex.quote(str(protenix_root_dir / 'common' / 'components.cif'))}",
        (
            "ccd_cache_file="
            f"{shlex.quote(str(protenix_root_dir / 'common' / 'components.cif.rdkit_mol.pkl'))}"
        ),
        'if [ ! -x "$python_bin" ]; then',
        '  echo "remote Python is missing or not executable: $python_bin" >&2',
        "  exit 2",
        "fi",
        "for required_file in "
        '"$builder_file" "$plan_file" "$rank_plan_file" '
        '"$ccd_file" "$ccd_cache_file"; do',
        '  if [ ! -f "$required_file" ]; then',
        '    echo "required remote file is missing: $required_file" >&2',
        "    exit 2",
        "  fi",
        "done",
        'if [ ! -d "$work_directory" ]; then',
        '  echo "remote work directory is missing: $work_directory" >&2',
        "  exit 2",
        "fi",
        'cd "$work_directory"',
        'exec 9>"$lock_file"',
        "if ! flock -n 9; then",
        f"  {lease_action}",
        "fi",
        f"nohup bash -lc {shlex.quote(builder_command)} "
        f">>{shlex.quote(str(log_path))} 2>&1 </dev/null 9>&9 &",
        'child_pid="$!"',
        'pid_tmp="${pid_file}.tmp.$$"',
        'printf "%s\\n" "$child_pid" >"$pid_tmp"',
        'mv -f "$pid_tmp" "$pid_file"',
        "sleep 0.5",
        'if ! kill -0 "$child_pid" 2>/dev/null; then',
        '  if [ -f "$status_file" ] && "$python_bin" -c '
        f"{shlex.quote(completed_status_check)} "
        '"$plan_file" "$status_file"; then',
        '    rm -f "$pid_file"',
        "    printf 'rank=%s already complete log=%s\\n' "
        f"{shlex.quote(str(rank))} \"$log_file\"",
        "    exit 0",
        "  fi",
        '  rm -f "$pid_file"',
        '  echo "rank process exited during startup; inspect $log_file" >&2',
        "  exit 6",
        "fi",
        "printf 'launched rank=%s pid=%s log=%s\\n' "
        f"{shlex.quote(str(rank))} \"$child_pid\" \"$log_file\"",
    ]
    return "\n".join(lines) + "\n"


def build_ssh_command(
    host: str,
    remote_script: str,
    *,
    ssh_options: Sequence[str] = (),
) -> tuple[str, ...]:
    if (
        host.startswith("-")
        or not _SAFE_HOST.fullmatch(host)
        or any(character.isspace() for character in host)
    ):
        raise ValueError(f"Unsafe SSH destination: {host!r}")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    for option in ssh_options:
        if not option or "\n" in option or "\r" in option:
            raise ValueError(f"Invalid SSH option: {option!r}")
        command.extend(["-o", option])
    command.extend(
        [
            host,
            "bash -lc " + shlex.quote(remote_script),
        ]
    )
    return tuple(command)


def _run_ssh(launch: RankLaunch) -> dict[str, Any]:
    completed = subprocess.run(
        launch.command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "rank": launch.rank,
        "host": launch.host,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def launch_ranks(
    *,
    hosts: Sequence[str],
    plan_dir: Path,
    builder: Path = DEFAULT_BUILDER,
    work_dir: Path = REPO_ROOT,
    protenix_root_dir: Path,
    python_executable: str = sys.executable,
    workers: int = 128,
    compresslevel: int = 3,
    log_dir: Optional[Path] = None,
    resume: bool = False,
    execute: bool = False,
    fast_path_mode: str = "auto",
    durable_output: bool = False,
    tail_max_lanes_per_leaf: int = 0,
    ssh_options: Sequence[str] = (),
) -> dict[str, Any]:
    plan_dir = plan_dir.expanduser().resolve()
    builder = builder.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    protenix_root_dir = protenix_root_dir.expanduser().resolve()
    plan = _load_plan(plan_dir)
    world_size = int(plan["world_size"])
    if len(hosts) != world_size:
        raise ValueError(
            f"Plan world_size={world_size}, but hosts file has "
            f"{len(hosts)} entries"
        )
    if not builder.is_file():
        raise FileNotFoundError(f"Builder does not exist: {builder}")
    if not work_dir.is_dir():
        raise NotADirectoryError(work_dir)
    if not protenix_root_dir.is_dir():
        raise NotADirectoryError(protenix_root_dir)
    log_dir = (
        log_dir.expanduser().resolve()
        if log_dir is not None
        else plan_dir / "logs"
    )
    try:
        log_dir.relative_to(plan_dir)
    except ValueError as error:
        raise ValueError(
            "--log-dir must remain inside the shared --plan-dir so rank "
            f"leases are cross-node: {log_dir}"
        ) from error

    launches = []
    for rank, host in enumerate(hosts):
        log_path = log_dir / f"rank-{rank:03d}.log"
        pid_path = log_dir / f"rank-{rank:03d}.pid"
        remote_script = build_remote_script(
            rank=rank,
            plan_dir=plan_dir,
            builder=builder,
            work_dir=work_dir,
            protenix_root_dir=protenix_root_dir,
            python_executable=python_executable,
            workers=workers,
            compresslevel=compresslevel,
            log_path=log_path,
            pid_path=pid_path,
            resume=resume,
            fast_path_mode=fast_path_mode,
            durable_output=durable_output,
            tail_max_lanes_per_leaf=tail_max_lanes_per_leaf,
        )
        launches.append(
            RankLaunch(
                rank=rank,
                host=host,
                log_path=str(log_path),
                pid_path=str(pid_path),
                command=build_ssh_command(
                    host,
                    remote_script,
                    ssh_options=ssh_options,
                ),
            )
        )

    if not execute:
        return {
            "executed": False,
            "plan_id": plan["plan_id"],
            "world_size": world_size,
            "launches": [
                {
                    **asdict(launch),
                    "command": shlex.join(launch.command),
                }
                for launch in launches
            ],
        }

    results = []
    with ThreadPoolExecutor(max_workers=min(16, len(launches))) as executor:
        futures = {
            executor.submit(_run_ssh, launch): launch for launch in launches
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["rank"]))
    failed = [
        result for result in results if int(result["returncode"]) != 0
    ]
    response = {
        "executed": True,
        "plan_id": plan["plan_id"],
        "world_size": world_size,
        "failed_launches": len(failed),
        "results": results,
    }
    if failed:
        raise RuntimeError(json.dumps(response, ensure_ascii=False))
    return response


def aggregate_status(
    plan_dir: Path,
    *,
    hosts: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    plan = _load_plan(plan_dir)
    world_size = int(plan["world_size"])
    if hosts is not None and len(hosts) != world_size:
        raise ValueError(
            f"Plan world_size={world_size}, but hosts file has "
            f"{len(hosts)} entries"
        )
    rows = []
    states: dict[str, int] = {}
    for rank in range(world_size):
        path = plan_dir / "status" / f"rank-{rank:03d}.json"
        if not path.is_file():
            row: dict[str, Any] = {
                "rank": rank,
                "state": "not_started",
            }
        else:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("plan_id") != plan["plan_id"]:
                    row = {"rank": rank, "state": "stale"}
            except (OSError, ValueError):
                row = {"rank": rank, "state": "invalid"}
        if hosts is not None:
            row["host"] = hosts[rank]
        state = str(row.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
        rows.append(row)
    return {
        "plan_id": plan["plan_id"],
        "world_size": world_size,
        "states": states,
        "completed": sum(int(row.get("completed", 0)) for row in rows),
        "failed": sum(int(row.get("failed", 0)) for row in rows),
        "ranks": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser(
        "launch",
        help="Print or execute one SSH launch per rank",
    )
    launch.add_argument("--hosts", type=Path, required=True)
    launch.add_argument("--plan-dir", type=Path, required=True)
    launch.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    launch.add_argument("--work-dir", type=Path, default=REPO_ROOT)
    launch.add_argument("--protenix-root-dir", type=Path, required=True)
    launch.add_argument("--python", dest="python_executable", default=sys.executable)
    launch.add_argument("--workers", type=int, default=128)
    launch.add_argument("--compresslevel", type=int, default=3)
    launch.add_argument("--log-dir", type=Path, default=None)
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--durable-output", action="store_true")
    launch.add_argument(
        "--tail-max-lanes-per-leaf",
        type=int,
        default=0,
        help=(
            "0 opens extra same-leaf lanes only in the tail; "
            "1 forbids concurrent writes to one leaf"
        ),
    )
    launch.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Additional ssh -o option; repeat as needed",
    )
    fast_group = launch.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--disable-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="off",
    )
    fast_group.add_argument(
        "--require-fast-path",
        dest="fast_path_mode",
        action="store_const",
        const="require",
    )
    launch.set_defaults(fast_path_mode="auto")

    status = subparsers.add_parser(
        "status",
        help="Read shared rank heartbeat files",
    )
    status.add_argument("--plan-dir", type=Path, required=True)
    status.add_argument("--hosts", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "launch":
            result = launch_ranks(
                hosts=read_hosts(args.hosts),
                plan_dir=args.plan_dir,
                builder=args.builder,
                work_dir=args.work_dir,
                protenix_root_dir=args.protenix_root_dir,
                python_executable=args.python_executable,
                workers=args.workers,
                compresslevel=args.compresslevel,
                log_dir=args.log_dir,
                resume=args.resume,
                execute=args.execute,
                fast_path_mode=args.fast_path_mode,
                durable_output=args.durable_output,
                tail_max_lanes_per_leaf=args.tail_max_lanes_per_leaf,
                ssh_options=args.ssh_option,
            )
        elif args.command == "status":
            hosts = (
                read_hosts(args.hosts)
                if args.hosts is not None
                else None
            )
            result = aggregate_status(args.plan_dir, hosts=hosts)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
