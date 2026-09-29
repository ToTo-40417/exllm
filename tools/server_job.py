#!/usr/bin/env python3
"""Start and inspect detached, auditable EXLLM server jobs."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start(args):
    job = args.root / args.name
    status_path = job / "status.json"
    if status_path.exists():
        existing = read_json(status_path)
        if existing.get("state") in ("starting", "running") and alive(existing.get("pid", -1)):
            raise SystemExit(f"job is already active: {args.name}")
        raise SystemExit(f"job directory already exists; choose a new immutable name: {job}")
    if not args.command:
        raise SystemExit("missing command after --")
    job.mkdir(parents=True)
    command = args.command[1:] if args.command[0] == "--" else args.command
    write_json(job / "command.json", {"cwd": str(Path.cwd()), "command": command, "created_at": now()})
    write_json(status_path, {"name": args.name, "state": "starting", "pid": -1, "created_at": now()})
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "worker", "--job", str(job), "--", *command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    record = read_json(status_path)
    if record.get("state") == "starting":
        record["pid"] = process.pid
        write_json(status_path, record)
    print(json.dumps({"name": args.name, "pid": process.pid, "job": str(job)}, ensure_ascii=False))


def worker(args):
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    job = args.job
    status_path = job / "status.json"
    record = read_json(status_path)
    record.update({"state": "running", "pid": os.getpid(), "started_at": now()})
    write_json(status_path, record)
    with (job / "output.log").open("ab", buffering=0) as log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    record.update({"state": "complete" if result.returncode == 0 else "failed", "returncode": result.returncode, "finished_at": now()})
    write_json(status_path, record)
    raise SystemExit(result.returncode)


def status(args):
    job = args.root / args.name
    record = read_json(job / "status.json")
    record["process_alive"] = alive(record.get("pid", -1))
    record["log"] = str(job / "output.log")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--root", type=Path, required=True)
    start_parser.add_argument("--name", required=True)
    start_parser.add_argument("command", nargs=argparse.REMAINDER)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--job", type=Path, required=True)
    worker_parser.add_argument("command", nargs=argparse.REMAINDER)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    status_parser.add_argument("--name", required=True)
    args = parser.parse_args()
    {"start": start, "worker": worker, "status": status}[args.action](args)


if __name__ == "__main__":
    main()
