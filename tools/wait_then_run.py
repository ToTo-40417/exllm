#!/usr/bin/env python3
"""Wait for a detached server job, then run the next pipeline command."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing command after --")
    while True:
        if args.status.exists():
            status = json.loads(args.status.read_text(encoding="utf-8"))
            state = status.get("state")
            if state == "complete":
                break
            if state == "failed":
                raise SystemExit(f"dependency failed: {args.status}")
        time.sleep(args.poll_seconds)
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
