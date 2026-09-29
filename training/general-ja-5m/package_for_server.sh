#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
destination=${1:-"$root/training/general-ja-5m/work"}
stamp=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$destination"
archive="$destination/exllm-general-ja-5m-$stamp.tar.gz"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.venv-training' \
  --exclude='.venv-server' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='training/general-ja-5m/work' \
  -C "$root" -czf "$archive" .
sha256sum "$archive" > "$archive.sha256"
printf '%s\n' "$archive"
cat "$archive.sha256"
