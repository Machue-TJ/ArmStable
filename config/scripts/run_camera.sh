#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec conda run --no-capture-output -n piper python "$project_dir/camera_demo.py" "$@"
