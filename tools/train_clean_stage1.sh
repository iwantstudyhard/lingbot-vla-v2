#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export WORKSPACE="$(cd "${WORKSPACE:-$PROJECT_ROOT}" && pwd)"
cd "$WORKSPACE"
exec python -m tools.launch_clean_stage1 "$@"
