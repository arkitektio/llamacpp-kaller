#!/usr/bin/env bash
# Run the app with one of the setups in setups.yaml.
#
#   ./run.sh                 # the file's default setup
#   ./run.sh qwen            # a named setup
#   ./run.sh qwen login      # any further words are an arkitekt subcommand
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  export LLAMA_PROFILE="$1"
  shift
fi

if [ $# -eq 0 ]; then
  exec uv run arkitekt run
fi
exec uv run arkitekt "$@"
