#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_path="${MODEL_PATH:-generals-ppo-8x8-expander-gpu-v5.eqx}"
policy_input="${POLICY_INPUT:-auto}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to start the Generals PPO match." >&2
  echo "Install uv or make sure it is available in PATH, then run this script again." >&2
  exit 1
fi

if [[ ! -f "$model_path" ]]; then
  echo "Missing PPO checkpoint: $model_path" >&2
  echo "Place generals-ppo-8x8-expander-gpu-v5.eqx in the repository root or set MODEL_PATH." >&2
  exit 1
fi

exec uv run --python 3.12 python examples/play_against_model.py "$model_path" \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --policy-input "$policy_input" \
  --human-player 0 \
  --fps 30 \
  --auto-tick \
  --tick-rate 2 \
  --preview-top-k 3 \
  "$@"
