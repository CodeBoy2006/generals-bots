#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_0_path="${MODEL_0_PATH:-${MODEL_PATH:-generals-ppo-8x8-expander-gpu-v5.eqx}}"
model_1_path="${MODEL_1_PATH:-${OPPONENT_MODEL_PATH:-$model_0_path}}"
model_0_policy_input="${MODEL_0_POLICY_INPUT:-${POLICY_INPUT:-auto}}"
model_1_policy_input="${MODEL_1_POLICY_INPUT:-${OPPONENT_POLICY_INPUT:-auto}}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to start the Generals PPO watch match." >&2
  echo "Install uv or make sure it is available in PATH, then run this script again." >&2
  exit 1
fi

if [[ ! -f "$model_0_path" ]]; then
  echo "Missing PPO checkpoint for player 0: $model_0_path" >&2
  echo "Place the checkpoint in the repository root or set MODEL_0_PATH." >&2
  exit 1
fi

if [[ ! -f "$model_1_path" ]]; then
  echo "Missing PPO checkpoint for player 1: $model_1_path" >&2
  echo "Place the checkpoint in the repository root or set MODEL_1_PATH." >&2
  exit 1
fi

exec uv run --python 3.12 python examples/play_against_model.py \
  --machine-vs-machine \
  --model-0-path "$model_0_path" \
  --model-1-path "$model_1_path" \
  --model-0-policy-input "$model_0_policy_input" \
  --model-1-policy-input "$model_1_policy_input" \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --opponent-policy-mode sample \
  --fps 30 \
  --auto-tick \
  --tick-rate 4 \
  --preview-top-k 3 \
  "$@"
