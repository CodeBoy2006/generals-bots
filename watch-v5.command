#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_0_path="${MODEL_0_PATH:-${MODEL_PATH:-generals-ppo-8x8-expander-gpu-v5.eqx}}"
model_1_path="${MODEL_1_PATH:-${OPPONENT_MODEL_PATH:-$model_0_path}}"
model_0_policy_input="${MODEL_0_POLICY_INPUT:-${POLICY_INPUT:-auto}}"
model_1_policy_input="${MODEL_1_POLICY_INPUT:-${OPPONENT_POLICY_INPUT:-auto}}"
model_0_search_policy="${MODEL_0_SEARCH_POLICY:-${SEARCH_POLICY:-0}}"
model_1_search_policy="${MODEL_1_SEARCH_POLICY:-${OPPONENT_SEARCH_POLICY:-0}}"
search_rollout_policy_mode="${SEARCH_ROLLOUT_POLICY_MODE:-sample}"
search_top_k="${SEARCH_TOP_K:-4}"
search_rollout_steps="${SEARCH_ROLLOUT_STEPS:-16}"
search_rollouts_per_action="${SEARCH_ROLLOUTS_PER_ACTION:-4}"
search_army_weight="${SEARCH_ARMY_WEIGHT:-12.0}"
search_land_weight="${SEARCH_LAND_WEIGHT:-8.0}"
search_prior_weight="${SEARCH_PRIOR_WEIGHT:-0.01}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

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

search_args=()
if is_truthy "$model_0_search_policy"; then
  search_args+=(--search-policy)
fi
if is_truthy "$model_1_search_policy"; then
  search_args+=(--opponent-search-policy)
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
  --search-rollout-policy-mode "$search_rollout_policy_mode" \
  --search-top-k "$search_top_k" \
  --search-rollout-steps "$search_rollout_steps" \
  --search-rollouts-per-action "$search_rollouts_per_action" \
  --search-army-weight "$search_army_weight" \
  --search-land-weight "$search_land_weight" \
  --search-prior-weight "$search_prior_weight" \
  --fps 30 \
  --auto-tick \
  --tick-rate 4 \
  --preview-top-k 3 \
  "${search_args[@]}" \
  "$@"
