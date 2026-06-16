#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_path="${MODEL_PATH:-generals-ppo-8x8-expander-gpu-v5.eqx}"
policy_input="${POLICY_INPUT:-auto}"
search_policy="${SEARCH_POLICY:-0}"
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
  echo "uv is required to start the Generals PPO match." >&2
  echo "Install uv or make sure it is available in PATH, then run this script again." >&2
  exit 1
fi

if [[ ! -f "$model_path" ]]; then
  echo "Missing PPO checkpoint: $model_path" >&2
  echo "Place generals-ppo-8x8-expander-gpu-v5.eqx in the repository root or set MODEL_PATH." >&2
  exit 1
fi

search_args=()
if is_truthy "$search_policy"; then
  search_args+=(--search-policy)
fi

exec uv run --python 3.12 python examples/play_against_model.py "$model_path" \
  --grid-size 8 \
  --map-generator generated \
  --policy-mode sample \
  --policy-input "$policy_input" \
  --search-rollout-policy-mode "$search_rollout_policy_mode" \
  --search-top-k "$search_top_k" \
  --search-rollout-steps "$search_rollout_steps" \
  --search-rollouts-per-action "$search_rollouts_per_action" \
  --search-army-weight "$search_army_weight" \
  --search-land-weight "$search_land_weight" \
  --search-prior-weight "$search_prior_weight" \
  --human-player 0 \
  --fps 30 \
  --auto-tick \
  --tick-rate 2 \
  --preview-top-k 3 \
  "${search_args[@]}" \
  "$@"
