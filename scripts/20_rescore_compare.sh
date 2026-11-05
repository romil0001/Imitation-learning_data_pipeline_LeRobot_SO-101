#!/usr/bin/env bash
# After DAgger fine-tune: print how to re-eval + compare pc_success vs a baseline summary.
# If BASELINE_SUMMARY and CANDIDATE_SUMMARY are set, runs compare-eval immediately.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/_common.sh
source "$ROOT/scripts/_common.sh"

DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
JOB_NAME="${JOB_NAME:-act_${DATASET_NAME}_dagger}"
POLICY_PATH="${POLICY_PATH:-$ROOT/outputs/train/$JOB_NAME}"
PROTOCOL="${PROTOCOL:-$ROOT/config/eval/bag_clip_in_bowl.yaml}"

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "No dagger checkpoint at $POLICY_PATH" >&2
  echo "1) bash scripts/18_dagger.sh   # ~10 grasp corrections" >&2
  echo "2) bash scripts/19_train_dagger.sh" >&2
  exit 1
fi

echo "============================================================"
echo " Re-score dagger policy (act25 + success saving)"
echo " Policy:   $POLICY_PATH"
echo " Protocol: $PROTOCOL"
echo "============================================================"
echo "Run:"
echo "  POLICY_PATH=$POLICY_PATH CONTROL_MODE=act25 PROTOCOL=$PROTOCOL \\"
echo "    bash scripts/11_eval_act.sh"
echo
echo "Then compare (pick summary.json paths from outputs/eval/...):"
echo "  python -m so101 compare-eval --baseline <baseline>/summary.json --candidate <new>/summary.json"
echo

if [[ -n "${BASELINE_SUMMARY:-}" && -n "${CANDIDATE_SUMMARY:-}" ]]; then
  require_conda_lerobot
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  run_logged compare_eval "$(conda_lerobot_python)" -m so101 compare-eval \
    --baseline="$BASELINE_SUMMARY" \
    --candidate="$CANDIDATE_SUMMARY"
fi
