#!/usr/bin/env bash
# Interactive eval prompts (policy / checkpoint / duration or protocol).
# Sourced by scripts/17_eval.sh and optionally by 11/15/16 eval wrappers.
#
# Env skips (non-interactive / CI):
#   POLICY, POLICY_PATH / CHECKPOINT, DURATION, PROTOCOL, EVAL_MODE
#   INTERACTIVE=0  → never prompt

_eval_list_train_runs() {
  local root train
  root="$(project_root)"
  train="$root/outputs/train"
  [[ -d "$train" ]] || return 0
  find "$train" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
}

_eval_list_datasets() {
  local root ds
  root="$(project_root)"
  ds="$root/data/datasets"
  [[ -d "$ds" ]] || return 0
  find "$ds" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | while IFS= read -r d; do
    [[ -f "$d/meta.json" ]] || continue
    basename "$d"
  done
}

_eval_checkpoint_kind() {
  # Prints: best_val | last | legacy | none  for a train run dir
  local run="$1"
  if [[ -f "$run/best_val/model.pt" || -f "$run/best_val/model.msgpack" ]]; then
    echo best_val
  elif [[ -f "$run/last/model.pt" || -f "$run/last/model.msgpack" ]]; then
    echo last
  elif [[ -f "$run/model.pt" || -f "$run/model.msgpack" ]]; then
    echo legacy
  else
    echo none
  fi
}

_eval_prompt_choice() {
  # usage: _eval_prompt_choice "Prompt" default_value option1 option2 ...
  local prompt="$1"
  local default="$2"
  shift 2
  local -a opts=("$@")
  local i reply
  echo "" >&2
  echo "$prompt" >&2
  for i in "${!opts[@]}"; do
    local mark=""
    [[ "${opts[$i]}" == "$default" ]] && mark=" (default)"
    printf "  %2d) %s%s\n" "$((i + 1))" "${opts[$i]}" "$mark" >&2
  done
  while true; do
    read -r -p "Select [1-${#opts[@]}] (Enter=${default}): " reply || true
    if [[ -z "${reply:-}" ]]; then
      printf '%s' "$default"
      return 0
    fi
    if [[ "$reply" =~ ^[0-9]+$ ]] && (( reply >= 1 && reply <= ${#opts[@]} )); then
      printf '%s' "${opts[$((reply - 1))]}"
      return 0
    fi
    echo "Invalid choice." >&2
  done
}

_eval_prompt_text() {
  local prompt="$1"
  local default="$2"
  local reply
  read -r -p "$prompt [$default]: " reply || true
  if [[ -z "${reply:-}" ]]; then
    printf '%s' "$default"
  else
    printf '%s' "$reply"
  fi
}

# Sets: POLICY, POLICY_BASE (for hybrid), POLICY_PATH, EVAL_MODE, DURATION, PROTOCOL, DATASET_NAME
prompt_eval_options() {
  local root
  root="$(project_root)"
  local interactive="${INTERACTIVE:-1}"

  if [[ "$interactive" == "0" ]] || [[ ! -t 0 ]]; then
    POLICY="${POLICY:-act}"
    DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
    if [[ ! -f "$root/data/datasets/$DATASET_NAME/meta.json" ]]; then
      echo "Dataset not found: data/datasets/$DATASET_NAME/meta.json" >&2
      return 1
    fi
    POLICY_PATH="${POLICY_PATH:-${CHECKPOINT:-$root/outputs/train/act_${DATASET_NAME}}}"
    if [[ -n "${PROTOCOL+x}" ]]; then
      : # keep PROTOCOL as set (may be empty for timed)
    else
      PROTOCOL="${PROTOCOL:-$root/config/eval/bag_clip_in_bowl.yaml}"
    fi
    DURATION="${DURATION:-60}"
    EVAL_MODE="${EVAL_MODE:-}"
    if [[ -z "$EVAL_MODE" ]]; then
      if [[ -n "${PROTOCOL:-}" ]]; then
        EVAL_MODE=protocol
      else
        EVAL_MODE=timed
      fi
    fi
    # Default act25 + low-pass (less jerk than reactive=1); override with CONTROL_MODE=
    CONTROL_MODE="${CONTROL_MODE:-act25}"
    # Alias: smooth → act25
    [[ "$CONTROL_MODE" == "smooth" ]] && CONTROL_MODE=act25
    case "$CONTROL_MODE" in
      reactive)
        N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
        RECIPE="${RECIPE:-$root/config/recipes/reactive_act1.yaml}"
        ;;
      act5)
        N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
        RECIPE="${RECIPE:-}"
        ;;
      act25)
        N_ACTION_STEPS="${N_ACTION_STEPS:-25}"
        RECIPE="${RECIPE:-$root/config/recipes/act25.yaml}"
        ;;
      act200|ensemble)
        CONTROL_MODE=act200
        N_ACTION_STEPS="${N_ACTION_STEPS:-200}"
        RECIPE="${RECIPE:-$root/config/recipes/act200.yaml}"
        ;;
    esac
    return 0
  fi

  echo ""
  echo "============================================================"
  echo " EVAL LAUNCHER — pick policy, checkpoint, time limit"
  echo "============================================================"

  DATASET_NAME="${DATASET_NAME:-so101_bag_clip_bowl}"
  local -a datasets=()
  local ds_name
  while IFS= read -r ds_name; do
    [[ -z "$ds_name" ]] && continue
    # Skip auto-saved eval/dagger side sets in the main picker (still selectable if already default)
    datasets+=("$ds_name")
  done < <(_eval_list_datasets)
  if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "No datasets under data/datasets/*/meta.json" >&2
    return 1
  fi
  # Prefer bag_clip default when present
  local ds_default="$DATASET_NAME"
  local found=0
  for ds_name in "${datasets[@]}"; do
    [[ "$ds_name" == "$ds_default" ]] && found=1 && break
  done
  if [[ "$found" -eq 0 ]]; then
    ds_default="${datasets[0]}"
  fi
  DATASET_NAME="$(_eval_prompt_choice "Dataset" "$ds_default" "${datasets[@]}")"

  local -a policies=(act act-jax smolvla hybrid gemfuse)
  POLICY="${POLICY:-act}"
  POLICY="$(_eval_prompt_choice "Policy" "$POLICY" "${policies[@]}")"

  POLICY_BASE="${POLICY_BASE:-smolvla}"
  if [[ "$POLICY" == "hybrid" ]]; then
    local -a bases=(smolvla act act-jax)
    POLICY_BASE="$(_eval_prompt_choice "Hybrid policy-base" "$POLICY_BASE" "${bases[@]}")"
  fi

  # Checkpoint discovery
  local -a runs=()
  local -a labels=()
  local run kind label
  while IFS= read -r run; do
    [[ -z "$run" ]] && continue
    kind="$(_eval_checkpoint_kind "$run")"
    [[ "$kind" == "none" ]] && continue
    label="$(basename "$run") [$kind]"
    runs+=("$run")
    labels+=("$label")
  done < <(_eval_list_train_runs)

  if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No checkpoints under outputs/train/ — train first (bash scripts/10_train_act.sh)." >&2
    return 1
  fi

  local default_run="$root/outputs/train/act_${DATASET_NAME}"
  local default_label=""
  local i
  for i in "${!runs[@]}"; do
    if [[ "${runs[$i]}" == "$default_run" ]] || [[ "${runs[$i]}" == "${POLICY_PATH:-}" ]]; then
      default_label="${labels[$i]}"
      break
    fi
  done
  [[ -z "$default_label" ]] && default_label="${labels[0]}"

  local chosen_label
  chosen_label="$(_eval_prompt_choice "Checkpoint (train run)" "$default_label" "${labels[@]}")"
  for i in "${!labels[@]}"; do
    if [[ "${labels[$i]}" == "$chosen_label" ]]; then
      POLICY_PATH="${runs[$i]}"
      break
    fi
  done

  # Prefer best_val when present
  kind="$(_eval_checkpoint_kind "$POLICY_PATH")"
  if [[ "$kind" == "best_val" || "$kind" == "last" ]]; then
    local -a weight_opts=()
    [[ -d "$POLICY_PATH/best_val" ]] && weight_opts+=(best_val)
    [[ -d "$POLICY_PATH/last" ]] && weight_opts+=(last)
    if [[ ${#weight_opts[@]} -gt 1 ]]; then
      local sub default_w="${weight_opts[0]}"
      sub="$(_eval_prompt_choice "Which weights under $(basename "$POLICY_PATH")?" "$default_w" "${weight_opts[@]}")"
      if [[ "$sub" == "last" && -d "$POLICY_PATH/last" ]]; then
        POLICY_PATH="$POLICY_PATH/last"
      fi
      # best_val: keep run root so so101 eval resolves best_val/
    elif [[ ${#weight_opts[@]} -eq 1 && "${weight_opts[0]}" == "last" ]]; then
      POLICY_PATH="$POLICY_PATH/last"
    fi
  fi

  EVAL_MODE="${EVAL_MODE:-protocol}"
  EVAL_MODE="$(_eval_prompt_choice "Eval mode" "$EVAL_MODE" protocol timed)"

  if [[ "$EVAL_MODE" == "timed" ]]; then
    PROTOCOL=""
    DURATION="${DURATION:-60}"
    DURATION="$(_eval_prompt_text "Time limit (seconds)" "$DURATION")"
  else
    local -a protos=()
    local p
    for p in "$root"/config/eval/*.yaml; do
      [[ -f "$p" ]] && protos+=("$p")
    done
    if [[ ${#protos[@]} -eq 0 ]]; then
      PROTOCOL="$root/config/eval/bag_clip_in_bowl.yaml"
    else
      local def_proto="$root/config/eval/bag_clip_in_bowl.yaml"
      [[ -f "$def_proto" ]] || def_proto="${protos[0]}"
      PROTOCOL="$(_eval_prompt_choice "Protocol YAML" "$def_proto" "${protos[@]}")"
    fi
    DURATION="${DURATION:-60}"
  fi

  # Default act25 (queue 25 + low-pass). Use reactive only when debugging closed-loop.
  local control_default="${CONTROL_MODE:-act25}"
  [[ "$control_default" == "smooth" ]] && control_default=act25
  CONTROL_MODE="$(_eval_prompt_choice \
    "Control reactivity (how fast it replans)" \
    "$control_default" \
    reactive act5 act25 act200)"
  RECIPE="${RECIPE:-}"
  N_ACTION_STEPS="${N_ACTION_STEPS:-}"
  case "$CONTROL_MODE" in
    reactive)
      N_ACTION_STEPS=1
      RECIPE="${RECIPE:-$root/config/recipes/reactive_act1.yaml}"
      ;;
    act5)
      N_ACTION_STEPS=5
      RECIPE=""
      ;;
    act25|smooth)
      CONTROL_MODE=act25
      N_ACTION_STEPS=25
      RECIPE="${RECIPE:-$root/config/recipes/act25.yaml}"
      ;;
    act200|ensemble)
      CONTROL_MODE=act200
      N_ACTION_STEPS=200
      RECIPE="${RECIPE:-$root/config/recipes/act200.yaml}"
      ;;
  esac

  echo ""
  echo "Selected:"
  echo "  dataset:    $DATASET_NAME"
  if [[ "$POLICY" == "hybrid" && -n "${POLICY_BASE:-}" ]]; then
    echo "  policy:     $POLICY (base=$POLICY_BASE)"
  else
    echo "  policy:     $POLICY"
  fi
  echo "  checkpoint: $POLICY_PATH"
  if [[ "$EVAL_MODE" == "timed" ]]; then
    echo "  mode:       timed  duration=${DURATION}s"
  else
    echo "  mode:       protocol  $PROTOCOL"
  fi
  echo "  control:    $CONTROL_MODE  n_action_steps=${N_ACTION_STEPS:-checkpoint}  recipe=${RECIPE:-none}"
  echo ""
}

# Populate EXTRA_EVAL_FLAGS from N_ACTION_STEPS / RECIPE (append into EVAL_ARGS/ARGS).
_eval_append_control_flags() {
  EXTRA_EVAL_FLAGS=()
  if [[ -n "${N_ACTION_STEPS:-}" ]]; then
    EXTRA_EVAL_FLAGS+=(--n-action-steps="$N_ACTION_STEPS")
  fi
  if [[ -n "${RECIPE:-}" ]]; then
    EXTRA_EVAL_FLAGS+=(--recipe="$RECIPE")
  fi
}
