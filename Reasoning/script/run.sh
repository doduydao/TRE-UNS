#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: ./script/run.sh <MATRES|I2B2|TDDMAN|TBD> [train|predict|evaluate] [baseline|baseline_psl|baseline_reasoning|TRER]"
  exit 1
fi

DATASET="${1^^}"
if [[ "$DATASET" != "MATRES" && "$DATASET" != "I2B2" && "$DATASET" != "TDDMAN" && "$DATASET" != "TBD" ]]; then
  echo "Error: dataset must be MATRES, I2B2, TDDMAN, or TBD"
  exit 1
fi

RUN_MODE="${2:-evaluate}"
RUN_MODE="${RUN_MODE,,}"
if [[ "$RUN_MODE" != "train" && "$RUN_MODE" != "predict" && "$RUN_MODE" != "evaluate" ]]; then
  echo "Error: mode must be train, predict, or evaluate"
  exit 1
fi

MODE_OVERRIDE="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENTRY="$ROOT/tre_reasoner/start_evaluation.py"
TRAIN_ENTRY="$ROOT/tre_reasoner/start_training.py"
DATASET_CONF="$ROOT/script/config/${DATASET}.conf"

if [[ ! -f "$DATASET_CONF" ]]; then
  echo "Error: config not found: $DATASET_CONF"
  exit 1
fi

read_conf_value() {
  local section="$1"
  local key="$2"
  python - "$DATASET_CONF" "$section" "$key" <<'PY'
import configparser
import sys

conf_path, section, key = sys.argv[1], sys.argv[2], sys.argv[3]
parser = configparser.ConfigParser()
ok = parser.read(conf_path)
if not ok or section not in parser or key not in parser[section]:
    raise SystemExit(1)
print(parser[section][key])
PY
}

read_conf_value_or_default() {
  local section="$1"
  local key="$2"
  local default_value="$3"
  if value="$(read_conf_value "$section" "$key" 2>/dev/null)"; then
    echo "$value"
  else
    echo "$default_value"
  fi
}

MODEL_DATA_MODE="$(read_conf_value model data_mode)"
MODEL_BATCH_SIZE="$(read_conf_value model batch_size)"
MODEL_EPOCHS="$(read_conf_value model epochs)"
MODEL_LR="$(read_conf_value model lr)"

EVAL_MODE="$(read_conf_value_or_default run eval_mode baseline_reasoning)"
SPLIT="$(read_conf_value_or_default run split test)"
DATA_MODE="$(read_conf_value_or_default run data_mode "$MODEL_DATA_MODE")"
BATCH_SIZE="$(read_conf_value_or_default run batch_size "$MODEL_BATCH_SIZE")"
TRAIN_MODE="$(read_conf_value_or_default run train_mode baseline_reasoning)"
TRAIN_EPOCHS="$(read_conf_value_or_default run train_epochs "$MODEL_EPOCHS")"
TRAIN_LR="$(read_conf_value_or_default run train_lr "$MODEL_LR")"
TRAIN_BATCH_SIZE="$(read_conf_value_or_default run train_batch_size "$MODEL_BATCH_SIZE")"
MAX_PAIRS_PER_DOC_BATCH="$(read_conf_value_or_default run max_pairs_per_doc_batch 0)"

OUT_DIR="$ROOT/$DATASET"
CKPT_OUT_DIR="$OUT_DIR/checkpoint"
LOG_FILE="$OUT_DIR/log_${RUN_MODE}.txt"
PRED_FILE="$OUT_DIR/prediction.csv"
CHECKPOINT_FILE="$(read_conf_value run checkpoint_file)"
if [[ -z "$CHECKPOINT_FILE" ]]; then
  echo "Error: checkpoint_file is missing in [run] of $DATASET_CONF"
  exit 1
fi
CKPT="$CKPT_OUT_DIR/$CHECKPOINT_FILE"

if [[ -n "$MODE_OVERRIDE" && "$RUN_MODE" == "train" ]]; then
  MODE_OVERRIDE="${MODE_OVERRIDE,,}"
  if [[ "$MODE_OVERRIDE" != "baseline" && "$MODE_OVERRIDE" != "baseline_psl" && "$MODE_OVERRIDE" != "baseline_reasoning" && "$MODE_OVERRIDE" != "trer" ]]; then
    echo "Error: train mode override must be one of baseline, baseline_psl, baseline_reasoning, TRER"
    exit 1
  fi

  if [[ "$MODE_OVERRIDE" == "trer" ]]; then
    MODE_OVERRIDE="TRER"
  fi

  TRAIN_MODE="$MODE_OVERRIDE"
  CKPT="$CKPT_OUT_DIR/${DATASET}_${TRAIN_MODE}_model.pt"
  LOG_FILE="$OUT_DIR/log_${RUN_MODE}_${TRAIN_MODE}.txt"
fi

mkdir -p "$OUT_DIR"
mkdir -p "$CKPT_OUT_DIR"

echo "Running dataset=$DATASET mode=$RUN_MODE"
echo "Config: $DATASET_CONF"
if [[ "$RUN_MODE" == "train" ]]; then
  echo "Train mode: $TRAIN_MODE"
fi
echo "Outputs: $OUT_DIR"

export TRE_REASONER_ROOT="$ROOT"

case "$RUN_MODE" in
train)
  TRAIN_SAVE_PATH="$CKPT"
  python "$TRAIN_ENTRY" \
    --dataset "$DATASET" \
    --conf "$DATASET_CONF" \
    --mode "$TRAIN_MODE" \
    --epochs "$TRAIN_EPOCHS" \
    --lr "$TRAIN_LR" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --data_mode "$DATA_MODE" \
    --max_pairs_per_doc_batch "$MAX_PAIRS_PER_DOC_BATCH" \
    --save_path "$TRAIN_SAVE_PATH" \
    > "$LOG_FILE" 2>&1
  ;;
predict)
  echo "Checkpoint: $CKPT"
  if [[ ! -f "$CKPT" ]]; then
    echo "Error: trained checkpoint not found: $CKPT"
    echo "Please run: ./script/run.sh $DATASET train"
    exit 1
  fi
  python "$ENTRY" \
    --dataset "$DATASET" \
    --conf "$DATASET_CONF" \
    --mode "$EVAL_MODE" \
    --split "$SPLIT" \
    --data_mode "$DATA_MODE" \
    --batch_size "$BATCH_SIZE" \
    --model_path "$CKPT" \
    --output_file "$PRED_FILE" \
    > "$LOG_FILE" 2>&1
  ;;
evaluate)
  echo "Checkpoint: $CKPT"
  if [[ ! -f "$CKPT" ]]; then
    echo "Error: trained checkpoint not found: $CKPT"
    echo "Please run: ./script/run.sh $DATASET train"
    exit 1
  fi
  python "$ENTRY" \
    --dataset "$DATASET" \
    --conf "$DATASET_CONF" \
    --mode "$EVAL_MODE" \
    --split "$SPLIT" \
    --data_mode "$DATA_MODE" \
    --batch_size "$BATCH_SIZE" \
    --model_path "$CKPT" \
    --eval_consistency \
    --output_file "$PRED_FILE" \
    > "$LOG_FILE" 2>&1
  ;;
esac

echo "Done."
echo "- checkpoint: $CKPT"
echo "- log_${RUN_MODE}.txt: $LOG_FILE"
if [[ -f "$PRED_FILE" ]]; then
  echo "- prediction.csv: $PRED_FILE"
fi
