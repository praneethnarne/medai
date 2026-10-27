#!/usr/bin/env bash
# Train overnight without the machine falling asleep halfway through.
#
#   ./train_overnight.sh                     # baseline, default settings
#   ./train_overnight.sh --arch resnet18     # any train.py flag passes through
#
# Before you walk away:
#   * leave the LID OPEN  -- closing it sleeps a MacBook whatever caffeinate says
#   * leave it PLUGGED IN -- caffeinate keeps it awake, which uses power
#
# Stop it any time with Ctrl-C or by closing the terminal; every epoch is
# checkpointed, so re-running with --resume continues where it stopped.

set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME=${RUN_NAME:-ham_baseline_effb0}
LOG="train_${RUN_NAME}.log"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run ./setup.sh first." >&2
  exit 1
fi

echo "Run:   $RUN_NAME"
echo "Log:   $LOG"
echo "Sleep: prevented while training (caffeinate)"
echo "Lid:   keep it OPEN, or the Mac sleeps anyway"
echo

# caffeinate runs training as its child and holds the no-sleep assertion for
# exactly that long -- the Mac sleeps normally again the moment training ends.
#   -i  no idle sleep
#   -s  no system sleep while on mains power
#   -m  no disk sleep
caffeinate -ism ./.venv/bin/python -u backend/src/train/train.py \
  --collection ham10000 \
  --target lesion_type \
  --arch efficientnet_b0 \
  --epochs-head 3 \
  --epochs-full 15 \
  --workers 4 \
  --run-name "$RUN_NAME" \
  "$@" 2>&1 | tee "$LOG"

echo
echo "Training finished. Next, the headline result:"
echo
echo "  ./.venv/bin/python backend/src/eval/evaluate.py \\"
echo "      --run $RUN_NAME --collection ham10000 --split test"
echo
echo "  ./.venv/bin/python backend/src/eval/evaluate.py \\"
echo "      --run $RUN_NAME --collection pad_ufes_20 --split test --collapse-to-triage"
