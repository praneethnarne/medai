#!/usr/bin/env bash
# One-command setup: environment, data, splits, tests.
#
#   ./setup.sh
#
# Safe to re-run. Downloads are resumable and already-present images are
# skipped, so an interrupted run just needs the script again.
#
# DL_ARGS=--manifest-only ./setup.sh  exercises the whole chain without
# fetching the ~560 MB of images -- used to verify this script actually works.

set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv

echo "==> 1/6  Python environment"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
  echo "    created $VENV"
else
  echo "    $VENV already exists"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r backend/requirements.txt
echo "    dependencies installed"

echo "==> 2/6  Hardware check"
"$VENV/bin/python" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"    CUDA GPU: {torch.cuda.get_device_name(0)}  <- fastest")
elif torch.backends.mps.is_available():
    print("    Apple Metal (MPS)  <- good")
else:
    print("    CPU only  <- training will be SLOW (hours per run).")
    print("    Consider Google Colab's free GPU instead.")
PYEOF

echo "==> 3/6  Downloading datasets (~560 MB, resumable)"
"$VENV/bin/python" backend/src/data/download.py pad_ufes_20 --workers 16 ${DL_ARGS:-}
"$VENV/bin/python" backend/src/data/download.py ham10000 --workers 16 ${DL_ARGS:-}

echo "==> 4/6  Verifying every image is complete"
if [ -z "${DL_ARGS:-}" ]; then "$VENV/bin/python" backend/src/data/verify.py ham10000 pad_ufes_20; else echo "    (skipped: manifest-only run)"; fi

echo "==> 5/6  Auditing and splitting"
"$VENV/bin/python" backend/src/data/audit_collection.py ham10000
"$VENV/bin/python" backend/src/data/audit_collection.py pad_ufes_20
"$VENV/bin/python" backend/src/data/split.py ham10000 pad_ufes_20

echo "==> 6/6  Tests"
"$VENV/bin/python" -m pytest backend/tests -q

cat <<'DONE'

Setup complete.

Train the baseline (resumable -- stop any time with Ctrl-C, continue with --resume):

  ./.venv/bin/python -u backend/src/train/train.py \
      --collection ham10000 --target lesion_type --arch efficientnet_b0 \
      --epochs-head 3 --epochs-full 15 --run-name ham_baseline_effb0

Then the cross-domain evaluation, which is the project's headline result:

  ./.venv/bin/python backend/src/eval/evaluate.py \
      --run ham_baseline_effb0 --collection ham10000 --split test

  ./.venv/bin/python backend/src/eval/evaluate.py \
      --run ham_baseline_effb0 --collection pad_ufes_20 --split test \
      --collapse-to-triage

DONE
