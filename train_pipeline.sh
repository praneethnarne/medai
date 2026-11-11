#!/usr/bin/env bash
# Post-presentation training pipeline (START_HERE "After the presentation", steps 1-2),
# run unattended on the owner's Mac once they have said they are away from it.
#
#   ./train_pipeline.sh            # both stages
#   STAGE=joint ./train_pipeline.sh
#   STAGE=groups ./train_pipeline.sh
#
# Every command is resumable (--resume); re-running after an interruption continues.
# Nothing here types a number: every metric is written to results/ by the scripts.

set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
STAGE=${STAGE:-all}
WORKERS=${WORKERS:-4}

if [[ $STAGE == all || $STAGE == joint ]]; then
  echo "==> [joint] verify images"
  $PY backend/src/data/verify.py ham10000 pad_ufes_20

  echo "==> [joint] train on HAM10000 + PAD-UFES-20 (init from the HAM10000 model)"
  $PY -u backend/src/train/train.py --collection ham10000+pad_ufes_20 --target lesion_type \
      --arch efficientnet_b0 --epochs-head 2 --epochs-full 15 --workers "$WORKERS" \
      --init-from results/runs/ham_effb0_colab/best.pt --phone-aug \
      --run-name joint_effb0_phoneaug --resume

  echo "==> [joint] calibrate on phone-photo validation (target malignant recall 0.90)"
  $PY backend/src/eval/calibrate.py --run joint_effb0_phoneaug --collection pad_ufes_20 --target-recall 0.90

  echo "==> [joint] evaluate on both test sets"
  $PY backend/src/eval/evaluate.py --run joint_effb0_phoneaug --collection pad_ufes_20 --split test --collapse-to-triage
  $PY backend/src/eval/evaluate.py --run joint_effb0_phoneaug --collection ham10000 --split test --collapse-to-triage

  echo "==> [joint] unfamiliar-image check fitted on both domains"
  $PY backend/src/eval/ood.py --run joint_effb0_phoneaug --fit pad_ufes_20 ham10000 --check pad_ufes_20 ham10000
fi

if [[ $STAGE == all || $STAGE == groups ]]; then
  echo "==> [groups] SCIN images"
  $PY backend/src/data/scin.py images --workers 16
  $PY backend/src/data/scin.py audit

  echo "==> [groups] train Level 1 (what kind of thing)"
  $PY -u backend/src/train/train.py --target group --arch efficientnet_b0 \
      --epochs-head 2 --epochs-full 15 --workers "$WORKERS" --run-name groups_effb0 --resume

  echo "==> [groups] evaluate, calibrate, unfamiliar-image check"
  $PY backend/src/eval/evaluate.py --run groups_effb0 --collection groups --split test
  $PY backend/src/eval/calibrate.py --run groups_effb0 --target-accuracy 0.90
  $PY backend/src/eval/ood.py --run groups_effb0 --fit groups --check groups ham10000
fi

echo "==> pipeline finished"
