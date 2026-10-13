"""Build train/val/test splits that cannot leak a lesion across sides.

The split unit is read from the audit rather than assumed: PAD-UFES-20 carries
patient ids, HAM10000 only lesion ids. Splitting at a coarser unit than the data
supports puts two photographs of the same lesion on both sides of the split and
inflates every metric that follows -- the single easiest way to produce results
that look good and mean nothing.

    python backend/src/data/split.py ham10000 pad_ufes_20

Writes data/<name>/splits.json (committed -- it is small and it is the record of
exactly which image went where) and results/<name>_split_summary.json.
"""

import argparse
import collections
import json
import pathlib
import sys

from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data import labels  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
SEED = 42

# 1/7 then 1/6 of the remainder gives roughly 71/14/14. Exact proportions are
# reported rather than asserted, because group sizes vary and the grouping
# constraint takes precedence over hitting a round number.
TEST_FOLDS = 7
VAL_FOLDS = 6


def group_key(record, unit):
    clinical = record["metadata"]["clinical"]
    if unit == "patient":
        return clinical.get("patient_id") or f"lesion:{clinical.get('lesion_id')}" \
            or f"image:{record['isic_id']}"
    if unit == "lesion":
        return clinical.get("lesion_id") or f"image:{record['isic_id']}"
    return record["isic_id"]


def split_one(name):
    base = ROOT / "data" / name
    records = json.loads((base / "metadata.json").read_text())
    audit = json.loads((ROOT / "results" / f"{name}_audit.json").read_text())
    unit = audit["finest_split_unit"]

    y = [labels.triage_of(r) for r in records]
    groups = [group_key(r, unit) for r in records]

    idx = list(range(len(records)))
    sgkf = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(idx, y, groups))

    rest_y = [y[i] for i in rest_idx]
    rest_groups = [groups[i] for i in rest_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)
    train_pos, val_pos = next(sgkf2.split(list(range(len(rest_idx))), rest_y, rest_groups))

    assignment = {}
    for pos in train_pos:
        assignment[records[rest_idx[pos]]["isic_id"]] = "train"
    for pos in val_pos:
        assignment[records[rest_idx[pos]]["isic_id"]] = "val"
    for i in test_idx:
        assignment[records[i]["isic_id"]] = "test"

    (base / "splits.json").write_text(json.dumps(
        {"seed": SEED, "split_unit": unit, "assignment": assignment}, indent=1) + "\n")

    summary = summarise(records, assignment, unit)
    (ROOT / "results" / f"{name}_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def summarise(records, assignment, unit):
    by_split = collections.defaultdict(lambda: {
        "n_images": 0,
        "triage": collections.Counter(),
        "lesion_type": collections.Counter(),
        "groups": set(),
    })
    for r in records:
        s = by_split[assignment[r["isic_id"]]]
        s["n_images"] += 1
        s["triage"][labels.triage_of(r)] += 1
        s["lesion_type"][labels.lesion_type_of(r)] += 1
        s["groups"].add(group_key(r, unit))

    total = len(records)
    return {
        "split_unit": unit,
        "seed": SEED,
        "n_images": total,
        "splits": {
            k: {
                "n_images": v["n_images"],
                "share": round(v["n_images"] / total, 4),
                "n_groups": len(v["groups"]),
                "triage": dict(v["triage"].most_common()),
                "lesion_type": dict(v["lesion_type"].most_common()),
            }
            for k, v in sorted(by_split.items())
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()

    for name in args.names:
        s = split_one(name)
        print(f"\n{name}  (split unit: {s['split_unit']}, seed {s['seed']})")
        for split, info in s["splits"].items():
            tri = info["triage"]
            print(f"  {split:5s} {info['n_images']:6d} images "
                  f"({info['share'] * 100:4.1f}%)  {info['n_groups']:5d} {s['split_unit']}s"
                  f"   malignant={tri.get('malignant', 0):4d} "
                  f"indeterminate={tri.get('indeterminate', 0):4d} "
                  f"benign={tri.get('benign', 0):4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
