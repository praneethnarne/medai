"""SCIN: crowd-sourced phone photos of common skin conditions (Google / Stanford).

    python backend/src/data/scin.py manifest   # labels -> data/scin/manifest.json + splits.json
    python backend/src/data/scin.py images     # download the images in the manifest (resumable)
    python backend/src/data/scin.py audit      # counts per group and split -> results/scin_audit.json

Source: https://github.com/google-research-datasets/scin, files served from the public
bucket gs://dx-scin-public-data. Licence: SCIN Data Use License (reuse and adaptation
allowed; re-identifying contributors is prohibited). Cite: Ward et al., "Creating an
Empathy-Driven, Diverse Dermatology Dataset with Crowdsourcing" (arXiv 2402.18545).

Labels: each case has up to three images and a dermatologist-weighted differential
(condition -> weight). We take the highest-weighted condition and map it to a Level 1
group with `groups.SCIN_TO_GROUP`; unmapped conditions are excluded and reported.

Images are stored shrunk to at most 512 px on the long side (JPEG q90). The originals are
~2 MB each; the model sees 224 px, so nothing it uses is lost.

Splits are by CASE (all images of one case on one side), stratified by group, seed 42,
70 / 15 / 15. The manifest and splits are committed; the images are not.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import io
import json
import pathlib
import sys
import threading
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bootstrap  # noqa: F401,E402

from data.groups import GROUPS, group_of_scin  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "scin"
BUCKET = "https://storage.googleapis.com/dx-scin-public-data/"
MAX_SIDE = 512
SEED = 42


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _csv(name: str) -> list[dict]:
    raw = BASE / "raw" / name
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(_fetch(BUCKET + "dataset/" + name))
    with open(raw, newline="") as f:
        return list(csv.DictReader(f))


def build_manifest() -> list[dict]:
    labels = {r["case_id"]: r for r in _csv("scin_labels.csv")}
    out = []
    for case in _csv("scin_cases.csv"):
        lab = labels.get(case["case_id"])
        if not lab or not lab["weighted_skin_condition_label"]:
            continue
        diff = ast.literal_eval(lab["weighted_skin_condition_label"])
        if not diff:
            continue
        condition = max(diff, key=diff.get)
        for i in (1, 2, 3):
            path = case.get(f"image_{i}_path")
            if not path:
                continue
            out.append({
                "image": f"{case['case_id']}_{i}.jpg",
                "source_path": path,
                "case_id": case["case_id"],
                "condition": condition,
                "condition_weight": round(float(diff[condition]), 4),
                "group": group_of_scin(condition),      # None = excluded
                "shot_type": case.get(f"image_{i}_shot_type") or None,
                "fitzpatrick_skin_type": lab.get("dermatologist_fitzpatrick_skin_type_label_1") or None,
                "monk_skin_tone_us": lab.get("monk_skin_tone_label_us") or None,
            })
    return out


def build_splits(manifest: list[dict]) -> dict:
    from sklearn.model_selection import train_test_split

    case_group = {}
    for r in manifest:
        if r["group"]:
            case_group[r["case_id"]] = r["group"]
    cases = sorted(case_group)
    groups = [case_group[c] for c in cases]
    train, rest, _, g_rest = train_test_split(cases, groups, test_size=0.30,
                                              stratify=groups, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=g_rest, random_state=SEED)
    assignment = {c: "train" for c in train} | {c: "val" for c in val} | {c: "test" for c in test}
    return {"unit": "case_id", "seed": SEED, "ratios": [0.7, 0.15, 0.15],
            "assignment": dict(sorted(assignment.items()))}


def download_images(workers: int = 16, limit: int | None = None) -> None:
    from PIL import Image

    manifest = json.loads((BASE / "manifest.json").read_text())
    todo = [r for r in manifest if r["group"]][:limit]
    (BASE / "images").mkdir(parents=True, exist_ok=True)
    lock, done, failed = threading.Lock(), [0], []

    def one(r):
        dst = BASE / "images" / r["image"]
        if dst.exists() and dst.stat().st_size > 0:
            return
        try:
            with Image.open(io.BytesIO(_fetch(BUCKET + r["source_path"]))) as im:
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                tmp = dst.with_suffix(".part")
                im.save(tmp, format="JPEG", quality=90)
                tmp.rename(dst)
        except Exception as e:  # noqa: BLE001 -- reported, re-run resumes
            with lock:
                failed.append((r["image"], str(e)[:120]))
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    have = sum(1 for r in todo if (BASE / "images" / r["image"]).exists())
    print(f"images present: {have}/{len(todo)}; failed this run: {len(failed)}")
    for name, err in failed[:10]:
        print("  failed", name, err)
    if failed:
        print("re-run the same command to retry the failures")


def audit() -> dict:
    from collections import Counter

    manifest = json.loads((BASE / "manifest.json").read_text())
    assignment = json.loads((BASE / "splits.json").read_text())["assignment"]
    per = {s: Counter() for s in ("train", "val", "test")}
    for r in manifest:
        if r["group"]:
            per[assignment[r["case_id"]]][r["group"]] += 1
    excluded = Counter(r["condition"] for r in manifest if not r["group"])
    rep = {
        "images_by_split_and_group": {s: {g: per[s][g] for g in GROUPS} for s in per},
        "cases_by_group": dict(Counter(r["group"] for r in {r["case_id"]: r for r in manifest
                                                            if r["group"]}.values())),
        "excluded_images": sum(excluded.values()),
        "excluded_conditions": dict(excluded.most_common()),
    }
    out = ROOT / "results" / "scin_audit.json"
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep["images_by_split_and_group"], indent=2))
    print(f"excluded images: {rep['excluded_images']} (see {out.relative_to(ROOT)})")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["manifest", "images", "audit"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="images step: only the first N (testing)")
    args = ap.parse_args()
    if args.step == "manifest":
        m = build_manifest()
        BASE.mkdir(parents=True, exist_ok=True)
        (BASE / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        (BASE / "splits.json").write_text(json.dumps(build_splits(m), indent=1) + "\n")
        print(f"manifest: {len(m)} images, {sum(1 for r in m if r['group'])} mapped to a group")
    elif args.step == "images":
        download_images(args.workers, args.limit)
    else:
        audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
