"""Audit a downloaded ISIC collection's metadata.

Reads data/<name>/metadata.json (produced by download.py) and writes
results/<name>_audit.json. Every number in docs/DATA_AUDIT.md is read from
those files, so no figure in the documentation is typed by hand.

    python backend/src/data/audit_collection.py pad_ufes_20
    python backend/src/data/audit_collection.py ham10000
"""

import argparse
import collections
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data import labels  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]


def audit(records):
    dx = collections.Counter()
    # Counts in the project's own label space, so downstream documents never
    # have to re-derive the mapping from raw diagnosis strings. HAM10000's
    # vascular records carry a null diagnosis_3, which a naive string-keyed
    # count silently drops.
    lesion_types = collections.Counter()
    triage = collections.Counter()
    malignancy = collections.Counter()
    fitz = collections.Counter()
    field_coverage = collections.Counter()
    licences = collections.Counter()
    image_types = collections.Counter()
    patients, lesions = set(), set()
    dims = []

    for r in records:
        c = r["metadata"]["clinical"]
        a = r["metadata"].get("acquisition", {})
        dx[c.get("diagnosis_3") or c.get("diagnosis_2") or c.get("diagnosis_1")] += 1
        lesion_types[labels.lesion_type_of(r)] += 1
        triage[labels.triage_of(r)] += 1
        malignancy[c.get("diagnosis_1")] += 1
        fitz[c.get("fitzpatrick_skin_type") or "unrecorded"] += 1
        licences[r.get("copyright_license")] += 1
        image_types[a.get("image_type")] += 1
        for key in c:
            field_coverage[key] += 1
        if c.get("patient_id"):
            patients.add(c["patient_id"])
        if c.get("lesion_id"):
            lesions.add(c["lesion_id"])
        if a.get("pixels_x"):
            dims.append((a["pixels_x"], a["pixels_y"]))

    widths = sorted(d[0] for d in dims)
    return {
        "n_images": len(records),
        "n_patients": len(patients),
        "n_lesions": len(lesions),
        # The split granularity we are actually able to use. Splitting at a
        # coarser unit than the data supports leaks the same lesion across
        # train and test and inflates every metric.
        "finest_split_unit": ("patient" if patients else
                              "lesion" if lesions else "image"),
        "licenses": dict(licences),
        "image_types": dict(image_types),
        "diagnosis_counts": dict(dx.most_common()),
        "lesion_type_counts": {t: lesion_types.get(t, 0) for t in labels.LESION_TYPES},
        "triage_counts": {t: triage.get(t, 0) for t in labels.TRIAGE},
        "malignancy_counts": dict(malignancy),
        "fitzpatrick_counts": dict(sorted(fitz.items(), key=lambda kv: str(kv[0]))),
        "field_coverage": dict(field_coverage.most_common()),
        "resolution": {
            "n_with_dims": len(dims),
            "width_min": widths[0] if widths else None,
            "width_median": widths[len(widths) // 2] if widths else None,
            "width_max": widths[-1] if widths else None,
        } if dims else None,
        "total_bytes": sum(r["files"]["full"]["size"] for r in records),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="collection name, as used by download.py")
    args = ap.parse_args()

    manifest = ROOT / "data" / args.name / "metadata.json"
    if not manifest.exists():
        print(f"no manifest at {manifest}\nrun: python backend/src/data/download.py "
              f"{args.name} --manifest-only", file=sys.stderr)
        return 1

    result = audit(json.loads(manifest.read_text()))
    result["collection"] = args.name
    out = ROOT / "results" / f"{args.name}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  {result['n_images']} images, {result['n_lesions']} lesions, "
          f"{result['n_patients']} patients")
    print(f"  finest split unit: {result['finest_split_unit']}")
    print(f"  licence: {result['licenses']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
