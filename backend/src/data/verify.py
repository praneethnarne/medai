"""Check a downloaded collection is complete and every image is a whole JPEG.

    python backend/src/data/verify.py ham10000 pad_ufes_20

Exits non-zero if anything is missing or truncated, so it can gate training.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data.download import looks_complete  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]


def verify(name):
    base = ROOT / "data" / name
    manifest = base / "metadata.json"
    if not manifest.exists():
        return {"collection": name, "error": "no manifest"}

    records = json.loads(manifest.read_text())
    images = base / "images"
    missing, truncated = [], []
    total_bytes = 0

    for r in records:
        path = images / f"{r['isic_id']}.jpg"
        if not path.exists():
            missing.append(r["isic_id"])
        elif not looks_complete(path):
            truncated.append(r["isic_id"])
        else:
            total_bytes += path.stat().st_size

    extra = {p.stem for p in images.glob("*.jpg")} - {r["isic_id"] for r in records}
    return {
        "collection": name,
        "expected": len(records),
        "present": len(records) - len(missing),
        "missing": missing[:20],
        "n_missing": len(missing),
        "truncated": truncated[:20],
        "n_truncated": len(truncated),
        "n_unexpected_files": len(extra),
        "n_partial_files": len(list(images.glob("*.part"))),
        "bytes_on_disk": total_bytes,
        "ok": not missing and not truncated,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()

    results = [verify(n) for n in args.names]
    out = ROOT / "results" / "data_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    all_ok = True
    for r in results:
        if r.get("error"):
            print(f"{r['collection']:14s} ERROR: {r['error']}")
            all_ok = False
            continue
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['collection']:14s} {status:4s} "
              f"{r['present']}/{r['expected']} images, "
              f"{r['bytes_on_disk'] / 1e6:.0f} MB, "
              f"missing={r['n_missing']} truncated={r['n_truncated']} "
              f"partial={r['n_partial_files']}")
        all_ok &= r["ok"]
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
