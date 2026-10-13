"""Download an ISIC archive collection: metadata manifest first, then images.

Resumable and idempotent -- an image already on disk at the expected size is
skipped, so an interrupted run can simply be restarted.

    python backend/src/data/download.py ham10000
    python backend/src/data/download.py pad_ufes_20
    python backend/src/data/download.py --list

Images land in data/<name>/images/, the manifest in data/<name>/metadata.json.
Both are gitignored; the manifest is the reproducible record of what was fetched.
"""

import argparse
import concurrent.futures
import json
import pathlib
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

COLLECTIONS = {
    "ham10000": {"id": "212", "note": "dermatoscopic, pretraining corpus"},
    "pad_ufes_20": {"id": "406", "note": "smartphone, primary domain"},
    "isic2018_masks": {"id": "531", "note": "segmentation masks"},
}

ROOT = pathlib.Path(__file__).resolve().parents[3]
API = "https://api.isic-archive.com/api/v2/images/search/"


def ssl_context():
    """python.org builds on macOS ship no CA bundle, so verification fails
    out of the box. Prefer certifi's when it is installed."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_manifest(collection_id, ctx):
    url = API + "?" + urllib.parse.urlencode(
        {"collections": collection_id, "limit": "100"})
    records, page_no = [], 0
    while url:
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
            page = json.load(resp)
        records.extend(page["results"])
        url = page.get("next")
        page_no += 1
        print(f"\r  manifest: {len(records)} records ({page_no} pages)", end="", flush=True)
    print()
    return records


def download_one(url, dest, expected_size, ctx, retries=3):
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return "skipped"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries - 1:
                print(f"\n  FAILED {dest.name}: {exc}")
                return "failed"
            time.sleep(2 ** attempt)
    return "failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=sorted(COLLECTIONS))
    ap.add_argument("--list", action="store_true", help="show known collections and exit")
    ap.add_argument("--manifest-only", action="store_true",
                    help="fetch metadata without downloading images")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel downloads (default 16); the archive is the "
                         "bottleneck, not bandwidth")
    args = ap.parse_args()

    if args.list or not args.name:
        for k, v in COLLECTIONS.items():
            print(f"  {k:16s} ISIC collection {v['id']:>4}  -- {v['note']}")
        return 0

    spec = COLLECTIONS[args.name]
    out = ROOT / "data" / args.name
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    ctx = ssl_context()

    print(f"{args.name}: ISIC collection {spec['id']} ({spec['note']})")
    manifest_path = out / "metadata.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text())
        print(f"  manifest: {len(records)} records (cached)")
    else:
        records = fetch_manifest(spec["id"], ctx)
        manifest_path.write_text(json.dumps(records, indent=1) + "\n")

    total_bytes = sum(r["files"]["full"]["size"] for r in records)
    print(f"  {len(records)} images, {total_bytes / 1e9:.2f} GB")
    if args.manifest_only:
        return 0

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    started = time.time()
    done_n = 0

    def work(record):
        nonlocal done_n
        f = record["files"]["full"]
        result = download_one(f["url"], images / f"{record['isic_id']}.jpg",
                              f["size"], ctx)
        with lock:
            counts[result] += 1
            done_n += 1
            if done_n % 50 == 0 or done_n == len(records):
                rate = done_n / max(time.time() - started, 1e-6)
                remaining = (len(records) - done_n) / max(rate, 1e-6)
                print(f"\r  {done_n}/{len(records)}  failed={counts['failed']}  "
                      f"{rate:.0f} img/s  eta {remaining / 60:.0f}m",
                      end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, records))

    print(f"\n  done in {(time.time() - started) / 60:.1f}m: {counts}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
