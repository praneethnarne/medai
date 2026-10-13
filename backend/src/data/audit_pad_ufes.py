"""Audit the PAD-UFES-20 collection in the ISIC archive.

Metadata only -- downloads no images. Produces results/pad_ufes20_audit.json,
which every number in docs/DATA_AUDIT.md is read from.

    python backend/src/data/audit_pad_ufes.py
"""

import collections
import json
import pathlib
import ssl
import urllib.parse
import urllib.request

COLLECTION = "406"  # PAD-UFES-20
API = "https://api.isic-archive.com/api/v2/images/search/"
OUT = pathlib.Path(__file__).resolve().parents[3] / "results" / "pad_ufes20_audit.json"


def _ssl_context():
    """A python.org install on macOS ships no CA bundle, so urllib fails TLS
    verification out of the box. Use certifi's bundle when it is available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_all():
    url = API + "?" + urllib.parse.urlencode({"collections": COLLECTION, "limit": "100"})
    ctx = _ssl_context()
    records = []
    while url:
        with urllib.request.urlopen(url, context=ctx) as resp:
            page = json.load(resp)
        records.extend(page["results"])
        url = page.get("next")
    return records


def audit(records):
    dx = collections.Counter()
    malignancy = collections.Counter()
    fitz = collections.Counter()
    field_coverage = collections.Counter()
    patients, lesions = set(), set()

    for r in records:
        c = r["metadata"]["clinical"]
        dx[c.get("diagnosis_3") or c.get("diagnosis_2")] += 1
        malignancy[c.get("diagnosis_1")] += 1
        fitz[c.get("fitzpatrick_skin_type") or "unrecorded"] += 1
        for key in c:
            field_coverage[key] += 1
        if c.get("patient_id"):
            patients.add(c["patient_id"])
        if c.get("lesion_id"):
            lesions.add(c["lesion_id"])

    return {
        "collection": "PAD-UFES-20 (ISIC collection 406)",
        "license": "CC-BY",
        "image_type": "clinical: close-up (smartphone)",
        "n_images": len(records),
        "n_patients": len(patients),
        "n_lesions": len(lesions),
        "diagnosis_counts": dict(dx.most_common()),
        "malignancy_counts": dict(malignancy),
        "fitzpatrick_counts": dict(sorted(fitz.items(), key=lambda kv: str(kv[0]))),
        "field_coverage": dict(field_coverage.most_common()),
    }


if __name__ == "__main__":
    result = audit(fetch_all())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(f"{result['n_images']} images, {result['n_patients']} patients, "
          f"{result['n_lesions']} lesions")
