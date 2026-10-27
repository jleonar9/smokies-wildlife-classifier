"""Download the photos in data/manifest.csv into data/images/<split>/<species>/<photo_id>.jpg,
check each one, and remove near-duplicates.

  python src/download.py [--workers 8] [--root data/images]

Safe to re-run: files already on disk are skipped. Writes data/download_report.json.
"""
import argparse
import csv
import io
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_SIDE = 160          # smaller images are thumbnails or crops too small to learn from
HASH_MAX_DIST = 4       # dHash bits that may differ for two photos to count as the same picture


def read_manifest(path=DATA / "manifest.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dest(row, root):
    return root / row["split"] / row["species"] / f"{row['photo_id']}.jpg"


def fetch_one(row, root, session, retries=3):
    path = dest(row, root)
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = session.get(row["url"], timeout=60)
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.load()
            if min(img.size) < MIN_SIDE:
                return "too_small"
            img.convert("RGB").save(path, "JPEG", quality=92)
            return "ok"
        except (requests.RequestException, OSError):
            time.sleep(2 * (attempt + 1))
    return "failed"


def dhash(img, size=8):
    """Difference hash as a 64-bit integer: robust to resizing and re-compression, so re-uploads match."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    return _POP8[np.ascontiguousarray(x, dtype=np.uint64).view(np.uint8)].reshape(-1, 8).sum(axis=1)


def find_duplicates(paths_by_split, max_dist=HASH_MAX_DIST):
    """Return files to delete. Splits are checked in the order train, val, test, so a test photo that
    matches a train/val photo is the copy removed (it would leak the answer into the test)."""
    kept = np.zeros(0, dtype=np.uint64)
    kept_paths, remove = [], []
    for split in ("train", "val", "test"):
        for p in paths_by_split.get(split, []):
            try:
                with Image.open(p) as im:
                    h = np.uint64(dhash(im))
            except OSError:
                remove.append((p, "unreadable"))
                continue
            if len(kept):
                d = popcount64(kept ^ h)
                j = int(d.argmin())
                if d[j] <= max_dist:
                    q = kept_paths[j]
                    remove.append((p, f"duplicate_of:{q.parent.parent.name}/{q.parent.name}/{q.name}"))
                    continue
            kept = np.append(kept, h)
            kept_paths.append(p)
    return remove


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=str(DATA / "images"))
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    rows = read_manifest()
    session = requests.Session()
    session.headers["User-Agent"] = "smokies-wildlife-classifier (portfolio project)"
    t = time.time()
    status = {}
    with ThreadPoolExecutor(a.workers) as pool:
        for i, s in enumerate(pool.map(lambda r: fetch_one(r, root, session), rows), 1):
            status[s] = status.get(s, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} ({time.time() - t:.0f}s) {status}", flush=True)
    print(f"Downloaded in {time.time() - t:.0f}s: {status}")

    by_split = {s: sorted((root / s).glob("*/*.jpg")) for s in ("train", "val", "test")}
    removed = find_duplicates(by_split)
    for p, _ in removed:
        p.unlink(missing_ok=True)
    counts = {s: {sp.name: len(list(sp.glob("*.jpg"))) for sp in sorted((root / s).glob("*"))} for s in by_split}
    report = {"download_status": status, "removed": [{"file": str(p.relative_to(root)), "reason": why} for p, why in removed],
              "counts": counts, "totals": {s: sum(c.values()) for s, c in counts.items()}}
    (DATA / "download_report.json").write_text(json.dumps(report, indent=2))
    print(f"Removed {len(removed)} duplicates/unreadable files. Images per split: {report['totals']}")


if __name__ == "__main__":
    sys.exit(main())
