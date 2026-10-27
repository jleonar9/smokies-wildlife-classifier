"""Offline tests for the dataset steps (no network, no PyTorch).  Run: python tests/test_data.py"""
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import build_manifest as bm  # noqa: E402
import download as dl  # noqa: E402


def check(c, msg):
    if not c:
        raise AssertionError(msg)


def fake_obs(oid, uid, licenses=("cc-by-nc",)):
    return {"id": oid, "user": {"id": uid, "login": f"user{uid}"}, "observed_on": "2025-05-01",
            "photos": [{"id": oid * 10 + i, "url": f"https://x/photos/{oid * 10 + i}/square.jpg", "license_code": lic,
                        "attribution": f"(c) user{uid}"} for i, lic in enumerate(licenses)]}


def test_rows_and_licenses():
    obs = [fake_obs(1, 7), fake_obs(2, 8, (None, "cc-by")), fake_obs(3, 9, (None,))]
    rows = bm.rows_for(obs, "elk", "test")
    check(len(rows) == 2, "observation with no open-licensed photo is skipped")
    check(rows[0]["url"].endswith("/medium.jpg"), "medium-size URL")
    check(rows[1]["photo_id"] == 21 and rows[1]["license"] == "cc-by", "first OPEN photo is used, not the first photo")


def test_user_split():
    users = range(5000)
    share = np.mean([bm.user_in_val(u) for u in users])
    check(0.12 < share < 0.18, f"about 15% of photographers in val, got {share:.3f}")
    check(all(bm.user_in_val(u) == bm.user_in_val(u) for u in range(50)), "deterministic")
    rows = bm.assign_region_split(bm.rows_for([fake_obs(i, i % 40) for i in range(400)], "elk", "train"))
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], set()).add(r["split"])
    check(all(len(s) == 1 for s in by_user.values()), "a photographer is never on both sides")


def test_caps_and_dedup():
    rows = [{"species": "elk", "split": "train", "obs_id": i, "photo_id": i} for i in range(700)]
    rows += [{"species": "elk", "split": "val", "obs_id": 1000 + i, "photo_id": 1000 + i} for i in range(200)]
    rows += [{"species": "elk", "split": "test", "obs_id": 5000 + i, "photo_id": 5000 + i} for i in range(300)]
    kept = bm.cap_per_species(rows, train_cap=500)
    n = {s: sum(r["split"] == s for r in kept) for s in ("train", "val", "test")}
    check(n == {"train": 500, "val": 88, "test": 300}, n)
    d = bm.drop_cross_split_duplicates([{"obs_id": 1, "photo_id": 1}, {"obs_id": 1, "photo_id": 2}, {"obs_id": 2, "photo_id": 3}])
    check([r["photo_id"] for r in d] == [1, 3], "same observation kept once")


def scene(seed, size=(320, 240)):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", size, tuple(int(v) for v in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        d.ellipse([x, y, x + rng.integers(20, 90), y + rng.integers(20, 90)], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return im


def test_dedup(tmp):
    root = tmp / "images"
    paths = {}
    for split, sp, name, im in [
        ("train", "elk", "1", scene(1)),
        ("train", "elk", "2", scene(2)),
        ("test", "elk", "3", scene(1).resize((300, 225)).filter(ImageFilter.GaussianBlur(0.6))),   # re-upload of train photo 1
        ("test", "elk", "4", scene(3)),
        ("val", "bear", "5", scene(2)),                                                         # exact copy of train photo 2
    ]:
        p = root / split / sp / f"{name}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, quality=80)
        paths.setdefault(split, []).append(p)
    removed = dl.find_duplicates(paths)
    names = sorted(p.name for p, _ in removed)
    check(names == ["3.jpg", "5.jpg"], f"near-duplicates in val/test removed, train kept: {names}")
    check(all("duplicate_of:train/elk/" in why for _, why in removed), "reason names the train original")
    # distinct scenes should be far apart
    with Image.open(paths["train"][0]) as a, Image.open(paths["test"][1]) as b:
        da, db = np.uint64(dl.dhash(a)), np.uint64(dl.dhash(b))
    check(int(dl.popcount64(np.array([da ^ db]))[0]) > dl.HASH_MAX_DIST * 3, "different photos have distant hashes")


def test_popcount():
    xs = np.array([0, 1, 0xFF, 2**64 - 1], dtype=np.uint64)
    check(dl.popcount64(xs).tolist() == [0, 1, 8, 64], dl.popcount64(xs))


def main():
    test_rows_and_licenses()
    test_user_split()
    test_caps_and_dedup()
    test_popcount()
    with tempfile.TemporaryDirectory() as d:
        test_dedup(pathlib.Path(d))
    print("All data tests passed")


if __name__ == "__main__":
    main()
