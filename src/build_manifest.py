"""Build the image list (data/manifest.csv) from iNaturalist: which photo, which species, which split, and its license.

Splits are chosen so the test really measures what the app will face:
  test   research-grade observations INSIDE Great Smoky Mountains National Park (never trained on)
  train  observations elsewhere in Tennessee and North Carolina
  val    a held-out 15% of the train-region photographers (split by person, so one photographer's
         camera, angles and backgrounds never appear on both sides)

Only photos under a Creative Commons license (or CC0) are used, and each row keeps the attribution
string iNaturalist requires. Run: python src/build_manifest.py   (about 3 minutes; ~1 API call per second)
"""
import csv
import hashlib
import json
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
API = "https://api.inaturalist.org/v1/observations"
PARK_PLACE = 72645            # Great Smoky Mountains National Park
REGION_PLACES = "45,30"       # Tennessee, North Carolina
OPEN_LICENSES = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd"]
TRAIN_CAP = 500               # train photos per species (after filtering)
REGION_FETCH = 900            # region observations fetched per species, to leave room for filtering
TEST_CAP = 150                # park observations per species
VAL_SHARE = 0.15              # share of region photographers held out for validation
PHOTOS_PER_OBS = 1            # first photo only: it's the one the observer chose to lead with
FIELDS = ["photo_id", "obs_id", "species", "split", "url", "license", "attribution", "user_id", "user_login",
          "observed_on", "obs_url"]


def load_species(path=DATA / "species.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_observations(taxon_id, cap, session, **place):
    """Newest-first research-grade observations with open-licensed photos, paged with id_below."""
    out, id_below = [], None
    while len(out) < cap:
        params = {"taxon_id": taxon_id, "quality_grade": "research", "photos": "true",
                  "photo_license": ",".join(OPEN_LICENSES), "per_page": 200,
                  "order_by": "id", "order": "desc", **place}
        if id_below:
            params["id_below"] = id_below
        for attempt in range(5):
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        results = r.json()["results"]
        time.sleep(1.0)                      # iNaturalist asks for about 1 request per second
        if not results:
            break
        out += results
        id_below = results[-1]["id"]
    return out[:cap]


def rows_for(obs_list, species_key, split):
    rows = []
    for o in obs_list:
        photos = [p for p in o.get("photos", []) if p.get("license_code") in OPEN_LICENSES][:PHOTOS_PER_OBS]
        for p in photos:
            rows.append({
                "photo_id": p["id"], "obs_id": o["id"], "species": species_key, "split": split,
                "url": p["url"].replace("/square.", "/medium."),
                "license": p["license_code"], "attribution": p.get("attribution", ""),
                "user_id": o["user"]["id"], "user_login": o["user"]["login"],
                "observed_on": o.get("observed_on") or "", "obs_url": f"https://www.inaturalist.org/observations/{o['id']}",
            })
    return rows


def user_in_val(user_id, share=VAL_SHARE):
    """Deterministic person-level split: a photographer is in validation for every species or none."""
    h = int(hashlib.sha1(f"smokies-{user_id}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000 < share


def assign_region_split(rows):
    for r in rows:
        r["split"] = "val" if user_in_val(r["user_id"]) else "train"
    return rows


def drop_cross_split_duplicates(rows):
    """The same observation can't be in two places, but keep the first occurrence of any photo/obs id."""
    seen_obs, seen_photo, out = set(), set(), []
    for r in rows:
        if r["obs_id"] in seen_obs or r["photo_id"] in seen_photo:
            continue
        seen_obs.add(r["obs_id"]); seen_photo.add(r["photo_id"])
        out.append(r)
    return out


def cap_per_species(rows, train_cap=TRAIN_CAP, val_cap=None):
    """Keep at most train_cap train and val_cap val photos per species (newest first, as fetched)."""
    val_cap = val_cap or round(train_cap * VAL_SHARE / (1 - VAL_SHARE))
    caps = {"train": train_cap, "val": val_cap}
    kept, n = [], {}
    for r in rows:
        if r["split"] in caps:
            k = (r["species"], r["split"])
            n[k] = n.get(k, 0) + 1
            if n[k] > caps[r["split"]]:
                continue
        kept.append(r)
    return kept


def main():
    species = load_species()
    session = requests.Session()
    session.headers["User-Agent"] = "smokies-wildlife-classifier (portfolio project; github.com/jleonar9)"
    rows = []
    for sp in species:
        tid = int(sp["taxon_id"])
        park = fetch_observations(tid, TEST_CAP, session, place_id=PARK_PLACE)
        region = fetch_observations(tid, REGION_FETCH, session, place_id=REGION_PLACES, not_in_place=PARK_PLACE)
        sp_rows = rows_for(park, sp["key"], "test") + assign_region_split(rows_for(region, sp["key"], "train"))
        rows += sp_rows
        counts = {s: sum(r["split"] == s for r in sp_rows) for s in ("train", "val", "test")}
        print(f"{sp['common_name']:28s} fetched: park {counts['test']:4d}, region {counts['train'] + counts['val']:4d}", flush=True)
    rows = drop_cross_split_duplicates(rows)
    # A photographer who also shot in the park would leak their style into test: drop them from train/val.
    test_users = {r["user_id"] for r in rows if r["split"] == "test"}
    before = len(rows)
    rows = [r for r in rows if r["split"] == "test" or r["user_id"] not in test_users]
    n_overlap = before - len(rows)
    rows = cap_per_species(rows)
    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    totals = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    per_species = {sp["key"]: {s: sum(r["species"] == sp["key"] and r["split"] == s for r in rows)
                               for s in ("train", "val", "test")} for sp in species}
    meta = {"built": time.strftime("%Y-%m-%d"), "park_place_id": PARK_PLACE, "region_place_ids": REGION_PLACES,
            "train_cap": TRAIN_CAP, "test_cap": TEST_CAP, "val_share": VAL_SHARE,
            "dropped_for_park_photographer_overlap": n_overlap, "totals": totals,
            "per_species": per_species, "licenses": {lic: sum(r["license"] == lic for r in rows) for lic in OPEN_LICENSES}}
    (DATA / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(rows)} photos: {totals}. Dropped {meta['dropped_for_park_photographer_overlap']} "
          "train/val photos by people who also photographed in the park.")


if __name__ == "__main__":
    sys.exit(main())
