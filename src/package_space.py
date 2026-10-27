"""Assemble the Hugging Face Space in build/space/ and optionally upload it.

  python src/package_space.py                 # just build the folder
  python src/package_space.py --upload USER   # also create/update the Space USER/smokies-wildlife-id

The app's "not sure" threshold is the lowest confidence at which test accuracy reaches 90%.
"""
import argparse
import json
import shutil

from common import DATA, MODELS, REPORTS, ROOT

SPACE_README = """---
title: Smokies Wildlife ID
emoji: 🐻
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

Identify 20 common animals of Great Smoky Mountains National Park from a photo.
Code, data sources and evaluation: {repo}
Training photos from iNaturalist observers under Creative Commons licenses (non-commercial use).
"""


def pick_threshold(target=0.90):
    m = json.loads((REPORTS / "metrics.json").read_text())
    for row in m["coverage"]:
        if row["accuracy"] is not None and row["accuracy"] >= target:
            return row["threshold"]
    return 0.5


def build(run="convnext_ft", repo="https://github.com/jleonar9/smokies-wildlife-classifier"):
    out = ROOT / "build" / "space"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", out)
    shutil.copy(ROOT / "app" / "requirements.txt", out)
    shutil.copy(DATA / "species.csv", out)
    shutil.copy(MODELS / run / "model.pt", out)
    cfg = json.loads((MODELS / run / "config.json").read_text())
    cfg["answer_threshold"] = pick_threshold()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "README.md").write_text(SPACE_README.format(repo=repo), encoding="utf-8")
    print(f"Space files in {out} (answer threshold {cfg['answer_threshold']:.0%})")
    return out


def upload(folder, user, name="smokies-wildlife-id"):
    from huggingface_hub import HfApi, login
    login()   # asks for a Hugging Face access token with write permission (you paste it; it isn't stored here)
    api = HfApi()
    repo_id = f"{user}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="space",
                      commit_message="Update Smokies Wildlife ID")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="HF_USERNAME")
    a = ap.parse_args()
    folder = build()
    if a.upload:
        upload(folder, a.upload)
