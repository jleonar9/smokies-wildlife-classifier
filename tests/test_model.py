"""CPU smoke test of the training, evaluation and Grad-CAM code on synthetic images (needs torch + timm).

Run: python tests/test_model.py    (about a minute; downloads nothing)
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import evaluate as ev  # noqa: E402
from common import MODELS, ImageList, build_transforms, class_names, list_split  # noqa: E402
from train import cosine_with_warmup, data_config, macro_f1, make_model, param_groups, predict  # noqa: E402


def check(c, msg):
    if not c:
        raise AssertionError(msg)


def make_images(root, classes, per_split={"train": 3, "val": 2, "test": 2}):
    rng = np.random.default_rng(0)
    for split, n in per_split.items():
        for i, key in enumerate(classes):
            d = root / split / key
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                arr = rng.integers(0, 255, (80, 96, 3), dtype=np.uint8)
                arr[..., i % 3] = 40 + 10 * i            # a learnable color cue per class
                Image.fromarray(arr).save(d / f"{i}{j}{split[0]}.jpg")


def test_pieces(root, classes):
    model = make_model("resnet18", len(classes), pretrained=False)
    cfg = data_config(model)
    check(cfg["img_size"] == 224 and len(cfg["mean"]) == 3, cfg)
    groups = param_groups(model, 1e-4, 10, 0.05, linear_probe=True)
    check(len(groups) == 1 and not next(model.parameters()).requires_grad, "linear probe freezes the backbone")
    model = make_model("resnet18", len(classes), pretrained=False)
    groups = param_groups(model, 1e-4, 10, 0.05, linear_probe=False)
    check(groups[1]["lr"] == 1e-3, "head gets the higher learning rate")
    check(abs(cosine_with_warmup(0, 100, 10) - 0.1) < 1e-9 and cosine_with_warmup(100, 100, 10) < 1e-9, "schedule")

    _, t_test = build_transforms(64, cfg["mean"], cfg["std"])
    items = list_split(root, "test", classes)
    check(len(items) == 2 * len(classes), len(items))
    loader = torch.utils.data.DataLoader(ImageList(items, t_test), batch_size=8)
    probs, labels = predict(model, loader, torch.device("cpu"), amp=False)
    check(probs.shape == (len(items), len(classes)) and np.allclose(probs.sum(1), 1, atol=1e-4), "probabilities")
    m = ev.metrics(probs, labels, len(classes))
    check(0 <= m["accuracy"] <= m["top3_accuracy"] <= 1 and 0 <= m["ece"] <= 1, m)
    rows = ev.per_class(probs, labels, classes)
    check(len(rows) == len(classes) and sum(r["test_images"] for r in rows) == len(items), "per-class rows")
    curve = ev.coverage_curve(probs, labels)
    check(curve[0]["answered"] == 1.0 and curve[-1]["answered"] <= 1.0, curve)

    x = t_test(Image.open(items[0][0]).convert("RGB")).unsqueeze(0)
    cam, p = ev.grad_cam(model, x)
    check(cam.shape[0] == 1 and float(cam.max()) <= 1.0 + 1e-6 and float(cam.min()) >= 0, "Grad-CAM in [0, 1]")
    y = np.array([0, 0, 1, 1]); check(abs(macro_f1(y, np.array([0, 1, 1, 1]), 2) - (2/3 + 0.8) / 2) < 1e-9, "macro-F1")


def test_training_script(root):
    run = "_smoke_test"
    try:
        r = subprocess.run([sys.executable, str(HERE / "src" / "train.py"), "--smoke", "--run", run, "--images", str(root)],
                           capture_output=True, text=True, timeout=600)
        check(r.returncode == 0, r.stdout + r.stderr)
        check((MODELS / run / "model.pt").exists() and (MODELS / run / "config.json").exists(), "checkpoint saved")
        model, cfg = ev.load_run(run, torch.device("cpu"))
        check(cfg["model"] == "resnet18" and len(cfg["classes"]) == 20, cfg)
    finally:
        shutil.rmtree(MODELS / run, ignore_errors=True)


def main():
    classes = class_names()
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        make_images(root, classes)
        test_pieces(root, classes)
        test_training_script(root)
    print("All model tests passed")


if __name__ == "__main__":
    main()
