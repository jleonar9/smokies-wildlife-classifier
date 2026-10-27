"""Score trained models on the held-out Smokies test set and make the report figures.

  python src/evaluate.py --runs convnext_probe convnext_ft     # compares runs; figures for the last one

Writes reports/metrics.json, reports/per_class.csv, reports/confusion_matrix.png, reports/errors.png,
reports/gradcam.png, reports/confidence.png and reports/image_credits.md.
"""
import argparse
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import DATA, MODELS, REPORTS, ImageList, build_transforms, list_split, load_species, save_json
from train import macro_f1, make_model, predict


def load_run(run, device):
    cfg = json.loads((MODELS / run / "config.json").read_text())
    model = make_model(cfg["model"], len(cfg["classes"]), pretrained=False)
    state = torch.load(MODELS / run / "model.pt", map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval(), cfg


def metrics(probs, labels, n):
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    conf = probs.max(1)
    # Expected calibration error over 10 confidence bins
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return {"n": int(len(labels)), "accuracy": float((pred == labels).mean()),
            "top3_accuracy": float(np.mean([labels[i] in top3[i] for i in range(len(labels))])),
            "macro_f1": macro_f1(labels, pred, n), "ece": float(ece)}


def per_class(probs, labels, classes):
    pred = probs.argmax(1)
    rows = []
    for c, key in enumerate(classes):
        tp = int(np.sum((pred == c) & (labels == c)))
        support = int(np.sum(labels == c))
        predicted = int(np.sum(pred == c))
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        confused = np.bincount(pred[(labels == c) & (pred != c)], minlength=len(classes))
        top_conf = classes[int(confused.argmax())] if confused.sum() else ""
        rows.append({"species": key, "test_images": support, "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(0 if tp == 0 else 2 * p * r / (p + r), 3), "most_confused_with": top_conf})
    return rows


def coverage_curve(probs, labels):
    """Accuracy when the app only answers above a confidence threshold, and how often it answers."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    out = []
    for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= t
        out.append({"threshold": t, "answered": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def grad_cam(model, x, target=None):
    """Grad-CAM for timm CNNs: weight the last feature map by the gradient of the class score."""
    model.eval()
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    target = logits.argmax(1) if target is None else target
    model.zero_grad()
    logits.gather(1, target.view(-1, 1)).sum().backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1))
    cam = cam / cam.flatten(1).max(1).values.clamp(min=1e-8).view(-1, 1, 1)
    return cam.detach(), torch.softmax(logits.detach(), 1)


def figures(run, model, cfg, items, probs, labels, classes, names, device, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    REPORTS.mkdir(exist_ok=True)
    n = len(classes)
    pred = probs.argmax(1)
    cm = np.zeros((n, n))
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n), [names[c] for c in classes], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), [names[c] for c in classes], fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.5 else "#333")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True species")
    ax.set_title("Test set (photos taken inside the park): counts, shaded by share of each true species")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=130); plt.close(fig)

    credits = {}

    def show(ax, path):
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.axis("off")
        row = manifest.get(path.stem)
        if row:
            credits[path.stem] = row

    # Most confident mistakes
    wrong = np.where(pred != labels)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:12]
    if len(wrong):
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(len(wrong) / cols)), cols, figsize=(12, 3.3 * np.ceil(len(wrong) / cols)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), wrong):
            show(ax, items[i][0])
            ax.set_title(f"true: {names[classes[labels[i]]]}\npred: {names[classes[pred[i]]]} ({probs[i].max():.0%})\n"
                         f"photo {items[i][0].stem}", fontsize=8)
        fig.suptitle("Most confident mistakes on the park test set"); fig.tight_layout()
        fig.savefig(REPORTS / "errors.png", dpi=110); plt.close(fig)

    # Grad-CAM on one correct example per species (first 12 species with a correct prediction)
    _, t_test = build_transforms(cfg["img_size"], cfg["mean"], cfg["std"])
    picks = []
    for c in range(n):
        idx = np.where((labels == c) & (pred == c))[0]
        if len(idx):
            picks.append(idx[np.argmax(probs[idx, c])])
    picks = picks[:12]
    if picks:
        fig, axes = plt.subplots(len(picks) // 4 + (len(picks) % 4 > 0), 4, figsize=(12, 3.2 * (len(picks) // 4 + 1)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), picks):
            path = items[i][0]
            with Image.open(path) as im:
                x = t_test(im.convert("RGB")).unsqueeze(0).to(device)
            cam, _ = grad_cam(model, x)
            cam = torch.nn.functional.interpolate(cam[None], size=x.shape[-2:], mode="bilinear")[0, 0].cpu().numpy()
            img = x[0].cpu().permute(1, 2, 0).numpy() * np.array(cfg["std"]) + np.array(cfg["mean"])
            ax.imshow(img.clip(0, 1)); ax.imshow(cam, cmap="jet", alpha=0.35)
            ax.set_title(f"{names[classes[labels[i]]]}\nphoto {path.stem}", fontsize=8)
            if manifest.get(path.stem):
                credits[path.stem] = manifest[path.stem]
        fig.suptitle("Grad-CAM: where the model looked"); fig.tight_layout()
        fig.savefig(REPORTS / "gradcam.png", dpi=110); plt.close(fig)

    # Confidence vs accuracy
    curve = coverage_curve(probs, labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c["answered"] for c in curve], [c["accuracy"] for c in curve], "o-", color="#2f6b3a")
    for c in curve:
        ax.annotate(f"≥{c['threshold']:.0%}", (c["answered"], c["accuracy"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("Share of photos the model answers"); ax.set_ylabel("Accuracy on those photos")
    ax.set_title("Trading coverage for accuracy with a confidence threshold"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "confidence.png", dpi=130); plt.close(fig)

    lines = ["# Image credits", "", "Photos shown in the report figures, from iNaturalist:", ""]
    for pid, r in sorted(credits.items()):
        lines.append(f"- Photo {pid}: {r['attribution']} — [{r['obs_url']}]({r['obs_url']})")
    (REPORTS / "image_credits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["convnext_ft"])
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species = load_species()
    names = {s["key"]: s["common_name"] for s in species}
    with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
        manifest = {r["photo_id"]: r for r in csv.DictReader(f)}
    results = {}
    for run in a.runs:
        model, cfg = load_run(run, device)
        classes = cfg["classes"]
        _, t_test = build_transforms(cfg["img_size"], cfg["mean"], cfg["std"])
        res = {}
        for split in ("val", "test"):
            items = list_split(a.images, split, classes)
            dl = DataLoader(ImageList(items, t_test), batch_size=64, num_workers=2)
            probs, labels = predict(model, dl, device, device.type == "cuda")
            res[split] = metrics(probs, labels, len(classes))
        results[run] = res
        print(f"{run:16s} val acc {res['val']['accuracy']:.3f}  test acc {res['test']['accuracy']:.3f}  "
              f"test top-3 {res['test']['top3_accuracy']:.3f}  test macro-F1 {res['test']['macro_f1']:.3f}")
    # Detailed outputs for the last (main) run
    rows = per_class(probs, labels, classes)
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    save_json({"runs": results, "main_run": a.runs[-1], "coverage": coverage_curve(probs, labels)}, REPORTS / "metrics.json")
    if not a.no_figures:
        figures(a.runs[-1], model, cfg, items, probs, labels, classes, names, device, manifest)
    print(f"Reports written to {REPORTS}")


if __name__ == "__main__":
    main()
