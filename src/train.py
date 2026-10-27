"""Fine-tune an ImageNet-pretrained model on the Smokies photos.

  python src/train.py --run convnext_ft                       # full fine-tune (default model)
  python src/train.py --run convnext_probe --linear-probe     # baseline: frozen backbone, train only the classifier
  python src/train.py --smoke                                 # tiny CPU run for testing the code

Saves models/<run>/model.pt (fp16 weights), config.json and history.json; the best epoch by
validation macro-F1 is kept.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import DATA, MODELS, ImageList, build_transforms, class_names, list_split, save_json

DEFAULT_MODEL = "convnext_tiny.fb_in22k_ft_in1k"


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def make_model(name, n_classes, pretrained=True):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)


def data_config(model):
    import timm
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"img_size": int(cfg["input_size"][-1]), "mean": list(cfg["mean"]), "std": list(cfg["std"])}


def param_groups(model, lr, head_lr_mult, weight_decay, linear_probe):
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    if linear_probe:
        for p in backbone:
            p.requires_grad = False
        return [{"params": list(head.parameters()), "lr": lr * head_lr_mult, "weight_decay": weight_decay}]
    return [{"params": backbone, "lr": lr, "weight_decay": weight_decay},
            {"params": list(head.parameters()), "lr": lr * head_lr_mult, "weight_decay": weight_decay}]


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_ft")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--linear-probe", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: resnet18, no pretrained weights, 1 epoch")
    a = ap.parse_args()
    if a.smoke:
        a.model, a.epochs, a.batch, a.workers = "resnet18", 1, 8, 0

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    classes = class_names()
    model = make_model(a.model, len(classes), pretrained=not a.smoke).to(device)
    cfg = data_config(model)
    if a.smoke:
        cfg["img_size"] = 64
    t_train, t_test = build_transforms(cfg["img_size"], cfg["mean"], cfg["std"])
    train_items = list_split(a.images, "train", classes)
    val_items = list_split(a.images, "val", classes)
    if not train_items or not val_items:
        raise SystemExit(f"No images under {a.images}/train or /val. Run src/download.py first.")
    print(f"{len(train_items)} train / {len(val_items)} val images, {len(classes)} classes, device {device}")

    # Balance classes by sampling, so a species with 500 photos doesn't drown one with 300.
    counts = np.bincount([y for _, y in train_items], minlength=len(classes))
    weights = 1.0 / counts[[y for _, y in train_items]]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
    train_dl = DataLoader(ImageList(train_items, t_train), batch_size=a.batch, sampler=sampler,
                          num_workers=a.workers, pin_memory=amp, drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(ImageList(val_items, t_test), batch_size=a.batch * 2, num_workers=a.workers, pin_memory=amp)

    opt = torch.optim.AdamW(param_groups(model, a.lr, a.head_lr_mult, a.weight_decay, a.linear_probe))
    total = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total, warmup=len(train_dl)))
    scaler = torch.amp.GradScaler(enabled=amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    out = MODELS / a.run
    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        if a.linear_probe:
            model.eval(); model.get_classifier().train()     # keep frozen batch-norm statistics fixed
        t, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item() * len(y); seen += len(y)
            if a.smoke and step >= 3:
                break
        probs, labels = predict(model, val_dl, device, amp)
        pred = probs.argmax(1)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_acc": float((pred == labels).mean()),
               "val_macro_f1": macro_f1(labels, pred, len(classes)), "seconds": round(time.time() - t, 1)}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['train_loss']:.3f}  val acc {row['val_acc']:.3f}  "
              f"macro-F1 {row['val_macro_f1']:.3f}  ({row['seconds']}s)", flush=True)
        if row["val_macro_f1"] > best:
            best = row["val_macro_f1"]
            out.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                       out / "model.pt")
    save_json({"model": a.model, "classes": classes, **cfg, "linear_probe": a.linear_probe,
               "best_val_macro_f1": best, "args": vars(a)}, out / "config.json")
    save_json(history, out / "history.json")
    print(f"Best val macro-F1 {best:.3f}; saved to {out}")


if __name__ == "__main__":
    main()
