"""Smokies Wildlife ID: upload a photo, get the top-3 species and a heatmap of what the model looked at.

Runs on a free Hugging Face Space (CPU). Expects model.pt, config.json and species.csv next to this file.
"""
import csv
import json
import pathlib

import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

HERE = pathlib.Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
with open(HERE / "species.csv", newline="", encoding="utf-8") as f:
    SPECIES = {r["key"]: r for r in csv.DictReader(f)}
THRESHOLD = float(CFG.get("answer_threshold", 0.5))
VENOMOUS = {"timber_rattlesnake", "eastern_copperhead"}

model = timm.create_model(CFG["model"], pretrained=False, num_classes=len(CFG["classes"]))
state = torch.load(HERE / "model.pt", map_location="cpu")
model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
model.eval()
prep = T.Compose([T.Resize(int(CFG["img_size"] * 256 / 224)), T.CenterCrop(CFG["img_size"]),
                  T.ToTensor(), T.Normalize(CFG["mean"], CFG["std"])])


def grad_cam(x):
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    top = logits.argmax(1)
    model.zero_grad()
    logits[0, top[0]].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1))[0]
    cam = cam / cam.max().clamp(min=1e-8)
    return cam.detach().numpy(), torch.softmax(logits.detach(), 1)[0].numpy()


def overlay(img, cam):
    import matplotlib
    heat = matplotlib.colormaps["jet"](np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)) / 255.0)[..., :3]
    base = np.asarray(img, dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8((0.62 * base + 0.38 * heat) * 255))


def identify(img):
    if img is None:
        return None, None, "Upload a photo to start."
    img = img.convert("RGB")
    x = prep(img).unsqueeze(0)
    cam, probs = grad_cam(x)
    order = np.argsort(-probs)
    labels = {SPECIES[CFG["classes"][i]]["common_name"]: float(probs[i]) for i in order[:3]}
    top = CFG["classes"][order[0]]
    heat = overlay(prep_view(img), cam)
    if probs[order[0]] < THRESHOLD:
        note = (f"**Not sure.** The best guess is {SPECIES[top]['common_name']} at {probs[order[0]]:.0%}, below the "
                f"{THRESHOLD:.0%} confidence this app needs to answer. It only knows the {len(CFG['classes'])} species listed below.")
    else:
        sp = SPECIES[top]
        note = f"**{sp['common_name']}** (*{sp['scientific_name']}*), {sp['group'].lower()}."
    if top in VENOMOUS and probs[order[0]] >= 0.2:
        note += "\n\n⚠️ Possibly venomous. Never handle or approach a snake you can't identify with certainty."
    return labels, heat, note


def prep_view(img):
    """The exact square crop the model sees, for drawing the heatmap on."""
    s = CFG["img_size"]
    v = T.Compose([T.Resize(int(s * 256 / 224)), T.CenterCrop(s)])(img)
    return v


species_list = ", ".join(sorted(r["common_name"] for r in SPECIES.values()))
with gr.Blocks(title="Smokies Wildlife ID") as demo:
    gr.Markdown("# Smokies Wildlife ID\nIdentify 20 common animals of Great Smoky Mountains National Park from a photo. "
                "Trained on iNaturalist photos from Tennessee and North Carolina and tested on photos taken inside the park.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Your photo")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Top 3")
            out_note = gr.Markdown()
            out_cam = gr.Image(label="Where the model looked (Grad-CAM)")
    inp.change(identify, inp, [out_label, out_cam, out_note])
    gr.Markdown(f"**Species it knows:** {species_list}.\n\nFor fun and learning, not for safety decisions. "
                "Model and data: see the project README; training photos © their iNaturalist observers, Creative Commons licensed.")

if __name__ == "__main__":
    demo.launch()
