"""Shared pieces: species list, datasets and transforms."""
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load_species(path=DATA / "species.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def class_names():
    return [s["key"] for s in load_species()]


def list_split(root, split, classes):
    """(path, label) for every image of a split, in a stable order."""
    items = []
    for label, key in enumerate(classes):
        for p in sorted((pathlib.Path(root) / split / key).glob("*.jpg")):
            items.append((p, label))
    return items


def build_transforms(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    from torchvision import transforms as T
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.35, 1.0)),   # wildlife is often small in the frame
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.2),
    ])
    test = T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, test


class ImageList:
    """Minimal Dataset over (path, label) pairs."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


def save_json(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
