# Smokies Wildlife ID: a 20-species image classifier for Great Smoky Mountains National Park

Upload a photo and get the top-3 species, a confidence-based "not sure" answer, and a Grad-CAM heatmap of what the model looked at. Built with PyTorch transfer learning on research-grade [iNaturalist](https://www.inaturalist.org) observations.

**Status:** data pipeline and code complete; training results and the live demo link are added after the first Colab run.

## Why the test set is the interesting part

Most image-classifier demos test on a random slice of the same photos they trained on. This one doesn't:

| Split | Where the photos come from | Purpose |
|---|---|---|
| Train | Tennessee and North Carolina, **outside** the park | learn the species |
| Validation | same region, but a separate 15% of **photographers** | pick the best epoch without leaking one person's camera or backyard into both sides |
| Test | **inside** Great Smoky Mountains National Park | the photos the app will actually see |

Photographers who appear in the park test set are also removed from train and validation, and near-duplicate photos (re-uploads, crops) are removed across splits with a perceptual hash.

## Species

American black bear, white-tailed deer, elk, coyote, American red squirrel, eastern gray squirrel, groundhog, North American river otter, wild turkey, pileated woodpecker, barred owl, dark-eyed junco, timber rattlesnake, eastern copperhead, common watersnake, common box turtle, eastern fence lizard, red-cheeked salamander (found only in the Smokies), eastern newt and American toad. Chosen from the most-observed animals in the park with enough openly licensed photos in the region (`data/species.csv` has the counts).

## Method

1. `src/build_manifest.py`: queries the iNaturalist API for research-grade observations with Creative Commons photos, assigns splits, and records each photo's license and attribution in `data/manifest.csv`.
2. `src/download.py`: downloads the photos, drops unreadable or tiny ones, and removes near-duplicates (64-bit difference hash, ≤4 bits apart).
3. `src/train.py`: ConvNeXt-Tiny (ImageNet-22k pretrained, via `timm`). A linear-probe baseline (frozen backbone) and a full fine-tune with AdamW, cosine schedule with warm-up, label smoothing, TrivialAugment, random erasing and class-balanced sampling. Best epoch by validation macro-F1.
4. `src/evaluate.py`: accuracy, top-3 accuracy, macro-F1, calibration error, per-species precision and recall, confusion matrix, the most confident mistakes, a coverage-vs-accuracy curve for the "not sure" threshold, and Grad-CAM.
5. `app/`: Gradio demo for Hugging Face Spaces (CPU).

## Run it

```powershell
.\run.ps1                                  # tests + build the photo list (PC)
```
Then open `notebooks/train_colab.ipynb` in Google Colab with a T4 GPU and **Run all** (download, train, evaluate). `python src/package_space.py --upload <hf-user>` publishes the demo.

Tests: `tests/test_data.py` (splits, licenses, de-duplication; no network) and `tests/test_model.py` (training, metrics and Grad-CAM on synthetic images, CPU).

## Data and licenses

Photos © their iNaturalist observers, used under the Creative Commons license recorded for each photo in `data/manifest.csv` (most are CC BY-NC). Photos are not redistributed in this repo; the scripts download them from iNaturalist's open-data bucket. Photos shown in report figures are credited in `reports/image_credits.md`. The demo is non-commercial.

---
Josh · Seymour, TN
