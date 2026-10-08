# flatfish-ml

Training and evaluation code for three flatfish (넙치) vision models:

| Directory | Task | Entry points |
|---|---|---|
| [`thin_cls_2026/`](thin_cls_2026/) | 3-class thinness grading (`normal` / `warning` / `severe`) from RGB + segmentation mask + tabular meta features | [`vision_mask_meta.py`](thin_cls_2026/vision_mask_meta.py), [`vision_mask_meta_eval.py`](thin_cls_2026/vision_mask_meta_eval.py) |
| [`angle_cls/`](angle_cls/) | Binary pose classification (`normal` / `abnormal` shooting angle), ResNet-18 | [`train.ipynb`](angle_cls/train.ipynb), [`train_transfer.ipynb`](angle_cls/train_transfer.ipynb) |
| [`yolo_detection/`](yolo_detection/) | Single-class fish detection, YOLOv8n / YOLO11n via Ultralytics | [`run.ipynb`](yolo_detection/run.ipynb) |

Model weights, datasets and run artifacts are **not** in this repo — see
[Configuration](#configuration) for how to point the code at your own.

## Setup

```bash
git clone <this repo> && cd flatfish-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the experiments were actually run with
(torch 1.10.1 / CUDA 11.1). Newer versions generally work; `ultralytics`,
`opencv-python`, `pycocotools` and `scipy` are only needed for
`yolo_detection/`.

## Configuration

No paths are hardcoded. Every script takes CLI flags, and the data root also
reads from an environment variable so you can set it once:

```bash
export FLATFISH_DATA_ROOT=/path/to/your/dataset
export FLATFISH_DEVICE=cuda:0        # optional, defaults to cuda:0
```

| Variable | Used by | Default |
|---|---|---|
| `FLATFISH_DATA_ROOT` | all | `./data` |
| `FLATFISH_DEVICE` | `thin_cls_2026/` | `cuda:0` (eval falls back to CPU) |
| `FLATFISH_CKPT_DIR` | `angle_cls/` notebooks | `./ckpt` |
| `FLATFISH_DATASET_ROOT` | `yolo_detection/run.ipynb` | `./data/det_seg_dataset` |
| `FLATFISH_YOLO_DATA_YAML` | `yolo_detection/run.ipynb` | `$FLATFISH_DATASET_ROOT/fishdataset.yaml` |
| `FLATFISH_YOLO_WEIGHTS` | `yolo_detection/run.ipynb` | `./fishdetect_result/train/weights/best.pt` |
| `FLATFISH_SAMPLE_IMAGE` | `yolo_detection/run.ipynb` | `$FLATFISH_DATA_ROOT/sample.jpg` |
| `FLATFISH_MMDET_RESULT_DIR` | `yolo_detection/run.ipynb` | `./mmdet_result` |

Notebooks read these in their first cell; edit that cell directly if you prefer.

---

## thin_cls_2026 — thinness grading

### Expected data layout

```
$FLATFISH_DATA_ROOT/
├── total.csv                # training split (5-fold CV is done over this file)
├── test.csv                 # held-out split, used by the *_eval.py scripts
├── images/<name>.{jpg,png,jpeg}
└── segment_images_v2/<name> # binary segmentation mask, same basename
```

Both CSVs need a `name` column, an integer `target` column (`0` normal,
`1` warning, `2` severe), and the meta feature columns. `--meta` selects which
feature set is fed to the model:

| `--meta` | Columns |
|---|---|
| `5` | `height, width, pixel, wdh, hdw` |
| `8` | the 5 above + `slope1, slope2, curvation` |
| `6` *(default)* | `pixel, slope2, curvation, slope3, slope4, curvation2` |

RGB images are resized to 900×1200 and ImageNet-normalized; masks are used at
native resolution. See [`dataset.py`](thin_cls_2026/dataset.py).

### Model

[`vision_mask_meta.py`](thin_cls_2026/vision_mask_meta.py) encodes each
modality separately, concatenates the three feature vectors, and classifies:

```
RGB  (3×900×1200) ──> ImageEncoder ──┐
mask (1×900×1200) ──> MaskEncoder  ──┼─> concat ─> FusionClassifier ─> 3 logits
meta (n_meta,)    ──> MetaEncoder  ──┘
```

All four modules live in [`models_1205.py`](thin_cls_2026/models_1205.py)
(~862K parameters with `--meta 6`). Other fusion strategies and the
single-/dual-modality ablations are not part of this repo.

### Training

5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`) over
`total.csv`, writing `best_fold{1..5}.pth` — the epoch with the best validation
accuracy for each fold.

```bash
cd thin_cls_2026

# output dir defaults to ./results_vision_mask_meta_6meta
python vision_mask_meta.py --meta 6

# resume a partially finished run (e.g. folds 1-3 already done)
python vision_mask_meta.py --meta 6 --start-fold 4
```

Run `python vision_mask_meta.py --help` for the full flag list.

### Evaluation

[`vision_mask_meta_eval.py`](thin_cls_2026/vision_mask_meta_eval.py) is
standalone — it scores one checkpoint on `test.csv` and prints 3-class and
binary (`normal` vs `warning|severe`) metrics plus ROC-AUC, then saves
`confusion_matrix_3class.png` and `confusion_matrix_binary.png`:

```bash
# defaults to ./results_vision_mask_meta_6meta/best_fold1.pth
python vision_mask_meta_eval.py --meta 6

# a specific fold, figures elsewhere
python vision_mask_meta_eval.py --meta 6 \
    --ckpt ./results_vision_mask_meta_6meta/best_fold3.pth \
    --save-dir ./eval_out/fold3
```

> **`--meta` must match what the checkpoint was trained with.** The encoder's
> input width and the dataset columns are both derived from it, and `5`/`8`/`6`
> are different column sets — not nested — so a mismatch either fails to load
> the state dict or silently feeds the model the wrong features.

To get fold-level spread, run it once per `best_fold{N}.pth`. Note this scores
the held-out `test.csv`, which is not the same as the per-fold validation
metrics the training run reports.

---

## angle_cls — pose classification

ResNet-18 (ImageNet weights via `torch.hub`) with a 2-way head.

### Expected data layout

```
$FLATFISH_DATA_ROOT/angle_cls/
├── train/{normal,abnormal}/*.jpg
└── test/{normal,abnormal}/*.jpg
```

Labels come from the directory name — `normal` → 0, `abnormal` → 1. Images are
resized to 256×256, no normalization. See [`dataset.py`](angle_cls/dataset.py).

### Notebooks

- **[`train.ipynb`](angle_cls/train.ipynb)** — trains from ImageNet init
  (20 epochs, AdamW, lr 3e-3), then loads a checkpoint and reports
  `classification_report` / `confusion_matrix` on the test split.
- **[`train_transfer.ipynb`](angle_cls/train_transfer.ipynb)** — same loop, but
  fine-tunes from a previously trained `angle_cls.pth`
  (30 epochs, lr 7e-3).

Both save one checkpoint per epoch to `$FLATFISH_CKPT_DIR/{epoch}.pth`. Run the
config cell first, then the cells top to bottom; the training cell and the
eval cell are independent, so you can skip straight to eval if you already have
a checkpoint.

---

## yolo_detection — fish detection

Ultralytics YOLO, single class (`fish`).

[`run.ipynb`](yolo_detection/run.ipynb) covers, in order:

1. Train `yolov8n.pt` on `$FLATFISH_YOLO_DATA_YAML` (20 epochs, batch 2).
2. `model.val()` on the same yaml.
3. Rename the class to `fish` and re-export the checkpoint.
4. Single-image prediction smoke test.
5. Mean-IoU against YOLO-format ground truth labels, computed manually with
   `torchvision.ops.box_iou`.
6. Baseline comparison against mmdetection Faster R-CNN / SOLOv2 `.pkl` dumps
   (requires `$FLATFISH_MMDET_RESULT_DIR`; the mmdetection configs themselves
   are not in this repo).
7. A simple L1-distance pose evaluation helper.
8. Dataset housekeeping cells — train/val label reshuffling driven by a CSV.

Cells 6–8 are ad-hoc analysis rather than part of the detection pipeline; the
baseline and housekeeping cells will not run without the corresponding external
files.

[`coco2yolo.py`](yolo_detection/coco2yolo.py) converts a COCO
`instances_*.json` into YOLO `.txt` labels:

```bash
python coco2yolo.py -j /path/to/instances_train2017.json -o /path/to/labels/train
```

---

## Notes

- Notebook outputs are stripped before committing, to keep diffs readable. If
  you use `nbstripout`, `nbstripout --install` in this repo will keep it that way.
- `.gitignore` excludes `*.pth`, `*.pt`, `results_*/`, `runs/` and dataset
  files. Keep weights out of git history — they are hundreds of MB each.
