import torch
import numpy as np
from torch.utils.data import DataLoader
from models_1205 import ImageEncoder, MaskEncoder, MetaEncoder, FusionClassifier
from dataset import Flatfish_dataset_v3
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import os
import argparse

# -----------------------------
# 1. Config
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--meta", type=str, default="6", choices=["5", "8", "6"],
                    help="meta feature set the checkpoint was TRAINED with — must match, "
                         "or the model is fed the wrong columns")
parser.add_argument("--data-root", default=os.environ.get("FLATFISH_DATA_ROOT", "./data"),
                    help="dataset root directory (env: FLATFISH_DATA_ROOT)")
parser.add_argument("--ckpt", default=None,
                    help="checkpoint to evaluate (default: ./results_vision_mask_meta_<meta>meta/best_fold1.pth)")
parser.add_argument("--save-dir", default=None,
                    help="where confusion-matrix figures are written "
                         "(default: ./results_vision_mask_meta_<meta>meta)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device", default=os.environ.get("FLATFISH_DEVICE"),
                    help="torch device (env: FLATFISH_DEVICE); defaults to cuda:0 when available")
args = parser.parse_args()

# Must stay in sync with vision_mask_meta.py — the checkpoint was trained on one
# of these column sets, and feeding a different one silently degrades the scores.
META_SETS = {
    "5": ['height', 'width', 'pixel', 'wdh', 'hdw'],
    "8": ['height', 'width', 'pixel', 'wdh', 'hdw', 'slope1', 'slope2', 'curvation'],
    "6": ['pixel', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2'],
}
META_COLS = META_SETS[args.meta]
NUM_META = len(META_COLS)

_default_dir = f"./results_vision_mask_meta_{args.meta}meta"

device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
root = args.data_root
ckpt_path = args.ckpt or f"{_default_dir}/best_fold1.pth"
batch_size = args.batch_size
class_names = ["normal", "warning", "severe"]  # 레이블 순서가 0/1/2라고 가정
SAVE_DIR = args.save_dir or _default_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# normal=0, warning=1, severe=2 라는 가정
NORMAL_LABEL = 0

# -----------------------------
# 2. Dataset & DataLoader
# -----------------------------
# Stage2 때 사용했던 validation 인덱스를 쓰려면 아래 주석 해제
# from torch.utils.data import Subset
# val_idx_path = "checkpoints/stage2_val_indices.npy"
# if os.path.exists(val_idx_path):
#     full_train = Flatfish_dataset(root, "train")
#     eval_dataset = Subset(full_train, np.load(val_idx_path))
#     print(f"✅ Loaded Stage2 validation subset: {len(eval_dataset)} samples")
# else:
#     eval_dataset = Flatfish_dataset(root, "test")

eval_dataset = Flatfish_dataset_v3(root, "test", meta_cols=META_COLS)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 3. Model Load
# -----------------------------
print("📦 Loading model checkpoint...")
ckpt = torch.load(ckpt_path, map_location=device)

img_enc = ImageEncoder().to(device)
mask_enc=  MaskEncoder().to(device)
meta_enc = MetaEncoder(in_dim=NUM_META).to(device)
clf = FusionClassifier().to(device)

img_enc.load_state_dict(ckpt["img"])
mask_enc.load_state_dict(ckpt["mask"])
meta_enc.load_state_dict(ckpt["meta"])
clf.load_state_dict(ckpt["clf"])
img_enc.eval(); mask_enc.eval(); meta_enc.eval(); clf.eval()
print("✅ Model loaded successfully.")

# -----------------------------
# 4. Evaluation (확률까지 수집)
# -----------------------------
@torch.no_grad()
def collect_predictions(loader):
    """
    Returns:
        y_true:      (N,) int
        y_pred:      (N,) int
        y_score_pos: (N,) float — P(warning|severe)
    """
    y_true, y_pred, pos_scores = [], [], []

    for rgb_imgs, mask_imgs, labels, metas in loader:
        rgb_imgs, mask_imgs, labels, metas = rgb_imgs.to(device), mask_imgs.to(device), labels.to(device), metas.to(device)
        img_feat = img_enc(rgb_imgs)
        mask_feat = mask_enc(mask_imgs)
        meta_feat = meta_enc(metas)
        logits = clf(img_feat, mask_feat, meta_feat)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)  # (B,3)
        # 클래스 순서가 ["normal","warning","severe"] 라는 가정
        pos_prob = probs[:, 1] + probs[:, 2]

        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())
        pos_scores.append(pos_prob.detach().cpu().numpy())

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(pos_scores),
    )

y_true, y_pred, y_score_pos = collect_predictions(eval_loader)

# -----------------------------
# 5. Multi-class (3-class) Metrics
# -----------------------------
acc = accuracy_score(y_true, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print("\n=== 📊 Evaluation Results (3-class) ===")
print(f"Accuracy:              {acc:.4f}")
print(f"Precision (macro):     {prec_m:.4f}")
print(f"Recall (macro):        {rec_m:.4f}")
print(f"F1-score (macro):      {f1_m:.4f}")
print(f"Precision (weighted):  {prec_w:.4f}")
print(f"Recall (weighted):     {rec_w:.4f}")
print(f"F1-score (weighted):   {f1_w:.4f}\n")

try:
    print("=== Per-Class Report (3-class) ===")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4, zero_division=0
    ))
except Exception:
    # target_names 길이 불일치 등 안전 가드
    print("=== Per-Class Report (3-class) ===")
    print(classification_report(
        y_true, y_pred,
        digits=4, zero_division=0
    ))

# -----------------------------
# 5-2. Binary Metrics (normal vs warning|severe)
# -----------------------------
def to_binary(y: np.ndarray) -> np.ndarray:
    # normal(0) -> 0, warning|severe(1|2) -> 1
    return (y != NORMAL_LABEL).astype(int)

y_true_bin = to_binary(y_true)
y_pred_bin = to_binary(y_pred)

acc_b = accuracy_score(y_true_bin, y_pred_bin)
prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
    y_true_bin, y_pred_bin, average="binary", zero_division=0
)
prec_macro_b, rec_macro_b, f1_macro_b, _ = precision_recall_fscore_support(
    y_true_bin, y_pred_bin, average="macro", zero_division=0
)

try:
    auc_b = roc_auc_score(y_true_bin, y_score_pos)
except Exception:
    auc_b = float("nan")

print("=== ✅ Binary Results (normal vs warning|severe) ===")
print(f"Binary Accuracy:       {acc_b:.4f}")
print(f"Binary Precision:      {prec_b:.4f}")
print(f"Binary Recall:         {rec_b:.4f}")
print(f"Binary F1:             {f1_b:.4f}")
print(f"Macro Precision:       {prec_macro_b:.4f}")
print(f"Macro Recall:          {rec_macro_b:.4f}")
print(f"Macro F1:              {f1_macro_b:.4f}")
print(f"ROC-AUC (binary):      {auc_b:.4f}\n")

print("=== Classification Report (Binary) ===")
print(classification_report(
    y_true_bin, y_pred_bin,
    target_names=["normal (0)", "warn|sev (1)"],
    digits=4, zero_division=0
))

# -----------------------------
# 6. Confusion Matrix (3-class)
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
im = ax.imshow(cm, cmap="Blues")

ax.set_title("Confusion Matrix (3-class)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=16,              # ← 여기만 추가
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
save_path_mc = os.path.join(SAVE_DIR, "confusion_matrix_3class.png")
plt.savefig(save_path_mc, bbox_inches="tight")
plt.close()
print(f"✅ Confusion matrix (3-class) saved to {save_path_mc}")

# -----------------------------
# 6-2. Confusion Matrix (Binary)
# -----------------------------
cm_b = confusion_matrix(y_true_bin, y_pred_bin)
fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
im = ax.imshow(cm_b, cmap="Greens")

ax.set_title("Confusion Matrix (2-class)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ticks_b = ["normal (0)", "warn|sev (1)"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(ticks_b, )
ax.set_yticklabels(ticks_b)

thresh_b = cm_b.max() / 2.0 if cm_b.max() > 0 else 0.5
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_b[i, j]),
                ha="center", va="center",
                fontsize=16,              # ← 여기만 추가
                color="white" if cm_b[i, j] > thresh_b else "black")

plt.tight_layout()
save_path_b = os.path.join(SAVE_DIR, "confusion_matrix_binary.png")
plt.savefig(save_path_b, bbox_inches="tight")
plt.close()
print(f"✅ Confusion matrix (Binary) saved to {save_path_b}")
