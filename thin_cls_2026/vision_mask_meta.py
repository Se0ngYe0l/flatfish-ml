import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from models_1205 import ImageEncoder, MaskEncoder, MetaEncoder, FusionClassifier
from dataset import Flatfish_dataset_v3

parser = argparse.ArgumentParser()
parser.add_argument("--meta", type=str, default="6", choices=["5", "8", "6"],
                    help="meta feature set: 5 / 8 / 6(default)")
parser.add_argument("--data-root", default=os.environ.get("FLATFISH_DATA_ROOT", "./data"),
                    help="dataset root directory (env: FLATFISH_DATA_ROOT)")
parser.add_argument("--output-dir", default=None,
                    help="where fold checkpoints are written (default: ./results_vision_mask_meta_<meta>meta)")
parser.add_argument("--start-fold", type=int, default=1,
                    help="resume from this fold (1-5); earlier folds are skipped")
parser.add_argument("--device", default=os.environ.get("FLATFISH_DEVICE", "cuda:0"),
                    help="torch device (env: FLATFISH_DEVICE)")
args = parser.parse_args()

META_SETS = {
    "5": ['height', 'width', 'pixel', 'wdh', 'hdw'],
    "8": ['height', 'width', 'pixel', 'wdh', 'hdw', 'slope1', 'slope2', 'curvation'],
    "6": ['pixel', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2']
}

META_COLS = META_SETS[args.meta]
NUM_META = len(META_COLS)

device = args.device
root = args.data_root
output_path = args.output_dir or f"results_vision_mask_meta_{args.meta}meta"
os.makedirs(output_path, exist_ok=True)

criterion = torch.nn.CrossEntropyLoss()


def build_models():
    img_enc = ImageEncoder().to(device)
    mask_enc = MaskEncoder().to(device)
    meta_enc = MetaEncoder(in_dim=NUM_META).to(device)
    clf = FusionClassifier().to(device)
    return img_enc, mask_enc, meta_enc, clf


def evaluate(loader, img_enc, mask_enc, meta_enc, clf):
    clf.eval(); img_enc.eval(); mask_enc.eval(); meta_enc.eval()
    correct = total = loss_total = 0
    with torch.no_grad():
        for rgb_imgs, mask_imgs, labels, metas in loader:
            rgb_imgs, mask_imgs, labels, metas = (
                rgb_imgs.to(device), mask_imgs.to(device), labels.to(device), metas.to(device)
            )
            img_feat = img_enc(rgb_imgs)
            mask_feat = mask_enc(mask_imgs)
            meta_feat = meta_enc(metas)
            logits = clf(img_feat, mask_feat, meta_feat)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item() * labels.size(0)
    return correct / total, loss_total / total


full_ds = Flatfish_dataset_v3(root, "total", meta_cols=META_COLS)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(full_ds)):
    if fold + 1 < args.start_fold:
        continue
    print(f"\n========== Fold {fold+1}/5 ==========")

    train_subset = Subset(full_ds, train_idx)
    val_subset = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    img_enc, mask_enc, meta_enc, clf = build_models()

    opt = torch.optim.Adam(
        list(clf.parameters()) + list(mask_enc.parameters()) +
        list(img_enc.parameters()) + list(meta_enc.parameters()),
        # lr=1e-3
        lr=5e-4
    )

    best_acc = 0.0
    best_loss = float("inf")

    # pbar = tqdm(range(100), desc=f"Fold {fold+1}/5")
    pbar = tqdm(range(150), desc=f"Fold {fold+1}/5")

    for ep in pbar:
        clf.train(); img_enc.train(); mask_enc.train(); meta_enc.train()
        train_correct = train_total = train_loss_total = 0

        for rgb_imgs, mask_imgs, labels, metas in train_loader:
            rgb_imgs, mask_imgs, labels, metas = (
                rgb_imgs.to(device), mask_imgs.to(device), labels.to(device), metas.to(device)
            )
            img_feat = img_enc(rgb_imgs)
            mask_feat = mask_enc(mask_imgs)
            meta_feat = meta_enc(metas)
            logits = clf(img_feat, mask_feat, meta_feat)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_total += loss.item() * labels.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss_total / train_total

        val_acc, val_loss = evaluate(val_loader, img_enc, mask_enc, meta_enc, clf)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_loss = val_loss
            torch.save({
                "fold": fold + 1,
                "epoch": ep + 1,
                "img": img_enc.state_dict(),
                "mask": mask_enc.state_dict(),
                "meta": meta_enc.state_dict(),
                "clf": clf.state_dict(),
                "best_acc": best_acc,
                "best_loss": best_loss
            }, f"{output_path}/best_fold{fold+1}.pth")

        pbar.set_postfix({
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "best_acc": f"{best_acc:.4f}"
        })

    fold_results.append(best_acc)
    print(f"Fold {fold+1} Best Acc: {best_acc:.4f}")

print("\n========== K-Fold Result ==========")
print("Fold Accs:", fold_results)
print("Mean Acc:", sum(fold_results) / len(fold_results))
