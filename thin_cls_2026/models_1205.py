import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Image Encoder (CNN)
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,7,2,3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1,1),

            nn.Conv2d(32, 64, 3,1,1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3,1,1, bias=False),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)     # (B,64)
        feat = self.proj(feat)               # (B,128)
        feat = F.normalize(feat, dim=1)      # contrastive normalize
        return feat


class MaskEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,7,2,3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1,1),

            nn.Conv2d(32, 64, 3,1,1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3,1,1, bias=False),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)     # (B,64)
        feat = self.proj(feat)               # (B,128)
        feat = F.normalize(feat, dim=1)      # contrastive normalize
        return feat


# -------------------------
# Metadata Encoder (MLP)
# -------------------------
class MetaEncoder(nn.Module):
    def __init__(self, in_dim=6, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        feat = self.mlp(x)
        feat = self.proj(feat)              # (B,128)
        feat = F.normalize(feat, dim=1)
        return feat


# -------------------------
# Fusion Classifier
# -------------------------
class FusionClassifier(nn.Module):
    def __init__(self, emb_dim=256, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_feat, mask_feat, meta_feat):
        fused = torch.cat([img_feat, mask_feat, meta_feat], dim=1)
        return self.fc(fused)
