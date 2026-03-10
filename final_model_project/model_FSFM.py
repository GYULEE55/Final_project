# ===============================================================
# 🧠 model_fsfm_arcface.py
# 🔥 FSFM Backbone + ArcFace Head
#  - train: forward(x, label=y) → (logits, emb)
#  - eval:  forward(x, label=None) → emb (512-D)
# ===============================================================

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

# FSFM backbone import
FSFM_CODE_DIR = "/workspace/FSFM-CVPR25/fsfm-3c"
if FSFM_CODE_DIR not in sys.path:
    sys.path.append(FSFM_CODE_DIR)

from FSFM_V5.models_fsfm import vit_target_network

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# 1) ArcFace Head (FP32 고정)
# ===============================================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        # 항상 FP32에서 ArcFace 계산
        with amp.autocast(enabled=False):
            x = F.normalize(x.float(), dim=1)
            W = F.normalize(self.weight.float(), dim=1)
            cosine = F.linear(x, W)  # (B, num_classes)

            # ----------------------
            # label=None → margin OFF (inference)
            # ----------------------
            if label is None:
                return cosine * self.s

            # ----------------------
            # label 존재 → margin ON (train)
            # ----------------------
            cosine_clamped = cosine.clamp(-1.0, 1.0)
            sine = torch.sqrt((1.0 - cosine_clamped ** 2).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)

            logits = (one_hot * phi + (1 - one_hot) * cosine) * self.s
            return logits


# ===============================================================
# 2) FSFM Backbone + ArcFace Head
# ===============================================================
class FSFM_ArcFace(nn.Module):

    def __init__(self, model_name="fsfm_vit_base_patch16",
                 embed_dim=512, num_classes=2, img_size=224):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size

        # FSFM backbone
        self.backbone = vit_target_network(model_name)

        # projection head → embedding
        self.head = nn.LazyLinear(embed_dim, bias=False)

        # ArcFace margin head
        self.arcface = ArcMarginProduct(embed_dim, num_classes, s=30.0, m=0.25)

    def _pool_features(self, feats: torch.Tensor) -> torch.Tensor:
        # FSFM output: (B, N, C)
        if feats.dim() == 3:
            return feats.mean(dim=1)  # (B, C)
        elif feats.dim() == 2:
            return feats
        else:
            raise RuntimeError(f"Unexpected FSFM feature shape: {feats.shape}")

    def forward(self, x, label=None):
        """
        train  : logits, emb = model(x, label=y)
        eval   : emb = model(x, label=None)
        """
        B = x.size(0)
        L = (self.img_size // 16) ** 2

        imgs_masks = torch.zeros((B, L), device=x.device, dtype=torch.int64)
        sfr_mask   = torch.zeros((B, L), device=x.device, dtype=torch.int64)

        # 1) FSFM backbone
        feats = self.backbone(x, imgs_masks, sfr_mask, mask_ratio=0.75)
        feats = self._pool_features(feats)               # (B, C)

        # 2) embedding
        emb = self.head(feats)                           # (B, embed_dim)
        emb = F.normalize(emb, dim=-1)                   # L2 normalize

        # ------------------------------
        # eval 모드: emb만 반환
        # ------------------------------
        if label is None:
            return emb

        # ------------------------------
        # train 모드: logits, emb 반환
        # ------------------------------
        logits = self.arcface(emb, label)
        return logits, emb

    # ----------------------------------------------------------
    # FSFM checkpoint loader
    # ----------------------------------------------------------
    def load_fsfm_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"[FSFM] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing:", missing[:5], "...")
        if len(unexpected) > 0:
            print("  unexpected:", unexpected[:5], "...")
