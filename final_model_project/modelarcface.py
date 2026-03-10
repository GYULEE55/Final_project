# ===============================================================
# 🧠 model_fsfm_arcface_v2.py
# 🎭 FSFM Backbone + ArcFace Loss
# ===============================================================

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

FSFM_REPO = "/workspace/FSFM-CVPR25"
sys.path.append(os.path.join(FSFM_REPO, "fsfm-3c"))
from models_fsfm import vit_target_network

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        with amp.autocast(enabled=False):
            x = F.normalize(x.float(), dim=1)
            W = F.normalize(self.weight.float(), dim=1)
            cosine = F.linear(x, W)
            sine = torch.sqrt(1.0 - cosine**2)
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            return (one_hot * phi + (1 - one_hot) * cosine) * self.s


class FSFM_ArcFace(nn.Module):
    def __init__(self, model_name="fsfm_vit_base_patch16", embed_dim=512):
        super().__init__()
        self.backbone = vit_target_network(model_name)
        self.head = nn.LazyLinear(embed_dim, bias=False)
        self.arcface = ArcMarginProduct(embed_dim, 2)

    def forward(self, x, label=None):
        B = x.size(0)
        L = (224 // 16) ** 2
        masks = torch.zeros((B, L), device=x.device, dtype=torch.int64)
        feats = self.backbone(x, masks, masks, mask_ratio=0.75)
        feats = feats.mean(dim=1)
        emb = F.normalize(self.head(feats), dim=-1)
        if label is not None:
            return self.arcface(emb, label), emb
        return emb
