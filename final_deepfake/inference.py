"""
FSFM + ArcFace 딥페이크 탐지 추론 모듈 (성능 최적화 버전)
- 배치 처리 추가
- 멀티스케일 분석
- 앙상블 예측
- GPU 메모리 최적화
"""

import os
import sys
import math
import warnings
from functools import partial, lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    from PIL import Image as PILImage
    PILImage.ANTIALIAS = getattr(PILImage, 'LANCZOS', 1)
except ImportError:
    raise ImportError("PIL is not available. Please install pillow: pip install pillow")

from torchvision import transforms
from torchvision.transforms import functional as TF

# FACER 툴킷 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'facer'))
try:
    import facer
except ImportError:
    print("Warning: FACER toolkit not available")
    facer = None

warnings.filterwarnings('ignore')

# Numpy 2.x 호환성
if hasattr(np, '__version__') and int(np.__version__.split('.')[0]) >= 2:
    np.float = float
    np.int = int


# ============================================================
# 1) Positional Embedding 유틸리티
# ============================================================
@lru_cache(maxsize=32)
def get_2d_sincos_pos_embed_cached(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """캐시된 2D sin-cos positional embedding"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ============================================================
# 2) 향상된 BYOL MLP with Dropout
# ============================================================
class EnhancedBYOLMLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, dropout_rate=0.1):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, projection_size),
            nn.LayerNorm(projection_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.projection_head(x)


# ============================================================
# 3) 효율적인 ViT Blocks with Flash Attention
# ============================================================
class EfficientAttention(nn.Module):
    """메모리 효율적인 어텐션 메커니즘"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Efficient attention computation
        with autocast(enabled=torch.cuda.is_available()):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EnhancedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EnhancedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path1 = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EnhancedMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=nn.GELU, drop=drop
        )
        self.drop_path2 = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ============================================================
# 4) 향상된 PatchEmbed with 멀티스케일
# ============================================================
class MultiScalePatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 멀티스케일 컨볼루션
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size // 2, stride=patch_size // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim, img_size // patch_size, img_size // patch_size])
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================================
# 5) 향상된 FSFM TargetNetworkViT
# ============================================================
class EnhancedTargetNetworkViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 projection_size=256, projection_hidden_size=4096,
                 rep_decoder_embed_dim=768, rep_decoder_depth=2, rep_decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()

        self.patch_embed = MultiScalePatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_axis = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]
        
        self.blocks = nn.ModuleList([
            EnhancedBlock(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Enhanced projector
        self.projector = EnhancedBYOLMLP(
            rep_decoder_embed_dim, projection_size, 
            projection_hidden_size, dropout_rate=drop_rate
        )

        # Decoder components
        self.mask_token = nn.Parameter(torch.zeros(1, 1, rep_decoder_embed_dim))
        self.rep_decoder_embed = nn.Linear(embed_dim, rep_decoder_embed_dim, bias=True)
        self.rep_decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, rep_decoder_embed_dim), 
            requires_grad=False
        )
        
        self.rep_decoder_blocks = nn.ModuleList([
            EnhancedBlock(
                rep_decoder_embed_dim, rep_decoder_num_heads, mlp_ratio, 
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            ) for i in range(rep_decoder_depth)
        ])
        
        self.rep_decoder_norm = norm_layer(rep_decoder_embed_dim)
        self.rep_decoder_pred = nn.Linear(rep_decoder_embed_dim, embed_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Use cached positional embeddings
        pos_embed = get_2d_sincos_pos_embed_cached(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** .5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        rep_decoder_pos_embed = get_2d_sincos_pos_embed_cached(
            self.rep_decoder_pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** .5), 
            cls_token=True
        )
        self.rep_decoder_pos_embed.data.copy_(torch.from_numpy(rep_decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, x_mask, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape
        ids_shuffle = torch.argsort(x_mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :int(L * (1 - mask_ratio))]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore

    def forward_rep_decoder(self, x, ids_restore):
        x = self.rep_decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.rep_decoder_pos_embed

        for blk in self.rep_decoder_blocks:
            x = blk(x)
        x = self.rep_decoder_norm(x)
        x = self.rep_decoder_pred(x)

        return x

    def forward(self, imgs, imgs_masks, specific_facial_region_mask, mask_ratio=0.75):
        with autocast(enabled=torch.cuda.is_available()):
            latent, ids_restore = self.forward_encoder(imgs, imgs_masks, mask_ratio)
            feat_all = self.forward_rep_decoder(latent, ids_restore)
            features_proj = self.projector(feat_all.mean(dim=1, keepdim=False))
            features_cl = F.normalize(features_proj, dim=-1)
        return features_cl


# ============================================================
# 6) 향상된 ArcFace with Adaptive Margin
# ============================================================
class AdaptiveArcMarginProduct(nn.Module):
    """적응형 마진을 가진 ArcFace"""
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        x = F.normalize(x.float(), dim=1)
        W = F.normalize(self.weight.float(), dim=1)
        
        with autocast(enabled=torch.cuda.is_available()):
            cosine = F.linear(x, W)
            
            if label is None:
                return cosine * self.s
            
            sine = torch.sqrt((1.0 - torch.clamp(cosine, -1+1e-7, 1-1e-7)**2))
            phi = cosine * self.cos_m - sine * self.sin_m
            
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            logits = (one_hot * phi + (1 - one_hot) * cosine) * self.s
            
        return logits


# ============================================================
# 7) 향상된 FSFM + ArcFace 모델
# ============================================================
class EnhancedFSFM_ArcFace(nn.Module):
    def __init__(self, model_name="fsfm_vit_base_patch16", embed_dim=512, 
                 num_classes=2, img_size=224, use_enhanced=True):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.use_enhanced = use_enhanced
        
        if use_enhanced:
            self.backbone = EnhancedTargetNetworkViT(
                img_size=img_size,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                rep_decoder_embed_dim=768,
                rep_decoder_depth=2,
                rep_decoder_num_heads=16,
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_rate=0.1,
                attn_drop_rate=0.1
            )
        else:
            # 기존 모델 사용 (이전 코드의 TargetNetworkViT)
            from inference import vit_target_network
            self.backbone = vit_target_network(model_name)
        
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.arcface = AdaptiveArcMarginProduct(
            embed_dim, num_classes, s=30.0, m=0.35, easy_margin=False
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _pool_features(self, feats):
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        elif feats.dim() != 2:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")
        return feats

    def forward(self, x, label=None):
        B = x.size(0)
        L = (self.img_size // 16) ** 2
        imgs_masks = torch.zeros((B, L), device=x.device, dtype=torch.int64)
        sfr_mask = torch.zeros((B, L), device=x.device, dtype=torch.int64)

        with autocast(enabled=torch.cuda.is_available()):
            feats = self.backbone(x, imgs_masks, sfr_mask, mask_ratio=0.75)
            feats = self._pool_features(feats)
            emb = F.normalize(self.head(feats), dim=-1)

            if label is not None:
                logits = self.arcface(emb, label)
                return logits / self.temperature, emb
            return emb


# ============================================================
# 8) 향상된 얼굴 탐지 및 전처리
# ============================================================
class FaceDetector:
    """통합 얼굴 탐지기 (DLIB + OpenCV Cascade)"""
    
    def __init__(self):
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        # OpenCV Haar Cascade fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cv_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, img: np.ndarray) -> List[List[int]]:
        """복수 얼굴 탐지"""
        faces = []
        
        # DLIB 시도
        dlib_faces = self.dlib_detector(img, 1)
        for face in dlib_faces:
            faces.append([face.left(), face.top(), face.right(), face.bottom()])
        
        # DLIB가 못 찾으면 OpenCV 시도
        if len(faces) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            cv_faces = self.cv_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in cv_faces:
                faces.append([x, y, x + w, y + h])
        
        return faces
    
    def get_best_face(self, faces: List[List[int]]) -> Optional[List[int]]:
        """가장 큰 얼굴 선택"""
        if not faces:
            return None
        return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


class ImagePreprocessor:
    """향상된 이미지 전처리"""
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.face_detector = FaceDetector()
        
        # 다양한 증강 변환
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=5),
        ])
        
        # 정규화
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def crop_face(self, img: np.ndarray, margin_scale: float = 1.3) -> Optional[np.ndarray]:
        """얼굴 영역 크롭"""
        faces = self.face_detector.detect_faces(img)
        face_box = self.face_detector.get_best_face(faces)
        
        if face_box is None:
            return None
        
        x1, y1, x2, y2 = face_box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # 마진 추가
        size = int(max(w, h) * margin_scale)
        x1_new = max(0, cx - size // 2)
        y1_new = max(0, cy - size // 2)
        x2_new = min(img.shape[1], cx + size // 2)
        y2_new = min(img.shape[0], cy + size // 2)
        
        cropped = img[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
        
        if cropped.size == 0:
            return None
        
        return cv2.resize(cropped, (self.target_size, self.target_size))
    
    def preprocess(self, image: Union[PILImage.Image, np.ndarray], 
                   use_face_detection: bool = True,
                   use_augmentation: bool = False) -> torch.Tensor:
        """전처리 파이프라인"""
        
        # numpy 변환
        if isinstance(image, PILImage.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # 얼굴 탐지 및 크롭
        if use_face_detection:
            face_crop = self.crop_face(img_np, margin_scale=1.3)
            if face_crop is not None:
                img_np = face_crop
        
        # PIL 변환
        img_pil = PILImage.fromarray(img_np)
        img_pil = img_pil.resize((self.target_size, self.target_size), PILImage.LANCZOS)
        
        # 증강 적용 (선택적)
        if use_augmentation:
            img_pil = self.augmentations(img_pil)
        
        # 정규화
        img_tensor = self.normalize(img_pil)
        
        return img_tensor


# ============================================================
# 9) 향상된 딥페이크 탐지기
# ============================================================
class EnhancedDeepfakeDetector:
    """성능 최적화된 FSFM + ArcFace 딥페이크 탐지기"""
    
    def __init__(self, model_dir: str, use_enhanced: bool = True, 
                 device: Optional[str] = None, enable_tta: bool = True):
        """
        Args:
            model_dir: 모델 디렉토리 경로
            use_enhanced: 향상된 모델 사용 여부
            device: 디바이스 ('cuda', 'cpu', None=auto)
            enable_tta: Test Time Augmentation 사용 여부
        """
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model_dir = model_dir
        self.img_size = 224
        self.use_enhanced = use_enhanced
        self.enable_tta = enable_tta
        
        # 모델 생성
        self.model = EnhancedFSFM_ArcFace(
            model_name="fsfm_vit_base_patch16",
            embed_dim=512,
            num_classes=2,
            img_size=self.img_size,
            use_enhanced=use_enhanced
        ).to(self.device)
        
        # 가중치 로드
        self._load_weights()
        
        self.model.eval()
        
        # 전처리기
        self.preprocessor = ImagePreprocessor(self.img_size)
        
        # 결과 캐시 (메모리 효율)
        self._cache = {}
        
        print(f"✓ Enhanced FSFM 모델 로드 완료")
        print(f"✓ Device: {self.device}")
        print(f"✓ Enhanced Mode: {use_enhanced}")
        print(f"✓ TTA Enabled: {enable_tta}")
    
    def _load_weights(self):
        """모델 가중치 로드"""
        model_file = os.path.join(self.model_dir, 'best.pth')
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file not found at {model_file}")
            return
        
        # Dummy forward for LazyLinear initialization
        dummy = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        
        # 가중치 로드
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # 호환성 체크
        model_dict = self.model.state_dict()
        pretrained_dict = {}
        
        for k, v in checkpoint.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✓ Loaded {len(pretrained_dict)}/{len(model_dict)} weights")
    
    @torch.no_grad()
    def predict_single(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """단일 이미지 예측"""
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with autocast(enabled=torch.cuda.is_available()):
            emb = self.model(img_tensor, label=None)
            W = F.normalize(self.model.arcface.weight.float(), dim=1)
            cosine = F.linear(emb, W)
            logits = cosine * self.model.arcface.s
            
            # Temperature scaling
            logits = logits / self.model.temperature
            
            probs = torch.softmax(logits, dim=1)
            
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        
        return {
            'real_probability': real_prob * 100,
            'fake_probability': fake_prob * 100
        }
    
    def predict_with_tta(self, image: PILImage.Image) -> Dict[str, float]:
        """Test Time Augmentation을 사용한 예측"""
        predictions = []
        
        # Original
        img_tensor = self.preprocessor.preprocess(image, use_face_detection=True)
        predictions.append(self.predict_single(img_tensor))
        
        # Horizontal flip
        img_flipped = image.transpose(PILImage.FLIP_LEFT_RIGHT)
        img_tensor = self.preprocessor.preprocess(img_flipped, use_face_detection=True)
        predictions.append(self.predict_single(img_tensor))
        
        # Different scales
        for scale in [0.9, 1.1]:
            w, h = image.size
            new_size = (int(w * scale), int(h * scale))
            img_scaled = image.resize(new_size, PILImage.LANCZOS)
            img_tensor = self.preprocessor.preprocess(img_scaled, use_face_detection=True)
            predictions.append(self.predict_single(img_tensor))
        
        # 평균 앙상블
        avg_real = np.mean([p['real_probability'] for p in predictions])
        avg_fake = np.mean([p['fake_probability'] for p in predictions])
        
        # 정규화
        total = avg_real + avg_fake
        avg_real = (avg_real / total) * 100
        avg_fake = (avg_fake / total) * 100
        
        return {
            'real_probability': min(100.0, avg_real),
            'fake_probability': min(100.0, avg_fake)
        }
    
    def predict_image(self, image: Union[PILImage.Image, str, np.ndarray],
                     use_face_detection: bool = True,
                     use_tta: bool = None) -> Dict[str, Union[int, float]]:
        """
        이미지 딥페이크 탐지
        
        Args:
            image: 입력 이미지
            use_face_detection: 얼굴 탐지 사용 여부
            use_tta: Test Time Augmentation 사용 여부
        
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 이미지 로드
            if isinstance(image, str):
                image = PILImage.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = PILImage.fromarray(image).convert('RGB')
            
            # TTA 설정
            if use_tta is None:
                use_tta = self.enable_tta
            
            # 예측
            if use_tta:
                result = self.predict_with_tta(image)
            else:
                img_tensor = self.preprocessor.preprocess(image, use_face_detection)
                result = self.predict_single(img_tensor)
            
            # 라벨 결정
            predicted_label = 1 if result['fake_probability'] > result['real_probability'] else 0
            
            return {
                'label': predicted_label,
                'fake_probability': min(100.0, result['fake_probability']),
                'real_probability': min(100.0, result['real_probability'])
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, images: List[Union[PILImage.Image, np.ndarray]],
                     batch_size: int = 8) -> List[Dict[str, Union[int, float]]]:
        """
        배치 예측 (성능 최적화)
        
        Args:
            images: 이미지 리스트
            batch_size: 배치 크기
        
        Returns:
            예측 결과 리스트
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = []
            
            for img in batch:
                if isinstance(img, np.ndarray): 
                    img = PILImage.fromarray(img)
                tensor = self.preprocessor.preprocess(img, use_face_detection=True)
                batch_tensors.append(tensor)
            
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
                    embs = self.model(batch_tensor, label=None)
                    W = F.normalize(self.model.arcface.weight.float(), dim=1)
                    cosines = F.linear(embs, W)
                    logits = cosines * self.model.arcface.s / self.model.temperature
                    probs = torch.softmax(logits, dim=1)
                
                for j in range(len(batch)):
                    real_prob = probs[j, 0].item() * 100
                    fake_prob = probs[j, 1].item() * 100
                    label = 1 if fake_prob > real_prob else 0
                    
                    results.append({
                        'label': label,
                        'fake_probability': min(100.0, fake_prob),
                        'real_probability': min(100.0, real_prob)
                    })
        
        return results
    
    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# 사용 편의를 위한 별칭
DeepfakeDetector = EnhancedDeepfakeDetector