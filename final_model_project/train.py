# ===============================================================
# 🧠 train_fsfm_arcface_v3_onecell.py
# 🎯 FSFM backbone 미세조정 + ArcFace margin 완화 + 로그/그래프
# ===============================================================

import os, sys, time, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 0) 경로/임포트
# ---------------------------------------------------------------
FSFM_MODEL_PATH = "/workspace/model_fsfm_arcface.py"     # 참고용(실제 import는 모듈명)
DATASET_PATH    = "/workspace/DATA_GeneV2.py"    # 참고용(실제 import는 모듈명)
CKPT_PATH       = "/workspace/fsfm_checkpoints/pretrained_models/VF2_ViT-B/checkpoint-400.pth"
DATA_ROOT       = "/workspace/sample_dataset"
OUT_DIR         = "./outputs_fsfm_arcface"

if "/workspace" not in sys.path:
    sys.path.append("/workspace")

from FSFM_V1.model_fsfm_arcface import FSFM_ArcFace
from model_detect.DATA_GeneV2 import DeepfakeDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------
# 1) 하이퍼파라미터
# ---------------------------------------------------------------
IMG_SIZE   = 224
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 3e-4     # ✅ 상향 조정
WD         = 1e-4
SEED       = 42
EARLY_STOP = 7
NUM_WORKERS = min(2, os.cpu_count() // 2)
RESUME     = False

# ---------------------------------------------------------------
# 2) 재현성
# ---------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------
# 3) 학습 루프
# ---------------------------------------------------------------
def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"✅ Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 데이터셋
    dataset_full = DeepfakeDataset(DATA_ROOT, train=True, img_size=IMG_SIZE)
    idx = np.arange(len(dataset_full))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    labels = np.array(dataset_full.labels)[tr_idx]
    binc = torch.bincount(torch.tensor(labels), minlength=2).float()
    cls_w = binc.sum() / (2.0 * (binc + 1e-12))  # class weight (CE용)
    weights = cls_w[torch.tensor(labels)]
    if len(weights) < len(tr_idx):
        reps = int(np.ceil(len(tr_idx) / len(weights)))
        weights = weights.repeat(reps)[:len(tr_idx)]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    train_loader = DataLoader(torch.utils.data.Subset(dataset_full, tr_idx),
                              batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset_full, va_idx),
                            batch_size=max(16, BATCH_SIZE//2),
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 모델
    model = FSFM_ArcFace(model_name="fsfm_vit_base_patch16",
                         embed_dim=512, num_classes=2).to(DEVICE)
    model.load_fsfm_checkpoint(CKPT_PATH)

    # ✅ Backbone 미세조정 허용
    for p in model.backbone.parameters():
        p.requires_grad = True

    # ✅ ArcFace margin 완화 (m=0.35)
    try:
        model.arcface.m = 0.35
        model.arcface.cos_m = math.cos(model.arcface.m)
        model.arcface.sin_m = math.sin(model.arcface.m)
        model.arcface.th    = math.cos(math.pi - model.arcface.m)
        model.arcface.mm    = math.sin(math.pi - model.arcface.m) * model.arcface.m
        print(f"✅ ArcFace margin set to m={model.arcface.m}")
    except Exception as e:
        print(f"⚠️ ArcFace margin 변경 실패: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=cls_w.to(DEVICE))
    scaler = amp.GradScaler(enabled=True)

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    last_ckpt, best_ckpt = f"{OUT_DIR}/last.pth", f"{OUT_DIR}/best.pth"
    logs = {"epoch": [], "train_loss": [], "val_f1": [], "val_acc": []}

    # Resume
    start_epoch = 1
    if RESUME and os.path.exists(last_ckpt):
        try:
            ck = torch.load(last_ckpt, map_location="cpu")
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            best_f1, best_epoch, no_improve = ck["best_f1"], ck["best_epoch"], ck["no_improve"]
            start_epoch = ck["epoch"] + 1
            print(f"🔁 Resumed from epoch {ck['epoch']} (best F1={best_f1:.4f})")
        except Exception as e:
            print(f"⚠️ Resume 실패: {e}")

    print(f"[INFO] Training start | workers={NUM_WORKERS}, bs={BATCH_SIZE}")

    # -----------------------------------------------------------
    # 학습
    # -----------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits, _ = model(x, y)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # -------------------------------------------------------
        # 검증
        # -------------------------------------------------------
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with amp.autocast():
                    logits, _ = model(x, y)
                preds += logits.argmax(1).cpu().tolist()
                gts   += y.cpu().tolist()

        val_acc = accuracy_score(gts, preds)
        val_f1  = f1_score(gts, preds, average="macro")

        # 리포트 & 푸터
        print(f"\n🎯 Epoch {epoch} | Macro F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
        print(classification_report(gts, preds, target_names=["Real(0)", "Fake(1)"], digits=2))
        print(f"🧠 Epoch {epoch}/{EPOCHS} 완료 | "
              f"TrainLoss={total_loss/len(train_loader):.4f} | "
              f"ValF1={val_f1:.4f} | BestF1={best_f1:.4f} (E{best_epoch}) | "
              f"Time={(time.time()-t0)/60:.1f}분")
        print("="*80)

        logs["epoch"].append(epoch)
        logs["train_loss"].append(total_loss / len(train_loader))
        logs["val_acc"].append(val_acc)
        logs["val_f1"].append(val_f1)

        # 체크포인트
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1, "best_epoch": best_epoch,
            "no_improve": no_improve
        }, last_ckpt)

        if val_f1 > best_f1:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Best saved (E{epoch}, F1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"⏹ Early stop @E{epoch}, Best F1={best_f1:.4f} (E{best_epoch})")
                break

    # -----------------------------------------------------------
    # 시각화
    # -----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(logs["epoch"], logs["train_loss"], label="Train Loss", marker="s")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss by Epoch")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loss_curve.png"); plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(logs["epoch"], logs["val_f1"], label="Val F1", marker="o")
    plt.plot(logs["epoch"], logs["val_acc"], label="Val Acc", marker="^")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Metrics by Epoch")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/f1_curve.png"); plt.close()

    print(f"\n✅ Done. Best F1={best_f1:.4f} @E{best_epoch}")
    print(f"📈 그래프 저장: {OUT_DIR}/f1_curve.png, {OUT_DIR}/loss_curve.png")

# ---------------------------------------------------------------
# 실행
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
