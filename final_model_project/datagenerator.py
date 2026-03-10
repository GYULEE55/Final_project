#!/usr/bin/env python3
# ===============================================================
# 🧠 DATA_GeneV3.py
# - CSV 기반 Dataset
# - 숫자/문자 라벨 정상 처리
# - WeightedRandomSampler 포함
# - BlindFace 안전 적용
# ===============================================================

import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

# ===============================================================
# 1) Dlib (blind_face)
# ===============================================================
try:
    import dlib
    from imutils import face_utils
    predictor_path = "/workspace/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(predictor_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        DLIB_AVAILABLE = True
        print("✅ Dlib predictor 활성화")
    else:
        DLIB_AVAILABLE = False
        print("⚠️ predictor 없음 → blind_face 비활성화")
except:
    DLIB_AVAILABLE = False
    print("⚠️ dlib import 실패 → blind_face 비활성화")


# ===============================================================
# Blind face polygon 계산
# ===============================================================
def get_poly(shape_, case_):
    input_shape = (224, 224)
    if case_ == 0:  # 눈
        pt2 = np.array([shape_[20][0], shape_[36][1] + (shape_[36][1] - shape_[20][1])])
        pt1 = np.array([shape_[23][0], shape_[45][1] + (shape_[45][1] - shape_[23][1])])
        pts = np.array([shape_[36], shape_[20], shape_[23], shape_[45], pt1, pt2], np.int32)
    elif case_ == 1:  # 이마
        pts = np.array([shape_[0], shape_[17], shape_[26], shape_[16]], np.int32)
    elif case_ == 2:  # 왼쪽 얼굴
        pts = np.array([shape_[0], shape_[17], shape_[21], shape_[27], shape_[8]], np.int32)
    elif case_ == 3:  # 오른쪽 얼굴
        pts = np.array([shape_[16], shape_[26], shape_[22], shape_[27], shape_[8]], np.int32)
    elif case_ == 4:  # 코
        pt1 = np.array([shape_[31][0], shape_[28][1]])
        pt2 = np.array([shape_[35][0], shape_[28][1]])
        pts = np.array([shape_[27], pt1, shape_[31], shape_[51], shape_[35], pt2], np.int32)
    else:  # 전체 이마
        pt1 = np.array([0, shape_[17][1]])
        pt2 = np.array([input_shape[0]-1, shape_[26][1]])
        pt3 = np.array([input_shape[0]-1, 0])
        pt4 = np.array([0, 0])
        pts = np.array([pt1, pt2, pt3, pt4], np.int32)
    return pts


def blind_face(img):
    """얼굴 일부 랜덤 가리기"""
    if not DLIB_AVAILABLE:
        return img
    if random.random() < 0.5:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rects = detector(gray, 0)
            if len(rects) > 0:
                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                case = random.randint(0, 5)
                pts = get_poly(shape, case)
                cv2.fillConvexPoly(img, pts, (0, 0, 0))
        except:
            pass
    return img


# ===============================================================
# 2) CSV Dataset
# ===============================================================
class DeepfakeDatasetCSV(Dataset):
    def __init__(self, csv_path, train=True, img_size=224):
        self.csv_path = csv_path
        self.train = train
        self.img_size = img_size

        # CSV 로드
        df = pd.read_csv(csv_path)
        if "status" in df.columns:
            df = df[df["status"] == "good"]

        self.image_paths = df["path"].tolist()

        # =========================================================
        # 🔥 라벨 처리 (숫자/문자 모두 지원)
        # =========================================================
        if "label" in df.columns:
            if df["label"].dtype == object:
                # 문자열 라벨
                self.labels = [1 if str(l).lower() == "fake" else 0 for l in df["label"]]
            else:
                # 숫자 라벨
                self.labels = df["label"].astype(int).tolist()
        else:
            # 라벨이 없을 경우 → 경로 기반 추출
            self.labels = [1 if "fake" in str(p).lower() else 0 for p in self.image_paths]

        # 요약 출력
        cnt = Counter(self.labels)
        print(f"\n📊 Dataset: {os.path.basename(csv_path)}")
        print(f"총 {len(self.image_paths)} | Real={cnt[0]} | Fake={cnt[1]}")

        self.transform = self._build_transforms()

    def _build_transforms(self):
        aug = [transforms.Resize((self.img_size, self.img_size))]
        if self.train:
            aug += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
            ]
        aug += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        return transforms.Compose(aug)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train:
            img = blind_face(img)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ===============================================================
# 3) DataLoader 생성
# ===============================================================
def get_loaders(batch_size=64, num_workers=4):

    train_csv = "/workspace/split_csv/train_main.csv"
    val_int_csv = "/workspace/split_csv/val_internal.csv"
    val_ext_csv = "/workspace/split_csv/val_external.csv"

    ds_train = DeepfakeDatasetCSV(train_csv, train=True)
    ds_val_internal = DeepfakeDatasetCSV(val_int_csv, train=False)
    ds_val_external = DeepfakeDatasetCSV(val_ext_csv, train=False)

    # ===========================================================
    # WeightedRandomSampler → Real/Fake 균형
    # ===========================================================
    labels = ds_train.labels
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )

    # Loader
    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    loader_val_internal = DataLoader(
        ds_val_internal,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    loader_val_external = DataLoader(
        ds_val_external,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader_train, loader_val_internal, loader_val_external


# ===============================================================
# 4) 단독 실행 테스트
# ===============================================================
if __name__ == "__main__":
    train_loader, val_i, val_e = get_loaders(batch_size=8)
    for img, lbl in train_loader:
        print("Batch OK:", img.shape, lbl[:8])
        break
