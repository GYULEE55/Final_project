# ===============================================================
# 📁 dataset_fsfm_v2.py
# 🎭 FSFM + ArcFace용 데이터셋 정의 (라벨링 + blind_face 안정화)
# ===============================================================

import os
import cv2
import random
import numpy as np
from glob import glob
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
import torch

# ===============================================================
# 1️⃣ Dlib 설정 (얼굴 랜드마크 기반 blind_face)
# ===============================================================
try:
    import dlib
    from imutils import face_utils

    predictor_path = "/workspace/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(predictor_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        DLIB_AVAILABLE = True
    else:
        DLIB_AVAILABLE = False
except Exception:
    DLIB_AVAILABLE = False


def get_poly(shape_, case_):
    input_shape = (224, 224)
    if case_ == 0:  # 눈
        pt2 = np.array([shape_[20][0], shape_[36][1] + shape_[36][1] - shape_[20][1]])
        pt1 = np.array([shape_[23][0], shape_[45][1] + shape_[45][1] - shape_[23][1]])
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
    else:  # 이마 전체
        pts = np.array([[0, shape_[17][1]], [223, shape_[26][1]], [223, 0], [0, 0]], np.int32)
    return pts


def blind_face(img):
    if not DLIB_AVAILABLE:
        return img
    if random.random() < 0.5:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rects = detector(gray, 0)
            if len(rects) > 0:
                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                pts = get_poly(shape, random.randint(0, 5))
                cv2.fillConvexPoly(img, pts, (0, 0, 0))
        except Exception:
            pass
    return img


def get_label_from_path(path):
    path = path.lower()
    if any(k in path for k in ["fake", "ai_people", "ai_video", "ff_fake", "kodf fake"]):
        return 1
    elif any(k in path for k in ["real", "korea_image", "ff_real", "original"]):
        return 0
    return 0


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, train=True, img_size=224):
        self.image_paths = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
        self.labels = [get_label_from_path(p) for p in self.image_paths]
        self.train = train
        self.transform = self._build_transforms(img_size)

    def _build_transforms(self, img_size):
        aug = [transforms.Resize((img_size, img_size))]
        if self.train:
            aug += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
            ]
        aug += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
        return transforms.Compose(aug)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        if self.train:
            img = blind_face(img)
        img = self.transform(Image.fromarray(img))
        return img, torch.tensor(self.labels[idx])
