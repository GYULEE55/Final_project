# ============================================================
# 🎯 crop_real_only_min_filter_v2.py
#  - Real만 추출
#  - extreme noise(minimal quality filter)만 제외
#  - RetinaFace decode 경고 제거 (clone().detach() 적용)
#  - 회전 → 반사패딩 → 224 크롭 동일
#  - Real video는 15프레임 균등 샘플
# ============================================================
import os
print(">>> 실행된 파일 경로:", __file__)
print(">>> 현재 working dir:", os.getcwd())

import os, sys, cv2, torch, random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE_DIR = Path("/workspace/final_dataset").resolve()
OUT_DIR  = Path("/workspace/DATA_cropV2_real").resolve()
META_DIR = OUT_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

FACE_SIZE = 224
CONF_THRESH = 0.7
MIN_FACE = 64
FRAMES_REAL = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# ============================================================
# 1) Real 판별
# ============================================================
def is_real(p: str) -> bool:
    s = p.lower()
    return any(k in s for k in [
        "real/korea_image", "korea_image_val",
        "celeb-real", "ff_real", "kodf_real"
    ])

# ============================================================
# 2) RetinaFace 로딩
# ============================================================
sys.path.append("/workspace/Pytorch_Retinaface")
from models.retinaface import RetinaFace
from data.config import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm

retina = RetinaFace(cfg=cfg_re50, phase="test")
w = torch.load("/workspace/Pytorch_Retinaface/weights/Resnet50_Final.pth", map_location="cpu")
if "state_dict" in w:
    w = w["state_dict"]

retina.load_state_dict({k.replace("module.", ""): v for k, v in w.items()}, strict=False)
retina.to(DEVICE).eval()

HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@torch.inference_mode()
def detect(img):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None, None

    sc = torch.tensor([w, h, w, h])
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    loc, conf, land = retina(x)
    conf = conf.squeeze(0).detach().cpu().numpy()[:, 1]
    loc = loc.squeeze(0).detach().cpu().numpy()
    land = land.squeeze(0).detach().cpu().numpy()

    pri = PriorBox(cfg_re50, image_size=(h, w)).forward()

    # ----- decode 경고 제거 버전 -----
    loc_t  = torch.as_tensor(loc).clone().detach()
    pri_t  = torch.as_tensor(pri).clone().detach()
    var_t  = torch.as_tensor(cfg_re50["variance"]).clone().detach()
    land_t = torch.as_tensor(land).clone().detach()

    boxes = decode(loc_t, pri_t, var_t) * sc
    land  = decode_landm(land_t, pri_t, var_t)

    boxes = boxes.cpu().numpy()
    land  = land.cpu().numpy()

    idx = np.where(conf > CONF_THRESH)[0]
    if len(idx) == 0:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = HAAR.detectMultiScale(g, 1.3, 5)
        if len(faces) == 0:
            return None, None
        x, y, wf, hf = faces[0]
        return [x, y, x + wf, y + hf], None

    j = idx[np.argmax(conf[idx])]
    box = boxes[j].astype(int)
    if (box[2] - box[0] < MIN_FACE) or (box[3] - box[1] < MIN_FACE):
        return None, None

    lm = land[j] * np.array([w, h] * 5)
    return box, lm.reshape(5, 2)

# ============================================================
# 3) 회전+크롭
# ============================================================
def rotate_pad_crop_224(img, box, lm):
    x1, y1, x2, y2 = map(int, box)

    # --- 회전 각도 ---
    if lm is not None:
        left, right = lm[0], lm[1]
        ang = np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0]))
    else:
        ang = 0.0

    h, w = img.shape[:2]
    pad = int(max(h, w) * 0.35)
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # --- 회전 ---
    c = (img_pad.shape[1]//2, img_pad.shape[0]//2)
    M = cv2.getRotationMatrix2D(c, ang, 1.0)
    rot = cv2.warpAffine(img_pad, M,
                         (img_pad.shape[1], img_pad.shape[0]),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101)

    pts = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], np.float32)
    pts += np.array([pad, pad])

    R = np.vstack([M, [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((4, 1))]) @ R.T

    xs, ys = pts_h[:, 0], pts_h[:, 1]
    cx, cy = int(xs.mean()), int(ys.mean())
    side   = int(max(xs.max()-xs.min(), ys.max()-ys.min()) * 1.15)

    x1n = max(cx - side//2, 0)
    y1n = max(cy - side//2, 0)
    x2n = min(cx + side//2, rot.shape[1])
    y2n = min(cy + side//2, rot.shape[0])

    face = rot[y1n:y2n, x1n:x2n]
    if face.size == 0:
        return None

    return cv2.resize(face, (FACE_SIZE, FACE_SIZE))

# ============================================================
# 4) Extreme noise filter (minimal)
# ============================================================
def is_extreme_noise(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_b = float(np.mean(g))
    sharp  = float(cv2.Laplacian(g, cv2.CV_64F).var())

    if mean_b < 10:   # 완전 검정
        return True
    if sharp < 3:     # 완전 블러
        return True
    return False

# ============================================================
# 5) Video frame extract
# ============================================================
def extract_frames(vpath, k=FRAMES_REAL):
    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total - 1, num=min(k, total), dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)

    cap.release()
    return frames

# ============================================================
# 6) Main — Real만 처리
# ============================================================
def process_real_only():
    rec = []

    for root, _, files in os.walk(BASE_DIR):
        for fn in files:
            src = Path(root) / fn
            if not is_real(str(src)):
                continue

            ext = src.suffix.lower()
            if ext not in IMG_EXT.union(VID_EXT):
                continue

            rel = src.relative_to(BASE_DIR)
            dst_base = OUT_DIR / rel
            dst_base.parent.mkdir(parents=True, exist_ok=True)

            # ----- 이미지 -----
            if ext in IMG_EXT:
                img = cv2.imread(str(src))
                if img is None:
                    continue
                if is_extreme_noise(img):
                    continue

                box, lm = detect(img)
                if box is None:
                    continue

                face = rotate_pad_crop_224(img, box, lm)
                if face is None:
                    continue

                cv2.imwrite(str(dst_base), face)
                rec.append({"path": str(dst_base), "label": "real"})
                continue

            # ----- 비디오 -----
            frames = extract_frames(src)
            for i, fr in enumerate(frames):
                if is_extreme_noise(fr):
                    continue

                box, lm = detect(fr)
                if box is None:
                    continue

                face = rotate_pad_crop_224(fr, box, lm)
                if face is None:
                    continue

                out = dst_base.with_name(dst_base.stem + f"_f{i:03d}.jpg")
                cv2.imwrite(str(out), face)
                rec.append({"path": str(out), "label": "real"})

    df = pd.DataFrame(rec)
    df.to_csv(META_DIR / "dataset_real_only_min_filter.csv", index=False)
    print("\n📄 저장 완료:", META_DIR / "dataset_real_only_min_filter.csv")
    print("총 Real 크롭:", len(df))

# 실행
process_real_only()
