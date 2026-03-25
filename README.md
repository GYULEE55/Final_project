<div align="center">

# FaceGuard

### Explainable Deep Learning System for Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

딥페이크 이미지와 영상을 탐지하고, **왜 그렇게 판단했는지 시각적으로 설명하는 XAI 기반 탐지 시스템**입니다.

> 단순 분류 모델이 아니라, **일반화 성능 개선 + 설명 가능성 + 실제 사용 가능한 웹 인터페이스**까지 연결한 프로젝트입니다.

</div>

---

## 한눈에 보기

- **무엇을 만들었나**: 딥페이크 이미지·영상 탐지 모델과 Grad-CAM 기반 설명, 추론 안정화 로직, Streamlit 웹 데모를 포함한 End-to-End 시스템
- **무슨 문제를 풀었나**: 학습 데이터에서는 잘 맞지만 외부 데이터에서 무너지는 일반화 문제와, 결과를 믿기 어려운 블랙박스 문제
- **핵심 성과**: Unseen data 기준 **Macro F1 0.43 -> 0.82**로 개선
- **내 역할**: 프로젝트 리더로서 모델 아키텍처 선정, 데이터 파이프라인 설계, 설명성 기능, 웹 시스템 구현 주도

---

## 핵심 성과

<p align="center">
  <img src="final_deepfake/images/image copy 10.png" width="880" alt="Validation Macro F1 (internal vs external)"/>
</p>

```text
[val data] Macro F1 = 0.9580 | Acc = 0.9675
[unseen data] Macro F1 = 0.8227 | Acc = 0.8776
```

- 내부 검증셋에서는 높은 정확도 유지
- 외부 데이터에서도 일반화 성능을 확보
- 설명 기능(Grad-CAM, 판정 근거, 신뢰도 추이)을 통해 결과 해석 가능성 강화

---

## 프로젝트 개요

딥페이크 기술이 빠르게 고도화되면서, 단순히 정확도만 높은 모델보다 **처음 보는 데이터에서도 버티고, 사람이 결과를 납득할 수 있는 탐지 시스템**이 중요해졌습니다.

이 프로젝트는 다음 세 가지를 동시에 해결하는 것을 목표로 했습니다.
1. 다양한 딥페이크 기법에 대응하는 **일반화 성능 향상**
2. 결과를 설명할 수 있는 **XAI 기반 해석 가능성 확보**
3. 실제 사용 가능한 **웹 서비스 형태의 시스템 구현**

### 프로젝트 정보
- **기간**: 2025.10.13 ~ 2025.11.20 (5주)
- **팀 구성**: 2명
- **역할**: 프로젝트 리더 / 모델 아키텍처 설계, 데이터 파이프라인 구축, 시스템 구현
- **성과**: 교육 과정 **최우수상 수상**

---

## 문제 정의

### 1. 일반화 성능 문제
기존 딥페이크 탐지 모델은 학습 데이터셋에서는 높은 성능을 보이지만, 다른 출처의 데이터에서는 성능이 급격히 떨어졌습니다.

### 2. 블랙박스 문제
사용자는 Real/Fake 결과만 보고는 왜 그렇게 판단됐는지 알기 어렵습니다.

### 3. 실사용성 문제
영상 기반 탐지는 프레임마다 결과가 흔들릴 수 있어, 실제 서비스 수준의 안정적인 추론 로직이 필요했습니다.

---

## 접근 방식

### 모델 아키텍처

<p align="center">
  <img src="final_deepfake/images/image copy 5.png" alt="모델 조사 기반" width="800"/>
</p>

11가지 딥페이크 탐지 모델을 공간·시간·주파수 관점에서 비교한 뒤, 초기 베이스라인으로 ViT를 선택했습니다.

**선정 과정**
- 베이스라인: Xception
- 공간 기반 후보: EfficientNet-B4
- 시간 기반 후보: ViT
- 기타 후보: PixelCNN

<p align="center">
  <img src="final_deepfake/images/image copy 8.png" alt="베이스라인 모델 성능" width="700"/>
</p>

하지만 ViT는 외부 데이터에서 일반화 한계를 보였습니다.

<p align="center">
  <img src="final_deepfake/images/image copy 6.png" alt="모델 한계성 분석" width="750"/>
</p>

### 핵심 해결 전략

1. **ViT -> FSFM + ArcFace 전환**
   - 얼굴 특화 표현을 더 잘 반영하는 FSFM 사용
   - 얼굴 간 특징 공간 분리를 강화하는 ArcFace 헤드 적용
2. **Synthetic Data Engineering**
   - StyleGAN-K 기반 한국인 얼굴 데이터를 추가해 데이터 편향 완화
3. **Inference Stabilization**
   - Top-K 프레임 선택 + Auto-Threshold를 도입해 영상 판정 흔들림 완화
4. **Explainability**
   - Grad-CAM, 신뢰도 추이, 판정 근거 시각화 기능 추가

---

## 데이터셋 구축

### StyleGAN-K 기반 한국인 데이터 생성

<p align="center">
  <img src="final_deepfake/images/image copy 7.png" alt="StyleGAN 한국인 데이터 생성" width="500"/>
</p>

외부 데이터 성능 저하 원인을 한국인 데이터와 생성형 AI 데이터 부족에 따른 편향으로 보고, synthetic face generation 파이프라인을 구성했습니다.

**생성 과정**
1. Latent vector 입력
2. StyleGAN Generator로 한국인 얼굴 생성
3. FaceNet 임베딩 추출
4. L2 distance 기반 최적화
5. 약 **5,000장** 수준의 한국인 fake 데이터 확보

### 최종 데이터 분포

```text
국적 분포:   KR:Global = 42:58
레이블 분포: Real:Fake = 26:74
모달리티:    Video:Image = 70:30
```

---

## 성능 개선 과정

<p align="center">
  <img src="final_deepfake/images/image copy 4.png" alt="성능 개선 그래프" width="750"/>
</p>

<p align="center">
  <img src="final_deepfake/images/image copy 9.png" alt="파인튜닝 3단계" width="900"/>
</p>

| 단계 | 모델/기법 | Macro F1 | 주요 개선 사항 |
|------|-----------|----------|---------------|
| Baseline | ViT (불균형) | 0.38 | 초기 베이스라인 |
| V1 | 얼굴 특화 베이스라인 | **0.60** | ViT -> FSFM ViT-B 전환 |
| V2 | 학습 안정화 | **0.74** | LR/ArcFace margin 조정, gradient clipping |
| **V3** | **파이프라인 고도화** | **0.82** | **Top-K + Auto-Threshold** |

### 최종 성과

- 베이스라인 대비 **90.7% 성능 향상**
- Macro F1 **43% -> 82%**
- 외부 데이터에서도 사용할 수 있는 수준의 일반화 성능 확보

---

## 시스템 구현

<p align="center">
  <img src="final_deepfake/images/image copy 3.png" alt="FaceGuard Vision 핵심 기능" width="900"/>
</p>

### 핵심 기능

1. **조작 부분 시각화**
   - Grad-CAM 기반 히트맵 제공
2. **딥페이크 구간 추이 분석**
   - 영상 프레임별 신뢰도 추이 시각화
3. **딥페이크 기법 판별**
   - 12가지 딥페이크 기법 및 AI 생성 이미지 분류
4. **얼굴 자동 검출**
   - RetinaFace 기반 얼굴 검출 및 margin 확장
5. **임계치 자동 조정**
   - Top-K + Auto-Threshold 기반 판정 안정화
6. **실시간 기록 대시보드**
   - 여러 입력의 판정 결과를 추적 가능하게 구성

### 결과 화면

<p align="center">
  <img src="final_deepfake/images/image copy.png" alt="실제 분석 결과 화면" width="900"/>
</p>

이 프로젝트는 모델 정확도만 높이는 데서 끝나지 않고,
- **판단 근거를 시각화**하고
- **사용자가 결과를 다시 확인할 수 있게 하고**
- **웹 환경에서 실제로 사용할 수 있는 흐름**으로 구성한 점이 핵심입니다.

---

## 이 프로젝트에서 보여주고 싶은 역량

- 문제를 정확도 개선 이슈로만 보지 않고, **일반화 / 설명성 / 시스템 관점**으로 확장해 해결한 경험
- 모델 선택, 데이터 보강, 추론 안정화, UI 연결까지 이어지는 **End-to-End AI 엔지니어링 역량**
- 실험 결과를 숫자로만 제시하지 않고, 왜 좋아졌는지 구조적으로 설명하는 역량
- 보안·미디어·플랫폼 등 다양한 도메인으로 확장 가능한 **컴퓨터 비전 시스템 설계 경험**

---

## 참고 문헌

> 30개 이상의 최신 논문을 리뷰해 연구 방향과 모델 선택 기준을 정리했습니다.

<details>
<summary><b>주요 참고 문헌 보기</b></summary>

- 데이터셋: FaceForensics++, Celeb-DF, DFDC, KoDF
- 탐지 모델: FSFM, ViT, Xception, EfficientNet
- XAI: Grad-CAM, Active Illumination
- 멀티모달: Audio-Visual Fusion, Vision-Language Models
- 서베이: Deepfake Meta-Review, Multimedia Survey

</details>

---

## 팀 정보

**이승규** - 프로젝트 리더
- 모델 아키텍처 설계
- 데이터 파이프라인 구축
- 추론 안정화 로직 설계
- 시스템 구현 및 데모 제작
