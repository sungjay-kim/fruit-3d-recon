# Fruit 3D Reconstruction (ArUco-scaled COLMAP)

ArUco 마커로 자동 스케일이 보정되는 **과일(참외 등) 3D 재구성** 파이프라인입니다. 다방향 RGB 촬영 → SAM3 객체 마스크 → COLMAP SfM/MVS → 실제 크기(미터 단위) PLY 출력까지 한 번에 처리합니다.

> 본 repo는 [`tcore` shape completion 데이터셋 파이프라인](https://github.com/sungjay-kim)의 GROUND TRUTH 생성 단계(Step A)와 SAM3 라벨링 스크립트(Step 1)를 묶어 정리한 것입니다.

---

## 요약

| 항목 | 내용 |
| --- | --- |
| 입력 | 다방향 RGB 이미지 (카메라 3대 × 36장 × 10° 간격 권장) |
| 마스크 | SAM3 text-prompt 기반 binary mask |
| 카메라 포즈 | COLMAP SIFT + cross-camera angular pair matching |
| Dense | COLMAP PatchMatch Stereo + Stereo Fusion (다중 GPU 지원) |
| Scale | ArUco 마커 (3 cm × 3 cm, DICT_4X4_50) 자동 삼각측량 |
| 출력 | `fused_scaled.ply` (단위: m, 노이즈 제거 후) |

---

## 디렉토리 구조

```
fruit-3d-recon/
├── README.md
├── bash/
│   ├── aruco_reconstruction.sh       # 메인 파이프라인
│   ├── manual_relaxed_crosscam.sh    # Sparse reconstruction (cross-cam matching)
│   ├── dense.sh                      # Dense reconstruction + mask filtering
│   └── pointcloud_to_mesh.sh         # (선택) Poisson meshing
├── scripts/
│   ├── generate_aruco_markers.py     # ArUco PDF 생성
│   ├── generate_aruco_masks.py       # 자동 ArUco 마스크 생성
│   ├── compute_aruco_scale.py        # 삼각측량으로 scale factor 계산
│   ├── apply_scale_to_ply.py         # fused.ply에 scale 적용
│   ├── remove_noise_ply.py           # 노이즈 제거
│   ├── copy_img_and_mask.py          # 데이터 디렉토리 구조 준비
│   ├── mesh_volume.py                # 메쉬 부피 계산
│   └── preprocess_pointcloud_for_meshing.py
├── sam3_labeling/
│   └── run_text_prompt_on_zip.py     # SAM3 text-prompt 라벨링
├── aruco_markers/
│   └── aruco_markers_3cm_x24.pdf     # 인쇄용 (24개 마커, 3cm)
└── third_party/
    └── colmap_python/
        └── read_write_model.py        # COLMAP 공식 helper (Apache 2.0)
```

---

## 의존성

### 1. COLMAP

`colmap` 바이너리가 PATH에 있거나 환경변수 `COLMAP_BIN`으로 지정되어 있어야 합니다.
GPU 빌드를 권장합니다 (PatchMatch Stereo가 CUDA를 사용).

```bash
# Ubuntu 예시
sudo apt-get install colmap

# 또는 source 빌드: https://colmap.github.io/install.html
```

### 2. Python 환경

테스트된 환경: Python 3.10+, micromamba/conda 기반 `colmap_env`

```bash
micromamba create -n colmap_env -c conda-forge \
  python=3.10 colmap opencv open3d numpy pillow scipy
micromamba activate colmap_env
```

### 3. SAM3 (라벨링 단계)

[Meta SAM3](https://github.com/facebookresearch/sam3)는 별도 설치가 필요합니다 (라벨링에만 사용, reconstruction과 환경 분리 권장):

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
# 본 repo 작성 시점 검증된 버전: main 브랜치 (Python 3.12+, PyTorch 2.7+)
pip install -e .
```

설치 후 본 repo의 `sam3_labeling/run_text_prompt_on_zip.py`를 사용하면 됩니다 (sam3 패키지 import만 필요하므로 SAM3 repo 안으로 복사하거나 PYTHONPATH에 추가).

---

## 데이터 준비

### 디렉토리 구조

```
runs/{SAMPLE}/
├── images/
│   ├── 1/    # 카메라 1 프레임 (PNG)
│   ├── 2/    # 카메라 2 프레임
│   └── 3/    # 카메라 3 프레임
└── masks/    # SAM3 객체 마스크 (물체만, 마커 미포함)
    ├── 1/
    ├── 2/
    └── 3/
```

이미지 파일명에 회전 각도가 들어있어야 합니다. 예: `CAM#1_xxx(Degree-0).png`, `CAM#1_xxx(Degree-10).png` ...

### ArUco 마커 셋업 (최초 1회)

1. `python scripts/generate_aruco_markers.py` → `aruco_markers/aruco_markers_3cm_x24.pdf` 생성 (이미 포함되어 있음)
2. 인쇄 후 검은색 수직 패널 3~4개에 배치 (마커당 흰색 테두리 0.3 cm 포함)
3. 패널을 턴테이블 가장자리에 세우고 **물체와 함께 회전**

**ArUco 역할:**
- ① Sparse reconstruction에서 feature matching / pose 추정 보조
- ② Scale factor 계산 (알려진 0.03 m 크기 → COLMAP arbitrary scale 보정)

---

## 파이프라인 실행

### Step 1. SAM3 라벨링

```bash
# SAM3 환경에서
python sam3_labeling/run_text_prompt_on_zip.py \
  --frames-path /path/to/{SAMPLE}_{CAM} \
  --prompt "melon" \
  --prompt-frame 0 \
  --output-dir /path/to/output_dir \
  --outputs masks overlays json \
  --direction both
```

각 SAMPLE × CAM 조합마다 실행 (예: `20260318_test_1`, `20260318_test_2`, `20260318_test_3`).

출력:
- `output_dir/{SAMPLE}_{CAM}/masks/*.png` — Binary mask (L-mode, 0/255)
- `output_dir/{SAMPLE}_{CAM}/overlays/*.png` — 시각화
- `output_dir/{SAMPLE}_{CAM}/predictions.json` — 메타데이터

### Step 2. 데이터 구조 준비

```bash
# colmap_env 환경에서
python scripts/copy_img_and_mask.py \
  --merged-root /path/to/phenobox/merged \
  --mask-root /path/to/sam3/output_dir \
  --runs-root ./runs
```

각 sample의 이미지/마스크를 `runs/{SAMPLE}/images/{cam}/`, `runs/{SAMPLE}/masks/{cam}/` 구조로 복사합니다.

소스 디렉토리 구조 (예시):
- `merged-root/{SAMPLE}/{1,2,3}/*.png` — 카메라별 RGB 프레임
- `mask-root/{SAMPLE}_{CAM}/masks/*_obj0.png` — SAM3 마스크 (또는 `mask-root/{SAMPLE}/{CAM}/masks/`)

### Step 3. ArUco 통합 Reconstruction

```bash
# 기본 실행
bash bash/aruco_reconstruction.sh {SAMPLE}

# 카메라2 밝기 보정 + 각도 임계값 + GPU 지정
bash bash/aruco_reconstruction.sh {SAMPLE} 1.5 25 0,1
```

**파라미터:**

| 순서 | 변수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| 1 | `SAMPLE` | — | 샘플 ID (`runs/{SAMPLE}/` 폴더명) |
| 2 | `BRIGHT_FACTOR` | 1.0 | 카메라2 밝기 보정 (>1.0 = 밝게) |
| 3 | `ANGLE_DELTA` | 20 | Cross-camera matching 각도 임계값(°) |
| 4 | `GPU_IDX` | 0,1 | PatchMatch GPU 인덱스 |

**내부 흐름:**

```
[1] 원본 object_masks/ 보존
[2] generate_aruco_masks.py → aruco_masks/ + sparse_masks/
[3] masks/ ← sparse_masks/  (물체+마커 마스크)
[4] Sparse reconstruction (manual_relaxed_crosscam.sh)
       SIFT feature → sequential + cross-cam angular pair matching
       → COLMAP Mapper → sparse/manual_relaxed/{idx}/
[5] masks/ ← object_masks/  (물체만 — dense용)
[6] Best sparse model 자동 선택 (등록 이미지 수 기준)
[7] compute_aruco_scale.py → scale_factor.json (마커 코너 삼각측량)
[8] dense.sh → fused.ply (mask-based filtering 포함)
[9] apply_scale_to_ply.py → fused_scaled.ply (단위: m)
[10] remove_noise_ply.py → 노이즈 제거 후 fused_scaled.ply 덮어쓰기
```

**출력:**

| 파일 | 내용 |
| --- | --- |
| `runs/{SAMPLE}/dense/manual_relaxed_{idx}/fused.ply` | Dense PCD (COLMAP arbitrary scale) |
| `runs/{SAMPLE}/dense/manual_relaxed_{idx}/fused_scaled.ply` | **최종 출력** — 단위 m, 노이즈 제거 |
| `runs/{SAMPLE}/scale_factor.json` | scale factor + 삼각측량 통계 |
| `runs/{SAMPLE}/aruco_masks/`, `sparse_masks/` | 자동 생성 마스크 |

### Step 4. (선택) Meshing

```bash
bash bash/pointcloud_to_mesh.sh {SAMPLE}
```

`fused.ply`를 Poisson surface reconstruction으로 메쉬화합니다.

환경변수로 동작 조정 가능:
- `COLMAP_BIN` — colmap 바이너리 경로 (기본: PATH의 `colmap`)
- `POISSON_DEPTH` (기본 9) — 품질 부족 시 10~11 시도
- `RUN_DELAUNAY=1` — Delaunay mesher도 함께 실행

---

## 실측 참고치

`20260318_test` (3대 카메라 × 36장 = 108장, A6000 × 2 GPU):

| 단계 | 소요시간 |
| --- | --- |
| ArUco 마스크 생성 | ~2분 |
| Sparse reconstruction | ~4분 |
| Scale factor 계산 | ~16초 |
| Dense reconstruction (2 GPU) | ~19분 |
| **총합** | **~26분** |

- Scale factor: 0.205792 m / COLMAP_unit (3개 이상 마커 평균)
- 결과 크기: 0.142 × 0.130 × 0.129 m (참외 1개)

---

## 트러블슈팅

| 증상 | 처리 |
| --- | --- |
| 카메라2가 어두워서 등록 실패 | `BRIGHT_FACTOR=1.5` 또는 `2.0` |
| Cross-cam matching 실패 | `ANGLE_DELTA=25~30`으로 완화 |
| 등록 이미지 수가 50% 미만 | `manual_relaxed_crosscam.sh`가 자동으로 fallback (exhaustive matcher + 더 공격적 settings) |
| ArUco 마커 미탐지 | 다중 전처리(CLAHE, gamma, blue channel) 합집합 적용. 유효 범위 50~250 px. 카메라가 너무 멀면 마커 크기 키울 것. |
| Poisson 결과가 거칠다 | `POISSON_DEPTH=10` 또는 `11` |

---

## 라이센스

- 본 repo의 코드: TBD
- `third_party/colmap_python/read_write_model.py`: COLMAP 공식 (BSD-3)
- SAM3는 별도 라이센스 (Meta) — 본 repo에는 미포함, 사용자 별도 설치
