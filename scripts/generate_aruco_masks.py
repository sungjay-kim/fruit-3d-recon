"""
ArUco 마커 마스크 생성 스크립트

입력:
  runs/{SAMPLE}/images/{1,2,3}/*.png        - 원본 이미지
  runs/{SAMPLE}/masks/{1,2,3}/*.png.png     - 물체 마스크 (SAM3)

출력:
  runs/{SAMPLE}/aruco_masks/{1,2,3}/        - 마커만 포함한 binary 마스크
  runs/{SAMPLE}/sparse_masks/{1,2,3}/       - 물체+마커 합성 마스크 (sparse reconstruction용)

마스크 파일명 포맷: {image_name}.png.png (기존 SAM3 출력과 동일)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MAX_MARKER_ID = 23  # 유효 마커 ID 범위 (0~23)
MIN_MARKER_SIDE_PX = 50  # 마커 변 길이 최소 픽셀 (노이즈 필터링)
MAX_MARKER_SIDE_PX = 250  # 마커 변 길이 최대 픽셀 (오탐지 필터링)
MARKER_PADDING_PX = 15  # 마커 주변 여백 (흰색 테두리 포함)


def _make_detector(relaxed: bool = False) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    if relaxed:
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 50
        params.adaptiveThreshWinSizeStep = 5
        params.minMarkerPerimeterRate = 0.01
        params.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(ARUCO_DICT, params)


def _preprocess_variants(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """어두운 UV 조명 환경에서 탐지율을 높이기 위해 다양한 전처리 결과를 반환."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.split(image_bgr)[0]
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    def gamma(img, g):
        return np.clip(255 * (img.astype(np.float32) / 255) ** g, 0, 255).astype(np.uint8)

    return [
        ("gray", gray),
        ("gray_clahe", clahe.apply(gray)),
        ("gray_gamma0.3", gamma(gray, 0.3)),
        ("blue_gamma0.3", gamma(b, 0.3)),
        ("blue_clahe", clahe.apply(b)),
    ]


def detect_aruco_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """다중 전처리 + 합집합으로 ArUco 마커를 탐지. (mask, n_markers) 반환."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    found_ids = {}  # {marker_id: corners}

    variants = _preprocess_variants(image_bgr)
    detectors = [_make_detector(relaxed=False), _make_detector(relaxed=True)]

    for _, processed in variants:
        for detector in detectors:
            corners, ids, _ = detector.detectMarkers(processed)
            if ids is None:
                continue
            for corner, mid in zip(corners, ids.flatten()):
                mid = int(mid)
                if 0 <= mid <= MAX_MARKER_ID and mid not in found_ids:
                    pts = corner[0]
                    sides = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
                    mean_side = np.mean(sides)
                    if MIN_MARKER_SIDE_PX <= mean_side <= MAX_MARKER_SIDE_PX:
                        found_ids[mid] = pts

    for mid, pts in found_ids.items():
        pts = pts.astype(np.float32)
        center = pts.mean(axis=0)
        direction = pts - center
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        pts_padded = pts + direction / norm * MARKER_PADDING_PX
        cv2.fillConvexPoly(mask, pts_padded.astype(np.int32), 255)

    return mask, len(found_ids)


def load_object_mask(mask_path: Path) -> np.ndarray | None:
    """물체 마스크를 로드한다. 없으면 None 반환."""
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def process_sample(sample: str, runs_root: Path, object_masks_dir: str | None = None):
    run_dir = runs_root / sample
    images_root = run_dir / "images"
    masks_root = Path(object_masks_dir) if object_masks_dir else run_dir / "object_masks"
    if not masks_root.exists():
        masks_root = run_dir / "masks"  # fallback
    aruco_masks_root = run_dir / "aruco_masks"
    sparse_masks_root = run_dir / "sparse_masks"

    if not images_root.exists():
        print(f"[ERROR] images 디렉토리 없음: {images_root}")
        return

    total_images = 0
    total_detected = 0

    for cam_dir in sorted(images_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam = cam_dir.name

        aruco_cam_dir = aruco_masks_root / cam
        sparse_cam_dir = sparse_masks_root / cam
        aruco_cam_dir.mkdir(parents=True, exist_ok=True)
        sparse_cam_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(cam_dir.glob("*.png"))
        cam_detected = 0

        for img_path in image_files:
            total_images += 1
            mask_filename = img_path.name + ".png"  # {image}.png.png

            # 원본 이미지 로드
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [WARN] 이미지 로드 실패: {img_path.name}")
                continue

            # ArUco 마커 마스크 생성 (다중 전처리 합집합)
            aruco_mask, n_markers = detect_aruco_mask(img_bgr)
            if n_markers:
                cam_detected += n_markers
                total_detected += n_markers

            # aruco_masks/ 저장
            cv2.imwrite(str(aruco_cam_dir / mask_filename), aruco_mask)

            # 물체 마스크 로드 (있으면)
            obj_mask_path = masks_root / cam / mask_filename
            obj_mask = load_object_mask(obj_mask_path)

            # sparse_masks/ = 물체 | ArUco
            if obj_mask is not None:
                # 크기 맞추기
                if obj_mask.shape != aruco_mask.shape:
                    aruco_mask = cv2.resize(aruco_mask, (obj_mask.shape[1], obj_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                sparse_mask = cv2.bitwise_or(obj_mask, aruco_mask)
            else:
                sparse_mask = aruco_mask

            cv2.imwrite(str(sparse_cam_dir / mask_filename), sparse_mask)

        print(f"  CAM {cam}: {len(image_files)}개 이미지, 총 {cam_detected}개 마커 탐지 (평균 {cam_detected/max(len(image_files),1):.1f}개/이미지)")

    print(f"\n완료: {total_images}개 이미지 처리, 총 {total_detected}개 마커 탐지 (평균 {total_detected/max(total_images,1):.1f}개/이미지)")
    print(f"  aruco_masks/  → {aruco_masks_root}")
    print(f"  sparse_masks/ → {sparse_masks_root}")


def main():
    parser = argparse.ArgumentParser(description="ArUco 마커 마스크 생성")
    parser.add_argument("--sample", required=True, help="샘플 이름 (예: 20260318_test)")
    parser.add_argument("--runs-root", default=None, help="runs 디렉토리 경로 (기본: 스크립트 위치/runs)")
    parser.add_argument("--object-masks-dir", default=None,
                        help="원본 물체 마스크 경로 (기본: runs/{sample}/object_masks/ 또는 masks/)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    runs_root = Path(args.runs_root) if args.runs_root else script_dir / "runs"

    print(f"샘플: {args.sample}")
    print(f"runs 경로: {runs_root}")
    process_sample(args.sample, runs_root, args.object_masks_dir)


if __name__ == "__main__":
    main()
