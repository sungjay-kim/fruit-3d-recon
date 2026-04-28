"""
ArUco 마커를 이용한 Scale Factor 계산 스크립트

COLMAP sparse reconstruction 결과의 카메라 포즈를 이용해
ArUco 마커 코너를 3D 삼각측량하고, 알려진 마커 크기와 비교하여
scale factor를 계산한다.

입력:
  runs/{SAMPLE}/sparse/manual_relaxed/0/  - COLMAP sparse 모델
  runs/{SAMPLE}/images/{1,2,3}/*.png      - 원본 이미지 (ArUco 탐지용)

출력:
  runs/{SAMPLE}/scale_factor.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# COLMAP model reader
COLMAP_SCRIPTS = Path(__file__).resolve().parent.parent / "colmap" / "scripts" / "python"
sys.path.insert(0, str(COLMAP_SCRIPTS))
from read_write_model import read_model, qvec2rotmat

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
MARKER_SIZE_M = 0.03  # 실제 마커 크기 (m)


def get_projection_matrix(image, cameras):
    """COLMAP Image와 Camera에서 투영 행렬 P = K @ [R | t] 반환."""
    cam = cameras[image.camera_id]
    # PINHOLE: params = [fx, fy, cx, cy]
    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    P = K @ np.hstack([R, t])
    return P, K, R, t


def detect_aruco_corners(img_bgr):
    """이미지에서 ArUco 마커 코너 탐지. {marker_id: corners(4,2)} 반환."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(gray)
    result = {}
    if ids is None:
        return result
    for corner, mid in zip(corners, ids.flatten()):
        result[int(mid)] = corner[0]  # shape (4, 2)
    return result


def triangulate_corners(observations, cameras, images):
    """
    observations: {image_name: {marker_id: corners(4,2)}}
    여러 뷰에서 보이는 마커의 3D 코너 좌표를 삼각측량.
    반환: {marker_id: {corner_idx: np.array(3,)}}
    """
    # 이미지 이름 → COLMAP Image 객체 매핑
    name_to_image = {img.name: img for img in images.values()}

    # 마커별 코너별 관측 수집: {marker_id: {corner_idx: [(P, pt2d), ...]}}
    marker_obs = {}
    for img_name, markers in observations.items():
        if img_name not in name_to_image:
            continue
        colmap_img = name_to_image[img_name]
        P, K, R, t = get_projection_matrix(colmap_img, cameras)

        for mid, corners in markers.items():
            if mid not in marker_obs:
                marker_obs[mid] = {i: [] for i in range(4)}
            for i, pt in enumerate(corners):
                marker_obs[mid][i].append((P, pt.astype(np.float64)))

    # 삼각측량
    result = {}
    for mid, corner_obs in marker_obs.items():
        corner_3d = {}
        valid = True
        for ci, obs_list in corner_obs.items():
            if len(obs_list) < 2:
                valid = False
                break
            # 모든 뷰 쌍에서 삼각측량 후 평균
            pts3d_list = []
            for i in range(len(obs_list)):
                for j in range(i + 1, len(obs_list)):
                    P1, pt1 = obs_list[i]
                    P2, pt2 = obs_list[j]
                    pts4d = cv2.triangulatePoints(P1, P2,
                                                  pt1.reshape(2, 1),
                                                  pt2.reshape(2, 1))
                    pts3d = (pts4d[:3] / pts4d[3]).flatten()
                    pts3d_list.append(pts3d)
            corner_3d[ci] = np.mean(pts3d_list, axis=0)

        if valid and len(corner_3d) == 4:
            result[mid] = corner_3d

    return result


def compute_marker_side_lengths(corner_3d):
    """마커 코너 순서: 0(TL)-1(TR)-2(BR)-3(BL). 4개 변의 길이 반환."""
    sides = []
    order = [0, 1, 2, 3, 0]  # 4변 순환
    for i in range(4):
        a = corner_3d[order[i]]
        b = corner_3d[order[i + 1]]
        sides.append(np.linalg.norm(b - a))
    return sides


def find_best_model(sparse_dir: Path) -> Path:
    """모델 디렉토리 중 cameras.bin이 있는 첫 번째 모델 반환."""
    for model_dir in sorted(sparse_dir.iterdir()):
        if (model_dir / "cameras.bin").exists():
            return model_dir
    raise FileNotFoundError(f"sparse 모델을 찾을 수 없음: {sparse_dir}")


def main():
    parser = argparse.ArgumentParser(description="ArUco 마커로 scale factor 계산")
    parser.add_argument("--sample", required=True, help="샘플 이름 (예: 20260318_test)")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--model-idx", default=None, type=int,
                        help="sparse 모델 인덱스 (기본: 자동 선택)")
    parser.add_argument("--marker-size-m", default=MARKER_SIZE_M, type=float,
                        help=f"마커 실제 크기 m (기본: {MARKER_SIZE_M})")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    runs_root = Path(args.runs_root) if args.runs_root else script_dir / "runs"
    run_dir = runs_root / args.sample

    # Sparse 모델 경로
    sparse_base = run_dir / "sparse" / "manual_relaxed"
    if args.model_idx is not None:
        model_dir = sparse_base / str(args.model_idx)
    else:
        model_dir = find_best_model(sparse_base)
    print(f"Sparse 모델: {model_dir}")

    # COLMAP 모델 읽기
    cameras, images, _ = read_model(str(model_dir), ext=".bin")
    print(f"등록된 카메라: {len(cameras)}, 이미지: {len(images)}")

    # 이미지별 ArUco 탐지
    images_root = run_dir / "images"
    observations = {}  # {image_name: {marker_id: corners}}
    total_detections = 0

    for img_obj in images.values():
        # COLMAP 이미지 이름: "1/CAM#1_...(Degree-0).png" 형태
        img_name = img_obj.name
        img_path = images_root / img_name
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        markers = detect_aruco_corners(img_bgr)
        if markers:
            observations[img_name] = markers
            total_detections += sum(1 for _ in markers)

    print(f"ArUco 탐지: {len(observations)}개 이미지, 총 {total_detections}개 마커 관측")

    if not observations:
        print("[ERROR] ArUco 마커를 탐지하지 못했습니다.")
        sys.exit(1)

    # 3D 삼각측량
    marker_3d = triangulate_corners(observations, cameras, images)
    print(f"삼각측량 성공 마커: {sorted(marker_3d.keys())}")

    if not marker_3d:
        print("[ERROR] 삼각측량에 성공한 마커가 없습니다. (각 마커가 최소 2개 뷰에서 보여야 함)")
        sys.exit(1)

    # Scale factor 계산
    all_sides = []
    measurements = []
    for mid, corner_3d in marker_3d.items():
        sides = compute_marker_side_lengths(corner_3d)
        mean_side = np.mean(sides)
        scale = args.marker_size_m / mean_side
        all_sides.extend(sides)
        measurements.append({
            "marker_id": mid,
            "sides_colmap_units": [float(s) for s in sides],
            "mean_side_colmap_units": float(mean_side),
            "scale_factor": float(scale),
        })
        print(f"  마커 ID {mid:2d}: 변 길이 {[f'{s:.4f}' for s in sides]} → scale={scale:.6f}")

    # 이상치 마커 제거 (median 기준 30% 이상 벗어나면 제외)
    per_marker_scales = [m["scale_factor"] for m in measurements]
    median_scale = float(np.median(per_marker_scales))
    excluded_ids = []
    filtered_sides = []
    for m in measurements:
        deviation = abs(m["scale_factor"] - median_scale) / median_scale
        if deviation > 0.3:
            excluded_ids.append(m["marker_id"])
            print(f"  [제외] 마커 ID {m['marker_id']}: scale={m['scale_factor']:.6f} (median 대비 {deviation*100:.0f}% 벗어남)")
        else:
            filtered_sides.extend(m["sides_colmap_units"])

    if filtered_sides:
        scale_factor = float(args.marker_size_m / np.mean(filtered_sides))
    else:
        scale_factor = float(args.marker_size_m / np.mean(all_sides))

    if excluded_ids:
        print(f"\n이상치 마커 제외: {excluded_ids}")
    print(f"\n최종 scale factor: {scale_factor:.6f} m/COLMAP_unit")
    print(f"  ({len(marker_3d) - len(excluded_ids)}개 마커 사용, {len(excluded_ids)}개 제외)")

    # 저장
    output = {
        "sample": args.sample,
        "scale_factor": scale_factor,
        "unit": "m",
        "marker_size_m": args.marker_size_m,
        "num_markers": len(marker_3d) - len(excluded_ids),
        "num_measurements": len(filtered_sides),
        "measurements": measurements,
        "excluded_marker_ids": excluded_ids,
    }
    out_path = run_dir / "scale_factor.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
