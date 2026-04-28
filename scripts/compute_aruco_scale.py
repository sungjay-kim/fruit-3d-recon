"""
Compute the metric scale factor from triangulated ArUco markers.

Uses camera poses from a COLMAP sparse reconstruction to triangulate ArUco
marker corners in 3D, then compares against the known physical marker size
to recover a scale factor.

Inputs:
  runs/{SAMPLE}/sparse/manual_relaxed/0/  - COLMAP sparse model
  runs/{SAMPLE}/images/{1,2,3}/*.png      - source RGB (for ArUco detection)

Output:
  runs/{SAMPLE}/scale_factor.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# COLMAP model reader (vendored in third_party/)
COLMAP_SCRIPTS = Path(__file__).resolve().parent.parent / "third_party" / "colmap_python"
sys.path.insert(0, str(COLMAP_SCRIPTS))
from read_write_model import read_model, qvec2rotmat

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
MARKER_SIZE_M = 0.03  # physical marker size (m)


def get_projection_matrix(image, cameras):
    """Build the projection matrix P = K @ [R | t] from a COLMAP Image and Camera."""
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
    """Detect ArUco markers in an image. Returns {marker_id: corners(4,2)}."""
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

    Triangulate marker corners across all views and return:
        {marker_id: {corner_idx: np.array(3,)}}
    """
    name_to_image = {img.name: img for img in images.values()}

    # Per-marker per-corner observations: {marker_id: {corner_idx: [(P, pt2d), ...]}}
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

    result = {}
    for mid, corner_obs in marker_obs.items():
        corner_3d = {}
        valid = True
        for ci, obs_list in corner_obs.items():
            if len(obs_list) < 2:
                valid = False
                break
            # Triangulate from every pair of views and average
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
    """Marker corner order: 0(TL)-1(TR)-2(BR)-3(BL). Return the 4 side lengths."""
    sides = []
    order = [0, 1, 2, 3, 0]  # close the loop
    for i in range(4):
        a = corner_3d[order[i]]
        b = corner_3d[order[i + 1]]
        sides.append(np.linalg.norm(b - a))
    return sides


def find_best_model(sparse_dir: Path) -> Path:
    """Return the first model directory containing cameras.bin."""
    for model_dir in sorted(sparse_dir.iterdir()):
        if (model_dir / "cameras.bin").exists():
            return model_dir
    raise FileNotFoundError(f"sparse model not found: {sparse_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute scale factor from ArUco markers")
    parser.add_argument("--sample", required=True, help="sample name (e.g. 20260318_test)")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--model-idx", default=None, type=int,
                        help="sparse model index (default: auto-select)")
    parser.add_argument("--marker-size-m", default=MARKER_SIZE_M, type=float,
                        help=f"physical marker size in m (default: {MARKER_SIZE_M})")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    runs_root = Path(args.runs_root) if args.runs_root else script_dir / "runs"
    run_dir = runs_root / args.sample

    sparse_base = run_dir / "sparse" / "manual_relaxed"
    if args.model_idx is not None:
        model_dir = sparse_base / str(args.model_idx)
    else:
        model_dir = find_best_model(sparse_base)
    print(f"Sparse model: {model_dir}")

    cameras, images, _ = read_model(str(model_dir), ext=".bin")
    print(f"Registered cameras: {len(cameras)}, images: {len(images)}")

    images_root = run_dir / "images"
    observations = {}  # {image_name: {marker_id: corners}}
    total_detections = 0

    for img_obj in images.values():
        # COLMAP image name format: "1/CAM#1_...(Degree-0).png"
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

    print(f"ArUco detection: {len(observations)} images, {total_detections} marker observations total")

    if not observations:
        print("[ERROR] no ArUco markers detected.")
        sys.exit(1)

    marker_3d = triangulate_corners(observations, cameras, images)
    print(f"Triangulated markers: {sorted(marker_3d.keys())}")

    if not marker_3d:
        print("[ERROR] no markers were successfully triangulated. (each marker must be visible in >=2 views)")
        sys.exit(1)

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
        print(f"  marker ID {mid:2d}: side lengths {[f'{s:.4f}' for s in sides]} → scale={scale:.6f}")

    # Remove outlier markers (>30% deviation from median scale)
    per_marker_scales = [m["scale_factor"] for m in measurements]
    median_scale = float(np.median(per_marker_scales))
    excluded_ids = []
    filtered_sides = []
    for m in measurements:
        deviation = abs(m["scale_factor"] - median_scale) / median_scale
        if deviation > 0.3:
            excluded_ids.append(m["marker_id"])
            print(f"  [excluded] marker ID {m['marker_id']}: scale={m['scale_factor']:.6f} ({deviation*100:.0f}% off median)")
        else:
            filtered_sides.extend(m["sides_colmap_units"])

    if filtered_sides:
        scale_factor = float(args.marker_size_m / np.mean(filtered_sides))
    else:
        scale_factor = float(args.marker_size_m / np.mean(all_sides))

    if excluded_ids:
        print(f"\nExcluded outlier markers: {excluded_ids}")
    print(f"\nFinal scale factor: {scale_factor:.6f} m/COLMAP_unit")
    print(f"  (used {len(marker_3d) - len(excluded_ids)} markers, excluded {len(excluded_ids)})")

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
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
