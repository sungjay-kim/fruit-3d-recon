"""
ArUco 마커 생성 스크립트
- 3cm x 3cm 마커 12개 (ID 0~11)
- 기둥 4개 x 마커 3개 배치용
- 인쇄 시 실제 크기 유지를 위해 DPI 설정 포함
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


MARKER_SIZE_CM = 3.0
MARKER_COUNT = 24  # 기둥 4개 x 높이 3개 x 앞뒤 2면
DPI = 300
MARGIN_CM = 0.5  # 마커 주변 여백 (절단 가이드)
WHITE_BORDER_CM = 0.3  # 마커 주변 흰색 테두리 (검은 배경 대비용)

# ArUco dictionary (4x4, 50개 ID)
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def cm_to_px(cm, dpi=DPI):
    return int(cm / 2.54 * dpi)


def generate_single_markers(output_dir: Path):
    """개별 마커 PNG 파일 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    marker_px = cm_to_px(MARKER_SIZE_CM)

    border_px = cm_to_px(WHITE_BORDER_CM)
    total_px = marker_px + border_px * 2

    for marker_id in range(MARKER_COUNT):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        # 흰색 테두리 추가 (검은 기둥 대비용)
        bordered = np.ones((total_px, total_px), dtype=np.uint8) * 255
        bordered[border_px:border_px + marker_px, border_px:border_px + marker_px] = marker_img
        filepath = output_dir / f"aruco_id{marker_id:02d}_{MARKER_SIZE_CM:.0f}cm.png"
        cv2.imwrite(str(filepath), bordered)
        print(f"Generated: {filepath.name} ({total_px}x{total_px}px, white border {WHITE_BORDER_CM}cm)")


def generate_print_sheet(output_dir: Path):
    """A4 인쇄용 시트 (모든 마커를 한 장에 배치)"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    marker_px = cm_to_px(MARKER_SIZE_CM)
    margin_px = cm_to_px(MARGIN_CM)
    cell_size = marker_px + margin_px * 2

    # A4 크기 (21cm x 29.7cm)
    a4_w = cm_to_px(21.0)
    a4_h = cm_to_px(29.7)

    cols = a4_w // cell_size
    rows = (MARKER_COUNT + cols - 1) // cols

    sheet = np.ones((a4_h, a4_w), dtype=np.uint8) * 255

    for i in range(MARKER_COUNT):
        r, c = divmod(i, cols)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_px)

        y = margin_px + r * cell_size
        x = margin_px + c * cell_size

        if y + marker_px > a4_h or x + marker_px > a4_w:
            print(f"Warning: marker ID {i} does not fit on sheet")
            continue

        sheet[y:y + marker_px, x:x + marker_px] = marker_img

        # ID 라벨 추가
        label_y = y + marker_px + cm_to_px(0.3)
        if label_y < a4_h:
            cv2.putText(sheet, f"ID:{i}", (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

    filepath = output_dir / f"aruco_print_sheet_{MARKER_SIZE_CM:.0f}cm_x{MARKER_COUNT}.png"
    cv2.imwrite(str(filepath), sheet)
    print(f"\nPrint sheet (PNG): {filepath.name}")


def generate_pdf(output_dir: Path):
    """A4 PDF 인쇄용 시트 (실제 크기 보장)"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_px = cm_to_px(MARKER_SIZE_CM)

    # A4 in cm
    a4_w_cm, a4_h_cm = 21.0, 29.7
    margin_cm = 1.0
    label_space_cm = 0.6
    cell_w_cm = MARKER_SIZE_CM + MARGIN_CM * 2
    cell_h_cm = MARKER_SIZE_CM + MARGIN_CM * 2 + label_space_cm

    cols = int((a4_w_cm - margin_cm * 2) // cell_w_cm)
    rows_per_page = int((a4_h_cm - margin_cm * 2) // cell_h_cm)
    per_page = cols * rows_per_page

    # A4 in inches for matplotlib
    a4_w_in = a4_w_cm / 2.54
    a4_h_in = a4_h_cm / 2.54

    filepath = output_dir / f"aruco_markers_{MARKER_SIZE_CM:.0f}cm_x{MARKER_COUNT}.pdf"

    with PdfPages(str(filepath)) as pdf:
        for page_start in range(0, MARKER_COUNT, per_page):
            fig = plt.figure(figsize=(a4_w_in, a4_h_in), dpi=DPI)

            for local_idx, marker_id in enumerate(range(page_start, min(page_start + per_page, MARKER_COUNT))):
                r, c = divmod(local_idx, cols)

                # 마커 위치 (cm → figure 비율)
                x_cm = margin_cm + c * cell_w_cm + MARGIN_CM
                y_cm = a4_h_cm - margin_cm - r * cell_h_cm - MARGIN_CM - MARKER_SIZE_CM

                x_frac = x_cm / a4_w_cm
                y_frac = y_cm / a4_h_cm
                w_frac = MARKER_SIZE_CM / a4_w_cm
                h_frac = MARKER_SIZE_CM / a4_h_cm

                marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
                # 흰색 테두리 추가
                border_px = cm_to_px(WHITE_BORDER_CM)
                total_px = marker_px + border_px * 2
                bordered = np.ones((total_px, total_px), dtype=np.uint8) * 255
                bordered[border_px:border_px + marker_px, border_px:border_px + marker_px] = marker_img

                # 흰색 테두리 포함 크기로 axes 조정
                total_cm = MARKER_SIZE_CM + WHITE_BORDER_CM * 2
                w_frac_b = total_cm / a4_w_cm
                h_frac_b = total_cm / a4_h_cm
                x_frac_b = x_frac - WHITE_BORDER_CM / a4_w_cm
                y_frac_b = y_frac - WHITE_BORDER_CM / a4_h_cm

                ax = fig.add_axes([x_frac_b, y_frac_b, w_frac_b, h_frac_b])
                ax.imshow(bordered, cmap='gray', interpolation='nearest')
                ax.axis('off')

                # ID 라벨
                label_x = x_frac + w_frac / 2
                label_y = y_frac - 0.01
                fig.text(label_x, label_y, f"ID:{marker_id}  {MARKER_SIZE_CM}cm",
                         ha='center', va='top', fontsize=8)

            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)

    print(f"\nPDF: {filepath.name}")
    print(f"  {MARKER_COUNT} markers, {MARKER_SIZE_CM}cm x {MARKER_SIZE_CM}cm each")
    print(f"  Print at 100% scale (no fit-to-page)")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "aruco_markers"
    generate_single_markers(output_dir)
    generate_print_sheet(output_dir)
    generate_pdf(output_dir)
    print(f"\nTotal: {MARKER_COUNT} markers (ID 0-{MARKER_COUNT - 1})")
    print(f"Dictionary: DICT_4X4_50")
    print(f"Marker size: {MARKER_SIZE_CM}cm x {MARKER_SIZE_CM}cm")
