import cv2
import numpy as np
from PIL import Image
import piexif

# ============================================================
#  Read DPI if available
# ============================================================
def read_dpi(image_path):
    try:
        img = Image.open(image_path)
        info = img.info
        if "dpi" in info:
            return float(info["dpi"][0])   # DPI stored as (x, y)
    except:
        pass
    return None  # No DPI found


# ============================================================
#  Detect 1 mm tick marks on radiograph
# ============================================================
def detect_mm_ticks(image_path):
    """
    Detects 1 mm tick spacing by looking for vertical bars
    typically printed on periapical radiographs.
    Returns pixel spacing OR None if not detected.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape

    # --- Crop bottom-left corner (where mm ticks usually appear)
    crop = img[int(h * 0.80):h, 0:int(w * 0.15)]

    # --- Edge enhance
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # --- Sum vertical pixel intensities
    vertical_profile = np.sum(edges, axis=0)

    # --- Find peaks (tick marks)
    threshold = np.max(vertical_profile) * 0.4
    peaks = np.where(vertical_profile > threshold)[0]

    if len(peaks) < 3:  # Not enough ticks found
        return None

    # --- Group peaks into tick positions
    groups = []
    current = [peaks[0]]

    for p in peaks[1:]:
        if p - current[-1] < 4:  # same peak cluster
            current.append(p)
        else:
            groups.append(current)
            current = [p]

    groups.append(current)

    # --- Compute centers of the peaks
    centers = [int(np.mean(g)) for g in groups]

    if len(centers) < 3:
        return None

    # --- Compute spacing between consecutive ticks
    spacings = np.diff(centers)

    # Remove outliers
    spacings = spacings[(spacings > 2) & (spacings < 80)]

    if len(spacings) == 0:
        return None

    return float(np.mean(spacings))  # pixel spacing per 1 mm


# ============================================================
#  Compute mm-per-pixel
# ============================================================
def compute_mm_per_pixel(image_path):
    # 1) Try reading DPI
    dpi = read_dpi(image_path)
    if dpi:
        return 25.4 / dpi, "dpi"

    # 2) Try detecting mm tick marks
    tick_px = detect_mm_ticks(image_path)
    if tick_px:
        return 1.0 / tick_px, "ticks"

    # 3) Fallback DPI = 300
    return 25.4 / 300.0, "fallback"


# ============================================================
#  Convert pixel results (“mean=XXpx”) into mm
# ============================================================
def convert_px_text_to_mm(text, mm_per_pixel):
    """
    Input:  "Tooth 1: mean=43.82px"
    Output: "Tooth 1: 2.92 mm"
    """
    lines = text.split("\n")
    final_lines = []

    for line in lines:
        if "mean=" in line and "px" in line:
            try:
                px = float(line.split("mean=")[1].split("px")[0])
                mm = px * mm_per_pixel
                line = line.split("mean=")[0] + f"{mm:.2f} mm"
            except:
                pass
        final_lines.append(line)

    return "\n".join(final_lines)
