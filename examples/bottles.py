#!/usr/bin/env python3
"""
parse_water_sort.py

Heuristic parser for Water-Sort style puzzle screenshots.

Reads:
- number of bottles detected
- rock bottle presence (bottle cannot pour out of)
- how many distinct colors (auto-clustered)
- how many of each color (unit counts)
- exactly which colors are in which bottle (top->bottom slots)

Dependencies:
  pip install opencv-python numpy scikit-learn pytesseract
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import re
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    pytesseract = None
    HAVE_TESSERACT = False

@dataclass(frozen=True)
class BottleBox:
    x: int
    y: int
    w: int
    h: int


def _iou(a: BottleBox, b: BottleBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    union = float(a.w * a.h + b.w * b.h - inter)
    return inter / union if union > 0 else 0.0

def detect_red_badges(
    img_bgr: np.ndarray,
    y_min_frac: float = 0.65,
    min_area: int = 3500,
    max_area: int = 25000,
) -> List[Tuple[int, int, int, int]]:
    """
    Finds the red circular count badges near the bottom of the screen.
    Returns list of (x,y,w,h) sorted left->right.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps hue: [0..10] and [170..179]
    lower1 = np.array([0, 120, 120], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 120], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img_bgr.shape[:2]
    candidates: List[Tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if y < H * y_min_frac:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        c_area = cv2.contourArea(cnt)
        circ = 4.0 * math.pi * c_area / (peri * peri)

        ar = w / float(h + 1e-6)
        if circ < 0.45:
            continue
        if not (0.7 < ar < 1.3):
            continue

        candidates.append((x, y, w, h, area))

    # take largest and sort left->right
    candidates.sort(key=lambda t: t[4], reverse=True)
    boxes = [(x, y, w, h) for (x, y, w, h, _) in candidates[:3]]
    boxes.sort(key=lambda b: b[0])
    return boxes


def ocr_badge_number(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
    """
    OCR a single red badge's number.
    Returns int or None if OCR fails or tesseract unavailable.
    """
    if not HAVE_TESSERACT:
        return None

    x, y, w, h = bbox
    crop = img_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # Digits are white-ish: low saturation, high value
    digit_mask = ((s < 90) & (v > 170)).astype(np.uint8) * 255
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Crop to digits only
    coords = cv2.findNonZero(digit_mask)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        pad = int(max(2, min(rw, rh) * 0.10))
        rx = max(0, rx - pad)
        ry = max(0, ry - pad)
        rw = min(digit_mask.shape[1] - rx, rw + 2 * pad)
        rh = min(digit_mask.shape[0] - ry, rh + 2 * pad)
        roi = digit_mask[ry:ry + rh, rx:rx + rw]
    else:
        roi = digit_mask

    # Tesseract likes black text on white background
    inv = 255 - roi
    inv = cv2.resize(inv, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    inv = cv2.morphologyEx(inv, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), iterations=1)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(inv, config=config)
    txt = re.sub(r"\D", "", txt)

    return int(txt) if txt else None


def read_powerup_counts(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Reads (retries, shuffles, add_bottles) from the bottom badges.
    Returns dict with values and badge boxes used.
    """
    badge_boxes = detect_red_badges(img_bgr)
    values = [ocr_badge_number(img_bgr, b) for b in badge_boxes]

    # Expect left->right: retries, shuffles, add_bottles
    result = {
        "retries": values[0] if len(values) > 0 else None,
        "shuffles": values[1] if len(values) > 1 else None,
        "add_bottles": values[2] if len(values) > 2 else None,
        "badge_boxes": badge_boxes,
        "tesseract_available": HAVE_TESSERACT,
    }
    return result



def detect_bottles(
    img_bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int] = (90, 20, 120),
    hsv_upper: Tuple[int, int, int] = (125, 255, 255),
    min_area: int = 40000,
    aspect_min: float = 2.0,
    aspect_max: float = 4.5,
    nms_iou_thresh: float = 0.30,
) -> List[BottleBox]:
    """
    Detect bottle bounding boxes by thresholding the cyan/blue bottle outline.

    NOTE: This is theme-dependent. If it detects 0 bottles, tune --outline-lower/--outline-upper.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, BottleBox]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = int(w * h)
        if area < min_area:
            continue
        aspect = float(h) / float(w + 1e-6)
        if not (aspect_min <= aspect <= aspect_max):
            continue
        candidates.append((area, BottleBox(x, y, w, h)))

    # Non-max suppression to remove overlapping duplicates
    candidates.sort(key=lambda t: t[0], reverse=True)
    selected: List[BottleBox] = []
    for _, box in candidates:
        if all(_iou(box, s) < nms_iou_thresh for s in selected):
            selected.append(box)

    if not selected:
        return []

    # Sort in a stable reading order: row-by-row, left-to-right
    ys = np.array([b.y + b.h / 2.0 for b in selected], dtype=np.float32).reshape(-1, 1)
    med_h = float(np.median([b.h for b in selected]))
    row_labels = DBSCAN(eps=max(40.0, med_h * 0.60), min_samples=1).fit_predict(ys)

    rows: List[Tuple[float, List[BottleBox]]] = []
    for lbl in sorted(set(row_labels)):
        row_boxes = [b for b, l in zip(selected, row_labels) if l == lbl]
        row_y = float(np.mean([b.y + b.h / 2.0 for b in row_boxes]))
        row_boxes.sort(key=lambda b: b.x)
        rows.append((row_y, row_boxes))
    rows.sort(key=lambda t: t[0])

    ordered: List[BottleBox] = []
    for _, row in rows:
        ordered.extend(row)
    return ordered


def is_filled_slot(hsv_med: Tuple[float, float, float]) -> bool:
    """
    Decide if a sampled slot is 'liquid' vs 'empty' using HSV medians.

    Tuned for:
      - bright saturated liquids
      - darker saturated liquids (e.g., brown) that have lower V
    """
    _, s, v = hsv_med
    if v >= 130 and s >= 40:
        return True
    if s >= 150 and v >= 80:
        return True
    return False


def bgr_list_to_lab(bgr_list: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    arr = np.array(bgr_list, dtype=np.uint8).reshape(-1, 1, 3)  # BGR
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab


def bgr_to_hex(bgr: Tuple[float, float, float]) -> str:
    b, g, r = [int(round(x)) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def detect_rock_bottles(
    img_bgr: np.ndarray,
    boxes: Sequence[BottleBox],
    below_extend: float = 0.30,
    s_max: int = 80,
    v_min: int = 160,
    ratio_thresh: float = 0.10,
) -> List[bool]:
    """
    Detect rock bottles by looking for a bright low-saturation 'rock pile'
    directly beneath the bottle bounding box.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, _ = hsv.shape[:2]
    flags: List[bool] = []
    for b in boxes:
        y1 = b.y + b.h
        y2 = min(H, int(b.y + b.h + b.h * below_extend))
        if y1 >= y2:
            flags.append(False)
            continue
        region = hsv[y1:y2, b.x : b.x + b.w]
        if region.size == 0:
            flags.append(False)
            continue
        _, s, v = cv2.split(region)
        rock_mask = (s < s_max) & (v > v_min)
        ratio = float(np.mean(rock_mask))
        flags.append(ratio >= ratio_thresh)
    return flags


def sample_bottle_slots(
    img_bgr: np.ndarray,
    boxes: Sequence[BottleBox],
    capacity: int = 4,
    inner_x_frac: float = 0.28,
    top_frac: float = 0.18,
    bottom_frac: float = 0.04,
    patch_half_h: int = 6,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float]]]:
    """
    For each bottle, sample `capacity` slots (top->bottom) and compute median HSV + BGR.
    Returns:
      - bottles list with per-slot sample indices into global `samples_bgr`
      - samples_bgr: BGR tuples for each filled slot
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    bottles: List[Dict[str, Any]] = []
    samples_bgr: List[Tuple[float, float, float]] = []

    for b in boxes:
        mx = int(round(b.w * inner_x_frac))
        top = int(round(b.h * top_frac))
        bot = int(round(b.h * bottom_frac))

        ix = b.x + mx
        iw = b.w - 2 * mx
        iy = b.y + top
        ih = b.h - top - bot

        iw = max(1, iw)
        ih = max(1, ih)
        ix = max(0, min(ix, img_bgr.shape[1] - 1))
        iy = max(0, min(iy, img_bgr.shape[0] - 1))

        slot_h = ih / float(capacity)
        slots: List[Dict[str, Any]] = []

        for s_idx in range(capacity):
            cy = int(round(iy + (s_idx + 0.5) * slot_h))
            y1 = max(iy, cy - patch_half_h)
            y2 = min(iy + ih, cy + patch_half_h)

            patch_bgr = img_bgr[y1:y2, ix : ix + iw]
            patch_hsv = hsv[y1:y2, ix : ix + iw]
            if patch_bgr.size == 0:
                slots.append({"filled": False, "sample_idx": None, "hsv": None, "bgr": None})
                continue

            hs, ss, vs = cv2.split(patch_hsv)
            med_h = float(np.median(hs))
            med_s = float(np.median(ss))
            med_v = float(np.median(vs))
            med_bgr = tuple(map(float, np.median(patch_bgr.reshape(-1, 3), axis=0)))

            filled = is_filled_slot((med_h, med_s, med_v))
            sample_idx: Optional[int] = None
            if filled:
                sample_idx = len(samples_bgr)
                samples_bgr.append(med_bgr)

            slots.append(
                {
                    "filled": filled,
                    "sample_idx": sample_idx,
                    "hsv": (med_h, med_s, med_v),
                    "bgr": med_bgr,
                }
            )

        bottles.append({"bbox": [b.x, b.y, b.w, b.h], "slots": slots})

    return bottles, samples_bgr


def cluster_colors_dbscan(
    samples_bgr: Sequence[Tuple[float, float, float]],
    eps: float = 16.0,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Cluster slot colors using DBSCAN in Lab space; automatically finds number of colors.
    """
    if len(samples_bgr) == 0:
        return np.array([], dtype=int), {}

    labs = bgr_list_to_lab(samples_bgr)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(labs)

    cluster_info: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == cid)[0]
        center_bgr = np.mean(np.array(samples_bgr, dtype=np.float32)[idx], axis=0)
        cluster_info[cid] = {
            "id": cid,
            "count": int(len(idx)),
            "center_bgr": [float(center_bgr[0]), float(center_bgr[1]), float(center_bgr[2])],
            "center_hex": bgr_to_hex(tuple(center_bgr.tolist())),
        }

    return labels, cluster_info


def build_result(
    img_bgr: np.ndarray,
    boxes: List[BottleBox],
    capacity: int,
    dbscan_eps: float,
    debug: bool = False,
    debug_out: str = "debug_annotated.png",
) -> Dict[str, Any]:
    bottles, samples_bgr = sample_bottle_slots(img_bgr, boxes, capacity=capacity)
    rock_flags = detect_rock_bottles(img_bgr, boxes)

    labels, cluster_info = cluster_colors_dbscan(samples_bgr, eps=dbscan_eps)

    # Attach rock flags and color IDs to each bottle slot
    for bi, b in enumerate(bottles):
        b["rock"] = bool(rock_flags[bi]) if bi < len(rock_flags) else False
        for slot in b["slots"]:
            if slot["filled"] and slot["sample_idx"] is not None:
                cid = int(labels[slot["sample_idx"]])
                slot["color_id"] = cid
                slot["color_hex"] = cluster_info[cid]["center_hex"]
            else:
                slot["color_id"] = None
                slot["color_hex"] = None

    color_counts = {str(cid): info["count"] for cid, info in cluster_info.items()}
    bottle_contents_ids = [[s["color_id"] for s in b["slots"]] for b in bottles]
    bottle_contents_hex = [[s["color_hex"] for s in b["slots"]] for b in bottles]
    colors_by_id = {str(cid): info["center_hex"] for cid, info in cluster_info.items()}

    result: Dict[str, Any] = {
        "num_bottles": len(bottles),
        "capacity": capacity,
        "powerups": read_powerup_counts(img_bgr),
        "rock_bottles": [i for i, f in enumerate(rock_flags) if f],
        "num_colors": len(cluster_info),
        "colors_by_id": colors_by_id,
        "colors": [cluster_info[cid] for cid in sorted(cluster_info)],
        "color_unit_counts": color_counts,
        "bottle_contents_ids": bottle_contents_ids,
        "bottle_contents_hex": bottle_contents_hex,
        "bottles": bottles,
    }

    if debug:
        dbg = img_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw bottle boxes
        for i, (b, rock) in enumerate(zip(boxes, rock_flags)):
            cv2.rectangle(dbg, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 255, 0), 2)
            label = f"{i}" + (" ROCK" if rock else "")
            cv2.putText(dbg, label, (b.x, max(20, b.y - 10)), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw sampled slot centers
        for bi, b in enumerate(boxes):
            x, y, w, h = b.x, b.y, b.w, b.h
            mx = int(round(w * 0.28))
            top = int(round(h * 0.18))
            bot = int(round(h * 0.04))
            ix = x + mx
            iw = w - 2 * mx
            iy = y + top
            ih = h - top - bot
            slot_h = ih / float(capacity)

            for si in range(capacity):
                cy = int(round(iy + (si + 0.5) * slot_h))
                cx = int(round(ix + iw / 2.0))
                slot = bottles[bi]["slots"][si]

                if slot["color_id"] is None:
                    cv2.circle(dbg, (cx, cy), 6, (128, 128, 128), -1)
                else:
                    # marker color = cluster center
                    hex_col = slot["color_hex"].lstrip("#")
                    r = int(hex_col[0:2], 16)
                    g = int(hex_col[2:4], 16)
                    bcol = int(hex_col[4:6], 16)
                    cv2.circle(dbg, (cx, cy), 8, (bcol, g, r), -1)
                    cv2.putText(dbg, str(slot["color_id"]), (cx + 10, cy + 5), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if "powerups" in result and result["powerups"]["badge_boxes"]:
            for i, (x, y, w, h) in enumerate(result["powerups"]["badge_boxes"]):
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(dbg, f"badge{i}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(debug_out, dbg)
        result["debug_image"] = debug_out

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a Water Sort puzzle screenshot into bottle/slot colors.")
    ap.add_argument("image", help="Path to screenshot image (png/jpg)")
    ap.add_argument("--capacity", type=int, default=4, help="Bottle capacity / number of slots (default: 4)")
    ap.add_argument("--dbscan-eps", type=float, default=16.0, help="DBSCAN eps in Lab space (default: 16)")
    ap.add_argument(
        "--outline-lower",
        type=int,
        nargs=3,
        default=(90, 20, 120),
        metavar=("H", "S", "V"),
        help="HSV lower bound for bottle outline detection",
    )
    ap.add_argument(
        "--outline-upper",
        type=int,
        nargs=3,
        default=(125, 255, 255),
        metavar=("H", "S", "V"),
        help="HSV upper bound for bottle outline detection",
    )
    ap.add_argument("--debug", action="store_true", help="Write an annotated debug image")
    ap.add_argument("--debug-out", default="debug_annotated.png", help="Debug image output path")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    boxes = detect_bottles(img, hsv_lower=tuple(args.outline_lower), hsv_upper=tuple(args.outline_upper))
    if not boxes:
        raise SystemExit(
            "No bottles detected. Try adjusting --outline-lower/--outline-upper (HSV) or check image quality."
        )

    result = build_result(
        img,
        boxes,
        capacity=args.capacity,
        dbscan_eps=args.dbscan_eps,
        debug=args.debug,
        debug_out=args.debug_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
