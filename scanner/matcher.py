# -*- coding: utf-8 -*-
"""
Visual Matcher - matches live camera view against saved labeled crops.
Uses ORB feature matching to find custom-labeled objects in live frames.
"""

import cv2
import os
import json
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

# Cache: list of {label, keypoints, descriptors, size}
_templates = []
_templates_loaded = False


def _load_templates():
    """Load all custom-labeled crops and compute ORB features."""
    global _templates, _templates_loaded
    _templates = []

    cust_path = os.path.join(LABELS_DIR, "custom_labels.json")
    if not os.path.exists(cust_path):
        _templates_loaded = True
        return

    with open(cust_path) as f:
        custom = json.load(f)

    orb = cv2.ORB_create(nfeatures=500)

    for fname, labels in custom.items():
        img_path = os.path.join(FRAMES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        for lb in labels:
            x1, y1, x2, y2 = lb["box"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(des) < 8:
                continue
            _templates.append({
                "label": lb["label"],
                "kp":    kp,
                "des":   des,
                "w":     x2 - x1,
                "h":     y2 - y1,
            })

    _templates_loaded = True
    print(f"Matcher: loaded {len(_templates)} labeled templates")


def reload_templates():
    """Force reload templates (call after new labels are saved)."""
    global _templates_loaded
    _templates_loaded = False
    _load_templates()


def match_frame(frame, min_matches=12):
    """
    Find custom-labeled objects in a live frame using ORB feature matching.
    Returns list of {label, box:[x1,y1,x2,y2], confidence}
    """
    global _templates_loaded
    if not _templates_loaded:
        _load_templates()

    if not _templates:
        return []

    orb     = cv2.ORB_create(nfeatures=500)
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if des2 is None or len(des2) < 8:
        return []

    results = []
    fh, fw  = frame.shape[:2]

    for tmpl in _templates:
        try:
            matches = bf.knnMatch(tmpl["des"], des2, k=2)
        except Exception:
            continue

        # Lowe's ratio test
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < min_matches:
            continue

        # Estimate bounding box from matched keypoints in frame
        pts = np.float32([kp2[m.trainIdx].pt for m in good])
        cx  = int(np.mean(pts[:, 0]))
        cy  = int(np.mean(pts[:, 1]))
        hw  = max(tmpl["w"] // 2, 40)
        hh  = max(tmpl["h"] // 2, 40)

        x1 = max(0,  cx - hw)
        y1 = max(0,  cy - hh)
        x2 = min(fw, cx + hw)
        y2 = min(fh, cy + hh)

        conf = min(len(good) / 30.0, 1.0)  # rough confidence
        results.append({
            "label":      tmpl["label"],
            "box":        [x1, y1, x2, y2],
            "confidence": round(conf, 2),
            "known":      True,
            "source":     "custom"
        })

    # Deduplicate overlapping boxes for same label
    return _deduplicate(results)


def _deduplicate(results, iou_thresh=0.4):
    """Remove duplicate detections of the same label with overlapping boxes."""
    kept = []
    for r in results:
        overlap = False
        for k in kept:
            if k["label"] == r["label"] and _iou(r["box"], k["box"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(r)
    return kept


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0
