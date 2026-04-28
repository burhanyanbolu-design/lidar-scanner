"""
RF-DETR Object Detector - Drop-in upgrade for the LiDAR Room Scanner.

Replaces YOLOv8 with RF-DETR (Real-time Detection Transformer) from Hugging Face.
RF-DETR uses a DINOv2 ViT backbone + deformable DETR decoder for better accuracy
on cluttered indoor scenes (rooms, furniture, equipment) vs YOLOv8 nano.

Usage:
    - Set USE_RFDETR = True in this file to switch from YOLOv8 to RF-DETR
    - Or import detect_frame / detect_all_frames directly from this module
    - All return formats are identical to detector.py so it's a drop-in swap

Model sizes (trade accuracy vs speed):
    - stevenbucaille/rf-detr-small   (~fastest, good for live feed)
    - stevenbucaille/rf-detr-medium  (~balanced, recommended)
    - stevenbucaille/rf-detr-large   (~best accuracy, slower)
"""

import cv2
import os
import json
import glob
import numpy as np
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Config ───────────────────────────────────────────────────────────────────
USE_RFDETR    = True                              # False = fall back to YOLOv8
RFDETR_MODEL  = "stevenbucaille/rf-detr-small"   # change to -medium or -large
CONFIDENCE    = 0.40                              # detection threshold (0.0–1.0)
# ─────────────────────────────────────────────────────────────────────────────

# Cached models
_rfdetr_model     = None
_rfdetr_processor = None
_yolo_model       = None


def _get_rfdetr():
    """Load RF-DETR model and processor (cached after first call)."""
    global _rfdetr_model, _rfdetr_processor
    if _rfdetr_model is None:
        try:
            from transformers import AutoImageProcessor, RfDetrForObjectDetection
            import torch
            print(f"Loading RF-DETR model: {RFDETR_MODEL} ...")
            _rfdetr_processor = AutoImageProcessor.from_pretrained(RFDETR_MODEL)
            _rfdetr_model     = RfDetrForObjectDetection.from_pretrained(RFDETR_MODEL)
            _rfdetr_model.eval()
            print("RF-DETR loaded successfully.")
        except ImportError:
            print("transformers not installed. Run: pip install transformers torch")
            _rfdetr_model = None
        except Exception as e:
            print(f"RF-DETR load error: {e}")
            _rfdetr_model = None
    return _rfdetr_model, _rfdetr_processor


def _get_yolo():
    """Load YOLOv8 model (cached after first call) - fallback."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            print("YOLOv8 fallback model loaded.")
        except ImportError:
            print("ultralytics not installed.")
            _yolo_model = None
    return _yolo_model


def _detect_rfdetr(frame_path, whitelist=None):
    """
    Run RF-DETR detection on a single frame.

    Args:
        frame_path: Path to the image file
        whitelist:  Optional set of label strings to filter results

    Returns:
        List of dicts: {label, confidence, box:[x1,y1,x2,y2]}
    """
    import torch

    model, processor = _get_rfdetr()
    if model is None:
        return []

    # Load image - RF-DETR expects PIL Image
    pil_img = Image.open(frame_path).convert("RGB")
    w, h    = pil_img.size

    # Preprocess and run inference
    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: convert logits → boxes with confidence scores
    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=CONFIDENCE
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label = model.config.id2label[label_id.item()]
        conf  = round(score.item(), 3)

        # Apply whitelist filter if provided
        if whitelist and label.lower() not in whitelist:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append({
            "label":      label,
            "confidence": conf,
            "box":        [x1, y1, x2, y2]
        })

    return detections


def _detect_yolo(frame_path, whitelist=None):
    """
    Fallback: run YOLOv8 detection on a single frame.
    Identical return format to _detect_rfdetr.
    """
    model = _get_yolo()
    if model is None:
        return []

    img     = cv2.imread(frame_path)
    if img is None:
        return []

    results = model(img, verbose=False, conf=CONFIDENCE, iou=0.4)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        if whitelist and label.lower() not in whitelist:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label":      label,
            "confidence": round(conf, 2),
            "box":        [x1, y1, x2, y2]
        })
    return detections


def detect_frame(frame_path, whitelist=None):
    """
    Run object detection on a single frame.
    Uses RF-DETR if USE_RFDETR=True, otherwise falls back to YOLOv8.

    Args:
        frame_path: Path to the image file
        whitelist:  Optional set of label strings to filter results

    Returns:
        List of dicts: {label, confidence, box:[x1,y1,x2,y2]}
    """
    if USE_RFDETR:
        results = _detect_rfdetr(frame_path, whitelist)
        # If RF-DETR fails (model not loaded), fall back to YOLO
        if results is None:
            print("RF-DETR unavailable, falling back to YOLOv8")
            return _detect_yolo(frame_path, whitelist)
        return results
    return _detect_yolo(frame_path, whitelist)


def detect_frame_live(bgr_frame, whitelist=None):
    """
    Run detection on a live OpenCV BGR frame (numpy array).
    Used for the live feed overlay in the scanner dashboard.

    Args:
        bgr_frame: OpenCV BGR numpy array from webcam
        whitelist: Optional set of label strings to filter

    Returns:
        List of dicts: {label, confidence, box:[x1,y1,x2,y2]}
    """
    import torch

    if not USE_RFDETR:
        # YOLOv8 live path
        model = _get_yolo()
        if model is None:
            return []
        results = model(bgr_frame, verbose=False, conf=CONFIDENCE, iou=0.4)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            if whitelist and label.lower() not in whitelist:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label, "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })
        return detections

    # RF-DETR live path - convert BGR numpy → PIL RGB
    model, processor = _get_rfdetr()
    if model is None:
        return []

    rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    h, w    = bgr_frame.shape[:2]

    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONFIDENCE
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label = model.config.id2label[label_id.item()]
        conf  = round(score.item(), 3)
        if whitelist and label.lower() not in whitelist:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append({
            "label": label, "confidence": conf,
            "box": [x1, y1, x2, y2]
        })
    return detections


def detect_all_frames():
    """
    Run detection on every frame in FRAMES_DIR.
    Only detects objects that have been custom labelled by the user.
    Saves results to output/detections.json.

    Returns:
        dict: {filename: [detections]}
    """
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
    if not files:
        return {}

    # Build whitelist from custom labels
    custom    = load_custom_labels()
    whitelist = set()
    for labels in custom.values():
        for lb in labels:
            whitelist.add(lb["label"].lower())

    model_name = RFDETR_MODEL if USE_RFDETR else "YOLOv8n"
    print(f"Running detection with {model_name} on {len(files)} frames...")

    all_results = {}
    for fp in files:
        fname = os.path.basename(fp)
        print(f"  Detecting: {fname}")
        all_results[fname] = detect_frame(
            fp, whitelist=whitelist if whitelist else None
        )

    # Persist results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "detections.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detections → {out_path}")
    return all_results


def load_detections():
    """Load previously saved detections from output/detections.json."""
    path = os.path.join(OUTPUT_DIR, "detections.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_custom_label(filename, label, box):
    """
    Save a user-drawn label box for a frame.

    Args:
        filename: Frame filename (e.g. frame_001.jpg)
        label:    Object name string
        box:      [x1, y1, x2, y2] pixel coordinates
    """
    os.makedirs(LABELS_DIR, exist_ok=True)
    path = os.path.join(LABELS_DIR, "custom_labels.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    if filename not in data:
        data[filename] = []
    data[filename].append({"label": label, "box": box, "custom": True})

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def load_custom_labels():
    """Load all user-drawn custom labels from labels/custom_labels.json."""
    path = os.path.join(LABELS_DIR, "custom_labels.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def search_object(query):
    """
    Search for an object name across all detections and custom labels.

    Args:
        query: Search string (e.g. "chair", "person")

    Returns:
        List of dicts: {filename, label, confidence, box, custom}
    """
    query   = query.lower().strip()
    results = []

    # Search auto detections
    for fname, dets in load_detections().items():
        for d in dets:
            if query in d["label"].lower():
                results.append({
                    "filename":   fname,
                    "label":      d["label"],
                    "confidence": d.get("confidence", 1.0),
                    "box":        d["box"],
                    "custom":     False
                })

    # Search custom labels
    for fname, labels in load_custom_labels().items():
        for lb in labels:
            if query in lb["label"].lower():
                results.append({
                    "filename":   fname,
                    "label":      lb["label"],
                    "confidence": 1.0,
                    "box":        lb["box"],
                    "custom":     True
                })

    return results


def get_all_object_names():
    """Return sorted list of all unique object names found across all frames."""
    names = set()
    for dets in load_detections().values():
        for d in dets:
            names.add(d["label"])
    for labels in load_custom_labels().values():
        for lb in labels:
            names.add(lb["label"])
    return sorted(names)


def draw_detections(frame_path, include_custom=True):
    """
    Return a JPEG image (bytes) with all detection boxes drawn on it.
    RF-DETR boxes drawn in cyan, custom labels in green.

    Args:
        frame_path:     Path to the frame image
        include_custom: Whether to overlay user-drawn labels too

    Returns:
        JPEG bytes or None if image not found
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None

    fname      = os.path.basename(frame_path)
    detections = load_detections().get(fname, [])
    custom     = load_custom_labels().get(fname, []) if include_custom else []

    # RF-DETR / auto detections - cyan boxes
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['label']} {d['confidence']:.0%}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Custom / user labels - green boxes
    for lb in custom:
        x1, y1, x2, y2 = lb["box"]
        label = f"★ {lb['label']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 100), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()
