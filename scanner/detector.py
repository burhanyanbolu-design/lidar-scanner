"""
Object Detector - uses YOLOv8 to detect and label objects in scan frames.
Supports: auto-detect with pretrained model + custom labels saved by user.
"""

import cv2
import os
import json
import glob

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Cached model
_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            _model = YOLO("yolov8n.pt")  # nano = fast, downloads automatically ~6MB
            print("YOLOv8 model loaded.")
        except ImportError:
            print("ultralytics not installed. Run: pip install ultralytics")
            _model = None
    return _model


def detect_frame(frame_path, whitelist=None):
    """
    Run YOLOv8 detection on a single frame.
    If whitelist is provided, only return detections matching those labels.
    Returns list of {label, confidence, box:[x1,y1,x2,y2]}
    """
    model = _get_model()
    if model is None:
        return []

    img = cv2.imread(frame_path)
    if img is None:
        return []

    results = model(img, verbose=False, conf=0.55, iou=0.4)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        # Filter to whitelist if provided
        if whitelist and label.lower() not in whitelist:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "box": [x1, y1, x2, y2]
        })
    return detections


def detect_all_frames():
    """
    Run detection on every frame in FRAMES_DIR.
    Only detects objects that have been custom labelled by the user.
    Returns dict: {filename: [detections]}
    """
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
    if not files:
        return {}

    # Build whitelist from custom labels
    custom = load_custom_labels()
    whitelist = set()
    for labels in custom.values():
        for lb in labels:
            whitelist.add(lb["label"].lower())

    results = {}
    for fp in files:
        fname = os.path.basename(fp)
        print(f"Detecting: {fname}")
        results[fname] = detect_frame(fp, whitelist=whitelist if whitelist else None)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "detections.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detections: {out}")
    return results


def load_detections():
    """Load previously saved detections."""
    path = os.path.join(OUTPUT_DIR, "detections.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_custom_label(filename, label, box):
    """
    Save a user-drawn label box for a frame.
    box = [x1, y1, x2, y2] in pixel coords
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
    """Load all user-drawn custom labels."""
    path = os.path.join(LABELS_DIR, "custom_labels.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def search_object(query):
    """
    Search for an object name across all detections + custom labels.
    Returns list of {filename, label, confidence, box}
    """
    query = query.lower().strip()
    results = []

    # Search auto detections
    detections = load_detections()
    for fname, dets in detections.items():
        for d in dets:
            if query in d["label"].lower():
                results.append({
                    "filename": fname,
                    "label": d["label"],
                    "confidence": d.get("confidence", 1.0),
                    "box": d["box"],
                    "custom": False
                })

    # Search custom labels
    custom = load_custom_labels()
    for fname, labels in custom.items():
        for lb in labels:
            if query in lb["label"].lower():
                results.append({
                    "filename": fname,
                    "label": lb["label"],
                    "confidence": 1.0,
                    "box": lb["box"],
                    "custom": True
                })

    return results


def get_all_object_names():
    """Return sorted list of all unique object names found."""
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
    Return a JPEG image with all detection boxes drawn on it.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None

    fname = os.path.basename(frame_path)
    detections = load_detections().get(fname, [])
    custom     = load_custom_labels().get(fname, []) if include_custom else []

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f"{d['label']} {d['confidence']:.0%}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    for lb in custom:
        x1, y1, x2, y2 = lb["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(img, f"★ {lb['label']}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()
