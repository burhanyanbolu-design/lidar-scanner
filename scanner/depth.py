"""
Depth map generator
Analyses captured frames and estimates depth using stereo/monocular methods
Outputs colour-coded depth images: blue=far, red=close
"""

import cv2
import numpy as np
import os
import glob
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
DEPTH_DIR  = os.path.join(BASE_DIR, "depth")


def estimate_depth(image):
    """
    Estimate depth from a single image using:
    - Edge detection (edges = depth boundaries)
    - Blur/focus analysis (blurry = far, sharp = close)
    - Brightness gradient (darker areas tend to be further)
    Returns a depth map as a grayscale image (0=far, 255=close)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Edge/detail map - sharp edges = close objects
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = np.abs(edges)
    edges = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)

    # 2. Local contrast map - high contrast = close
    kernel = np.ones((21, 21), np.float32) / (21 * 21)
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    contrast = np.abs(gray.astype(np.float32) - local_mean)

    # 3. Vertical gradient - objects lower in frame tend to be closer
    y_gradient = np.linspace(1.0, 0.2, h).reshape(h, 1)
    y_gradient = np.tile(y_gradient, (1, w))

    # Combine all cues
    depth = (edges * 0.5 + contrast * 0.3 + y_gradient * 50 * 0.2)

    # Normalise to 0-255
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    return depth


def depth_to_colour(depth_gray):
    """Convert grayscale depth to colour map: blue=far, green=mid, red=close"""
    coloured = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
    return coloured


def process_all_frames():
    """Process all frames in the frames folder"""
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))

    if not files:
        print("No frames found! Run the scanner first.")
        return []

    os.makedirs(DEPTH_DIR, exist_ok=True)
    results = []

    print(f"Processing {len(files)} frames...")

    for i, filepath in enumerate(files):
        img = cv2.imread(filepath)
        if img is None:
            continue

        print(f"  Analysing frame {i+1}/{len(files)}: {os.path.basename(filepath)}")

        # Generate depth map
        depth_gray = estimate_depth(img)
        depth_colour = depth_to_colour(depth_gray)

        # Create side-by-side comparison
        img_resized    = cv2.resize(img, (640, 480))
        depth_resized  = cv2.resize(depth_colour, (640, 480))

        # Add labels
        cv2.putText(img_resized, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(depth_resized, "DEPTH MAP (red=close, blue=far)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([img_resized, depth_resized])

        # Save
        fname = f"depth_{i+1:04d}_{os.path.basename(filepath)}"
        out_path = os.path.join(DEPTH_DIR, fname)
        cv2.imwrite(out_path, combined)
        results.append(out_path)

    return results


def show_results(results):
    """Show depth maps one by one"""
    if not results:
        return

    print(f"\nShowing {len(results)} depth maps...")
    print("Press any key to go to next, Q to quit")

    for path in results:
        img = cv2.imread(path)
        if img is None:
            continue

        # Scale down if too wide for screen
        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        cv2.imshow("Depth Map - any key=next, Q=quit", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("  DEPTH MAP GENERATOR")
    print("=" * 50)
    print()
    print("This analyses your captured frames and estimates")
    print("depth: RED = close objects, BLUE = far objects")
    print()

    results = process_all_frames()

    if results:
        print(f"\nDone! {len(results)} depth maps saved to: {DEPTH_DIR}")
        print()
        show = input("View depth maps now? (y/n): ").strip().lower()
        if show == 'y':
            show_results(results)
    else:
        print("No frames to process.")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
