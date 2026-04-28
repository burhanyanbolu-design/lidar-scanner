"""
Room Measurements
Analyses captured frames to estimate room dimensions
"""

import cv2
import numpy as np
import os
import glob
import json
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Calibration - adjust these based on your camera/room
# Approximate focal length in pixels (640px wide camera ~ 600)
FOCAL_LENGTH_PX = 600
# Known reference height - average ceiling height in meters
KNOWN_CEILING_HEIGHT = 2.4


def detect_lines(image):
    """Detect strong lines in image (walls, floor, ceiling)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=80,
                             minLineLength=100,
                             maxLineGap=20)
    return lines, edges


def classify_lines(lines, image_shape):
    """Classify lines as horizontal (floor/ceiling) or vertical (walls)"""
    h, w = image_shape[:2]
    horizontal = []
    vertical   = []

    if lines is None:
        return horizontal, vertical

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

        if angle < 20:        # nearly horizontal
            horizontal.append(line[0])
        elif angle > 70:      # nearly vertical
            vertical.append(line[0])

    return horizontal, vertical


def estimate_wall_distance(vertical_lines, image_width, focal_length):
    """
    Estimate distance to walls using vertical line positions.
    Lines near edges = walls are close.
    Lines near centre = walls are far.
    """
    if not vertical_lines:
        return None

    # Find leftmost and rightmost vertical lines
    x_positions = [(l[0] + l[2]) / 2 for l in vertical_lines]
    left_x  = min(x_positions)
    right_x = max(x_positions)

    centre = image_width / 2

    # Distance from centre to wall lines
    left_dist_px  = centre - left_x
    right_dist_px = right_x - centre

    # Convert pixel distance to approximate metres
    # Using similar triangles: real_dist = focal * real_size / pixel_size
    # Assuming average room width ~4m as reference
    scale = 4.0 / image_width

    left_m  = round(left_dist_px  * scale * 2, 2)
    right_m = round(right_dist_px * scale * 2, 2)
    total_m = round(left_m + right_m, 2)

    return {
        "left_wall_m":  left_m,
        "right_wall_m": right_m,
        "total_width_m": total_m
    }


def estimate_ceiling_distance(horizontal_lines, image_height):
    """Estimate ceiling height from horizontal lines"""
    if not horizontal_lines:
        return None

    y_positions = [(l[1] + l[3]) / 2 for l in horizontal_lines]
    top_y    = min(y_positions)
    bottom_y = max(y_positions)

    # Top line = ceiling, bottom line = floor
    ceiling_ratio = top_y / image_height
    floor_ratio   = bottom_y / image_height

    # Estimate height using known ceiling height reference
    estimated_height = round(KNOWN_CEILING_HEIGHT * (1 / ceiling_ratio) * 0.5, 2)

    return {
        "ceiling_y_ratio": round(ceiling_ratio, 2),
        "floor_y_ratio":   round(floor_ratio, 2),
        "estimated_height_m": min(estimated_height, 4.0)  # cap at 4m
    }


def draw_measurements(image, h_lines, v_lines, wall_data, ceiling_data):
    """Draw detected lines and measurements on image"""
    display = image.copy()
    h, w = display.shape[:2]

    # Draw horizontal lines in blue
    for line in h_lines:
        x1, y1, x2, y2 = line
        cv2.line(display, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # Draw vertical lines in green
    for line in v_lines:
        x1, y1, x2, y2 = line
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw measurement overlay
    cv2.rectangle(display, (0, 0), (w, 130), (0, 0, 0), -1)

    cv2.putText(display, "ROOM MEASUREMENTS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if wall_data:
        cv2.putText(display,
                    f"Width: ~{wall_data['total_width_m']}m  "
                    f"(L:{wall_data['left_wall_m']}m  R:{wall_data['right_wall_m']}m)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    if ceiling_data:
        cv2.putText(display,
                    f"Height: ~{ceiling_data['estimated_height_m']}m",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.putText(display,
                f"Lines found: {len(v_lines)} vertical  {len(h_lines)} horizontal",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return display


def process_frames():
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))

    if not files:
        print("No frames found! Run the scanner first.")
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    all_measurements = []

    print(f"Analysing {len(files)} frames for room measurements...")

    for i, filepath in enumerate(files):
        img = cv2.imread(filepath)
        if img is None:
            continue

        print(f"  Frame {i+1}/{len(files)}: {os.path.basename(filepath)}")

        lines, edges = detect_lines(img)
        h_lines, v_lines = classify_lines(lines, img.shape)

        wall_data    = estimate_wall_distance(v_lines, img.shape[1], FOCAL_LENGTH_PX)
        ceiling_data = estimate_ceiling_distance(h_lines, img.shape[0])

        display = draw_measurements(img, h_lines, v_lines, wall_data, ceiling_data)

        # Save annotated image
        fname    = f"measure_{i+1:04d}.jpg"
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, display)
        results.append(out_path)

        if wall_data or ceiling_data:
            all_measurements.append({
                "frame": i + 1,
                "wall":    wall_data,
                "ceiling": ceiling_data
            })

    # Save measurements to JSON
    if all_measurements:
        json_path = os.path.join(OUTPUT_DIR, "measurements.json")
        with open(json_path, "w") as f:
            json.dump(all_measurements, f, indent=2)
        print(f"\nMeasurements saved to: {json_path}")

    return results, all_measurements


def print_summary(measurements):
    """Print average room measurements"""
    if not measurements:
        print("No measurements found.")
        return

    widths  = [m["wall"]["total_width_m"]
               for m in measurements if m["wall"]]
    heights = [m["ceiling"]["estimated_height_m"]
               for m in measurements if m["ceiling"]]

    print()
    print("=" * 40)
    print("  ROOM MEASUREMENT SUMMARY")
    print("=" * 40)

    if widths:
        avg_w = round(sum(widths) / len(widths), 2)
        min_w = round(min(widths), 2)
        max_w = round(max(widths), 2)
        print(f"  Width:  avg={avg_w}m  min={min_w}m  max={max_w}m")

    if heights:
        avg_h = round(sum(heights) / len(heights), 2)
        print(f"  Height: avg={avg_h}m")

    if widths and heights:
        avg_w = round(sum(widths) / len(widths), 2)
        avg_h = round(sum(heights) / len(heights), 2)
        area  = round(avg_w * avg_w, 1)  # approximate square room
        print(f"  Est. floor area: ~{area} m²")

    print("=" * 40)
    print()
    print("Note: These are estimates based on image analysis.")
    print("For accurate measurements use a tape measure!")


def main():
    print("=" * 50)
    print("  ROOM MEASUREMENT ANALYSER")
    print("=" * 50)
    print()

    results, measurements = process_frames()

    if results:
        print_summary(measurements)

        show = input("View annotated frames? (y/n): ").strip().lower()
        if show == 'y':
            print("Any key = next frame, Q = quit")
            for path in results:
                img = cv2.imread(path)
                if img is not None:
                    cv2.imshow("Room Measurements", img)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
            cv2.destroyAllWindows()

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
