"""
Image stitcher - combines captured frames into a panorama
"""

import cv2
import numpy as np
import os
import glob
from datetime import datetime

# Always find frames relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_frames():
    """Load all captured frames in order"""
    files = sorted(glob.glob(f"{FRAMES_DIR}/frame_*.jpg"))
    if not files:
        print("No frames found in 'frames/' folder!")
        return []

    frames = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            frames.append(img)
            print(f"Loaded: {os.path.basename(f)}")

    print(f"\nLoaded {len(frames)} frames total.")
    return frames


def stitch_panorama(frames):
    """Stitch frames into a panorama using OpenCV"""
    print("\nStitching panorama...")

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        print("Panorama created successfully!")
        return panorama
    else:
        error_codes = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images - capture more frames with more overlap",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Images too different - make sure frames overlap",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera adjustment failed - try more frames",
        }
        msg = error_codes.get(status, f"Unknown error code: {status}")
        print(f"Stitching failed: {msg}")
        return None


def create_contact_sheet(frames):
    """Create a grid of all frames as a contact sheet"""
    print("\nCreating contact sheet...")

    cols = 4
    rows = (len(frames) + cols - 1) // cols
    thumb_w, thumb_h = 320, 240

    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        thumb = cv2.resize(frame, (thumb_w, thumb_h))

        # Add frame number label
        cv2.putText(thumb, f"Frame {i+1}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        y1, y2 = row * thumb_h, (row + 1) * thumb_h
        x1, x2 = col * thumb_w, (col + 1) * thumb_w
        sheet[y1:y2, x1:x2] = thumb

    return sheet


def save_output(image, name):
    """Save output image"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")
    return filename


def main():
    print("=" * 50)
    print("  IMAGE STITCHER")
    print("=" * 50)

    frames = load_frames()
    if not frames:
        return

    print(f"\nWhat would you like to create?")
    print("1. Panorama (stitched wide image)")
    print("2. Contact sheet (grid of all frames)")
    print("3. Both")

    choice = input("\nEnter 1, 2 or 3: ").strip()

    if choice in ("1", "3"):
        if len(frames) < 2:
            print("Need at least 2 frames for panorama!")
        else:
            panorama = stitch_panorama(frames)
            if panorama is not None:
                path = save_output(panorama, "panorama")
                cv2.imshow("Panorama - Press any key to close", panorama)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if choice in ("2", "3"):
        sheet = create_contact_sheet(frames)
        path = save_output(sheet, "contact_sheet")
        cv2.imshow("Contact Sheet - Press any key to close", sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\nDone! Check the '{OUTPUT_DIR}/' folder for your images.")


if __name__ == "__main__":
    main()
