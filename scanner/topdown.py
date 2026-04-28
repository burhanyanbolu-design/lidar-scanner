"""
Live Top-Down Map
Builds a bird's eye view of the room as frames are captured
"""

import cv2
import numpy as np
import os
import glob
import json

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Map config
MAP_SIZE    = 600   # pixels
MAP_SCALE   = 50    # pixels per metre
MAP_ORIGIN  = (MAP_SIZE // 2, MAP_SIZE - 50)  # camera starts at bottom centre


class TopDownMap:
    def __init__(self):
        self.map_img    = self._blank_map()
        self.positions  = []   # camera positions
        self.walls      = []   # detected wall points
        self.frame_count = 0

    def _blank_map(self):
        img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        # Grid lines
        for i in range(0, MAP_SIZE, MAP_SCALE):
            cv2.line(img, (i, 0), (i, MAP_SIZE), (30, 30, 30), 1)
            cv2.line(img, (0, i), (MAP_SIZE, i), (30, 30, 30), 1)
        # Border
        cv2.rectangle(img, (2, 2), (MAP_SIZE-2, MAP_SIZE-2), (60, 60, 60), 2)
        return img

    def _metres_to_px(self, x_m, y_m):
        px = int(MAP_ORIGIN[0] + x_m * MAP_SCALE)
        py = int(MAP_ORIGIN[1] - y_m * MAP_SCALE)
        return (px, py)

    def add_frame(self, frame_img, frame_index, pan_angle_deg=None):
        """
        Add a frame to the map.
        Estimates camera movement from optical flow between frames.
        """
        self.frame_count += 1

        # Estimate horizontal pan from frame index
        # Assume user pans left to right across ~180 degrees
        if pan_angle_deg is None:
            total_frames = max(len(self.positions) + 1, 10)
            pan_angle_deg = (frame_index / total_frames) * 120 - 60  # -60 to +60 degrees

        angle_rad = np.radians(pan_angle_deg)

        # Estimate distance to wall from vertical lines
        wall_dist = self._estimate_wall_dist(frame_img)

        # Camera position (moves slightly with each frame)
        cam_x = frame_index * 0.1  # slight movement
        cam_y = 0.0
        cam_pos = (cam_x, cam_y)
        self.positions.append(cam_pos)

        # Wall point in direction of camera pan
        wall_x = cam_x + wall_dist * np.sin(angle_rad)
        wall_y = cam_y + wall_dist * np.cos(angle_rad)
        self.walls.append((wall_x, wall_y, pan_angle_deg))

        self._redraw()

    def _estimate_wall_dist(self, frame):
        """Estimate distance to nearest wall in metres"""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                minLineLength=80, maxLineGap=15)

        if lines is None:
            return 3.0  # default 3m

        # Count vertical lines - more lines = closer wall
        v_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
            if angle > 70:
                v_count += 1

        # More vertical lines = closer
        dist = max(1.0, 5.0 - v_count * 0.3)
        return round(dist, 1)

    def _redraw(self):
        """Redraw the full map"""
        self.map_img = self._blank_map()

        # Draw scale bar
        cv2.putText(self.map_img, "1m", (10, MAP_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.line(self.map_img,
                 (10, MAP_SIZE - 20),
                 (10 + MAP_SCALE, MAP_SIZE - 20),
                 (100, 100, 100), 1)

        # Draw wall points and rays
        for wx, wy, angle in self.walls:
            wall_px = self._metres_to_px(wx, wy)
            # Draw wall point
            cv2.circle(self.map_img, wall_px, 4, (0, 100, 255), -1)

        # Connect wall points to form room outline
        if len(self.walls) > 1:
            pts = [self._metres_to_px(wx, wy) for wx, wy, _ in self.walls]
            for i in range(len(pts) - 1):
                cv2.line(self.map_img, pts[i], pts[i+1], (0, 60, 180), 1)

        # Draw camera path
        if len(self.positions) > 1:
            for i in range(len(self.positions) - 1):
                p1 = self._metres_to_px(*self.positions[i])
                p2 = self._metres_to_px(*self.positions[i+1])
                cv2.line(self.map_img, p1, p2, (0, 200, 0), 2)

        # Draw camera (latest position)
        if self.positions:
            cam_px = self._metres_to_px(*self.positions[-1])
            cv2.circle(self.map_img, cam_px, 8, (0, 255, 0), -1)
            cv2.circle(self.map_img, cam_px, 8, (255, 255, 255), 1)

        # Draw scan rays from camera to walls
        if self.positions and self.walls:
            cam_px = self._metres_to_px(*self.positions[-1])
            for wx, wy, _ in self.walls[-5:]:  # last 5 rays
                wall_px = self._metres_to_px(wx, wy)
                cv2.line(self.map_img, cam_px, wall_px, (0, 80, 0), 1)

        # Title and frame count
        cv2.putText(self.map_img, "TOP-DOWN MAP", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(self.map_img, f"Frames: {self.frame_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Legend
        cv2.circle(self.map_img, (MAP_SIZE - 80, 20), 5, (0, 255, 0), -1)
        cv2.putText(self.map_img, "Camera", (MAP_SIZE - 70, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.circle(self.map_img, (MAP_SIZE - 80, 40), 4, (0, 100, 255), -1)
        cv2.putText(self.map_img, "Wall", (MAP_SIZE - 70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    def get_map_jpg(self):
        """Return map as JPEG bytes for dashboard"""
        _, buf = cv2.imencode('.jpg', self.map_img)
        return buf.tobytes()

    def save_map(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "topdown_map.jpg")
        cv2.imwrite(path, self.map_img)
        print(f"Map saved: {path}")
        return path

    def build_from_frames(self):
        """Build map from all existing frames"""
        files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
        if not files:
            print("No frames found!")
            return

        print(f"Building map from {len(files)} frames...")
        for i, filepath in enumerate(files):
            img = cv2.imread(filepath)
            if img is not None:
                self.add_frame(img, i)
                print(f"  Processed frame {i+1}/{len(files)}")

        self.save_map()
        print("Map complete!")


def main():
    print("=" * 50)
    print("  TOP-DOWN MAP BUILDER")
    print("=" * 50)
    print()

    mapper = TopDownMap()
    mapper.build_from_frames()

    if mapper.frame_count > 0:
        print()
        print("Showing map - press any key to close")
        cv2.imshow("Top-Down Room Map", mapper.map_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
