"""
Webcam capture module
"""

import cv2
import serial
import os
import time
import threading
from datetime import datetime

# Config
COM_PORT = "COM5"
BAUD_RATE = 9600
WEBCAM_INDEX = 0
PREVIEW_RESOLUTION = (640, 480)

# Always save frames next to the script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")


class Scanner:
    def __init__(self, on_frame_captured=None):
        self.on_frame_captured = on_frame_captured
        self.running = False
        self.frame_count = 0
        self.latest_frame = None
        self.status = "idle"
flag for browser button

        # Create frames directory
        os.makedirs(FRAMES_DIR, exist_ok=True)
        print(f"Frames will be saved to: {FRAMES_DIR}")

        # Init webcam
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_RESOLUTION[1])

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {WEBCAM_INDEX}")

        print("Webcam ready!")

        # Init Arduino serial
        try:
            self.arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"Arduino connected on {COM_PORT}")
        except serial.SerialException as e:
            print(f"No Arduino found - keyboard/browser mode only")
            self.arduino = None

    def request_capture(self):
        """Called from browser dashboard button"""
        self._capture_requested = True

    def start(self):
        self.running = True
        self.frame_count = 0
        self.status = "scanning"

        if self.arduino:
            self.arduino.write(b"START\n")
            self.arduino_thread = threading.Thread(target=self._listen_arduino, daemon=True)
            self.arduino_thread.start()

        print("Scanner started!")
        print(">> Use the CAPTURE button in the browser")
        print(">> OR press SPACE in the webcam window")
        print(">> Press Q in webcam window to stop")

        # Run preview on main thread
        self._preview_loop()

    def stop(self):
        self.running = False
        self.status = "idle"
        if self.arduino:
            self.arduino.write(b"STOP\n")
        print(f"Stopped. {self.frame_count} frames saved to: {FRAMES_DIR}")

    def capture_frame(self):
        """Capture and save a frame"""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read from webcam")
            return None

        self.frame_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(FRAMES_DIR, f"frame_{self.frame_count:04d}_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        self.latest_frame = frame.copy()
        print(f"Saved frame {self.frame_count}: {filename}")

        if self.on_frame_captured:
            self.on_frame_captured(self.frame_count, filename, frame)

        return filename

    def _listen_arduino(self):
        while self.running:
            try:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode("utf-8").strip()
                    if line.startswith("CAPTURE:"):
                        self.capture_frame()
            except Exception as e:
                print(f"Arduino error: {e}")
                break

    def _preview_loop(self):
        """Main preview loop - runs on main thread"""
        print("Opening webcam window...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.latest_frame = frame.copy()

            # Check if browser button was pressed
            if self._capture_requested:
                self._capture_requested = False
                self.capture_frame()
                # Flash green border
                frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10,
                                           cv2.BORDER_CONSTANT, value=(0, 255, 0))

            # Draw overlay on display copy
            display = frame.copy()
            h, w = display.shape[:2]

            # Background bar for text
            cv2.rectangle(display, (0, 0), (w, 110), (0, 0, 0), -1)

            cv2.putText(display, f"Frames captured: {self.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "SPACE = capture  |  Q = stop", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
            cv2.putText(display, f"Saving to: {FRAMES_DIR}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Crosshair
            cx, cy = w // 2, h // 2
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            cv2.imshow("Scanner - SPACE to capture, Q to stop", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.stop()
                break
            elif key == 32:  # SPACE
                self.capture_frame()

        cv2.destroyAllWindows()

    def get_latest_frame_jpg(self):
        if self.latest_frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', self.latest_frame)
        return buffer.tobytes()

    def cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.arduino:
            self.arduino.close()
        cv2.destroyAllWindows()
