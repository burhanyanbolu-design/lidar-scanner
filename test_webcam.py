"""
Simple webcam test - run this first to check webcam works
"""
import cv2

print("Testing webcam...")
print("Press SPACE to capture, Q to quit")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam index 0, trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No webcam found!")
        input("Press Enter to exit...")
        exit()

print("Webcam opened! Window should appear now...")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame!")
        break

    cv2.putText(frame, f"Frames saved: {count} | SPACE=save Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Webcam Test", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # SPACE
        count += 1
        import os
        os.makedirs("frames", exist_ok=True)
        filename = f"frames/test_frame_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print(f"Done! {count} frames saved in 'frames' folder")
input("Press Enter to exit...")
