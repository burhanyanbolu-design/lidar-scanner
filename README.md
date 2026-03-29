# LiDAR Room Scanner

Arduino UNO R4 WiFi + USB Webcam room scanner with live dashboard.

## Hardware Required

- Arduino UNO R4 WiFi
- USB Webcam (external)
- USB cable for Arduino

## Setup

### Step 1: Upload Arduino Code

1. Open Arduino IDE
2. Open `arduino/scanner.ino`
3. Select board: **Arduino UNO R4 WiFi**
4. Select port: **COM5**
5. Click Upload

### Step 2: Install Python Dependencies

Double-click `setup.bat`

### Step 3: Run Scanner

Double-click `START_SCANNER.bat`

Browser will open automatically at http://localhost:5000

## How to Use

1. Click **START SCAN** in the dashboard
2. Walk around the room holding the webcam
3. Press the **Arduino button** to capture a frame
4. Or press **SPACE** in the preview window
5. Click **STOP SCAN** when done
6. All frames saved in `frames/` folder

## Config

Edit `scanner/capture.py` to change:
- `COM_PORT` - Arduino port (default: COM5)
- `WEBCAM_INDEX` - Camera index (default: 1 for external USB)

## Files

```
lidar-scanner/
├── arduino/
│   └── scanner.ino      # Upload to Arduino
├── scanner/
│   ├── capture.py       # Webcam + Arduino capture
│   └── dashboard.py     # Web dashboard
├── frames/              # Captured frames saved here
├── main.py              # Run this to start
├── setup.bat            # First time setup
└── START_SCANNER.bat    # Start the scanner
```
