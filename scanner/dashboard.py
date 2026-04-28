"""
Live web dashboard - monitor scanner in browser
"""

import base64
import os
import time
import threading
from flask import Flask, render_template_string, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scanner-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global scanner reference
scanner = None
capture_flag = False        # set True when browser button pressed
request_capture_fn = None   # set by main.py
get_frame_fn = None         # set by main.py

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LiDAR Room Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a0a; color: #00ff88; font-family: monospace; }
        
        .header {
            background: #111;
            padding: 15px 20px;
            border-bottom: 1px solid #00ff88;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        h1 { font-size: 1.4em; letter-spacing: 3px; }
        
        .status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            border: 1px solid #00ff88;
        }
        .status.scanning { background: #00ff8833; color: #00ff88; }
        .status.idle { background: #ff000033; color: #ff4444; }
        
        .main {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 15px;
            padding: 15px;
            height: calc(100vh - 60px);
        }
        
        .panel {
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            overflow: hidden;
        }
        
        .panel h2 {
            font-size: 0.8em;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 10px;
            border-bottom: 1px solid #222;
            padding-bottom: 8px;
        }
        
        #live-feed {
            width: 100%;
            border-radius: 4px;
            border: 1px solid #333;
        }
        
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat-box {
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            color: #00ff88;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 0.7em;
            color: #666;
            margin-top: 4px;
            letter-spacing: 1px;
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.9em;
            letter-spacing: 2px;
            cursor: pointer;
            margin-bottom: 8px;
            transition: all 0.2s;
        }
        
        .btn-start { background: #00ff88; color: #000; }
        .btn-start:hover { background: #00cc66; }
        .btn-stop { background: #ff4444; color: #fff; }
        .btn-stop:hover { background: #cc0000; }
        .btn-capture { background: #4488ff; color: #fff; }
        .btn-capture:hover { background: #2266dd; }
        
        #log {
            background: #0a0a0a;
            border: 1px solid #222;
            border-radius: 4px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-size: 0.75em;
            color: #888;
        }
        
        .log-entry { margin-bottom: 4px; }
        .log-entry.capture { color: #00ff88; }
        .log-entry.info { color: #4488ff; }
        .log-entry.error { color: #ff4444; }
        
        #frames-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .thumb {
            width: 100%;
            border-radius: 3px;
            border: 1px solid #333;
            cursor: pointer;
        }
        .thumb:hover { border-color: #00ff88; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⬡ LIDAR ROOM SCANNER</h1>
        <div class="status idle" id="status-badge">IDLE</div>
    </div>
    
    <div class="main">
        <!-- Left: Live Feed -->
        <div>
            <div class="panel" style="margin-bottom:15px">
                <h2>LIVE FEED</h2>
                <img id="live-feed" src="/video_feed" alt="Live Feed">
            </div>
            
            <div class="panel">
                <h2>CAPTURED FRAMES</h2>
                <div id="frames-grid"></div>
            </div>
        </div>
        
        <!-- Right: Controls & Stats -->
        <div>
            <div class="panel" style="margin-bottom:15px">
                <h2>STATS</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value" id="frame-count">0</div>
                        <div class="stat-label">FRAMES</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="session-time">0s</div>
                        <div class="stat-label">SESSION</div>
                    </div>
                </div>
            </div>
            
            <div class="panel" style="margin-bottom:15px">
                <h2>CONTROLS</h2>
                <button class="btn btn-start" onclick="startScan()">▶ START SCAN</button>
                <button class="btn btn-stop" onclick="stopScan()">■ STOP SCAN</button>
                <button class="btn btn-capture" onclick="manualCapture()">◉ CAPTURE FRAME</button>
            </div>
            
            <div class="panel">
                <h2>LOG</h2>
                <div id="log"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let frameCount = 0;
        let sessionStart = null;
        let timerInterval = null;

        function log(msg, type='info') {
            const logEl = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            const time = new Date().toLocaleTimeString();
            entry.textContent = `[${time}] ${msg}`;
            logEl.prepend(entry);
        }

        function startScan() {
            fetch('/start').then(r => r.json()).then(d => {
                log('Scan started', 'info');
                document.getElementById('status-badge').textContent = 'SCANNING';
                document.getElementById('status-badge').className = 'status scanning';
                sessionStart = Date.now();
                timerInterval = setInterval(updateTimer, 1000);
            });
        }

        function stopScan() {
            fetch('/stop').then(r => r.json()).then(d => {
                log(`Scan stopped. ${frameCount} frames captured.`, 'info');
                document.getElementById('status-badge').textContent = 'IDLE';
                document.getElementById('status-badge').className = 'status idle';
                clearInterval(timerInterval);
            });
        }

        function manualCapture() {
            fetch('/capture').then(r => r.json()).then(d => {
                log(`Manual capture: frame ${d.frame}`, 'capture');
            });
        }

        function updateTimer() {
            if (!sessionStart) return;
            const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
            document.getElementById('session-time').textContent = elapsed + 's';
        }

        // Socket events from server
        socket.on('frame_captured', function(data) {
            frameCount = data.frame_count;
            document.getElementById('frame-count').textContent = frameCount;
            log(`Frame ${frameCount} captured`, 'capture');

            // Add thumbnail
            const grid = document.getElementById('frames-grid');
            const img = document.createElement('img');
            img.src = '/thumbnail/' + data.filename;
            img.className = 'thumb';
            img.title = `Frame ${frameCount}`;
            grid.prepend(img);
        });

        socket.on('status', function(data) {
            log(data.message, data.type || 'info');
        });

        log('Dashboard ready. Click START SCAN to begin.', 'info');
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/start')
def start_scan():
    if scanner:
        scanner.start()
        socketio.emit('status', {'message': 'Scanning started', 'type': 'info'})
    return jsonify({'status': 'started'})


@app.route('/stop')
def stop_scan():
    if scanner:
        scanner.stop()
        socketio.emit('status', {'message': 'Scanning stopped', 'type': 'info'})
    return jsonify({'status': 'stopped'})


@app.route('/capture')
def manual_capture():
    global capture_flag
    capture_flag = True
    return jsonify({'status': 'capturing'})


@app.route('/video_feed')
def video_feed():
    """Live MJPEG stream"""
    def generate():
        while True:
            if get_frame_fn is not None:
                frame = get_frame_fn()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/thumbnail/<path:filename>')
def thumbnail(filename):
    """Serve captured frame thumbnails"""
    import cv2
    img = cv2.imread(f"frames/{os.path.basename(filename)}")
    if img is not None:
        img = cv2.resize(img, (120, 90))
        _, buffer = cv2.imencode('.jpg', img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return '', 404


def notify_frame_captured(frame_count, filename, frame):
    """Called by scanner when a frame is captured"""
    socketio.emit('frame_captured', {
        'frame_count': frame_count,
        'filename': os.path.basename(filename)
    })


def run_dashboard():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
