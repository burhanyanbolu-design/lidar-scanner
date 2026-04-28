# -*- coding: utf-8 -*-
"""
LiDAR Room Scanner - Main Entry Point
Features: live feed, top-down map, auto capture, manual capture, object detection
"""

import cv2
import os
import time
import threading
import webbrowser
from datetime import datetime
from flask import Flask, render_template_string, jsonify, Response, request
from flask_socketio import SocketIO
from scanner.topdown import TopDownMap
from scanner.detector_rfdetr import (
    detect_all_frames, load_detections, load_custom_labels,
    save_custom_label, search_object, get_all_object_names,
    draw_detections, detect_frame_live,
    FRAMES_DIR as DET_FRAMES_DIR,
    OUTPUT_DIR, LABELS_DIR
)

# ── Config ──────────────────────────────────────────────
WEBCAM_INDEX = 0
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR   = os.path.join(BASE_DIR, "frames")

# ── Flask ────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'scanner'
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared state
state = {
    "frame_count":     0,
    "latest_frame":    None,
    "annotated_frame": None,
    "capture_now":     False,
    "auto_interval":   0,
    "last_auto":       0,
    "live_detect":     False,
    "live_boxes":      [],
    "last_detect":     0,
    "cam_source":      "usb",   # "usb" or "phone"
}

mapper = TopDownMap()

# ── HTML Dashboard ───────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Room Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#0a0a0a; color:#00ff88; font-family:monospace; padding:15px; }
        h1 { letter-spacing:3px; margin-bottom:15px; font-size:1.2em; }
        .grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
        .panel { background:#111; border:1px solid #333; border-radius:8px; padding:12px; }
        .panel h2 { color:#666; font-size:0.7em; letter-spacing:2px; margin-bottom:8px; }
        img.feed { width:100%; border-radius:4px; border:1px solid #222; }
        .stat { font-size:2.5em; color:#00ff88; text-align:center; margin:8px 0; }
        .label { color:#666; font-size:0.7em; text-align:center; letter-spacing:2px; }
        .btn { width:100%; padding:12px; border:none; border-radius:6px;
               font-family:monospace; font-size:0.85em; letter-spacing:2px;
               cursor:pointer; margin-bottom:8px; }
        .blue  { background:#4488ff; color:#fff; }
        .green { background:#00ff88; color:#000; }
        .red   { background:#ff4444; color:#fff; }
        .blue:hover  { background:#2266dd; }
        .green:hover { background:#00cc66; }
        .red:hover   { background:#cc0000; }
        .row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
        .row label { font-size:0.7em; color:#666; white-space:nowrap; }
        input[type=range] { flex:1; accent-color:#00ff88; }
        #ival { color:#00ff88; font-size:0.8em; min-width:28px; }
        .bar { background:#222; border-radius:4px; height:5px; margin-bottom:8px; }
        .fill { background:#00ff88; height:5px; border-radius:4px; width:0%; transition:width 0.1s; }
        #log { background:#000; border:1px solid #222; border-radius:4px;
               padding:8px; height:120px; overflow-y:auto; font-size:0.7em; margin-top:8px; }
        .c{color:#00ff88;} .i{color:#4488ff;} .w{color:#ffaa00;}
        #thumbs { display:grid; grid-template-columns:repeat(3,1fr); gap:3px;
                  max-height:160px; overflow-y:auto; margin-top:8px; }
        #thumbs img { width:100%; border-radius:2px; border:1px solid #333; }
    </style>
</head>
<body>
    <h1>⬡ ROOM SCANNER</h1>
    <div class="grid">

        <div class="panel">
            <h2>LIVE FEED</h2>
            <img class="feed" src="/video_feed">
        </div>

        <div class="panel">
            <h2>TOP-DOWN MAP</h2>
            <img class="feed" id="map" src="/map_feed">
        </div>

        <div class="panel">
            <h2>CONTROLS</h2>
            <div class="label">FRAMES SAVED</div>
            <div class="stat" id="count">0</div>

            <div class="row">
                <label>INTERVAL</label>
                <input type="range" id="slider" min="1" max="10" value="3"
                       oninput="document.getElementById('ival').textContent=this.value+'s'">
                <span id="ival">3s</span>
            </div>
            <div class="bar"><div class="fill" id="fill"></div></div>

            <button class="btn green" id="btn-auto" onclick="toggleAuto()">▶ START AUTO CAPTURE</button>
            <button class="btn blue" onclick="captureNow()">◉ CAPTURE NOW</button>
            <button class="btn red" onclick="clearFrames()">🗑 NEW SCAN (DELETE ALL)</button>

            <div id="log"></div>
            <div id="thumbs"></div>
        </div>
    </div>

    <script>
        const socket = io();
        let autoOn = false, fillAnim = null, fillStart = null;

        setInterval(() => {
            document.getElementById('map').src = '/map_feed?t=' + Date.now();
        }, 1000);

        function log(msg, cls='i') {
            const d = document.getElementById('log');
            const t = new Date().toLocaleTimeString();
            d.innerHTML = `<div class="${cls}">[${t}] ${msg}</div>` + d.innerHTML;
        }

        function captureNow() {
            fetch('/capture');
        }

        function clearFrames() {
            if (!confirm('Delete ALL captured frames and start a new scan?')) return;
            fetch('/clear_frames').then(r => r.json()).then(d => {
                document.getElementById('count').textContent = '0';
                document.getElementById('thumbs').innerHTML = '';
                log('All frames deleted. Ready for new scan!', 'w');
            });
        }

        function toggleAuto() {
            autoOn = !autoOn;
            const btn = document.getElementById('btn-auto');
            const interval = document.getElementById('slider').value;
            if (autoOn) {
                btn.textContent = '■ STOP AUTO CAPTURE';
                btn.className = 'btn red';
                fetch('/auto_start?interval=' + interval);
                log('Auto capture every ' + interval + 's', 'w');
                fillStart = Date.now();
                animateFill(parseInt(interval));
            } else {
                btn.textContent = '▶ START AUTO CAPTURE';
                btn.className = 'btn green';
                fetch('/auto_stop');
                log('Auto capture stopped', 'w');
                cancelAnimationFrame(fillAnim);
                document.getElementById('fill').style.width = '0%';
            }
        }

        function animateFill(interval) {
            function step() {
                if (!autoOn) return;
                const pct = Math.min(((Date.now() - fillStart) / (interval * 1000)) * 100, 100);
                document.getElementById('fill').style.width = pct + '%';
                fillAnim = requestAnimationFrame(step);
            }
            step();
        }

        socket.on('captured', function(d) {
            document.getElementById('count').textContent = d.count;
            log('Frame ' + d.count + ' saved!', 'c');
            const t = document.getElementById('thumbs');
            const img = document.createElement('img');
            img.src = '/thumb/' + d.filename + '?t=' + Date.now();
            t.prepend(img);
            fillStart = Date.now();
        });

        socket.on('cleared', function(d) {
            log(d.msg, 'w');
        });

        socket.on('status', function(d) { log(d.msg, d.type || 'i'); });

        log('Ready. Click START AUTO CAPTURE or press SPACE in webcam window.', 'i');
    </script>
</body>
</html>
"""

# ── HTML Dashboard ───────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Room Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#0a0a0a; color:#00ff88; font-family:monospace; padding:15px; }
        h1 { letter-spacing:3px; margin-bottom:12px; font-size:1.2em; }
        /* Tabs */
        .tabs { display:flex; gap:4px; margin-bottom:12px; }
        .tab { padding:8px 18px; border:1px solid #333; border-radius:6px 6px 0 0;
               cursor:pointer; font-family:monospace; font-size:0.8em; letter-spacing:2px;
               background:#111; color:#666; }
        .tab.active { background:#1a1a1a; color:#00ff88; border-bottom-color:#1a1a1a; }
        .tab-content { display:none; }
        .tab-content.active { display:block; }
        /* Panels */
        .grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
        .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .panel { background:#111; border:1px solid #333; border-radius:8px; padding:12px; }
        .panel h2 { color:#666; font-size:0.7em; letter-spacing:2px; margin-bottom:8px; }
        img.feed { width:100%; border-radius:4px; border:1px solid #222; }
        .stat { font-size:2.5em; color:#00ff88; text-align:center; margin:8px 0; }
        .label { color:#666; font-size:0.7em; text-align:center; letter-spacing:2px; }
        .btn { width:100%; padding:10px; border:none; border-radius:6px;
               font-family:monospace; font-size:0.8em; letter-spacing:2px;
               cursor:pointer; margin-bottom:8px; }
        .blue   { background:#4488ff; color:#fff; }
        .green  { background:#00ff88; color:#000; }
        .red    { background:#ff4444; color:#fff; }
        .orange { background:#ff8800; color:#000; }
        .blue:hover   { background:#2266dd; }
        .green:hover  { background:#00cc66; }
        .red:hover    { background:#cc0000; }
        .orange:hover { background:#cc6600; }
        .row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
        .row label { font-size:0.7em; color:#666; white-space:nowrap; }
        input[type=range] { flex:1; accent-color:#00ff88; }
        input[type=text]  { background:#000; border:1px solid #333; color:#00ff88;
                            font-family:monospace; padding:6px 8px; border-radius:4px;
                            font-size:0.8em; flex:1; }
        #ival { color:#00ff88; font-size:0.8em; min-width:28px; }
        .bar  { background:#222; border-radius:4px; height:5px; margin-bottom:8px; }
        .fill { background:#00ff88; height:5px; border-radius:4px; width:0%; transition:width 0.1s; }
        #log  { background:#000; border:1px solid #222; border-radius:4px;
                padding:8px; height:100px; overflow-y:auto; font-size:0.7em; margin-top:8px; }
        .c{color:#00ff88;} .i{color:#4488ff;} .w{color:#ffaa00;} .e{color:#ff4444;}
        #thumbs { display:grid; grid-template-columns:repeat(3,1fr); gap:3px;
                  max-height:140px; overflow-y:auto; margin-top:8px; }
        #thumbs img { width:100%; border-radius:2px; border:1px solid #333; cursor:pointer; }
        /* Objects tab */
        #obj-frames { display:grid; grid-template-columns:repeat(4,1fr); gap:6px;
                      max-height:320px; overflow-y:auto; margin-top:8px; }
        #obj-frames .fcard { position:relative; cursor:pointer; border:2px solid #333;
                             border-radius:4px; overflow:hidden; }
        #obj-frames .fcard:hover { border-color:#00ff88; }
        #obj-frames .fcard.selected { border-color:#ff8800; }
        #obj-frames img { width:100%; display:block; }
        #obj-frames .fname { position:absolute; bottom:0; left:0; right:0;
                             background:rgba(0,0,0,0.7); font-size:0.55em;
                             padding:2px 4px; color:#aaa; }
        #obj-canvas-wrap { position:relative; display:inline-block; width:100%; }
        #obj-canvas { width:100%; border:1px solid #333; border-radius:4px;
                      cursor:crosshair; display:block; }
        #search-results { margin-top:8px; max-height:200px; overflow-y:auto; }
        .result-card { display:flex; align-items:center; gap:8px; padding:6px;
                       border:1px solid #333; border-radius:4px; margin-bottom:4px;
                       cursor:pointer; }
        .result-card:hover { border-color:#00ff88; }
        .result-card img { width:80px; height:60px; object-fit:cover; border-radius:3px; }
        .result-info { font-size:0.75em; }
        .result-info .rlabel { color:#00ff88; font-size:1.1em; }
        .result-info .rfile  { color:#666; }
        .tag { display:inline-block; padding:2px 8px; border-radius:10px;
               font-size:0.65em; margin:2px; cursor:pointer; border:1px solid #444; color:#aaa; }
        .tag:hover { border-color:#00ff88; color:#00ff88; }
        #obj-log { background:#000; border:1px solid #222; border-radius:4px;
                   padding:6px; height:60px; overflow-y:auto; font-size:0.65em; margin-top:6px; }
    </style>
</head>
<body>
    <h1>⬡ ROOM SCANNER</h1>

    <div class="tabs">
        <div class="tab active" onclick="showTab('scan', this)">📷 SCAN</div>
        <div class="tab" onclick="showTab('objects', this)">🔍 OBJECTS</div>
    </div>

    <!-- ── SCAN TAB ── -->
    <div id="tab-scan" class="tab-content active">
        <div class="grid">
            <div class="panel">
                <h2>LIVE FEED</h2>
                <img class="feed" src="/video_feed">
                <div style="margin-top:6px;display:flex;align-items:center;gap:6px;">
                    <span id="cam-label" style="font-size:0.65em;color:#666;">SOURCE: USB CAM</span>
                    <button class="btn blue" id="btn-cam"
                            style="padding:4px 10px;font-size:0.65em;margin:0;width:auto;"
                            onclick="switchCam()">📱 USE PHONE CAM</button>
                </div>
            </div>
            <div class="panel">
                <h2>CONTROLS</h2>
                <div class="label">FRAMES SAVED</div>
                <div class="stat" id="count">0</div>
                <div class="row">
                    <label>INTERVAL</label>
                    <input type="range" id="slider" min="1" max="10" value="3"
                           oninput="document.getElementById('ival').textContent=this.value+'s'">
                    <span id="ival">3s</span>
                </div>
                <div class="bar"><div class="fill" id="fill"></div></div>
                <button class="btn green" id="btn-auto" onclick="toggleAuto()">▶ START AUTO CAPTURE</button>
                <button class="btn blue" onclick="captureNow()">◉ CAPTURE NOW</button>
                <button class="btn orange" id="btn-live" onclick="toggleLive()">👁 LIVE DETECT: OFF</button>
                <button class="btn red" onclick="clearFrames()">🗑 NEW SCAN (DELETE ALL)</button>
                <button class="btn orange" onclick="deleteUnlabeled()">🧹 DELETE UNLABELED FRAMES</button>
                <div id="log"></div>
                <div id="thumbs"></div>
            </div>
            <div class="panel">
                <h2>📱 UPLOAD FROM PHONE / FILE</h2>
                <p style="font-size:0.7em;color:#666;margin-bottom:10px;">
                    Transfer photos from your phone via USB, WhatsApp, email — then drag them here.
                </p>
                <div id="drop-zone"
                     ondragover="event.preventDefault();this.style.borderColor='#00ff88';"
                     ondragleave="this.style.borderColor='#333';"
                     ondrop="handleDrop(event);"
                     style="border:2px dashed #333;border-radius:8px;padding:24px;
                            text-align:center;cursor:pointer;margin-bottom:8px;"
                     onclick="document.getElementById('file-input').click()">
                    <div style="font-size:2em;margin-bottom:6px;">📂</div>
                    <div style="color:#666;font-size:0.75em;">Drag &amp; drop photos here<br>or click to browse</div>
                </div>
                <input type="file" id="file-input" multiple accept="image/*"
                       style="display:none" onchange="uploadFiles(this.files)">
                <div id="upload-log" style="background:#000;border:1px solid #222;border-radius:4px;
                     padding:6px;height:80px;overflow-y:auto;font-size:0.7em;margin-top:4px;"></div>
                <div style="margin-top:8px;font-size:0.65em;color:#444;">
                    💡 Scan QR to open phone camera:<br>
                    <span id="local-ip" style="color:#4488ff;">http://YOUR-PC-IP:5000/phone</span><br>
                    <div id="qrcode" style="margin-top:8px;background:#fff;padding:6px;display:inline-block;border-radius:4px;"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- ── OBJECTS TAB ── -->
    <div id="tab-objects" class="tab-content">
        <div class="grid2">

            <!-- Left: frame browser + canvas -->
            <div class="panel">
                <h2>SELECT FRAME TO LABEL</h2>
                <button class="btn blue" onclick="captureNow();setTimeout(loadObjFrames,800);">◉ CAPTURE FRAME NOW</button>
                <div id="obj-frames"></div>
                <div id="obj-canvas-wrap" style="margin-top:8px;">
                    <canvas id="obj-canvas" width="640" height="480"></canvas>
                </div>
                <div class="row" style="margin-top:8px;">
                    <input type="text" id="label-input" placeholder="Object name (e.g. chair, TV, door)">
                    <button class="btn orange" style="width:auto;padding:8px 14px;margin:0;"
                            onclick="saveLabel()">SAVE LABEL</button>
                </div>
                <button class="btn red" onclick="deleteSelected()">🗑 DELETE SELECTED FRAME</button>
                <div id="obj-log"></div>
            </div>

            <!-- Right: detect + search -->
            <div class="panel">
                <h2>AUTO DETECT OBJECTS</h2>
                <p style="font-size:0.7em;color:#666;margin-bottom:8px;">
                    Runs YOLOv8 on all frames. Detects 80+ common objects automatically.
                </p>
                <button class="btn orange" id="btn-detect" onclick="runDetect()">
                    ⚡ DETECT ALL FRAMES
                </button>
                <button class="btn blue" id="btn-detect-one" onclick="runDetectOne()">
                    🔍 DETECT SELECTED FRAME
                </button>
                <div id="known-tags" style="margin-bottom:8px;"></div>

                <h2 style="margin-top:12px;">SEARCH OBJECTS</h2>
                <div class="row">
                    <input type="text" id="search-input" placeholder="Search: chair, TV, person..."
                           onkeydown="if(event.key==='Enter') searchObj()">
                    <button class="btn blue" style="width:auto;padding:8px 14px;margin:0;"
                            onclick="searchObj()">FIND</button>
                </div>
                <div id="search-results"></div>
                <div id="frame-labels" style="margin-top:10px;"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let autoOn = false, fillAnim = null, fillStart = null;
        let selectedFrame = null;
        let drawing = false, startX, startY, currentBox = null;
        const canvas = document.getElementById('obj-canvas');
        const ctx = canvas.getContext('2d');
        let canvasImg = new Image();

        // ── Tab switching ──
        function showTab(name, el) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            el.classList.add('active');
            if (name === 'objects') loadObjFrames();
        }

        // ── Scan tab ──

        function log(msg, cls='i') {
            const d = document.getElementById('log');
            const t = new Date().toLocaleTimeString();
            d.innerHTML = `<div class="${cls}">[${t}] ${msg}</div>` + d.innerHTML;
        }

        function captureNow() { fetch('/capture'); }

        function toggleLive() {
            fetch('/live_detect_toggle').then(r => r.json()).then(d => {
                const btn = document.getElementById('btn-live');
                if (d.live_detect) {
                    btn.textContent = '👁 LIVE DETECT: ON';
                    btn.className = 'btn green';
                    log('Live detection ON - known objects will be labeled on feed', 'c');
                } else {
                    btn.textContent = '👁 LIVE DETECT: OFF';
                    btn.className = 'btn orange';
                    log('Live detection OFF', 'w');
                }
            });
        }

        function clearFrames() {
            if (!confirm('Delete ALL captured frames and start a new scan?')) return;
            fetch('/clear_frames').then(r => r.json()).then(d => {
                document.getElementById('count').textContent = '0';
                document.getElementById('thumbs').innerHTML = '';
                log('All frames deleted. Ready for new scan!', 'w');
            });
        }

        function deleteUnlabeled() {
            if (!confirm('Delete all frames that have NO labels? Labeled frames will be kept.')) return;
            fetch('/obj/delete_unlabeled', {method:'POST'}).then(r => r.json()).then(d => {
                log(`Deleted ${d.deleted} unlabeled frames. Kept ${d.kept} labeled frames.`, 'w');
            });
        }

        function handleDrop(e) {
            e.preventDefault();
            document.getElementById('drop-zone').style.borderColor = '#333';
            uploadFiles(e.dataTransfer.files);
        }

        function uploadFiles(files) {
            if (!files.length) return;
            const ulog = document.getElementById('upload-log');
            ulog.innerHTML = `<div class="i">Uploading ${files.length} photo(s)...</div>`;
            const fd = new FormData();
            for (const f of files) fd.append('photos', f);
            fetch('/obj/upload', {method:'POST', body: fd})
                .then(r => r.json()).then(d => {
                    ulog.innerHTML = `<div class="c">✓ Saved ${d.saved} photo(s) as frames!</div>` + ulog.innerHTML;
                    log(`Uploaded ${d.saved} photo(s) from file`, 'c');
                }).catch(() => {
                    ulog.innerHTML = `<div class="e">Upload failed</div>` + ulog.innerHTML;
                });
        }

        // Show local IP hint + QR code
        function updateQR(url) {
            document.getElementById('local-ip').textContent = url + '/phone';
            document.getElementById('qrcode').innerHTML = '';
            new QRCode(document.getElementById('qrcode'), {
                text:   url + '/phone',
                width:  128, height: 128,
                colorDark: '#000000', colorLight: '#ffffff',
            });
        }
        fetch('/local_ip').then(r=>r.json()).then(d=>{
            updateQR(d.url);
            if (d.https) log('HTTPS via ngrok - phone live view enabled!', 'c');
            else log('HTTP only - phone camera uses photo mode. Waiting for ngrok...', 'w');
        }).catch(()=>{});

        socket.on('ngrok_url', function(d) {
            updateQR(d.url);
            log('ngrok ready! Phone live view now available: ' + d.url + '/phone', 'c');
        });

        function switchCam() {
            fetch('/cam_switch').then(r => r.json()).then(d => {
                const isPhone = d.cam_source === 'phone';
                document.getElementById('cam-label').textContent =
                    isPhone ? 'SOURCE: PHONE CAM' : 'SOURCE: USB CAM';
                document.getElementById('btn-cam').textContent =
                    isPhone ? '💻 USE USB CAM' : '📱 USE PHONE CAM';
                log(isPhone
                    ? 'Switched to phone - scan QR and tap START STREAMING on phone'
                    : 'Switched back to USB camera', 'w');
            });
        }

        function toggleAuto() {
            autoOn = !autoOn;
            const btn = document.getElementById('btn-auto');
            const interval = document.getElementById('slider').value;
            if (autoOn) {
                btn.textContent = '■ STOP AUTO CAPTURE';
                btn.className = 'btn red';
                fetch('/auto_start?interval=' + interval);
                log('Auto capture every ' + interval + 's', 'w');
                fillStart = Date.now();
                animateFill(parseInt(interval));
            } else {
                btn.textContent = '▶ START AUTO CAPTURE';
                btn.className = 'btn green';
                fetch('/auto_stop');
                log('Auto capture stopped', 'w');
                cancelAnimationFrame(fillAnim);
                document.getElementById('fill').style.width = '0%';
            }
        }

        function animateFill(interval) {
            function step() {
                if (!autoOn) return;
                const pct = Math.min(((Date.now() - fillStart) / (interval * 1000)) * 100, 100);
                document.getElementById('fill').style.width = pct + '%';
                fillAnim = requestAnimationFrame(step);
            }
            step();
        }

        socket.on('captured', function(d) {
            document.getElementById('count').textContent = d.count;
            log('Frame ' + d.count + ' saved!', 'c');
            const t = document.getElementById('thumbs');
            const img = document.createElement('img');
            img.src = '/thumb/' + d.filename + '?t=' + Date.now();
            t.prepend(img);
            fillStart = Date.now();
        });
        socket.on('cleared', function(d) { log(d.msg, 'w'); });
        socket.on('status',  function(d) { log(d.msg, d.type || 'i'); });
        socket.on('live_detections', function(d) {
            const unique = [...new Set(d.labels)];
            log('Seeing: ' + unique.join(', '), 'c');
        });

        log('Ready. Click START AUTO CAPTURE or press SPACE in webcam window.', 'i');

        // ── Objects tab ──
        function objLog(msg, cls='i') {
            const d = document.getElementById('obj-log');
            d.innerHTML = `<div class="${cls}">${msg}</div>` + d.innerHTML;
        }

        function loadObjFrames() {
            fetch('/obj/frames').then(r => r.json()).then(data => {
                const grid = document.getElementById('obj-frames');
                grid.innerHTML = '';
                if (!data.frames.length) {
                    grid.innerHTML = '<div style="color:#666;font-size:0.7em;padding:8px;">No frames yet. Go to SCAN tab and capture some frames first.</div>';
                    return;
                }
                data.frames.forEach(f => {
                    const card = document.createElement('div');
                    card.className = 'fcard';
                    card.innerHTML = `<img src="/thumb/${f}?t=${Date.now()}"><div class="fname">${f.substring(0,16)}</div>`;
                    card.onclick = () => selectFrame(f, card);
                    grid.appendChild(card);
                });
                loadKnownTags();
            });
        }

        function selectFrame(fname, card) {
            document.querySelectorAll('.fcard').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            selectedFrame = fname;
            currentBox = null;
            const url = '/obj/frame_img/' + fname + '?t=' + Date.now();
            canvasImg.onload = () => {
                canvas.width  = canvasImg.naturalWidth;
                canvas.height = canvasImg.naturalHeight;
                redrawCanvas();
            };
            canvasImg.src = url;
            objLog('Selected: ' + fname, 'i');
            loadFrameLabels(fname);
        }

        function loadFrameLabels(fname) {
            fetch('/obj/frame_labels?filename=' + encodeURIComponent(fname))
                .then(r => r.json()).then(data => {
                const wrap = document.getElementById('frame-labels');
                if (!data.labels.length) {
                    wrap.innerHTML = '<div style="color:#444;font-size:0.7em;margin-top:4px;">No labels on this frame yet.</div>';
                    return;
                }
                wrap.innerHTML = '<div style="color:#666;font-size:0.65em;margin-bottom:4px;">LABELS ON THIS FRAME:</div>';
                data.labels.forEach((lb, idx) => {
                    const row = document.createElement('div');
                    row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:4px 6px;border:1px solid #333;border-radius:4px;margin-bottom:3px;';
                    const src = lb.custom ? '★ custom' : 'auto ' + Math.round((lb.confidence||1)*100) + '%';
                    const lbl = document.createElement('span');
                    lbl.style.cssText = 'color:#00ff88;font-size:0.8em;';
                    lbl.textContent = lb.label;
                    const badge = document.createElement('span');
                    badge.style.cssText = 'color:#555;font-size:0.7em;';
                    badge.textContent = src;
                    const btn = document.createElement('button');
                    btn.textContent = '✕';
                    btn.style.cssText = 'background:#ff4444;border:none;color:#fff;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:0.7em;';
                    btn.dataset.fname  = fname;
                    btn.dataset.idx    = idx;
                    btn.dataset.custom = lb.custom;
                    btn.addEventListener('click', function() {
                        deleteLabel(this.dataset.fname, parseInt(this.dataset.idx), this.dataset.custom === 'true');
                    });
                    row.appendChild(lbl);
                    row.appendChild(badge);
                    row.appendChild(btn);
                    wrap.appendChild(row);
                });
            });
        }

        function redrawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(canvasImg, 0, 0);
            if (currentBox) {
                ctx.strokeStyle = '#ff8800';
                ctx.lineWidth = 2;
                ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
            }
        }

        // Draw box on canvas
        canvas.addEventListener('mousedown', e => {
            if (!selectedFrame) return;
            const r = canvas.getBoundingClientRect();
            const scaleX = canvas.width  / r.width;
            const scaleY = canvas.height / r.height;
            startX = (e.clientX - r.left) * scaleX;
            startY = (e.clientY - r.top)  * scaleY;
            drawing = true;
        });

        canvas.addEventListener('mousemove', e => {
            if (!drawing) return;
            const r = canvas.getBoundingClientRect();
            const scaleX = canvas.width  / r.width;
            const scaleY = canvas.height / r.height;
            const mx = (e.clientX - r.left) * scaleX;
            const my = (e.clientY - r.top)  * scaleY;
            currentBox = { x: startX, y: startY, w: mx - startX, h: my - startY };
            redrawCanvas();
        });

        canvas.addEventListener('mouseup', () => { drawing = false; });

        function deleteLabel(fname, idx, isCustom) {
            fetch('/obj/delete_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: fname, index: idx, custom: isCustom})
            }).then(r => r.json()).then(d => {
                objLog('Label removed.', 'w');
                loadFrameLabels(fname);
                loadKnownTags();
                const url = '/obj/frame_img/' + fname + '?t=' + Date.now();
                canvasImg.onload = () => { canvas.width = canvasImg.naturalWidth; canvas.height = canvasImg.naturalHeight; redrawCanvas(); };
                canvasImg.src = url;
            });
        }

        function deleteSelected() {
            if (!selectedFrame) { objLog('No frame selected!', 'w'); return; }
            if (!confirm('Delete frame: ' + selectedFrame + '?')) return;
            fetch('/obj/delete_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: selectedFrame})
            }).then(r => r.json()).then(d => {
                objLog('Deleted: ' + selectedFrame, 'w');
                selectedFrame = null;
                currentBox = null;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                loadObjFrames();
                loadKnownTags();
            }).catch(() => objLog('Delete failed', 'e'));
        }

        function saveLabel() {
            if (!selectedFrame) { objLog('Select a frame first!', 'w'); return; }
            if (!currentBox)    { objLog('Draw a box around the object first!', 'w'); return; }
            const label = document.getElementById('label-input').value.trim();
            if (!label) { objLog('Enter an object name!', 'w'); return; }

            const box = [
                Math.round(Math.min(currentBox.x, currentBox.x + currentBox.w)),
                Math.round(Math.min(currentBox.y, currentBox.y + currentBox.h)),
                Math.round(Math.max(currentBox.x, currentBox.x + currentBox.w)),
                Math.round(Math.max(currentBox.y, currentBox.y + currentBox.h))
            ];

            fetch('/obj/label', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ filename: selectedFrame, label, box })
            }).then(r => r.json()).then(d => {
                objLog('Saved: ' + label + ' in ' + selectedFrame, 'c');
                currentBox = null;
                document.getElementById('label-input').value = '';
                loadKnownTags();
                loadFrameLabels(selectedFrame);
            });
        }

        function loadKnownTags() {
            fetch('/obj/names').then(r => r.json()).then(data => {
                const wrap = document.getElementById('known-tags');
                wrap.innerHTML = '<div style="color:#666;font-size:0.65em;margin-bottom:4px;">KNOWN OBJECTS:</div>';
                if (!data.names.length) {
                    wrap.innerHTML += '<span style="color:#444;font-size:0.7em;">None yet</span>';
                    return;
                }
                data.names.forEach(n => {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = n;
                    tag.onclick = () => {
                        document.getElementById('search-input').value = n;
                        searchObj();
                    };
                    wrap.appendChild(tag);
                });
            });
        }

        function runDetect() {
            const btn = document.getElementById('btn-detect');
            btn.textContent = '⏳ DETECTING...';
            btn.disabled = true;
            fetch('/obj/detect').then(r => r.json()).then(d => {
                btn.textContent = '⚡ DETECT ALL FRAMES';
                btn.disabled = false;
                objLog('Detection done! Found objects in ' + d.frame_count + ' frames.', 'c');
                loadKnownTags();
                if (selectedFrame) redrawCanvas();
            }).catch(() => {
                btn.textContent = '⚡ DETECT ALL FRAMES';
                btn.disabled = false;
                objLog('Detection failed. Is ultralytics installed?', 'e');
            });
        }

        function runDetectOne() {
            if (!selectedFrame) { objLog('Select a frame first!', 'w'); return; }
            const btn = document.getElementById('btn-detect-one');
            btn.textContent = '⏳ DETECTING...';
            btn.disabled = true;
            fetch('/obj/detect_one?filename=' + encodeURIComponent(selectedFrame))
                .then(r => r.json()).then(d => {
                    btn.textContent = '🔍 DETECT SELECTED FRAME';
                    btn.disabled = false;
                    objLog('Found ' + d.count + ' object(s) in ' + selectedFrame, 'c');
                    loadKnownTags();
                    // Reload canvas with new boxes
                    const url = '/obj/frame_img/' + selectedFrame + '?t=' + Date.now();
                    canvasImg.onload = () => { canvas.width = canvasImg.naturalWidth; canvas.height = canvasImg.naturalHeight; redrawCanvas(); };
                    canvasImg.src = url;
                }).catch(() => {
                    btn.textContent = '🔍 DETECT SELECTED FRAME';
                    btn.disabled = false;
                    objLog('Detection failed.', 'e');
                });
        }

        function searchObj() {
            const q = document.getElementById('search-input').value.trim();
            if (!q) return;
            fetch('/obj/search?q=' + encodeURIComponent(q)).then(r => r.json()).then(data => {
                const wrap = document.getElementById('search-results');
                wrap.innerHTML = '';
                if (!data.results.length) {
                    wrap.innerHTML = '<div style="color:#666;font-size:0.7em;padding:8px;">No results for "' + q + '"</div>';
                    return;
                }
                objLog('Found ' + data.results.length + ' result(s) for "' + q + '"', 'c');
                data.results.forEach((r, idx) => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px;border:1px solid #333;border-radius:4px;margin-bottom:4px;';
                    const badge = r.custom ? '★ Custom' : Math.round(r.confidence*100) + '% conf';

                    const thumb = document.createElement('img');
                    thumb.src = '/obj/frame_thumb/' + r.filename + '?t=' + Date.now();
                    thumb.style.cssText = 'width:70px;height:52px;object-fit:cover;border-radius:3px;cursor:pointer;flex-shrink:0;';
                    thumb.onclick = () => jumpToFrame(r.filename);

                    const info = document.createElement('div');
                    info.style.cssText = 'flex:1;font-size:0.75em;';
                    info.innerHTML = '<div style="color:#00ff88;">' + r.label + '</div>'
                        + '<div style="color:#666;">' + r.filename.substring(0,22) + '</div>'
                        + '<div style="color:#555;">' + badge + '</div>';
                    info.style.cursor = 'pointer';
                    info.onclick = () => jumpToFrame(r.filename);

                    const delBtn = document.createElement('button');
                    delBtn.textContent = '✕';
                    delBtn.title = 'Delete this label';
                    delBtn.style.cssText = 'background:#ff4444;border:none;color:#fff;border-radius:3px;padding:4px 8px;cursor:pointer;font-size:0.75em;flex-shrink:0;';
                    delBtn.dataset.filename = r.filename;
                    delBtn.dataset.label    = r.label;
                    delBtn.dataset.custom   = r.custom;
                    delBtn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        deleteLabelByName(this.dataset.filename, this.dataset.label, this.dataset.custom === 'true', card);
                    });

                    card.appendChild(thumb);
                    card.appendChild(info);
                    card.appendChild(delBtn);
                    wrap.appendChild(card);
                });
            });
        }

        function jumpToFrame(fname) {
            document.querySelectorAll('.fcard').forEach(c => {
                if (c.querySelector('.fname') && c.querySelector('.fname').textContent === fname.substring(0,16)) {
                    c.click();
                }
            });
        }

        function deleteLabelByName(filename, label, isCustom, cardEl) {
            if (!confirm('Remove label "' + label + '" from ' + filename + '?')) return;
            fetch('/obj/delete_label_by_name', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename, label: label, custom: isCustom})
            }).then(r => r.json()).then(d => {
                objLog('Removed "' + label + '" from ' + filename, 'w');
                if (cardEl) cardEl.remove();
                loadKnownTags();
                if (selectedFrame === filename) {
                    loadFrameLabels(filename);
                    const url = '/obj/frame_img/' + filename + '?t=' + Date.now();
                    canvasImg.onload = () => { canvas.width = canvasImg.naturalWidth; canvas.height = canvasImg.naturalHeight; redrawCanvas(); };
                    canvasImg.src = url;
                }
            });
        }

        // Load tags on page load
        loadKnownTags();
    </script>
</body>
</html>
"""

# ── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/capture')
def capture_route():
    if state["cam_source"] == "phone":
        # Save the latest phone frame directly
        save_frame(state["latest_frame"])
    else:
        state["capture_now"] = True
    return jsonify({"status": "ok"})

@app.route('/clear_frames')
def clear_frames():
    """Delete all captured frames and all label/detection data."""
    import glob, json
    files = glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg"))
    for f in files:
        os.remove(f)
    state["frame_count"]     = 0
    state["live_boxes"]      = []
    state["annotated_frame"] = None
    mapper.__init__()  # reset map

    # Wipe detection and label files
    for path in [
        os.path.join(OUTPUT_DIR, "detections.json"),
        os.path.join(LABELS_DIR, "custom_labels.json")
    ]:
        if os.path.exists(path):
            with open(path, "w") as f:
                json.dump({}, f)

    # Reload matcher so it forgets old templates
    try:
        from scanner.matcher import reload_templates
        reload_templates()
    except Exception:
        pass

    socketio.emit('cleared', {"msg": f"Deleted {len(files)} frames and all labels"})
    return jsonify({"status": "cleared", "deleted": len(files)})

@app.route('/auto_start')
def auto_start():
    interval = int(request.args.get('interval', 3))
    state["auto_interval"] = interval
    state["last_auto"]     = time.time()
    return jsonify({"status": "auto_started", "interval": interval})

@app.route('/auto_stop')
def auto_stop():
    state["auto_interval"] = 0
    return jsonify({"status": "auto_stopped"})

@app.route('/map_feed')
def map_feed():
    return Response(mapper.get_map_jpg(), mimetype='image/jpeg')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            # Use annotated frame if live detect is on, else raw frame
            if state["live_detect"] and state.get("annotated_frame") is not None:
                f = state["annotated_frame"]
            else:
                f = state["latest_frame"]
            if f is not None:
                _, buf = cv2.imencode('.jpg', f)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thumb/<path:name>')
def thumb(name):
    path = os.path.join(FRAMES_DIR, os.path.basename(name))
    img  = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (160, 120))
        _, buf = cv2.imencode('.jpg', img)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    return '', 404


# ── Object Detection Routes ──────────────────────────────
@app.route('/obj/frames')
def obj_frames():
    import glob
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
    return jsonify({"frames": [os.path.basename(f) for f in files]})

@app.route('/obj/frame_img/<path:name>')
def obj_frame_img(name):
    """Return full-size frame with detection boxes drawn"""
    path = os.path.join(FRAMES_DIR, os.path.basename(name))
    data = draw_detections(path)
    if data:
        return Response(data, mimetype='image/jpeg')
    return '', 404

@app.route('/obj/frame_thumb/<path:name>')
def obj_frame_thumb(name):
    path = os.path.join(FRAMES_DIR, os.path.basename(name))
    img  = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (160, 120))
        _, buf = cv2.imencode('.jpg', img)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    return '', 404

@app.route('/obj/detect')
def obj_detect():
    results = detect_all_frames()
    return jsonify({"status": "ok", "frame_count": len(results)})


@app.route('/obj/detect_one')
def obj_detect_one():
    """Detect objects in a single frame and save results."""
    import json
    from scanner.detector import detect_frame, OUTPUT_DIR
    filename = request.args.get('filename', '')
    path     = os.path.join(FRAMES_DIR, os.path.basename(filename))
    dets     = detect_frame(path)

    # Merge into detections.json
    det_path = os.path.join(OUTPUT_DIR, "detections.json")
    data = {}
    if os.path.exists(det_path):
        with open(det_path) as f:
            data = json.load(f)
    data[filename] = dets
    with open(det_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "ok", "count": len(dets), "detections": dets})

@app.route('/obj/label', methods=['POST'])
def obj_label():
    data = request.get_json()
    save_custom_label(data['filename'], data['label'], data['box'])
    # Reload visual matcher templates so live detect picks up the new label
    try:
        from scanner.matcher import reload_templates
        reload_templates()
    except Exception:
        pass
    return jsonify({"status": "saved"})


@app.route('/obj/frame_labels')
def obj_frame_labels():
    """Return all labels (custom + auto) for a single frame."""
    import json
    filename   = request.args.get('filename', '')
    detections = load_detections().get(filename, [])
    custom     = load_custom_labels().get(filename, [])
    labels = []
    for d in detections:
        labels.append({"label": d["label"], "confidence": d.get("confidence", 1.0),
                        "box": d["box"], "custom": False})
    for lb in custom:
        labels.append({"label": lb["label"], "confidence": 1.0,
                        "box": lb["box"], "custom": True})
    return jsonify({"labels": labels})


@app.route('/obj/delete_label', methods=['POST'])
def obj_delete_label():
    """Delete one specific label from a frame by index."""
    import json
    data     = request.get_json()
    filename = data['filename']
    idx      = data['index']
    is_custom = data['custom']

    # Build combined list same way as frame_labels to get correct index
    detections = load_detections().get(filename, [])
    custom     = load_custom_labels().get(filename, [])
    combined   = [(d, False) for d in detections] + [(lb, True) for lb in custom]

    if idx >= len(combined):
        return jsonify({"status": "error", "msg": "index out of range"}), 400

    _, item_is_custom = combined[idx]

    if item_is_custom:
        # Remove from custom_labels.json
        cust_path = os.path.join(LABELS_DIR, "custom_labels.json")
        if os.path.exists(cust_path):
            with open(cust_path) as f:
                all_custom = json.load(f)
            # Find which index within this file's list
            custom_idx = idx - len(detections)
            if filename in all_custom and custom_idx < len(all_custom[filename]):
                all_custom[filename].pop(custom_idx)
                if not all_custom[filename]:
                    del all_custom[filename]
            with open(cust_path, "w") as f:
                json.dump(all_custom, f, indent=2)
    else:
        # Remove from detections.json
        det_path = os.path.join(OUTPUT_DIR, "detections.json")
        if os.path.exists(det_path):
            with open(det_path) as f:
                all_dets = json.load(f)
            if filename in all_dets and idx < len(all_dets[filename]):
                all_dets[filename].pop(idx)
                if not all_dets[filename]:
                    del all_dets[filename]
            with open(det_path, "w") as f:
                json.dump(all_dets, f, indent=2)

    return jsonify({"status": "deleted"})


@app.route('/obj/delete_label_by_name', methods=['POST'])
def obj_delete_label_by_name():
    """Delete all labels matching a name from a specific frame."""
    import json
    data      = request.get_json()
    filename  = data['filename']
    label     = data['label']
    is_custom = data['custom']

    if is_custom:
        cust_path = os.path.join(LABELS_DIR, "custom_labels.json")
        if os.path.exists(cust_path):
            with open(cust_path) as f:
                all_custom = json.load(f)
            if filename in all_custom:
                all_custom[filename] = [lb for lb in all_custom[filename]
                                        if lb['label'].lower() != label.lower()]
                if not all_custom[filename]:
                    del all_custom[filename]
            with open(cust_path, "w") as f:
                json.dump(all_custom, f, indent=2)
    else:
        det_path = os.path.join(OUTPUT_DIR, "detections.json")
        if os.path.exists(det_path):
            with open(det_path) as f:
                all_dets = json.load(f)
            if filename in all_dets:
                all_dets[filename] = [d for d in all_dets[filename]
                                      if d['label'].lower() != label.lower()]
                if not all_dets[filename]:
                    del all_dets[filename]
            with open(det_path, "w") as f:
                json.dump(all_dets, f, indent=2)

    return jsonify({"status": "deleted"})


@app.route('/obj/delete_frame', methods=['POST'])
def obj_delete_frame():
    """Delete a single frame and remove its labels/detections."""
    import json
    data     = request.get_json()
    filename = os.path.basename(data.get('filename', ''))
    path     = os.path.join(FRAMES_DIR, filename)

    # Delete the image file
    if os.path.exists(path):
        os.remove(path)

    # Remove from detections.json
    det_path = os.path.join(OUTPUT_DIR, "detections.json")
    if os.path.exists(det_path):
        with open(det_path) as f:
            dets = json.load(f)
        dets.pop(filename, None)
        with open(det_path, "w") as f:
            json.dump(dets, f, indent=2)

    # Remove from custom_labels.json
    cust_path = os.path.join(LABELS_DIR, "custom_labels.json")
    if os.path.exists(cust_path):
        with open(cust_path) as f:
            custom = json.load(f)
        custom.pop(filename, None)
        with open(cust_path, "w") as f:
            json.dump(custom, f, indent=2)

    return jsonify({"status": "deleted", "filename": filename})

@app.route('/obj/search')
def obj_search():
    q = request.args.get('q', '')
    results = search_object(q)
    return jsonify({"results": results})

@app.route('/obj/names')
def obj_names():
    return jsonify({"names": get_all_object_names()})


@app.route('/obj/delete_unlabeled', methods=['POST'])
def delete_unlabeled():
    """Delete frames with no labels, and clean up any orphan label/detection data."""
    import glob, json
    all_files  = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
    detections = load_detections()
    custom     = load_custom_labels()

    # Frames that have at least one label
    labeled = set()
    for fname, dets in detections.items():
        if dets:
            labeled.add(fname)
    for fname, lbls in custom.items():
        if lbls:
            labeled.add(fname)

    # Delete image files that are not labeled
    deleted = []
    for fp in all_files:
        fname = os.path.basename(fp)
        if fname not in labeled:
            os.remove(fp)
            deleted.append(fname)

    # Clean detections.json - remove entries for deleted frames
    clean_det = {k: v for k, v in detections.items() if k not in deleted}
    det_path  = os.path.join(OUTPUT_DIR, "detections.json")
    if os.path.exists(det_path):
        with open(det_path, "w") as f:
            json.dump(clean_det, f, indent=2)

    # Clean custom_labels.json - remove entries for deleted frames
    clean_cust = {k: v for k, v in custom.items() if k not in deleted}
    cust_path  = os.path.join(LABELS_DIR, "custom_labels.json")
    if os.path.exists(cust_path):
        with open(cust_path, "w") as f:
            json.dump(clean_cust, f, indent=2)

    return jsonify({"status": "ok", "deleted": len(deleted), "kept": len(labeled)})


@app.route('/obj/upload', methods=['POST'])
def upload_photos():
    """Accept photo uploads (from phone or any device) and save to frames folder."""
    if 'photos' not in request.files:
        return jsonify({"status": "error", "msg": "No files"}), 400

    os.makedirs(FRAMES_DIR, exist_ok=True)
    saved = []
    for f in request.files.getlist('photos'):
        if f.filename == '':
            continue
        # Read image via OpenCV to validate and normalise
        import numpy as np
        data = np.frombuffer(f.read(), np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        state["frame_count"] += 1
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"frame_{state['frame_count']:04d}_{ts}.jpg"
        path  = os.path.join(FRAMES_DIR, fname)
        cv2.imwrite(path, img)
        mapper.add_frame(img, state["frame_count"] - 1)
        socketio.emit('captured', {"count": state["frame_count"], "filename": fname})
        saved.append(fname)

    return jsonify({"status": "ok", "saved": len(saved), "files": saved})


@app.route('/local_ip')
def local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    ngrok = state.get("ngrok_url")
    return jsonify({
        "ip":  ip,
        "url": ngrok if ngrok else f"http://{ip}:5000",
        "https": bool(ngrok)
    })


@app.route('/qr')
def qr_code():
    """Generate a QR code image for the local URL."""
    import socket, io, qrcode
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    url = f"http://{ip}:5000"
    qr  = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img     = qr.make_image(fill_color="black", back_color="white")
    pil_img = img._img.convert("RGB")
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')


@app.route('/live_detect_toggle')
def live_detect_toggle():
    state["live_detect"] = not state["live_detect"]
    if not state["live_detect"]:
        state["live_boxes"]      = []
        state["annotated_frame"] = None
    return jsonify({"live_detect": state["live_detect"]})


@app.route('/live_detect_on')
def live_detect_on():
    state["live_detect"] = True
    return '', 204

@app.route('/live_detect_off')
def live_detect_off():
    state["live_detect"] = False
    state["live_boxes"]      = []
    state["annotated_frame"] = None
    return '', 204


@app.route('/cam_switch')
def cam_switch():
    """Toggle between USB webcam and phone camera."""
    src = request.args.get('src', None)
    if src in ('usb', 'phone'):
        state["cam_source"] = src
    else:
        state["cam_source"] = "phone" if state["cam_source"] == "usb" else "usb"
    # Clear annotated frame on switch
    state["annotated_frame"] = None
    return jsonify({"cam_source": state["cam_source"]})


@app.route('/phone_frame', methods=['POST'])
def phone_frame():
    """Receive a JPEG frame from the phone browser and use it as latest_frame."""
    import numpy as np
    data = np.frombuffer(request.data, np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return '', 400

    # Always update latest_frame when phone sends (regardless of cam_source)
    state["latest_frame"]    = img
    state["annotated_frame"] = None

    # Save to frames folder if requested
    if request.headers.get('X-Capture') == '1':
        save_frame(img)

    return '', 204



# ── Phone camera page ────────────────────────────────────
PHONE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Room Scanner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#000;color:#00ff88;font-family:monospace;display:flex;flex-direction:column;height:100vh;overflow:hidden;}
#viewbox{position:relative;width:100%;flex:1;min-height:0;background:#000;}
video{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;display:none;}
#drawcanvas{position:absolute;top:0;left:0;width:100%;height:100%;display:none;}
#preview{width:100%;height:100%;object-fit:contain;display:block;}
#statusbar{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.75);
  padding:4px 8px;font-size:0.65em;color:#00ff88;text-align:center;}
.bar{background:#111;padding:6px;display:flex;flex-wrap:wrap;gap:5px;justify-content:center;flex-shrink:0;}
.btn{padding:10px 6px;border:none;border-radius:6px;font-family:monospace;font-size:0.7em;cursor:pointer;flex:1;min-width:70px;}
.green{background:#00ff88;color:#000;} .blue{background:#4488ff;color:#fff;}
.red{background:#ff4444;color:#fff;} .orange{background:#ff8800;color:#000;}
#count{color:#444;font-size:0.6em;text-align:center;padding:2px;flex-shrink:0;}
input[type=file]{display:none;}
.upload-btn{display:block;width:calc(100% - 16px);margin:6px 8px;padding:10px;
  background:#1a1a1a;border:2px dashed #333;border-radius:6px;
  text-align:center;cursor:pointer;font-family:monospace;font-size:0.7em;color:#555;}
</style>
</head>
<body>
<div id="viewbox">
  <video id="vid" autoplay playsinline muted></video>
  <canvas id="drawcanvas"></canvas>
  <img id="preview" src="" alt="">
  <div id="statusbar">Loading...</div>
</div>
<div id="count"></div>
<div class="bar" id="live-bar" style="display:none;">
  <button class="btn green" id="btn-stream" onclick="toggleStream()">STREAM</button>
  <button class="btn blue"  onclick="captureFrame()">CAPTURE</button>
  <button class="btn orange" id="btn-auto" onclick="toggleAuto()">AUTO</button>
  <button class="btn blue"  onclick="flipCamera()">FLIP</button>
</div>
<div class="bar" id="photo-bar">
  <button class="btn green" onclick="openCamera('environment')">BACK CAM</button>
  <button class="btn blue"  onclick="openCamera('user')">FRONT CAM</button>
</div>
<div class="bar">
  <button class="btn orange" onclick="runDetect()">DETECT ALL</button>
  <button class="btn red"    onclick="clearAll()">CLEAR ALL</button>
</div>
<label class="upload-btn" for="gallery-input">UPLOAD FROM GALLERY</label>
<input type="file" id="gallery-input" multiple accept="image/*" onchange="uploadGallery(this.files)">
<input type="file" id="cam-input" accept="image/*" onchange="handlePhoto(this.files[0])">

<script>
const isHttps = location.protocol === 'https:';
let streaming=false, autoOn=false, autoTimer=null, sendTimer=null, pollTimer=null;
let facingMode='environment', stream=null, framesSent=0;
const vid  = document.getElementById('vid');
const dc   = document.getElementById('drawcanvas');
const dctx = dc.getContext('2d');
const prev = document.getElementById('preview');

window.onload = function() {
  if (isHttps) {
    document.getElementById('live-bar').style.display = 'flex';
    document.getElementById('photo-bar').style.display = 'none';
    prev.style.display = 'none';
    vid.style.display = 'block';
    dc.style.display  = 'block';
    startCamera();
    // Draw loop - composites video + detection boxes onto canvas
    requestAnimationFrame(drawLoop);
    setStatus('Camera starting...');
  } else {
    setStatus('Photo mode - tap BACK or FRONT CAM');
  }
};

// ── Draw loop: video + boxes on canvas ──
let lastBoxes = [];
function drawLoop() {
  requestAnimationFrame(drawLoop);
  if (!vid.videoWidth) return;
  const vb = document.getElementById('viewbox');
  dc.width  = vb.clientWidth;
  dc.height = vb.clientHeight;
  // Draw video frame
  dctx.drawImage(vid, 0, 0, dc.width, dc.height);
  // Draw detection boxes
  if (lastBoxes.length) {
    const sx = dc.width  / vid.videoWidth;
    const sy = dc.height / vid.videoHeight;
    lastBoxes.forEach(b => {
      const [x1,y1,x2,y2] = b.box;
      const color = b.known ? '#00ff88' : '#00ccff';
      dctx.strokeStyle = color;
      dctx.lineWidth   = 3;
      dctx.strokeRect(x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy);
      const txt = b.label + ' ' + Math.round(b.confidence*100) + '%';
      dctx.font = 'bold 14px monospace';
      const tw  = dctx.measureText(txt).width + 8;
      dctx.fillStyle = color;
      dctx.fillRect(x1*sx, y1*sy - 20, tw, 20);
      dctx.fillStyle = '#000';
      dctx.fillText(txt, x1*sx + 4, y1*sy - 5);
    });
  }
}

async function startCamera() {
  if (stream) stream.getTracks().forEach(t=>t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video:{facingMode, width:{ideal:1280}, height:{ideal:720}}
    });
    vid.srcObject = stream;
    setStatus('Ready - tap STREAM to start');
  } catch(e) { setStatus('Camera error: ' + e.message); }
}

function flipCamera() {
  facingMode = facingMode==='environment' ? 'user' : 'environment';
  startCamera();
}

function sendLiveFrame(andCapture) {
  if (!vid.videoWidth) return;
  const c = document.createElement('canvas');
  c.width = vid.videoWidth; c.height = vid.videoHeight;
  c.getContext('2d').drawImage(vid, 0, 0);
  c.toBlob(blob => {
    if (!blob) return;
    fetch('/phone_frame', {method:'POST', body:blob,
      headers:{'Content-Type':'application/octet-stream','X-Capture':andCapture?'1':'0'}
    }).then(()=>{
      framesSent++;
      document.getElementById('count').textContent = 'Frames: ' + framesSent;
    }).catch(()=>{});
  }, 'image/jpeg', 0.8);
}

function captureFrame() {
  sendLiveFrame(true);
  setStatus('Captured!');
  setTimeout(()=>setStatus('Streaming...'), 1000);
}

function toggleStream() {
  streaming = !streaming;
  const btn = document.getElementById('btn-stream');
  if (streaming) {
    fetch('/cam_switch?src=phone');
    fetch('/live_detect_on');
    sendTimer = setInterval(()=>sendLiveFrame(false), 200);
    pollTimer = setInterval(fetchBoxes, 350);
    btn.textContent='STOP'; btn.className='btn red';
    setStatus('Streaming + detecting...');
  } else {
    clearInterval(sendTimer); clearInterval(autoTimer); clearInterval(pollTimer);
    autoOn = false;
    fetch('/cam_switch?src=usb');
    fetch('/live_detect_off');
    lastBoxes = [];
    btn.textContent='STREAM'; btn.className='btn green';
    document.getElementById('btn-auto').textContent='AUTO';
    document.getElementById('btn-auto').className='btn orange';
    setStatus('Stopped');
  }
}

function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('btn-auto');
  if (autoOn) {
    autoTimer = setInterval(()=>sendLiveFrame(true), 3000);
    btn.textContent='AUTO ON'; btn.className='btn red';
  } else {
    clearInterval(autoTimer);
    btn.textContent='AUTO'; btn.className='btn orange';
  }
}

function fetchBoxes() {
  fetch('/live_boxes').then(r=>r.json()).then(d=>{
    lastBoxes = d.boxes || [];
    if (lastBoxes.length) {
      const labels = [...new Set(lastBoxes.map(b=>b.label))];
      setStatus('Seeing: ' + labels.join(', '));
    } else {
      setStatus('Streaming...');
    }
  }).catch(()=>{});
}

// ── Photo mode ──
function openCamera(facing) {
  const inp = document.getElementById('cam-input');
  inp.setAttribute('capture', facing); inp.click();
}

function handlePhoto(file) {
  if (!file) return;
  prev.src = URL.createObjectURL(file);
  setStatus('Sending...');
  fetch('/phone_frame', {method:'POST', body:file,
    headers:{'Content-Type':'application/octet-stream','X-Capture':'1'}
  }).then(()=>{
    framesSent++;
    document.getElementById('count').textContent = 'Saved: ' + framesSent;
    setStatus('Saved!');
  }).catch(()=>setStatus('Failed - check WiFi'));
}

function uploadGallery(files) {
  if (!files.length) return;
  setStatus('Uploading ' + files.length + ' photo(s)...');
  const fd = new FormData();
  for (const f of files) fd.append('photos', f);
  fetch('/obj/upload', {method:'POST', body:fd})
    .then(r=>r.json()).then(d=>{
      setStatus('Saved ' + d.saved + ' photo(s)!');
      framesSent += d.saved;
      document.getElementById('count').textContent = 'Saved: ' + framesSent;
    });
}

function runDetect() {
  setStatus('Detecting all frames...');
  fetch('/obj/detect').then(r=>r.json()).then(d=>setStatus('Done! ' + d.frame_count + ' frames'));
}

function clearAll() {
  if (!confirm('Delete ALL frames?')) return;
  fetch('/clear_frames').then(()=>{
    framesSent=0; prev.src='';
    document.getElementById('count').textContent='';
    setStatus('Cleared');
  });
}

function setStatus(msg) { document.getElementById('statusbar').textContent = msg; }
</script>
</body>
</html>"""

@app.route('/phone')
def phone_page():
    return render_template_string(PHONE_HTML)


@app.route('/live_boxes')
def live_boxes():
    """Return current detection boxes for phone overlay polling."""
    return jsonify({"boxes": state.get("live_boxes", [])})


# ── Save frame ───────────────────────────────────────────
def save_frame(frame):
    if frame is None:
        return
    state["frame_count"] += 1
    os.makedirs(FRAMES_DIR, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"frame_{state['frame_count']:04d}_{ts}.jpg"
    path  = os.path.join(FRAMES_DIR, fname)
    cv2.imwrite(path, frame)
    print(f"Saved frame {state['frame_count']}: {path}")
    mapper.add_frame(frame, state["frame_count"] - 1)
    socketio.emit('captured', {"count": state["frame_count"], "filename": fname})


def run_dashboard():
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, allow_unsafe_werkzeug=True, use_reloader=False)
    cert = os.path.join(BASE_DIR, "cert.pem")
    key  = os.path.join(BASE_DIR, "key.pem")

    # Get local IP
    try:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Generate cert if missing
    if not os.path.exists(cert) or not os.path.exists(key):
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import datetime as dt

            pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"scanner")])
            now  = dt.datetime.now(dt.timezone.utc)
            c = (x509.CertificateBuilder()
                 .subject_name(subj).issuer_name(subj)
                 .public_key(pk.public_key())
                 .serial_number(x509.random_serial_number())
                 .not_valid_before(now)
                 .not_valid_after(now + dt.timedelta(days=365))
                 .add_extension(x509.SubjectAlternativeName([
                     x509.DNSName(u"localhost"),
                     x509.IPAddress(ipaddress.IPv4Address(local_ip)),
                     x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                 ]), critical=False)
                 .sign(pk, hashes.SHA256()))
            with open(cert, "wb") as f:
                f.write(c.public_bytes(serialization.Encoding.PEM))
            with open(key, "wb") as f:
                f.write(pk.private_bytes(serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()))
            print(f"SSL cert generated for {local_ip}")
        except Exception as e:
            print(f"SSL cert failed: {e}")
            cert = key = None

    # Build SSL context and wrap the server socket manually
    if cert and key and os.path.exists(cert):
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            print(f"HTTPS running at https://{local_ip}:5000")
            print(f"Phone camera page: https://{local_ip}:5000/phone")
            socketio.run(app, host='0.0.0.0', port=5000,
                         ssl_context=ctx,
                         debug=False, allow_unsafe_werkzeug=True, use_reloader=False)
            return
        except Exception as e:
            print(f"HTTPS failed ({e}), falling back to HTTP")

    print(f"HTTP running at http://{local_ip}:5000  (phone camera needs HTTPS)")
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, allow_unsafe_werkzeug=True, use_reloader=False)


def run_live_detector():
    """Background thread: runs RF-DETR (or YOLOv8 fallback) + visual matcher on live frames."""
    from scanner.detector_rfdetr import detect_frame_live
    from scanner.matcher         import match_frame, reload_templates
    DETECT_INTERVAL = 0.4
    reload_templates()  # load saved labeled crops on startup

    while True:
        time.sleep(DETECT_INTERVAL)
        # Run detection if live_detect is on OR phone is actively streaming
        if not state["live_detect"] and state["cam_source"] != "phone":
            continue
        frame = state["latest_frame"]
        if frame is None:
            continue

        boxes = []

        # ── 1. RF-DETR detections — only show objects you have custom labels for ──
        known = set(n.lower() for n in get_all_object_names())
        try:
            rfdetr_boxes = detect_frame_live(
                frame,
                whitelist=known if known else None
            )
            for det in rfdetr_boxes:
                boxes.append({
                    "label":      det["label"],
                    "confidence": det["confidence"],
                    "box":        det["box"],
                    "known":      True,
                    "source":     "rf-detr"
                })
        except Exception as e:
            print(f"RF-DETR live error: {e}")

        # ── 2. Visual matcher (your custom labeled objects) ──
        try:
            custom_hits = match_frame(frame)
            boxes.extend(custom_hits)
        except Exception as e:
            print(f"Matcher error: {e}")

        state["live_boxes"] = boxes

        # Draw boxes onto a copy of the frame and store as annotated_frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        # Scale font/thickness based on image size (iPhone 14 Pro is very high res)
        scale     = max(0.6, min(w, h) / 1000.0)
        thickness = max(2, int(scale * 2))
        font_scale = scale * 0.7
        label_h   = max(28, int(36 * scale))

        used_label_rects = []  # track label positions to avoid overlap

        for det in boxes:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf  = det["confidence"]
            color = (0, 255, 80) if det.get("known") else (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            txt = f"{label} {int(conf*100)}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Try placing label above box, shift down if it overlaps another label
            lx1, ly1 = x1, max(0, y1 - label_h)
            lx2, ly2 = x1 + tw + 8, ly1 + label_h
            # If overlapping a previous label, place inside box top instead
            for (rx1, ry1, rx2, ry2) in used_label_rects:
                if lx1 < rx2 and lx2 > rx1 and ly1 < ry2 and ly2 > ry1:
                    ly1 = y1 + 2
                    ly2 = ly1 + label_h
                    break
            used_label_rects.append((lx1, ly1, lx2, ly2))

            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(annotated, txt, (lx1 + 4, ly2 - int(label_h * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        state["annotated_frame"] = annotated

        if boxes:
            labels = list({b["label"] for b in boxes})
            socketio.emit('live_detections', {"count": len(boxes), "labels": labels})


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("  ROOM SCANNER")
    print("=" * 45)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        input("Press Enter to exit...")
        exit()

    print("Webcam ready!")

    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    t2 = threading.Thread(target=run_live_detector, daemon=True)
    t2.start()
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")

    # Try to start ngrok for phone HTTPS access
    def start_ngrok():
        import subprocess, json, time as _t
        from urllib.request import urlopen
        from urllib.error import URLError

        # Find ngrok
        import shutil
        ngrok_path = shutil.which("ngrok")
        if not ngrok_path:
            print("ngrok not found in PATH")
            return

        try:
            # Kill any existing ngrok
            subprocess.run(["taskkill", "/F", "/IM", "ngrok.exe"],
                           capture_output=True)
            _t.sleep(1)

            # Start ngrok
            subprocess.Popen(
                [ngrok_path, "http", "5000"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("ngrok starting...")
            _t.sleep(4)

            # Get tunnel URL
            for attempt in range(5):
                try:
                    resp = urlopen("http://localhost:4040/api/tunnels", timeout=3)
                    data = json.loads(resp.read())
                    for tun in data.get("tunnels", []):
                        if tun.get("proto") == "https":
                            url = tun["public_url"]
                            state["ngrok_url"] = url
                            print(f"\n{'='*45}")
                            print(f"  PHONE LIVE VIEW (HTTPS):")
                            print(f"  {url}/phone")
                            print(f"  Scan QR on dashboard")
                            print(f"{'='*45}\n")
                            socketio.emit('ngrok_url', {"url": url})
                            return
                except URLError:
                    _t.sleep(2)
            print("ngrok tunnel not found after retries")
        except Exception as e:
            print(f"ngrok error: {e}")

    state["ngrok_url"] = None
    t3 = threading.Thread(target=start_ngrok, daemon=True)
    t3.start()

    print("Dashboard: http://localhost:5000")
    print("SPACE = capture  |  Q = quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_phone = state["cam_source"] == "phone"

        # Only update latest_frame from USB when not in phone mode
        if not is_phone:
            state["latest_frame"]    = frame.copy()
            state["annotated_frame"] = None

        # Skip all capture logic when phone is active - phone handles its own captures
        if not is_phone:
            if state["capture_now"]:
                state["capture_now"] = False
                save_frame(frame)

            if state["auto_interval"] > 0:
                if time.time() - state["last_auto"] >= state["auto_interval"]:
                    state["last_auto"] = time.time()
                    save_frame(frame)

        # Draw overlay on OpenCV window
        display = frame.copy()
        h, w    = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
        if is_phone:
            cv2.putText(display, "PHONE CAM ACTIVE - controls on phone",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(display, "USB cam paused",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        else:
            auto_txt = (f"AUTO: every {state['auto_interval']}s"
                        if state["auto_interval"] > 0 else "AUTO: off")
            cv2.putText(display, f"Frames: {state['frame_count']}  |  {auto_txt}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "SPACE = capture  |  Q = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Scanner - SPACE to capture | Q to quit", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == 32 and not is_phone:
            save_frame(frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! {state['frame_count']} frames in: {FRAMES_DIR}")
    input("Press Enter to exit...")
