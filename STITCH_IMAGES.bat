@echo off
echo ================================
echo  Image Stitcher
echo ================================
call venv\Scripts\activate
python scanner/stitch.py
pause
