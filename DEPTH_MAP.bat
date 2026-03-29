@echo off
echo ================================
echo  Depth Map Generator
echo ================================
call venv\Scripts\activate
python scanner/depth.py
pause
