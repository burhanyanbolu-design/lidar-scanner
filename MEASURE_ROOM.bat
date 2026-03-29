@echo off
echo ================================
echo  Room Measurement Analyser
echo ================================
call venv\Scripts\activate
python scanner/measure.py
pause
