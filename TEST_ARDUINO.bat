@echo off
echo ================================
echo  Arduino Connection Test
echo ================================
call venv\Scripts\activate
python test_arduino.py
pause
