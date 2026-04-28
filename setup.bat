@echo off
echo ================================
echo  LiDAR Scanner Setup
echo  With RF-DETR Object Detection
echo ================================
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing all dependencies...
pip install -r requirements.txt

echo.
echo ================================
echo  Setup Complete!
echo  Run START_SCANNER.bat to begin
echo ================================
pause
