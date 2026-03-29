@echo off
echo ================================
echo  LiDAR Scanner Setup
echo ================================
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================
echo  Setup Complete!
echo  Run START_SCANNER.bat to begin
echo ================================
pause
