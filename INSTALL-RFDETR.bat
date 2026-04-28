@echo off
echo ========================================
echo   LIDAR SCANNER - RF-DETR UPGRADE
echo ========================================
echo.
echo Installing RF-DETR dependencies...
echo This may take a few minutes (downloading PyTorch + Transformers)
echo.

call venv\Scripts\activate.bat

pip install "transformers>=4.50.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ========================================
echo Testing RF-DETR installation...
echo ========================================
python -c "from transformers import AutoImageProcessor, RfDetrForObjectDetection; print('RF-DETR ready!')"

echo.
echo ✅ Done! RF-DETR is installed.
echo.
echo The model will download automatically (~300MB) on first run.
echo.
pause
