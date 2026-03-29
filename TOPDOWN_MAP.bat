@echo off
echo ================================
echo  Top-Down Map Builder
echo ================================
call venv\Scripts\activate
python scanner/topdown.py
pause
