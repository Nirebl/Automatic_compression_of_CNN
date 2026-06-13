@echo off
REM Batch script to apply HardSwish optimization to all YOLO models
REM Run from project root: scripts\apply_hardswish.bat

echo ================================================
echo  YOLO ncnn Model Optimizer - HardSwish
echo ================================================
echo.

cd /d "%~dp0.."

set ASSETS=app\src\main\assets

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Optimize each model
echo Optimizing yolov8n_320.param...
python scripts\optimize_yolo_ncnn.py -i %ASSETS%\yolov8n_320.param -o %ASSETS%\yolov8n_320_hs.param -m hardswish
echo.

echo Optimizing yolov8n_480.param...
python scripts\optimize_yolo_ncnn.py -i %ASSETS%\yolov8n_480.param -o %ASSETS%\yolov8n_480_hs.param -m hardswish
echo.

echo Optimizing yolov8n_640.param...
python scripts\optimize_yolo_ncnn.py -i %ASSETS%\yolov8n_640.param -o %ASSETS%\yolov8n_640_hs.param -m hardswish
echo.

echo ================================================
echo  Done! Optimized models saved with _hs suffix
echo ================================================
echo.
echo To use optimized models, update yolov8.cpp to load:
echo   yolov8n_XXX_hs.param instead of yolov8n_XXX.param
echo.
echo The .bin files remain the same (no need to copy)
echo.
pause
