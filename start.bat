@echo off
title Handwritten Math Solver

echo Activating conda environment...
call C:\Users\saury\anaconda3\Scripts\activate.bat C:\Users\saury\anaconda3

echo Starting backend server...
cd /d "%~dp0backend\src"
python app.py
pause
