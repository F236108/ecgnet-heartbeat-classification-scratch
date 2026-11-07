@echo off
title Push ECGNet to GitHub
echo ============================================================
echo   Pushing ECGNet to GitHub: https://github.com/F236108/ecgnet-heartbeat-classification-scratch
echo ============================================================
echo.

set "PATH=C:\Program Files\Git\cmd;C:\Program Files\Git\bin;%PATH%"

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   [SUCCESS] All files pushed to GitHub successfully!
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo   [NOTE] If prompted, sign in via browser or provide your GitHub Personal Access Token.
    echo ============================================================
)

echo.
pause
