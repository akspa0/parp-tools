@echo off
echo === Running Clean (WoWToolbox.AnalysisTool) ===
call clean.bat
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === Running Build (WoWToolbox.AnalysisTool) ===
call build.bat
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === All WoWToolbox.AnalysisTool steps completed successfully === 