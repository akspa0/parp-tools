@echo off
echo === Running Clean (WoWToolbox.Tests) ===
call clean.bat
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === Running Build (WoWToolbox.Tests) ===
call build.bat
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === Running Tests (WoWToolbox.Tests) ===
call test.bat
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo === All WoWToolbox.Tests steps completed successfully === 