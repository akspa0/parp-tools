@echo off
REM AlphaFixes Build Script for Windows

echo ============================================
echo AlphaFixes - WoW Alpha 0.5.3 Performance Patches
echo ============================================
echo.

REM Check for Visual Studio
if not defined VS140COMNTOOLS (
    echo [ERROR] Visual Studio not found!
    echo Please install Visual Studio 2015 or later with C++ support.
    pause
    exit /b 1
)

REM Set up build environment
call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x86

echo [INFO] Building AlphaFixes...

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo [INFO] Running CMake...
cmake .. -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_FLAGS="/O2 /GS-" ^
    -DCMAKE_CXX_FLAGS="/O2 /GS-"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

REM Build
echo [INFO] Compiling...
nmake

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Build complete!
echo.
echo The DLL has been built at: build\alphafixes.dll
echo.
echo To use AlphaFixes:
echo 1. Copy alphafixes.dll to your WoW Alpha directory
echo 2. Copy alphafixes.ini to configure (optional)
echo 3. Inject using a launcher or AppInit_DLLs
echo.
pause

cd ..
exit /b 0
