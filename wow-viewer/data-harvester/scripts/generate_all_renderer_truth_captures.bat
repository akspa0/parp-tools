@echo off
REM Generate MdxViewer renderer-truth captures for all 6 builds.
REM
REM Prerequisites:
REM   1. Generate per-tile stubs:
REM      cd wow-viewer\data-harvester
REM      uv run python scripts/build_v16_dataset.py generate-viewer-stubs --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
REM
REM   2. Run this batch file to capture renderer-truth artifacts.
REM
REM   3. Patch into stores:
REM      uv run python scripts/build_v16_dataset.py patch-renderer-truth --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
REM
REM Each MdxViewer session:
REM   1. Reads per-tile JSON stubs from --validation-dataset-root/dataset/
REM   2. Renders each tile with the game client at --game-path
REM   3. Writes PNGs to --validation-output/
REM   4. Exits after all captures done (--exit-after-validation)

setlocal

set MDXVIEWER=gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0-windows\ParpToolsWoWViewer.exe
set CAPTURE_ROOT=i:\parp\parp-tools\output\tmp\mdxviewer_validation_smoke
set CLIENT_ROOT=i:\parp\parp-tools\output\tmp\wowarchive-clients
set Resolution=512

echo === 1/6: 0_5_3_3368 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\0_5_3_3368\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\0_5_3_3368" ^
  --validation-output "%CAPTURE_ROOT%\0_5_3_3368" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo === 2/6: 0_5_5_3494 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\0_5_5_3494\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\0_5_5_3494" ^
  --validation-output "%CAPTURE_ROOT%\0_5_5_3494" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo === 3/6: 0_7_0_3694 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\0_7_0_3694\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\0_7_0_3694" ^
  --validation-output "%CAPTURE_ROOT%\0_7_0_3694" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo === 4/6: 3_0_1_8303 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\3_0_1_8303\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\3_0_1_8303" ^
  --validation-output "%CAPTURE_ROOT%\3_0_1_8303" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo === 5/6: 3_3_5_12340 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\3_3_5_12340\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\3_3_5_12340" ^
  --validation-output "%CAPTURE_ROOT%\3_3_5_12340" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo === 6/6: 4_0_0_11927 ===
"%MDXVIEWER%" ^
  --game-path "%CLIENT_ROOT%\4_0_0_11927\World of Warcraft" ^
  --validation-dataset-root "%CAPTURE_ROOT%\4_0_0_11927" ^
  --validation-output "%CAPTURE_ROOT%\4_0_0_11927" ^
  --validation-resolution %Resolution% ^
  --force-validation-regeneration ^
  --exit-after-validation

echo.
echo === All captures generated. Now patch into stores ===
echo cd wow-viewer\data-harvester
echo uv run python scripts/build_v16_dataset.py patch-renderer-truth --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927

endlocal
