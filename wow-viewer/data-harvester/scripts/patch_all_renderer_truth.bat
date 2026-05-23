@echo off
REM Patch renderer-truth signals into all 6 V16 Zarr stores.
REM Run from repo root after MdxViewer captures have been generated.
REM
REM Prerequisites:
REM   - MdxViewer captures exist under output\tmp\mdxviewer_validation_smoke\<build>\
REM   - V16 stores exist under wow-viewer\output\datasets\v16\<build>.zarr

setlocal

cd i:\parp\parp-tools\wow-viewer\data-harvester

echo === Patching all 6 builds ===
uv run python scripts/build_v16_dataset.py patch-renderer-truth ^
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927

echo === Done. ===

endlocal
