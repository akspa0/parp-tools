# V11.1 Terrain Model Training Guide

## Overview

Single-stage ConvNeXt V2-based terrain reconstruction model with frequency-banded loss.
Predicts height (3 scales, frequency-decomposed) + MCAL alpha + MCLY class + hole mask from 26 input signals.

Detail pulses fire every 25 epochs from epoch 60 onwards, each spiking high-frequency loss
weight for 8 epochs to sharpen fine detail.

## Prerequisites

- .NET 10 SDK
- Python 3.11+ with PyTorch 2.11+
- Staged game clients under `output/tmp/wowarchive-clients/`
- ~8GB VRAM minimum (batch 8), 16GB recommended (batch 16)

## Step 1 — Install Python Dependencies

```powershell
& '.venv-train/Scripts/python.exe' -m pip install timm lion-pytorch
```

## Step 2 — Extract Dataset

### Scan all clients
```powershell
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" `
  --map Azeroth --build 3.3.5.12340 --output output/tmp/v11_scan/scan_335_Azeroth.json
# All maps: Azeroth, Kalimdor, Northrend, PVPZone01-04, EmeraldDream

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft" `
  --map Azeroth --build 4.0.0.11927 --output output/tmp/v11_scan/scan_400_Azeroth.json
# All maps: Azeroth, Kalimdor, EmeraldDream, LostIsles, LostIslesPhase1-2, MountHyjalPhase1

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/3_0_1_8303/World of Warcraft" `
  --map Northrend --build 3.0.1.8303 --output output/tmp/v11_scan/scan_301_Northrend.json

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_7_0_3694/World of Warcraft" `
  --map Azeroth --build 0.7.0.3694 --output output/tmp/v11_scan/scan_070_Azeroth.json
# Maps: Azeroth, Kalimdor

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_5_5_3494/World of Warcraft" `
  --map Azeroth --build 0.5.5.3494 --output output/tmp/v11_scan/scan_055_Azeroth.json
# Maps: Azeroth, Kalimdor, EmeraldDream

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft" `
  --map Azeroth --build 0.5.3.3368 --output output/tmp/v11_scan/scan_053_Azeroth.json
# Maps: Azeroth, Kalimdor
```

### Merge, audit, curate, build cache
```powershell
# Merge all scans
$scans = Get-ChildItem output/tmp/v11_scan/scan_*.json | % { $_.FullName }
$args = @("run","--project","wow-viewer/tools/converter/WowViewer.Tool.Converter","--",
  "dataset-merge","--output","output/tmp/v11_scan/merged.json")
foreach ($s in $scans) { $args += @("--input",$s) }
& dotnet $args

# Audit (reads ADTs from archives to compute metrics, ~3-5 min for 10K tiles)
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- `
  dataset-audit --input output/tmp/v11_scan/merged.json `
  --output output/tmp/v11_scan/audited.json --limit 0

# Curate (filters low-quality tiles)
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- `
  dataset-curate --input output/tmp/v11_scan/audited.json `
  --output output/tmp/v11_scan/curated.json `
  --report output/tmp/v11_scan/curation_report.json `
  --no-require-minimap --no-require-wdl

# Build NPZ cache (includes minimaps from MPQ archives)
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- `
  dataset-build-cache --input output/tmp/v11_scan/curated.json `
  --output-dir output/tmp/v11_cache --overwrite
```

## Step 3 — Train

```powershell
& '.venv/Scripts/python.exe' scripts/train_v11.py `
  'output/tmp/v11_cache/v9_tensor_cache_manifest.json' `
  --output-dir 'output/ml-training/runs/v11.1_prod' `
  --epochs 300 --batch-size 8 --num-workers 0 --max-samples 2000 `
  --decoder-dim 96 --gradient-accumulation 1 --use-compile
```

### Key Parameters

| Arg | Default | Notes |
|-----|---------|-------|
| `--batch-size` | 16 | 8 for 8GB VRAM, 16 for 16GB |
| `--max-samples` | 1200 | Training tiles sampled from curated pool |
| `--decoder-dim` | 256 | 96 is enough, saves ~50MB VRAM |
| `--use-compile` | off | torch.compile, first epoch slow then 20-40% faster |
| `--optimizer` | adamw | adamw or lion |

### Monitoring

Loss prints: `lf` (coarse 17x17), `mid` (mid 65-up(17)), `hf` (detail 257-up(65)).

Detail pulses fire at epochs 75, 100, 125, 150, 175, 200, 225, 250, 275 — `hf` value
will spike to ~0.6+ then decay over 8 epochs, sharpening details.

Expected trajectory at batch 10 with 1600 samples:
- Epoch 0-20: lf≈0.6→0.3, mid≈0.2→0.04, hf≈0.06→0.02
- Epoch 20-60: lf≈0.3→0.08, mid≈0.03, hf≈0.01 — shape catching up
- Epoch 60+: lf≈0.08→0.04, mid≈0.02, hf≈0.01 — detail pulses fire
- mcal_l1: settles ~0.1 (easy signal)
- mcly_ce: starts ~3.5, ends ~1.8

## Step 4 — Validate

```powershell
# The validate script copies last.pt automatically to avoid file locks
python scripts/validate_v11.py output/ml-training/runs/v11.1_prod/last.pt `
  output/tmp/v11_cache --output-dir output/ml-training/runs/v11.1_prod/val
```

Or manually with copy to avoid conflicts:

```powershell
Copy-Item output/ml-training/runs/v11.1_prod/last.pt output/ml-training/runs/v11.1_prod/_val.pt -Force
python scripts/validate_v11.py output/ml-training/runs/v11.1_prod/_val.pt `
  output/tmp/v11_cache --output-dir output/ml-training/runs/v11.1_prod/val
Remove-Item output/ml-training/runs/v11.1_prod/_val.pt
```

Validation preview saved to `val/previews/validation.png`. 4-panel: Minimap | Target | Prediction | Error.

## Architecture (V11.1)

- **Encoder:** ConvNeXt V2 Tiny stages with overlapping conv stem (7×7+3×3 stride 2×2)
- **Decoder:** U-Net with ConvNeXt refinement blocks at 256×256
- **Input:** 26 channels (minimap, MCAL, normals, MCCV at 3× dropout, coarse height, liquid, objects, PM4, hole, derived)
- **Outputs:** height_17/65/257 + MCAL alpha (4ch sigmoid) + MCLY class + hole mask
- **Loss:** Frequency-banded L1 (lf/mid/hf) with detail-first ramp + detail pulses every 25 epochs + uncertainty-weighted auxiliary tasks
- **Size:** 29.6M params with decoder_dim=96, ~60MB in bf16
- **VRAM:** 5.8GB at batch 8 with compile, 10.9GB at batch 16
