# V11 / V11.1 Terrain Model Training Guide

## Overview

Single-stage ConvNeXt V2-based terrain reconstruction model.
Predicts height (3 scales) + MCAL alpha + MCLY class + hole mask from 26 input signals.

## Prerequisites

- .NET 10 SDK
- Python 3.11+ with PyTorch 2.11+
- Staged game clients under `output/tmp/wowarchive-clients/`
- ~17GB VRAM recommended (8GB minimum at batch 8)

## Step 1 — Install Python Dependencies

```powershell
& '.venv-train/Scripts/python.exe' -m pip install timm accelerate lion-pytorch
```

## Step 2 — Extract Dataset

### Scan all clients
```powershell
# 3.3.5 Wrath
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" `
  --map Azeroth --build 3.3.5.12340 --output output/tmp/v11_scan/scan_335_Azeroth.json

# Repeat for: Kalimdor, Northrend, PVPZone01-04, EmeraldDream

# 4.0.0 Cata beta
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft" `
  --map Azeroth --build 4.0.0.11927 --output output/tmp/v11_scan/scan_400_Azeroth.json

# Repeat for: Kalimdor, EmeraldDream, LostIsles, LostIslesPhase1, LostIslesPhase2, MountHyjalPhase1

# 3.0.1 pre-Wrath
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/3_0_1_8303/World of Warcraft" `
  --map Northrend --build 3.0.1.8303 --output output/tmp/v11_scan/scan_301_Northrend.json

# 0.7.0 pre-BC
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_7_0_3694/World of Warcraft" `
  --map Azeroth --build 0.7.0.3694 --output output/tmp/v11_scan/scan_070_Azeroth.json
# Repeat for: Kalimdor

# 0.5.5 / 0.5.3 Alpha
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_5_5_3494/World of Warcraft" `
  --map Azeroth --build 0.5.5.3494 --output output/tmp/v11_scan/scan_055_Azeroth.json
# Repeat for: Kalimdor, EmeraldDream

dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- dataset-scan `
  --client-root "output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft" `
  --map Azeroth --build 0.5.3.3368 --output output/tmp/v11_scan/scan_053_Azeroth.json
# Repeat for: Kalimdor
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
  --decoder-dim 96 --gradient-accumulation 1 --use-compile `
  --freq-ramp-epochs 60
```

### Key Parameters

| Arg | Default | Notes |
|-----|---------|-------|
| `--batch-size` | 16 | 8 for 8GB VRAM, 16 for 16GB |
| `--max-samples` | 1200 | Total training tiles. 2000+ recommended. |
| `--decoder-dim` | 256 | 96 enough for decoder, saves VRAM |
| `--use-compile` | off | torch.compile, first epoch slow |
| `--freq-ramp-epochs` | 60 | Detail-first loss schedule length |
| `--optimizer` | adamw | adamw or lion |

### Monitoring

Loss components: `hf_l1` (high-freq detail), `mid_l1`, `lf_l1` (coarse shape).

Early epochs (0-30): hf_l1 should drop fast (model learns detail first).
Mid epochs (30-60): lf_l1 catches up (shape refines).
Late epochs (60+): all losses plateau.

Expected loss trajectory (heights z-scored, ~1.0 = good):
- lf_l1: starts ~0.5, ends ~0.05
- mid_l1: starts ~0.3, ends ~0.03
- hf_l1: starts ~0.2, ends ~0.02
- mcal_l1: stays ~0.1 (alpha is easy)
- mcly_ce: starts ~3.5, ends ~1.5

## Step 4 — Validate

Run inference on the latest checkpoint separately (doesn't slow training):

```powershell
python scripts/infer_v11.py output/ml-training/runs/v11.1_prod/last.pt `
  output/tmp/v11_cache --output-dir output/ml-training/runs/v11.1_prod/validation --limit 4 --export-obj
```

## Architecture (V11.1)

- **Encoder:** ConvNeXt V2 Tiny stages with overlapping conv stem (7×7 stride 2 ×2)
- **Decoder:** U-Net with ConvNeXt refinement blocks
- **Input:** 26 channels (minimap, MCAL, normals, MCCV, coarse height, liquid, objects, PM4, hole, derived)
- **Outputs:** height_17/65/257 + MCAL alpha (4ch) + MCLY class + hole mask
- **Loss:** Frequency-banded L1 with detail-first ramp over 60 epochs + uncertainty-weighted auxiliary tasks
- **Vocab size:** 35.5M params (29.6M with decoder_dim=96)
