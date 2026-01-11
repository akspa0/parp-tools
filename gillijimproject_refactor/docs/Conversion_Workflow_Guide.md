# WoW Map Conversion Workflow Guide

**Complete workflows for converting map data between World of Warcraft versions.**

---

## Overview

| Workflow | Source | Target | Status |
|:---|:---|:---|:---|
| Alpha → WotLK | 0.5.3 | 3.3.5a | Automated |
| Modern → Alpha | 11.0+ | 0.5.3 | Manual steps |
| WMO v14 → v17 | 0.5.3 | 3.3.5a | Automated |

---

# Part 1: Alpha (0.5.3) to WotLK (3.3.5)

Fully automated by the **WoWRollback** toolchain.

## Prerequisites
- **WoWRollback** (built from source)
- **Alpha Data**: Extracted 0.5.3 client data (WDT/ADT/DBC)
- **LK Data**: 3.3.5 DBFilesClient (for crosswalks)

## Data Layout
```
test_data/
├── 0.5.3/tree/
│   ├── DBFilesClient/     (AreaTable.dbc, Map.dbc)
│   └── World/Maps/<Map>/
└── 3.3.5/tree/DBFilesClient/
```

## Workflow

### Step 1: Run Conversion
```powershell
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ../test_data \
  --lk-dbc-dir ../test_data/3.3.5/tree/DBFilesClient \
  --serve --port 8080
```

### Step 2: Verify Results
The tool will:
- Generate AreaID crosswalks
- Convert ADTs to 3.3.5 format
- Fix coordinates (Y/Z swap)
- Launch web viewer

### Output Locations
- ADTs: `parp_out/session_.../03_adts`
- Crosswalks: `parp_out/session_.../02_crosswalks`

---

# Part 2: WMO v14 to v17 Conversion

Converts Alpha WMO files to WotLK-compatible format.

## Prerequisites
- **WoWMapConverter.Cli** (built)
- Source WMO files from 0.5.3 client

## Workflow

### Step 1: Run Converter
```powershell
dotnet run --project WoWMapConverter.Cli -- \
  convert-wmo \
  --input path/to/Ironforge.wmo \
  --output path/to/output.wmo \
  --verbose
```

### Step 2: Copy Textures
The converter lists referenced textures. Copy them from Alpha data to your output:
```
DUNGEONS\TEXTURES\...
```

### Step 3: Verify in Noggit
Load the converted WMO in Noggit 3.x to verify:
- No geometry drop-outs
- Correct lighting/colors
- Proper material assignment

## Key Conversions Applied

| Feature | v14 | v17 |
|:---|:---|:---|
| MOMT size | 44 bytes | 64 bytes |
| MOGI size | 40 bytes | 32 bytes |
| BSP node | (varies) | 16 bytes |
| Batch startIndex | uint16 | uint32 |
| Lightmaps | MOLM/MOLD | MOCV only |

---

# Part 3: Modern (11.0+) to Alpha (0.5.3)

Backporting modern terrain to Alpha client. **Manual process**.

## Prerequisites
- **CASC Explorer** or `wow.tools.local`
- **BlpResize** tool
- **AlphaWdtAnalyzer**

## Workflow

### Step 1: Extract Modern Data
Extract WDT, ADTs, and assets (BLP, M2, WMO) from CASC.

### Step 2: Texture Reprocessing
Alpha has texture size limits (256x256 or 512x512).
```powershell
BlpResize --input "extracted/World/Textures" \
          --output "alpha/World/Textures" \
          --max-size 256
```

### Step 3: Coordinate Transformation

> [!IMPORTANT]
> Modern WoW: Z-up (Height = Z)  
> Alpha: Y-up (Height = Y)

```csharp
// Modern (X, Y, Z_Height) → Alpha (X, Z_Height, Y)
float alphaX = modernX;
float alphaY = modernZ;  // Height moves to Y
float alphaZ = modernY;  // Forward moves to Z
```

### Step 4: ADT Conversion
- Strip modern chunks (MFBO, MTXF)
- Rebuild MCNK headers
- Downgrade liquid (MH2O → MCLQ)

### Step 5: Import to Client
- Place files in `World/Maps/<MapName>/`
- Update `Map.dbc` and `AreaTable.dbc`

## Troubleshooting

| Issue | Cause | Fix |
|:---|:---|:---|
| Missing objects | Y/Z not swapped | Apply coordinate transform |
| Black textures | Invalid BLP format | Use BlpResize |
| Crashes | Modern chunks | Strip unsupported chunks |
| Floating objects | Z (height) wrong | Check coordinate order |

---

# Part 4: Tool Reference

## WoWMapConverter.Cli

```
Commands:
  convert-wmo    Convert WMO v14 to v17
  
Options:
  --input        Source WMO file
  --output       Output WMO path
  --verbose      Enable debug output
```

## WoWRollback.Orchestrator

```
Options:
  --maps         Comma-separated map names
  --versions     Source version (0.5.3)
  --alpha-root   Path to test_data root
  --lk-dbc-dir   Path to 3.3.5 DBFilesClient
  --serve        Launch web viewer
  --port         Web viewer port
```

---

# Part 5: Coordinate Systems Quick Reference

## Alpha (0.5.3) - XZY
- X: East-West
- Z: **Height** (Up)
- Y: North-South

## Modern/WotLK - XYZ
- X: East-West
- Y: North-South
- Z: **Height** (Up)

## Transform Formula
```
Alpha.X = Modern.X
Alpha.Y = Modern.Z  // Height
Alpha.Z = Modern.Y
```
