# WoW Map Conversion Workflow Guide

**Complete workflows for converting map data between World of Warcraft versions.**

> **Tool**: [WoWMapConverter v3](../src/WoWMapConverter/README.md)  
> Bidirectional conversion toolkit supporting Alpha 0.5.3 through modern retail.

---

## Overview

| Workflow | Source | Target | Tool |
|:---|:---|:---|:---|
| Alpha → WotLK | 0.5.3 | 3.3.5a | WoWMapConverter.Cli |
| WMO v14 → v17 | 0.5.3 | 3.3.5a | WoWMapConverter.Cli |
| Modern → Alpha | 11.0+ | 0.5.3 | Manual (WoWMapConverter + BlpResize) |

---

# Part 1: WoWMapConverter v3 Quick Start

## Prerequisites
- .NET 9.0 SDK
- Community listfile CSV (optional but recommended)

## Build
```powershell
dotnet build src/WoWMapConverter/WoWMapConverter.Cli
```

## Supported Versions
| Version | ADT | WMO | Models | Status |
|:---|:---|:---|:---|:---|
| Alpha 0.5.3 | Monolithic WDT | v14 | MDX | ✅ Ghidra-verified |
| Classic-WotLK | v18 | v17 | M2 | ✅ Full support |
| Cata+ | Split (_tex0/_obj0) | v17+ | M2/M3 | 🔧 In progress |

> [!NOTE]
> **CASC Loading**: The core library includes `CascReader.cs` for direct modern client loading. This feature is implemented but **untested** - revisit when needed.


---

# Part 2: WMO v14 to v17 Conversion

Converts Alpha WMO files to WotLK-compatible format.

## Command
```powershell
dotnet run --project src/WoWMapConverter/WoWMapConverter.Cli -- \
  convert-wmo \
  --input path/to/Ironforge.wmo \
  --output path/to/output.wmo \
  --verbose
```


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

### Step 2: Textures (Automatic)
The converter automatically exports all referenced textures alongside the WMO:
```
output/
├── Ironforge.wmo
└── DUNGEONS/TEXTURES/...  (all referenced BLPs)
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

### WMO Troubleshooting
> [!TIP]
> **Corrupt Groups in Noggit / Visible Gaps**:
> This often indicates a mismatch in `MOGP` header size handling. Alpha WMOs on disk may have 68-byte headers, while the client expects 128 bytes. The Converter (v3) uses **Adaptive Header Skipping** to handle this automatically. If you see corrupt groups, ensure you are using the latest build.
>
> **Upside-Down Textures**:
> If textures appear vertically flipped in Noggit, it may be due to incorrect V-coordinate flipping in the converter. The current converter uses a **Clean Pass-Through** strategy (no flip on read, no flip on write), which preserves the original Alpha orientation. This usually produces correct results for WotLK/Noggit.

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
