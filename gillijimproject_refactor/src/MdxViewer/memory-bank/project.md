# MdxViewer

## Overview
WoW world viewer. .NET 10, Silk.NET (OpenGL 3.3), ImGui.NET, SereniaBLPLib, DBCD, ImageSharp.
Fully supports 0.5.3–0.12. 3.3.5 in progress (MH2O + terrain texturing broken).

## Architecture
- `MpqDataSource` → `NativeMpqService` for MPQ reading; builds `_fileSet` from listfiles + nested WMO MPQs
- OpenGL rendering with GPU skinning, `MdxAnimator` for skeletal animation
- `AlphaTerrainAdapter` (0.5.3 monolithic) / `StandardTerrainAdapter` (0.6.0/3.3.5 split)

## Key Formats
- **MDX** (Alpha): MDLX magic, PRE2 particles, ATSQ geoset anim, compressed quat rotation
- **M2/MD20** (3.3.5): via `WarcraftNetM2Adapter` → MdxFile runtime format
- **WMO**: v14 rendering (4-pass), v14↔v17 converters exist
- **ADT**: Alpha monolithic WDT / 0.6.0 split / 3.3.5 split (root+obj0+tex0)
- **WDL**: 0.5.3-specific parser only (MVER v0x12 → MAOF → MARE)
- **BLP**: DXT1/3/5, palette, JPEG via SereniaBLPLib

## Known Issues
- **WMO stained glass**: wrong geometry mapping (root cause unknown)
- **MDX cylindrical stretch**: texture wrap mode fix failed (root cause unknown)
- **Patch MPQs**: deterministic priority with BZip2 support (working)

## Build
```powershell
dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj
```
