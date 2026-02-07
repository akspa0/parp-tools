# MdxViewer - Memory Bank

## Project Overview

**Purpose**: World of Warcraft Alpha 0.5.3 model viewer application.

**Key Technologies**:
- .NET 9.0 Windows Forms
- Silk.NET for OpenGL rendering
- SixLabors.ImageSharp for image processing
- DBCD for DBC database access

## Architecture

### Data Sources
- `MpqDataSource` - Main data source using MPQ archives
- Uses `NativeMpqService` for MPQ file reading
- Builds file list from MPQ internal listfiles
- Supports loose files on disk
- Custom scanning for nested WMO MPQ archives

### Rendering Pipeline
- OpenGL via Silk.NET
- Model rendering with vertex buffers
- Texture support for BLP format
- Animation system for models

## Key File Formats

### MDX (Alpha Model Format)
- Binary 3D model format used in Alpha 0.5.3
- Contains vertices, faces, textures, animations
- Uses C3Color type for color values
- Supports geoset animations

### WMO (World Map Object)
- Indoor/outdoor building models
- Supports BSP collision
- Uses portal system for visibility

### BLP (Blizzard Texture)
- Custom texture format with mipmaps
- JPEG or DXT compression
- Palette-based for older versions

### ADT/WDT (Terrain)
- Alpha client terrain format
- 16x16 chunks (MCNK)
- Supports alpha maps, heightmaps

## Key Classes

| Class | Purpose |
|-------|---------|
| `ViewerApp` | Main application entry point |
| `MpqDataSource` | MPQ file access and file list building |
| `FileBrowser` | File navigation UI |
| `ModelRenderer` | 3D model rendering |

## Known Issues & Solutions

### MPQ File Discovery
- **Symptom**: Only BLP/WMO files show, no MDX/M2
- **Fix**: Call `_mpq.GetAllKnownFiles()` and add to `_fileSet`
- **Debug**: Check "[MpqDataSource] Added X MPQ internal files"

### Case Sensitivity
- **Symptom**: Files not found despite existing
- **Cause**: Alpha uses uppercase extensions (.MPQ)
- **Fix**: Use `StringComparer.OrdinalIgnoreCase`

### WMO Nested Archives
- **Symptom**: WMO files won't load
- **Cause**: WMO data stored in `.wmo.MPQ` archives
- **Fix**: Use `ScanWmoMpqArchives()` for nested scanning

## Build & Run

```powershell
cd src/MdxViewer
dotnet build --no-restore
dotnet run
```

## Debug Output Examples

### File Discovery
```
[MpqDataSource] Ready. X known files:
  .blp: Y files
  .wmo: Z files
```

### WMO Archive Scanning
```
[MpqDataSource] Added WMO MPQ: World\wmo\Dungeon\test.wmo
```

## TODO

- [ ] Verify MDX/M2 file discovery
- [ ] Test WMO loading from nested MPQ
- [ ] Implement animation system
- [ ] Add particle rendering
- [ ] BLP texture loading with alpha
- [ ] Transparency blend modes
- [ ] Frustum culling
- [ ] Terrain rendering
- [ ] Detail doodad system
