# Phase 6C: WMOv14 Support & Conversion 🏰

**Goal**: Add Alpha WMOv14 (World Map Object) support for assets that crash in 3.3.5 clients.

---

## 📋 Why WMOv14 Support?

1. **Early assets only exist in v14** - Never updated to later versions
2. **3.3.5 crashes** - Non-manifold surfaces, rounding errors
3. **Data preservation** - Only way to access these assets
4. **Complete coverage** - Phase 6 needs WMO support

### Version History
- **v14**: Alpha (0.5.3-0.5.5) - **Single file**
- **v16**: Beta (0.6.0) - Split into root + groups
- **v17**: Release (1.0+) - Modern format

---

## 🏗️ Key Format Differences

### File Structure
```
v14 (Alpha): single_file.wmo
├── MVER (14)
└── MOMO (wrapper)
    ├── MOHD, MOTX, MOMT... (root)
    └── MOGP chunks (inline!)

v17 (LK): root.wmo + root_000.wmo + root_001.wmo...
```

### Major Changes
1. **Monolithic → Split**: v14 is one file, v17 splits root + groups
2. **Lightmaps → Vertex Colors**: v14 uses MOLM/MOLV, v17 uses MOCV
3. **Batch Structure**: v14 embeds in header, v17 has separate MOBA
4. **Material Indexing**: v14 uses MOTX offsets, v17 can use FileDataIDs

---

## 🎯 Reference Materials

### 1. WoWToolbox (C#) - Best Starting Point
**Path**: `lib/parp/parpToolbox/WoWToolbox/`
- ✅ Already C#, reads WMOv14

### 2. mirrormachine (C++) - Conversion Logic  
**Path**: `lib/mirrormachine/src/WMO_exporter.cpp`
- ✅ Shows v14 → v17 conversion

### 3. wow.export (TypeScript) - v17 Writing
**Path**: `lib/wow.export/src/js/3D/exporters/WMOExporter.js`
- ✅ Modern v17 format

### 4. wowdev.wiki
**Path**: `reference_data/wowdev.wiki/WMO.md`
- ✅ Complete specification

---

## 📦 Implementation

### New Files
```
WoWRollback.Core/Formats/Alpha/Wmo/
├── WmoV14Reader.cs       // Read v14
├── WmoV14Structures.cs   // Data structures
├── WmoV17Writer.cs       // Write v17
└── WmoConverter.cs       // v14 → v17
```

### Key Conversions

#### 1. Lightmaps → Vertex Colors
```csharp
// Sample lightmap at UV, apply to MOCV
var color = SampleLightmap(lightmap, uv);
vertexColors.Add(color);
```

#### 2. Batch Structure
```csharp
// Extract IntBatch[4] + ExtBatch[4] from v14 header
// Convert to v17 MOBA format with bounding boxes
```

#### 3. Fix Non-Manifold Surfaces
```csharp
// Validate mesh, remove degenerate triangles
// Weld duplicate vertices within epsilon
```

---

## 🎨 CLI Commands

```powershell
# Convert v14 → v17
wowrollback convert-wmo AlphaBuilding.wmo \
  --output converted/AlphaBuilding.wmo

# Export as GLB
wowrollback export-3d-wmo AlphaBuilding.wmo \
  --output AlphaBuilding.glb \
  --convert-lightmaps

# Batch convert
wowrollback convert-wmos \
  --input-dir ../test_data/0.5.3.3368/World/wmo \
  --output-dir converted_wmos \
  --threads 8
```

---

## ✅ Success Criteria

- [ ] Read all Alpha WMOv14 files
- [ ] Convert v14 → v17 without crashes
- [ ] Export as GLB with textures
- [ ] Lightmap conversion looks correct
- [ ] Non-manifold surface fixing

---

## 📅 Timeline

- **Week 1**: WmoV14Reader (based on WoWToolbox)
- **Week 2**: WmoV17Writer (based on wow.export)
- **Week 3**: Conversion logic (v14 → v17)
- **Week 4**: GLB export integration
- **Week 5**: Testing & validation

**Total: 5 weeks** (parallel with Phase 6B)

---

**Complete WMO support makes WoWRollback handle ALL Alpha 3D assets!** 🚀
