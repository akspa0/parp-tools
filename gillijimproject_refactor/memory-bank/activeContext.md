# Active Context

## Current Focus: MDX-L_Tool - Alpha 0.5.3 Archaeology (Feb 6, 2026)

### Critical Status

**Alpha 0.5.3 MDX Parsing**: ✅ WORKING - Validated `GEOS` chunk layout (Tag-Count structure, `UVAS` optimization).
**Texture Resolution**: ✅ WORKING - DBC-driven resolution via `DbcService` (DisplayInfo & Extra support).
**OBJ Export**: ✅ WORKING - Multi-geoset (split) export functional; verified complex models.
**M2 Export**: 🔧 IMPLEMENTING - Phase 2 (WotLK target) in progress.

### MDX Archaeology Findings (Alpha 0.5.3)

| Aspect | Findings | Notes |
|--------|----------|-------|
| **GEOS Chunk** | Tag(4), Count(4), Data(...) | Robust scanner handles Alpha null padding. |
| **ReplaceableId** | 11+n, 1+n mapping | Resolves to `CreatureDisplayInfo` variations. |
| **DBC Service** | CDI + Extra lookup | Maps `ModelID` to Variations and Baked Skins. |
| **Output Path** | `mdx-l_outputs/` | Standardized artifact storage directory. |

### Root Causes Resolved

1. **Missing Mesh Data**: Validated that `UVAS` Count=1 in Version 1300 contains the raw UV data directly, skipping the `UVBS` tag used in later versions.
2. **Missing Textures**: Resolved by implementing `DbcService` to query `CreatureDisplayInfo.csv` and `CreatureDisplayInfoExtra.csv` for variation strings and baked skins.

### Current Workstream

- [x] Implement robust `GEOS` scanner with smart padding detection.
- [x] Integrate `DbcService` for automated texture resolution.
- [x] Verify complex creature exports (Basilisk, Ogre, Lore).
- [ ] Implement M2 (v264) binary writer for 3.3.5 compatibility.

---

## Architecture: MDX-L_Tool

### Parsing Flow
```
1. FileStream → MdxFile.Load()
2. Chunk Scan: VERS → MODL → GEOS → BONE → ...
3. GEOS Scanner: Identify sub-chunks (VRTX, NRMS, TVRT/UVAS)
4. Texture Resolution: Resolve ReplaceableId via TextureService
5. Writer Dispatch: MDL (Text) or OBJ (Geometry)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `MdxFile.cs` | Main parser/scanner - handles Alpha padding. |
| `TextureService.cs`| Archaeology-driven texture resolution (DBC/Name fallback). |
| `ObjWriter.cs` | Exports split-geoset bodies for variants. |
| `MdxToM2Converter.cs`| (Upcoming) Bone/Animation mapping to WotLK v264. |

---

## Technical Notes

- **GEOS Alignment**: Always seek to the next valid 4-character TAG if a chunk appears truncated or followed by null padding.
- **UV Scaling**: Alpha UVs are standard floats [0..1] but may requires V-flip depending on the target renderer.
- **ReplaceableId Mapping**: ID 11+n / 1+n resolved via `CreatureDisplayInfo` variations. Fallback to `ModelNameSkin.blp`.
