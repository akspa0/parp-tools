# WoWRollback Active Context

## Current Focus: VLM ADT Interchange Format (Jan 2026)

### Session Update (2026-01-13 15:44 EST)

**Status**: MAJOR MILESTONE ACHIEVED ✓

**Per-layer alpha extraction now working!** First time in multiple attempts to build this tooling.

---

## VLM Tool - ADT Interchange Format

**Goal**: Liberate artwork from proprietary ADT format for:
- Artist preservation (20+ years of WoW modding artwork)
- AI/LLM QLoRA training for world generation
- Cross-game asset interchange
- Pre-Alpha restoration from magazine photos

### Implemented ✓

| Data | Storage | Status |
|------|---------|--------|
| Heights (MCVT) | JSON 145 floats/chunk | ✓ via McnkAlpha.McvtData |
| Normals (MCNR) | Accessor ready | ✓ McnkAlpha.McnrData |
| Shadows (MCSH) | PNG 64×64/chunk | ✓ |
| Alpha (MCAL) | PNG per layer (l1,l2,l3) | ✓ **NEW** |
| Layers (MCLY) | tex_id, flags, effect_id | ✓ **NEW** |
| Holes | int per chunk | ✓ McnkAlpha.Holes |
| Textures (MTEX) | String list | ✓ |
| Objects (MDDF/MODF) | JSON placements | ✓ |

### In Progress

| Data | Storage | Status |
|------|---------|--------|
| Tile stitching | 1024×1024 PNG | ✓ |
| Liquids (MH2O/MCLQ) | JSON + PNG | ✓ |
| Depth maps | DepthAnything3 | ✓ |

---

## Key Files Updated

| File | Changes |
|------|---------|
| `McnkAlpha.cs` | Added public accessors for all chunk data |
| `VlmDatasetExporter.cs` | Refactored to use McnkAlpha accessors |
| | Per-layer MCAL extraction with MCLY parsing |

---

## Output Structure

```
vlm_output/
├── images/         # Minimap PNGs
├── shadows/        # MCSH per-chunk (256 per tile)
├── masks/          # MCAL per-layer (_l1, _l2, _l3)
├── dataset/        # JSON with all metadata
└── texture_database.json
```

---

## Vision

This is the foundation for:
1. **ADT Interchange**: Artists extract their work from proprietary format
2. **AI World Building**: Train LLM/VLM on terrain-to-minimap correlation
3. **Pre-Alpha Restoration**: Reconstruct lost content from magazine photos
