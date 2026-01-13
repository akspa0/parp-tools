# WoWRollback Active Context

## Current Focus: VLM Dataset Tool Migration (Jan 2026)

### Session Update (2026-01-13 12:34 EST)

**Status**: APPROVED → IMPLEMENTING

**Decision**: Migrate VLM export from `WoWRollback.MinimapModule` to `WoWMapConverter.Core`.

**Plan**: Comprehensive bidirectional VLM dataset tool:
- **Export**: ADT → JSON + images (shadows, alphas, liquids, depth maps)
- **Decode**: JSON → ADT (for VLM output → game/editor)

---

## VLM Tool Data Export

| Data | Storage | Notes |
|------|---------|-------|
| Heights | JSON array | 256 chunks × 145 floats |
| Positions | JSON array | 768 floats (256×3) |
| Holes | JSON array | 256 ints |
| Shadows (MCSH) | PNG per chunk | 64×64 grayscale |
| Alpha (MCAL) | PNG per layer | Noggit-style: RLE/4-bit/big |
| Layers (MCLY) | JSON per chunk | texture_id, flags, effect_id |
| Textures (MTEX) | String list | All unique paths |
| Liquids (MH2O/MCLQ) | JSON + PNG | Type, heights, mask |
| Objects (MDDF/MODF) | JSON array | M2/WMO placements |
| Depth map | PNG | DepthAnything3 from minimap |

---

## Files to Create in WoWMapConverter.Core

| File | Purpose |
|------|---------|
| `VLM/VlmDataModels.cs` | All record types |
| `VLM/VlmDatasetExporter.cs` | Main export logic |
| `VLM/VlmAdtDecoder.cs` | JSON → ADT reconstruction |
| `VLM/AlphaMapService.cs` | Noggit-style MCAL read/write |
| `VLM/ShadowMapService.cs` | MCSH → PNG |
| `VLM/LiquidService.cs` | MH2O/MCLQ extraction |

---

## CLI Commands

```bash
vlm-export --client <path> --map <name> --out <dir> [--depth] [--limit N]
vlm-decode --input <tile.json> --output <tile.adt>
```

---

## Previous Focus: PM4 → ADT Pipeline (Dec 2025)

## CK24 Structure
```
CK24 = [Type:8bit][ObjectID:16bit]
- 0x00XXXX = Nav mesh (SKIP)
- 0x40XXXX = Has pathfinding data
- 0x42XXXX / 0x43XXXX = WMO-type objects
```

## Do NOT
- Match CK24=0x000000 to WMOs (it's nav mesh)
- Read PM4 tiles independently for cross-tile objects
