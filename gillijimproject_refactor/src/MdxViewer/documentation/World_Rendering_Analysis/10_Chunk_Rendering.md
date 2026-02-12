# Chunk Rendering System

## Chunk Rendering Pipeline

**Main Entry Point:** `CMapChunk::Render` @ 0x006a6d80

### Pipeline Phases

1. **Opaque Pass (`RenderOpaque` @ 0x006a6e00)**
   - Purpose: Renders terrain base layers and opaque doodads.
2. **Transparent Pass (`RenderTransparent` @ 0x006a6e50)**
   - Purpose: Handles alpha-blended terrain layers and transparent doodads.
3. **Liquid Pass (`RenderLiquid` @ 0x006a6ea0)**
   - Purpose: Renders water, lava, or slime surfaces.
4. **Detail Doodad Pass (`RenderDetailDoodads` @ 0x006a6ef0)**
   - Purpose: Renders grass, flowers, and other small decorative meshes.

### References

- `CMapChunk::Render` (0x006a6d80)
- `CMapChunk::RenderOpaque` (0x006a6e00)
- `CMapChunk::RenderTransparent` (0x006a6e50)
- `CMapChunk::RenderLiquid` (0x006a6ea0)
- `CMapChunk::RenderDetailDoodads` (0x006a6ef0)