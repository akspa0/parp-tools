# Research: Chunk Manipulation & Transposition in WoW ADT Formats

## 1. Noggit / Noggit-Red Chunk Manipulator Precedents

In legacy Noggit and Noggit-Red, the "Chunk Manipulator" was designed to address a critical limitation of the original Blizzard WoWEdit: the inability to shift terrain features across tile boundaries.

### Key Capabilities of Noggit Chunk Manipulator:
1. **Sub-Tile Grid Addressing**: Operating on the $16 \times 16$ chunk grid rather than monolithic $533.334\text{m}$ tiles.
2. **Multi-Chunk Copy/Paste Buffer**: Serializing chunk structures into an intermediate clipboard holding heights, normals, flags, and texture alpha layers.
3. **Tile Boundary Crossing**: Calculating destination chunk indices across the $16 \times 16$ boundary ($C'_x = (C_x + \Delta x) \pmod{16}$, $T'_x = T_x + \lfloor (C_x + \Delta x) / 16 \rfloor$).

### Limitations in Noggit that Parp Tools WoWViewer Solves:
- **Corrupted Texture Layer Indices**: Noggit often duplicated MTEX filename entries or exceeded the 4-layer-per-chunk limit, crashing the 3.3.5 client renderer.
- **Lost Object Placements**: Noggit did not transpose MDDF (doodads) and MODF (WMOs) associated with the moved chunks, leaving floating trees and buildings behind.
- **Normal Seam Tearing**: Noggit did not recompute normals along the outer boundary of the moved chunks where they meet unchanged terrain.
- **Destructive Edits with No Undo**: Direct file mutations without an operational undo/redo session stack.

---

## 2. ADT Coordinate Systems & Placements Translation

### Terrain Coordinates:
In the WoW engine, world space uses a right-handed coordinate system:
- $+X$: North
- $+Y$: West
- $+Z$: Up

Top-Left of the entire world map is $(X = 17066.666, Y = 17066.666)$.
Tile $(T_x, T_y)$ begins at:
$$\text{TileTopLeft}_X = 17066.666 - (T_x \times 533.33333)$$
$$\text{TileTopLeft}_Y = 17066.666 - (T_y \times 533.33333)$$

Chunk $(C_x, C_y)$ within tile $(T_x, T_y)$ begins at:
$$\text{ChunkCenter}_X = \text{TileTopLeft}_X - ((C_x + 0.5) \times 33.33333)$$
$$\text{ChunkCenter}_Y = \text{TileTopLeft}_Y - ((C_y + 0.5) \times 33.33333)$$

### Doodad / WMO Placement Offsets:
When a selection of chunks spanning global bounding box $[G_{x\text{min}}, G_{y\text{min}}] \to [G_{x\text{max}}, G_{y\text{max}}]$ is shifted by $(\Delta G_x, \Delta G_y)$, any object placement whose world position $(P_x, P_y)$ falls within the world bounding box of the source selection is shifted by:
$$\Delta P_x = - \Delta G_x \times 33.33333\text{m}$$
$$\Delta P_y = - \Delta G_y \times 33.33333\text{m}$$
$$\Delta P_z = \Delta Z$$

---

## 3. Texture Layer & Alpha Splat Palette Harmonization

An MCNK chunk has:
- `MCLY`: Array of up to 4 texture layer entries. Each layer references an index into the tile's root `MTEX` chunk ($0 \le \text{TextureId} < N$).
- `MCAL`: Compressed or uncompressed $64 \times 64$ 8-bit alpha maps for layers $1..3$ (layer 0 has implicit 100% opacity).

### Transposition Rule:
When moving a chunk from Tile A to Tile B:
1. For each texture in the chunk's MCLY table:
   - Identify the source texture filename in Tile A's `MTEX` table.
   - Look up or insert the texture filename into Tile B's `MTEX` table, assigning target index $T'_{\text{id}}$.
   - Update `MCLY.TextureId = T'_{id}`.
2. If Tile B's palette capacity or chunk layer limit is exceeded, execute `TerrainLayerAllocator.Normalize` to preserve the top 4 dominant layers by energy and renormalize alpha splats.
