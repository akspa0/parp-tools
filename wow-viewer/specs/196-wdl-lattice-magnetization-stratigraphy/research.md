# Spec 196 Research: WDL Lattice Magnetization, Polarity Inversion & Spatial Height Alignment

## 1. Dragon Isles & Compressed Inversion Phenomenon
In early WoW development (such as Dragon Isles, prototype Outland tiles, and Alpha 0.5.3 subterranean builds), developmental map data was frequently compressed into small numeric bands. In several instances:
- Heights were compressed relative to an upper ceiling boundary ($Z' = Z_{\text{ceiling}} - \alpha \cdot \Delta z$) or inverted to prevent interference with baseline terrain / water levels.
- When an archaeology/restoration tool applies a simple positive multiplier from the lowest floor ($Z' = Z_{\text{floor}} + \alpha \cdot (z - Z_{\text{floor}})$), low points stay at the floor while small upward variances explode into massive 100m+ vertical cliffs and spikes (as captured in the user screenshot).
- **Solution**: Polarity inversion option ($\text{Polarity} \in \{+1, -1\}$) and configurable anchor datum ($\text{AnchorMode} \in \{\text{Floor}, \text{Ceiling}, \text{Mean}, \text{NeighborBorder}\}$).

## 2. Neighboring Mesh Height Auto-Fitting Algorithm
Within a tile or across tile junctions, terrain often contains intact or partially active chunks within 1–3 chunk radii ($33.334\text{m} \to 100\text{m}$).
- Along the border between an active chunk $A$ and a compressed chunk $B$, there are 9 (or 17) shared boundary vertices.
- For known scale bands $S \in \{1.0, 3.333, 10.0, 16.0, 33.334, 64.0, 80.0, 128.0\}$ and polarities $P \in \{+1, -1\}$:
  $$\Delta Z(S, P) = \frac{1}{N} \sum_{i=1}^N \left( Z_A(i) - P \cdot S \cdot Z_B(i) \right)$$
  $$\text{RMSE}(S, P) = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( Z_A(i) - [P \cdot S \cdot Z_B(i) + \Delta Z] \right)^2}$$
- Selecting the $(S, P, \Delta Z)$ tuple that minimizes boundary RMSE yields seamless, mathematically optimal edge alignment without manual trial-and-error.

## 3. WDL Lattice Magnetization & Micro-Relief Fitting
A standard WDL file stores low-resolution terrain heightmaps:
- $17 \times 17$ tile vertex heights ($33.334\text{m}$ grid) across outer tile perimeter and chunk corners.
- $16 \times 16$ chunk center heights ($33.334\text{m}$ centers).
- For any point $(u, v) \in [0, 1]^2$ within a chunk, the macro surface height $Z_{\text{wdl}}(u, v)$ can be evaluated via bilinear/bicubic interpolation.
- **Magnetization Formula**:
  $$Z_{\text{final}}(u, v) = Z_{\text{wdl}}(u, v) + P \cdot S \cdot \left( Z_{\text{adt}}(u, v) - \bar{Z}_{\text{adt}} \right)$$
  where $\bar{Z}_{\text{adt}}$ is the chunk/tile local baseline.
- This anchors the compressed high-frequency ADT relief directly to the WDL macro landscape, eliminating vertical floating/tearing.

## 4. Async Tile Restoration Pipeline & 0-Hitch Rendering
Currently, `RefreshTerrainWeakSignalRestoreForLoadedTiles()` executes full-tile $257\times 257$ grid conversions, height transforms, normal re-generation, and OpenGL vertex buffer uploads synchronously inside the camera update loop.
- When traversing into a new tile, this causes 2–4 second main thread stalls.
- **Async Architecture**:
  - Offload tile analysis, height scaling, neighbor fitting, and normal solving to `Task.Run()` / `ThreadPool`.
  - Pass the completed `List<TerrainChunkData>` back to the main thread via a ConcurrentQueue / double-buffer.
  - The main thread performs only lightweight GPU buffer updates ($<1\text{ms}$ per tile), keeping the frame rate smooth and responsive.
