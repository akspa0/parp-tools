# Research & Archaeology: Temporal Stratigraphy & Weak Signal Development Mesh Restoration

## 1. Domain Discovery: Temporal Stratigraphy in WoW Map Tiles

### The "Weak Signal" Phenomenon
In early World of Warcraft terrain builds (specifically Alpha 0.5.3 build 3368, but also persisting into 1.12.1 Vanilla, 3.3.5 WotLK, and 4.0.0 Cataclysm), certain map tiles and chunks exhibit compressed, sub-meter elevation profiles. These were historically labelled "weak signal tiles".

Archaeological measurement (Spec 127/132 research & `memory-bank/weak-signal-tile-archaeology.md`) proved that these tiles are **not random floating point noise**:
- **Continuous Relief**: Sub-millimetre relief (e.g. $5.19 \times 10^{-4}$ amplitude at $Z = -501$) continues seamlessly across ADT tile boundaries. Random noise cannot maintain spatial continuity across independent tile files.
- **The Scale Degradation Factor ($1/0.0333... \approx 30\text{x}-33.334\text{x}$)**: When Blizzard's level designers used early WoWEdit terrain tools (e.g., flattening brush, eraser tool, or whole-zone scaling), height values were mathematically multiplied down by $\approx 0.03$ or $\frac{1}{33.334}$ (coinciding with the $33.334\text{m}$ MCNK chunk metric or reciprocal integer divisors).
- **Abandoned Historical Geometry**: In 0.5.3 Azeroth and Kalimdor, weak tiles concentrate heavily in northern latitudes (Kalimdor $y=10..19$ is enriched ~30x; Azeroth $y=20..29$ is enriched ~7x). Cross-build correlation against 4.0.0 proves these were **not early drafts of later zones**, but **authentic abandoned landscapes** from the pre-November 2001 world before the map was resized and split.

### Three Categories of Hidden Development Geometry

| Category | Geometry State | Underlying Mechanism | Recovery Method |
|---|---|---|---|
| **Squeezed Stratum (Weak Signal)** | Present, amplitude crushed ($0.001\text{m} - 0.5\text{m}$) | Multiplied by $0.03$ or downscaled into low bits | Stratigraphic level analysis + SIMD proportional amplification ($\times 33.334$, $\times 16$, $\times 64$) |
| **Holed Stratum (Dev Mesh)** | Present, full $1\times$ fidelity ($100\%$ uncompressed) | MCNK header `HoleMask` (offset 0x40) flags quads as unrendered | Ignore/bypass `HoleMask` rendering culling to expose dev caves, subterranean paths, Outland blockouts |
| **Submerged Stratum (Bathymetry)** | Present, untextured or obscured by liquid planes | Deep ocean floor modeling ($Z < -50\text{m}$) with distinct underwater features | Bathymetric elevation boosting + liquid layer isolation |

---

## 2. Quantitative Metric: `surviving_height_levels`

Amplitude alone cannot distinguish squeezed terrain from flat ocean. The authoritative discriminator is `surviving_height_levels`:
$$\text{Levels}(T) = |\{ h \in T.\text{Heights} \}|$$

- **Bit-Exact Flat** ($\le 1$ level): Pure flat plane, no authored relief.
- **Trace** ($2 - 8$ levels): Quantized or extreme low-precision trace.
- **Coarse Terrain** ($9 - 64$ levels): Quantized step-like historical terrain.
- **Rich Development Terrain** ($\ge 65$ levels, up to $27,000+$ distinct values): Intact historical landscape whose amplitude was compressed down to sub-meter range.

---

## 3. Internal Seam Profiles & Sub-Tile Merge Stratigraphy

Modern ADTs contain $16 \times 16$ MCNK chunks ($533.333\text{m} \times 533.333\text{m}$). During engine evolution (from 1999 single-world $128 \times 128$ MCNK cells to the modern 64x64 ADT container):
- **2x2 Merges**: Seam discontinuities spike at internal chunk index **8**.
- **4x4 Merges**: Seam discontinuities spike at internal chunk indices **4, 8, 12**.
- **MCNK Cell Quilt**: Seam discontinuities elevated across all 15 boundaries.

By evaluating C1 (gradient) and C0 (step) discontinuities across internal boundaries, the stratigraphic scanner identifies whether a tile represents a merged fragment of earlier worlds.

---

## 4. Bottlenecks in Existing Viewer Implementation

Inspection of `ViewerApp.cs` (lines 4445–4800) revealed critical performance and architectural defects:
1. **Synchronous Render-Thread Rebuilding**: `RefreshTerrainWeakSignalRestoreForLoadedTiles()` ran directly on the render thread, allocating full 257x257 float arrays and rebuilding GPU vertex buffers (`ReplaceTileChunksAndRebuild`), inducing 100ms–400ms frame hitches.
2. **Brittle $|Z| < 50\text{m}$ Gate**: The existing `IsTerrainWeakSignalRestoreCandidateHeightmap` ignored any tile where $|Z| \ge 50\text{m}$, discarding mountain tiles, high-altitude plateaus (e.g. Kalimdor $Z=441\text{m}$), and deep ocean floors.
3. **Flat Single-Factor Amplification**: Existing code applied one crude multiplier to the entire tile without respecting surrounding active terrain boundaries, resulting in sharp cliff seams at tile borders.
4. **Zero SIMD Optimization**: Height arrays and normal calculations looped element-by-element on the CPU without vectorization.

---

## 5. Architectural Solution: Spec 194

1. **Zero-Allocation In-Place SIMD Mesh Transform**: Update vertex height and normal buffers in-place using `System.Numerics.Vector<float>` without tearing down GPU tile scene nodes.
2. **Multi-Strata Temporal Gradient Model**: Continuous gradient scaling ($1.0\times \to 512.0\times$) with preset snap factors ($\frac{1}{0.03} \approx 33.334\times$, $16\times$, $64\times$, $128\times$) and C0/C1 boundary stitching to adjacent full-scale terrain.
3. **Comprehensive Stratigraphic Scanner**: Fast background multi-threaded scanner computing `surviving_height_levels`, raw MCVT deltas, MCNK seam profiles, and hole-mask geometry.
4. **Interactive In-Viewer Workbench Plugin**: Real-time false-color heatmap shaders (Levels, Stratum, Amplitude), instant amplification sliders, and hole-mask toggles.
5. **Dual-Era Offline Patcher & Manifest Exporter**: CLI tools producing `stratigraphy_manifest.json` and generating pre-patched loose ADT / monolithic Alpha WDT maps.
