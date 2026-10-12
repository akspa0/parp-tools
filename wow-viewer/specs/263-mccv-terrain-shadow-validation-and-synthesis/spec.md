# Specification: Spec 263 — 1.60 MCCV Terrain Shadow Ground-Truth Validation & Bidirectional Renderer Synthesis

## 1. Executive Summary

In WoW 1.60 (`wow_classic_beta` / 11.2.7 / 12.0 client engine used in **WoW: Forever**), Blizzard fundamentally restructured terrain shadowing: instead of relying on legacy 1-bit `MCSH` chunk bitmasks or runtime shadow cascades, the continuous terrain self-shadow and ambient occlusion field was **baked directly into `MCCV`** (MCNK Chunk Vertex Colors: 145 vertices per chunk, 580 bytes BGRA).

Because 1.60 Classic maps (Azeroth and Kalimdor) were generated directly from 1.12.1 terrain assets, 1.60 `MCCV` provides an authoritative, empirical **ground-truth baseline** for Blizzard's own baked terrain shadow field. However, **1.60 maps are not a uniform 1:1 copy of 1.12.1**: they represent an unholy composite of multiple eras (Vanilla, Alpha/Beta remnants, Cataclysm previews, and modern tooling revisions). Consequently, cross-validation requires strict **geographic stratification**:

- **Tier 1: Trusted Baseline Validation Zones (Pristine Terrain Lineage)**:
  - **Westfall** (e.g. tiles `[28..29, 48..49]`): Open plains, minimal structural deformation, direct 1.12.1 heightmap lineage.
  - **Elwynn Forest** (e.g. tiles `[30..31, 48..49]`): Rolling hills, standard canopy, preserved slope geometry.
  - **Stranglethorn Vale (STV)** (e.g. tiles `[30..33, 53..57]`): Intact ridges, jungle relief, authentic baked shadow contrast.
- **Tier 2: Excluded / Divergent Mashup Zones (Excluded from Baseline Gate)**:
  - **Major Capital Cities** (Stormwind, Ironforge, Orgrimmar, Undercity): Modern structural re-sculpting, high object occlusion.
  - **Wetlands**: Shoreline overhauls and water plane adjustments.
  - **Badlands & Redridge Mountains**: Re-tooled mountain passes, cliff boundary deformations.
  - **Thousand Needles**: Cataclysm / Vanilla terrain mashup reconciliation.

This specification establishes:
1. **Empirical Cross-Validation**: Direct mathematical comparison between 1.12.1 minimap-extracted residual shadow fields ($\Delta S$) and 1.60 `MCCV` vertex shadow arrays across Tier 1 matching continent tiles.
2. **Bidirectional Synthesis Engine**:
   - **Forward**: Sampling 1.12.1 / 0.5.3 minimap residual shadows into authentic 1.60-compliant `MCCV` chunks (145 vertices BGRA) to inject authentic terrain depth and ambient occlusion hints into legacy/restored maps running under WoW: Forever.
   - **Reverse**: Utilizing 1.60 `MCCV` as an exact photometric supervision target to calibrate and validate terrain height reconstruction.

---

## 2. User Stories & Acceptance Criteria

### User Stories
- **US-1 (Empirical Validation on Trusted Zones)**: As a researcher/developer, I want to mathematically compare the bare terrain shadow residual extracted from 1.12.1 minimaps against the actual 1.60 `MCCV` vertex shadow field on corresponding ADT tiles in trusted baseline zones (Westfall, Elwynn, STV), so that I have proof that our residual extraction matches Blizzard's offline baking physics without pollution from modern composite mashups.
- **US-2 (WoW: Forever MCCV Injection)**: As an editor/world builder, I want to take any 0.5.3 or 1.12.1 map that lacks terrain vertex shadows and synthesize authentic 1.60 `MCCV` chunk streams from its minimap, so that restored maps render with full terrain depth and self-shadowing in WoW: Forever.
- **US-3 (Cross-Era Terrain Quality & Zone Filter)**: As an archivist, I want an automated CLI and batch runner that processes continent tiles with automatic zone tier tagging, evaluates correlation metrics, and exports comparison sheets and synthesized ADT chunks.

### Acceptance Criteria (AC)
- **AC-001 (MCCV Extraction & Rasterization)**: Accurately extract 1.60 `MCCV` vertex arrays (145 vertices: 81 outer + 64 inner) from CASC/ADT files and interpolate them onto a continuous $256 \times 256$ tile grid with $\ge 99\%$ geometric coordinate precision.
- **AC-002 (Empirical Cross-Correlation on Trusted Zones)**: Achieve Normalized Cross-Correlation (NCC) $\ge 0.70$ between 1.12.1 minimap bare residual shadow and 1.60 `MCCV` terrain shadow on unoccluded terrain slopes in Tier 1 trusted zones (Westfall, Elwynn, STV).
- **AC-003 (Ridge Coincidence)**: Directional ridges extracted from 1.12.1 minimap shadows must match valley/crease minimums in 1.60 `MCCV` with $\ge 75\%$ spatial coincidence within a 2-vertex tolerance envelope in Tier 1 zones.
- **AC-004 (1.60 MCCV Synthesis)**: Synthesize 145-vertex BGRA chunks from 2D residual shadows that roundtrip into the ADT writer with bit-exact adherence to MCNK sub-chunk offsets and FourCC headers (`Mccv`).
- **AC-005 (Renderer Modulation Verification)**: Verify that the WoWViewer terrain shader evaluates `MCCV` correctly, modulating ambient and diffuse light passes without clipping or discoloration.
- **AC-006 (Governance & Receipts)**: Complete test coverage across C# and Python with all verification receipts recorded per `AGENTS.md` §9.2.

---

## 3. Architectural Constraints & Non-Goals

- **Strict Containment**: All file operations remain within `I:\parp\parp-tools`.
- **God-Class Freeze**: Zero member additions to `WorldScene.cs` or `ViewerApp.cs` per `AGENTS.md` §10. All new features live in isolated service classes.
- **Non-Goals**:
  - Re-baking lighting from scratch using runtime raytracing on the CPU; we leverage the baked minimap residuals and 1.60 `MCCV` directly.
  - Modifying existing stable format readers unless an explicit format bug is proven.
