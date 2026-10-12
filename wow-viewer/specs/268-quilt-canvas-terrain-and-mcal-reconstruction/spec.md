# Spec 268: Multi-Tile Quilt Canvas Terrain Reconstruction, MCAL Multi-Layer Deciphering & 3D Fractal Brush Lineage (Inches-Scale Resolution)

## 1. Overview & Problem Statement

Since February 2026, the core ambition of the terrain reconstruction pipeline has been synthesizing authentic, WDL-quality macro-mesh geometry and rich ADT terrain data for tiles where **only minimap image data exists** (e.g., the 2,252 development tiles, unreleased prototype continents, and pre-alpha world builds). 

While recent milestones (Spec 262–267) established essential primitives—such as photometric lighting calibration, SAM 3.1 object sieving, lean macro-trestle elevation prediction (`TrestleElevationUNet`), and tight OBB footprinting—several fundamental architectural gaps remain that prevent producing authentic, production-grade terrain:

1. **Disconnected Independent Tiles in the Quilt**:
   Existing tools process each $256 \times 256$ tile in isolation. In the rendered map or exported mesh quilt, adjacent tiles remain independent disconnected islands, exhibiting vertical seam cliffs ($Z$ mismatch between $x=256$ on tile $A$ and $x=0$ on tile $B$), discordant surface normals, and broken texture boundaries.
2. **Entangled Minimap Albedo vs. Shading**:
   Minimaps are composite photographic renderings containing 2D texture splats (grass, rock, snow, dirt), 3D diffuse and self-shadow relief, building rooftops, and doodads. Without deciphering composite colors back into discrete texture layers and stripping albedo, elevation models hallucinate geometric bumps from texture patterns (e.g., cobblestone pebbles or dark grass patches turned into terrain spikes).
3. **Missing MCAL Multi-Layer Deciphering & Dynamic Layer Stacking**:
   In authentic game data, terrain texturing is stored as multi-layer alpha splats (`MCLY`/`MCAL`) referencing discrete tileset textures (`MTEX`). The authentic terrain editor treated terrain as a continuous artist's canvas spanning multi-tile quilts. Because artists painted freely across chunk and tile boundaries, the editor implemented **dynamic layer assignment**: when more than 3 textures intersect on adjacent chunks, the system dynamically optimizes the layer order (L0–L3) to fit the 4-layer hardware constraint per chunk. Existing tools (Noggit / Noggit-Red) merely fill L0 across entire tiles, losing this authentic structure.
4. **Inches Resolution vs. Client Yards Resolution**:
   Decoded DAT project files (v22, v23, and v26)—the authentic project files exported directly from the authoring tools—reveal a critical architectural discovery: **The authoring environment operates in Inches, not Yards** ($1\text{ yard} = 36\text{ inches}$). Height offsets, vertex displacements, and sculpting strokes were stored at $36\times$ higher spatial and vertical resolution than client ADTs! Modern game clients interpret this underlying scale dynamically. Reconstructing terrain directly at coarse client yards resolution loses the fine editor intent. We must target authentic inches-level resolution during prediction and refinement, and then deterministically downsample to client yards resolution.
5. **v7-Era "Pastes & Scars" and 3D Fractal Brush Lineage**:
   Archaeological research from early 2026 (the v7 experimental era) revealed that original world builders heavily utilized **3D fractal brushes**, **terrain pastes** (multi-tile prefabricated terrain motifs like hills, ramps, and trenches), and left behind **brush scars** (historical seams and fossil records where terrain was re-textured without re-sculpting). Both heightmaps and MCAL layers must be analyzed jointly as a multi-tile quilt to detect and fit these fractal motifs.

---

## 2. User Stories

### User Story 1 - Multi-Tile Quilt Canvas Stitching & Seam Boundary Solver (Priority: P1)
As a world builder, I want adjacent reconstructed tiles to be solved jointly as a continuous multi-tile quilt canvas, so that boundary edges achieve exact C0 elevation continuity and C1 normal smoothness without visible seams or tears.

### User Story 2 - Minimap Albedo De-Mixing & Dynamic MCAL Layer Deciphering (Priority: P1)
As a terrain engineer, I want the pipeline to de-mix composite minimap RGB colors into discrete authentic tileset textures (`MTEX`) and multi-layer alpha splats (`MCLY`/`MCAL`), respecting the dynamic layer assignment rule (optimizing layer allocations when $>3$ textures intersect across chunk boundaries).

### User Story 3 - Bare-Terrain Shadow Sieve & WDL Macro-Trestle Synthesis (Priority: P1)
As an ML researcher, I want the pipeline to isolate bare terrain photometric shadows from object albedo and texture noise, feeding the clean signal into the lean WDL macro-trestle model (`TrestleElevationUNet`) to synthesize continuous macro-elevation lattices spanning authentic $>250\text{--}425$ yard mountain relief.

### User Story 4 - High-Resolution Inches-Scale Refinement & Fractal Pastes/Scars (Priority: P1)
As a technical artist, I want the refinement stage to operate at authentic inches-scale resolution ($36\times$ sub-cell density), fitting 3D fractal editor brushes, multi-tile prefabricated pastes, and historical brush scars across the quilt canvas before deterministically reducing to client 145-vertex MCVT yards.

### User Story 5 - Full Monolithic ADT / WDL / Mesh Materialization (Priority: P2)
As a pipeline developer, I want the output to produce fully conforming monolithic Wrath (v18) ADTs, updated WDT/WDL continent manifests, and 3D OBJ/GLB meshes with 100% authentic chunk transfer (MCVT, MCNR, MCLY, MCAL, MCCV, MMDX, MWMO, MDDF, MODF, MFBO, MCLQ).

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | Multi-Tile Quilt Seam Continuity | Solve boundary vertices across adjacent tiles in a quilt | Boundary height mismatch $\|\Delta Z\| \le 0.05$ yards; border normal cosine similarity $\ge 0.98$ |
| **AC-002** | MCAL Multi-Layer Deciphering | Decompose minimap RGB into discrete tileset textures & alpha splats | $\le 4$ layers per MCNK chunk; dynamic layer stack optimizes adjacent transitions; real BLP IDs assigned |
| **AC-003** | Bare Terrain Shadow Extraction | Strip object albedo and texture noise to isolate bare photometric shading | Residual shadow field achieves $\ge 85\%$ correlation to ground-truth terrain illumination |
| **AC-004** | Continuous WDL Macro Trestle Synthesis | Synthesize seamless $64 \times 64$ continent macro-elevation lattices | Mountain vertical relief $\ge 250$ yards; zero boundary shear between adjacent WDL cells |
| **AC-005** | 3D Fractal Brush & Paste/Scar Fitting | Detect and fit recurring 3D fractal brushes and multi-tile prefab pastes/scars | Normalized cross-correlation (NCC) $\ge 0.80$ on recognized terrain motifs |
| **AC-006** | Inches-to-Yards Deterministic Scaling | Sculpt at sub-cell inches resolution ($36\times$) and downsample to 145-vertex MCVT | Zero high-frequency aliasing or ringing; deterministic reduction to yards |
| **AC-007** | Monolithic ADT Chunk Transfer & Export | Export monolithic v18 ADTs with 100% chunk integrity loadable in WoWViewer | Valid MCVT, MCNR, MCLY, MCAL, MCCV, MMDX, MWMO, MDDF, MODF; clean render in WoWViewer |

---

## 4. Architectural Constraints & Code Ownership

- **Code Ownership (AGENTS.md §4)**: Core format I/O remains in `src/core/WowViewer.Core.IO/`. Python dataset, ML, and harvest tooling lives strictly under `wow-viewer/data-harvester/` using `uv`.
- **God-Class Freeze (AGENTS.md §10)**: Zero member additions to `WorldScene.cs` or `ViewerApp.cs`.
- **Format Reader Integrity (AGENTS.md §4)**: Do NOT modify existing MPQ readers, ADT readers, or `AlphaWdtWriter.cs`.
- **Environment & Portability (AGENTS.md §5)**: Zero hardcoded machine-local client paths in source code or tests. Local client roots configured via CLI arguments or environment variables.
- **PowerShell 7 Syntax (AGENTS.md §5)**: All commands formatted for PowerShell 7 with backtick (`` ` ``) continuation.
- **Scope Freeze & Receipts (AGENTS.md §9.1, §9.2)**: Scope frozen upon approval; all checkboxes require verification receipts with exact commands, exit codes, and real output.
