# Spec 276 — TrinityCore Multi-Version Population Pipeline & Ground Effect Fidelity

**Feature Branch**: `v0.6.1-dev`  
**Created**: 2026-10-18  
**Status**: Specified; awaiting phase implementation  
**Parent**: [Epic 252 (World Simulation, Audio & Interaction)](../252-epic-world-simulation-and-audio/spec.md) & [Epic 249 (Renderer Performance, Lighting & Correctness)](../249-epic-renderer-performance-and-correctness/spec.md)  
**Governance**: `AGENTS.md` §4 (Code Ownership & Architectural Boundaries), §9 (Governance, Scope Fidelity & Receipts), §10 (God-Class Freeze)

---

## 1. Problem Statement & Baseline Evidence

### 1.1 SQL World Population Limitations
1. **Hardcoded Legacy Alpha-Core Schema**:
   - `SqlWorldPopulationService.cs` and `AlphaCoreDbReader.cs` are hardcoded to the pre-release `alpha-core` schema (`spawns_creatures`, `spawns_gameobjects`, `world.sql`, `dbc.sql`).
   - The viewer has zero support for modern or retail-adjacent emulator databases such as **TrinityCore TDB** (3.3.5a Wrath of the Lich King, 4.3.4 Cataclysm, and later versions).
2. **Missing Authentic World Content & Map 451 (`development`)**:
   - TrinityCore's full 3.3.5a TDB (`H:\trinitycore\TDB_full_world_335.26091_2026_09_09.sql`, 292 MB) contains 151,815 creatures across 111 maps and 94,992 gameobjects across 91 maps.
   - Importantly, this includes Map ID 451 (`development`), which preserves authentic developer testing objects (`TEST BRONZE`, `TEST MARBLE`, `TEST STONE`, `Firepit of Death`, `Scott's Marker`, and `Field Repair Bot 74A`) that do not exist in client placement files (`_obj0.adt`).
3. **Alpha-Core Dev Corner Transposition to Map 451**:
   - In pre-release 0.5.x WoW, Blizzard placed test fixtures, NPCs, and test objects directly in the upper-left (e.g. tiles 0_0, 1_1) and upper-right corners of the Azeroth (Map 0) grid.
   - Around WoW 1.0.0, Blizzard severed those test areas from Map 0 and moved them into Map 451 (`development`).
   - `alpha-core` preserves reconstructed/scraped spawn data for those prototype Azeroth corner tiles. For client versions $\ge 1.0.0$ (including 3.3.5a), this data is missing from retail client maps unless explicitly transposed into Map 451.
   - The population pipeline must support transposing these alpha-core corner dev spawns into the Map 451 population dataset, allowing the viewer to populate all authentic developer assets on Map 451.
4. **Runtime SQL Parsing Cost**:
   - Parsing a 292 MB monolithic SQL text file on-the-fly at runtime in the viewer creates multi-second UI locks and gigabytes of GC pressure.
   - An offline or on-demand extractor is needed to ingest TrinityCore SQL and emit compact, partitioned, per-map dataset files (e.g. `output/population/{version}/map_{mapId}.json`) that the viewer can stream instantly.
5. **Multi-Version Extensibility**:
   - `H:\trinitycore\TrinityCore-master\sql\old` contains cumulative SQL migration data spanning `2.4.3`, `3.0.9`, `3.1.3`, `3.2.2a`, `3.3.2`, `3.3.3a`, `3.3.5a`, `4.2.2`, `4.3.4`, `6.x`, `7`, `8.x`, `9.x`, `10.x`, `11.x`, and `12.x`.
   - Tooling must be structured to accommodate multiple game version schemas with version-tagged data directories.
6. **NPC Animation Stance**:
   - When external creature spawns are injected into the scene, `MdxAnimator` defaults to sequence index 0 (`SetSequence(0)`).
   - In models where sequence 0 is `Walk` or `Run`, NPCs appear perpetually running or walking in place. NPCs should default to `Stand` / `Idle` animations.

### 1.2 Ground Effect Foliage Transparency & Fading (Decomp Ground-Truth)
1. **Decomp Analysis (`H:\wowdecomps\4.3.4\wowdecomp15595-w15595\WoW\Source\WorldClient\DetailDoodad.cpp`)**:
   - Authentic client render state:
     - `GxRsSet(GxRs_BlendingMode, GxBlend_Alpha)`: Alpha blending is active (`SrcAlpha, OneMinusSrcAlpha`).
     - `GxRsSet(GxRs_AlphaRef, static_cast<uint8_t>(World::detailDoodadAlphaRef))` where `World::detailDoodadAlphaRef = 128` (0.502f cutoff).
     - `GxRsSet(GxRs_DepthWrite, 1)`: Depth writing is enabled.
     - Distance fading via `CDetailDoodad::s_alphaRampTexture` scaled over the outer 15% of `detailDoodadDist`.
2. **Current `wow-viewer` Discrepancy**:
   - In `ModelRenderer.cs` line 1414, `isAlphaCutout` disables blending (`_gl.Disable(EnableCap.Blend)`) and uses a fixed threshold of `0.15f`.
   - At threshold 0.15 without blending, semi-transparent blade edges are rendered fully opaque, creating thick, chunky foliage edges.
   - In `GroundEffectSceneService.cs` line 541, distance fade is passed to `RenderInstance(transforms[i].Matrix, RenderPass.Opaque, transforms[i].Fade)`, but because `needsBlend` does not account for per-instance fade when `isAlphaCutout` is active, distant grass pops abruptly rather than softly fading out.

---

## 2. User Stories

- **US1 (TrinityCore 3.3.5a Spawns)**: As a viewer operator, I want to load and view creature and gameobject spawns from TrinityCore's full 3.3.5a database across any map, including developer test Map 451 (`development`).
- **US2 (Tooling-Extracted Compact Population Datasets)**: As an engine developer, I want an extraction tool to parse TrinityCore SQL data into lightweight, per-map partition files so that the viewer loads map population in milliseconds without parsing raw SQL text dumps.
- **US3 (Multi-Version Schema Extensibility)**: As an archive researcher, I want the extraction tooling and viewer population system to support multiple client versions (3.3.5a, 4.3.4, and older/newer eras) through versioned dataset selectors.
- **US4 (Authentic NPC Idle Stance)**: As a viewer operator, I want spawned NPCs to default to `Stand` or `Idle` animations instead of running or walking in place.
- **US5 (Ground Effects Transparency Fidelity)**: As a viewer operator, I want ground grass and foliage doodads to match client ground-truth with 0.50f alpha test threshold, clean alpha blending, and smooth distance-based fadeout.

---

## 3. Acceptance Criteria

- **AC-001 (Map 451 Spawns & Alpha-Core Transposition)**: When Map 451 (`development`) is opened and SQL population is enabled on builds $\ge 1.0.0$, the native TrinityCore 3.3.5a spawns (Field Repair Bot 74A, test materials, forges, anvils) and transposed prototype dev spawns from `alpha-core`'s upper-left/right Azeroth corner tiles render at their authentic world coordinates.
- **AC-002 (All Map IDs Supported)**: The population pipeline supports all maps in the 3.3.5a database (including Azeroth 0, Kalimdor 1, Outland 530, Northrend 571, and dungeons/arenas).
- **AC-003 (DBC Display Resolution)**: Creature templates resolve via `CreatureDisplayInfo.dbc` $\to$ `CreatureModelData.dbc` to authentic M2 paths; GameObject templates resolve via `GameObjectDisplayInfo.dbc`.
- **AC-004 (Compact Dataset Format & Tooling)**: An extractor utility (`TrinityCorePopulationExtractor` / CLI) converts raw TrinityCore SQL into partitioned, version-tagged map datasets (`output/population/{version}/map_{mapId}.json`). Loading a map's population takes $\le 10\text{ ms}$.
- **AC-005 (NPC Idle Animation Selection)**: When creature spawns are created, the animator selects `Stand` / `Idle` (animation ID 0 / names containing "stand" or "idle") and explicitly avoids `Walk` or `Run` sequences.
- **AC-006 (Ground Effects Render State Alignment)**: Ground detail doodads render with alpha testing threshold aligned to client ground truth ($\approx 0.50\text{f}$ / $128/255$) and smooth alpha fadeout over distance.
- **AC-007 (Compilation & Test Parity)**: All C# projects build with 0 errors (`dotnet build WowViewer.slnx -c Debug`); existing unit tests and new population tests pass.
