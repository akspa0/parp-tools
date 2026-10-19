# Plan — Spec 276: TrinityCore Multi-Version Population Pipeline & Ground Effect Fidelity

## 1. Architectural Strategy

The TrinityCore population pipeline decouples heavy offline database ingestion from high-speed runtime world rendering, following `AGENTS.md` §4 (Core Library First) and §10 (God-Class Freeze).

```
                                  [H:\trinitycore]
                       TDB_full_world_335.26091_2026_09_09.sql
                       (or TrinityCore-master/sql/old)
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │ TrinityCorePopulationExtractor  │
                      │       (CLI / Harvester)         │
                      └───────────────┬─────────────────┘
                                      │ Ingests templates & spawns
                                      │ Resolves DBC model paths
                                      ▼
                        [output/population/3.3.5a/]
                      ┌─────────────────────────────────┐
                      │  manifest.json                  │
                      │  map_0.json    map_1.json       │
                      │  map_451.json  map_530.json ... │
                      └───────────────┬─────────────────┘
                                      │ Streamed per active map
                                      ▼
                      ┌─────────────────────────────────┐
                      │    SqlWorldPopulationService    │
                      └───────────────┬─────────────────┘
                                      │ Injects WorldSpawnRecords
                                      ▼
                      ┌─────────────────────────────────┐
                      │       ExternalSpawnLayer        │
                      └───────────────┬─────────────────┘
                                      │ Selects Stand/Idle sequences
                                      ▼
                      ┌─────────────────────────────────┐
                      │      Scene Render Pipeline      │
                      │  (MdxRenderer / Instanced Mesh) │
                      └─────────────────────────────────┘
```

---

## 2. Technical Design by Component

### Component 1: Core Population Data Contracts & SQL Ingestion
- **Location**: `wow-viewer/src/core/WowViewer.Core/Population/`
- **Data Models**:
  - `PopulationRecord`:
    - `uint SpawnId`: Unique spawn identifier.
    - `uint EntryId`: Template ID.
    - `WorldSpawnType Type`: Creature or GameObject.
    - `string Name`: Display name from template.
    - `string? Subname`: Title/subname (e.g. `<Repair>` or `<Innkeeper>`).
    - `string ModelPath`: Resolved virtual model path (`Creature\FieldRepairBot\FieldRepairBot.m2` or `World\Generic\Active\Anvil\Anvil.mdx`).
    - `float Scale`: Combined model scale $\times$ template scale.
    - `Vector3 Position`: World position in WoW coordinates $(X, Y, Z)$.
    - `float Orientation`: Heading in radians.
    - `Quaternion? Rotation`: Full 4D rotation for gameobjects $(X, Y, Z, W)$.
    - `uint Faction`: Template faction ID.
    - `uint Flags`: NPC flags or gameobject flags.
    - `bool IsTransposed`: Flag indicating whether this record was transposed from prototype corner dev tiles.
  - `PopulationMapManifest`: Registry of available maps, total creature counts, gameobject counts, and client version tag.
- **Parser (`TrinityCoreSqlReader.cs`)**:
  - High-throughput streaming parser targeting `INSERT INTO \`creature\``, `INSERT INTO \`creature_template\``, `INSERT INTO \`gameobject\``, and `INSERT INTO \`gameobject_template\``.
  - Memory-efficient line-by-line regex/span scanner without loading entire 292 MB files into memory.
- **Prototype Corner Transposer (`AlphaCoreCornerTransposer.cs`)**:
  - Scans `alpha-core`'s `world.sql` Map 0 spawns for the historical test tiles in the upper-left (`tx <= 5, ty <= 5`, e.g. `(0, 0)`, `(1, 1)`) and upper-right corners.
  - Remaps these spawns into Map 451 (`development`) with coordinate preservation and source lineage tagging.
  - Merges them seamlessly with native 3.3.5a Map 451 test objects (`Field Repair Bot 74A`, `TEST BRONZE`, `Forge`, `Anvil`, etc.).

### Component 2: DBC Model Path Resolution
- **Resolution Chain**:
  - **Creatures**:
    1. Spawn `modelid` (override) $\to$ if $0$, fallback to `creature_template.modelid1` (or `modelid2`..`4`).
    2. `CreatureDisplayInfo.dbc`: DisplayID $\to$ `ModelID`, `CreatureModelScale`.
    3. `CreatureModelData.dbc`: ModelID $\to$ `ModelName` $\to$ normalized `.m2` path.
  - **GameObjects**:
    1. `gameobject_template.displayId`.
    2. `GameObjectDisplayInfo.dbc`: DisplayID $\to$ `ModelName` $\to$ normalized `.mdx` / `.m2` path.
- **Extractor CLI (`tools/PopulationExtractor`)**:
  - Standalone utility compiled into `tools/PopulationExtractor/`.
  - Commands:
    ```powershell
    dotnet run --project wow-viewer/tools/PopulationExtractor -- `
        --sql "H:/trinitycore/TDB_full_world_335.26091_2026_09_09.sql" `
        --dbc "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft/DBFilesClient" `
        --version "3.3.5a" `
        --output "wow-viewer/data/population/3.3.5a"
    ```

### Component 3: Runtime Population Streaming & UI Controls
- **Location**: `wow-viewer/src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/`
- **Features**:
  - Support pre-extracted directory path (fast path, $<10\text{ ms}$) with raw SQL fallback.
  - Population source dropdown: `Alpha-Core (Legacy)` vs `TrinityCore (3.3.5a)` vs `Custom Directory`.
  - Version selector for multi-version datasets (`3.3.5a`, `4.3.4`, etc.).
  - Map 451 (`development`) verification check: confirms 1 creature and 50 gameobjects load when Map 451 is open.

### Component 4: NPC Animation Stance Alignment
- **Location**: `wow-viewer/src/viewer/WoWViewer/Rendering/MdxAnimator.cs`
- **Problem**: When a model loads, `SetSequence(0)` is invoked. If sequence 0 is `Walk` or `Run`, stationary NPCs walk or run in place.
- **Solution**:
  - Implement `SelectDefaultIdleSequence()`:
    1. Look for sequence with `AnimationId == 0` (Stand in M2 AnimationData).
    2. Look for sequence with name containing "Stand" or "Idle" (case-insensitive).
    3. If none found, choose first sequence whose name does NOT contain "Walk", "Run", "Death", or "Attack".
    4. Fall back to sequence 0 only if no better sequence exists.

### Component 5: Ground Effect Foliage Transparency & Fading
- **Decomp Alignment (`DetailDoodad.cpp`)**:
  - `World::detailDoodadAlphaRef = 128` ($0.50196\text{f}$).
  - In `ModelRenderer.cs`:
    - Set `alphaCutoutThreshold = 0.50f` for foliage/cutout layers (eliminating thick, opaque leaf edges).
    - When `fadeAlpha < 0.999f`, enable alpha blending so distant ground effects fade smoothly rather than rendering as opaque or popping out.
  - In `GroundEffectSceneService.cs`:
    - Ensure fading detail doodads are passed through the proper render pass with blending enabled.

---

## 3. Phase Roadmap

- **Phase 1: Ground-Truth Decomp Verification & Foliage Transparency Fix**
  - Adjust alpha cutout threshold from 0.15f to 0.50f matching `World::detailDoodadAlphaRef = 128`.
  - Enable smooth distance fade blending in `ModelRenderer.cs` and `GroundEffectSceneService.cs`.
  - Validate with existing render quality tests.

- **Phase 2: Core Data Contracts & TrinityCore SQL Parser**
  - Add `TrinityCoreSqlReader` in `WowViewer.Core.IO` with unit tests for `creature`, `gameobject`, and templates.
  - Parse Map 451 test records (`Field Repair Bot 74A`, `TEST BRONZE`, `Forge`, `Anvil`, etc.).

- **Phase 3: DBC Model Path Resolver & Population Extractor Tool**
  - Implement DBC display info resolution (`CreatureDisplayInfo`, `CreatureModelData`, `GameObjectDisplayInfo`).
  - Create `PopulationExtractor` CLI tool to export partitioned JSON datasets by map ID.
  - Run extractor on 3.3.5a TDB; generate Map 451 and standard map partitions.

- **Phase 4: Viewer Integration & NPC Idle Animation Selection**
  - Enhance `MdxAnimator` to prioritize `Stand` / `Idle` over `Walk` / `Run`.
  - Update `SqlSpawnStreamingService.cs` and `SqlWorldPopulationService.cs` to consume TrinityCore datasets.
  - Verify Map 451 and world maps in viewer build.
