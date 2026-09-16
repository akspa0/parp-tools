# Feature Specification: Modern Client Assets (Post-5.0.1, FileDataID Era)

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-16

**Status**: Draft

**Depends on**: [Spec 238 CASC Data Source](../238-casc-data-source/spec.md)

**Input**: User description: "...and then support clients above version 5.0.1, to read the assets properly. We ultimately can learn from existing implementations like wow.export ... as well as what we need to make it happen in the renderer."

## Context

Once Spec 238 can pull bytes out of a modern client, the viewer still has to understand them. From
6.0 onward, and decisively from 8.1, the client stopped naming files by path and started
referencing them by **FileDataID**. That change shows up in:

- **WDT**: per-tile file ids for root/obj/tex/lod ADTs, minimap and map textures (`MAID`, 8.1+), plus light/occlusion/fog companion WDTs.
- **ADT**: placements whose "name index" is a file id when a flag is set, texture ids for diffuse/height (`MDID`/`MHID`), and texturing parameters (`MTXP`).
- **M2**: chunked `MD21` files with id tables for skins, textures, skeleton, bones and animations (`SFID`, `TXID`, `SKID`, `BFID`, `AFID`, ...).
- **WMO**: group file ids (`GFID`), doodad ids (`MODI`), texture ids in materials, and new era chunks.
- **Client databases**: `.db2` in WDB5/6 and WDC1–5 layouts, needed for maps, areas, lights and liquids.

**Repo facts (measured 2026-09-16):**

- The core has **no FileDataID resolution at all** (0 matches for `MAID`/`SFID`/`TXID`/`GFID`/`MODI`).
- The viewer's supported range tops out at MoP 5.0.1 split ADTs (Spec 197). `MD21` is only detected, plus used in a Warcraft.NET M2 adapter.
- The vendored **DBCD already reads WDB2 through WDC5**, so database *decoding* exists. What's missing is id-addressed database access and per-era table usage.
- The vendored **Warcraft.NET** has chunk definitions for M2 (Legion/BfA/SL/DF/TWW), WMO (WoD/Legion/BfA), and WDT/ADT (root, obj, tex, lod, light, occlusion, fog). That is format prior art, but Core.IO remains the canonical owner (Constitution II).
- Related specs: 197 (split ADT, height texturing), 198 (M2/WMO shader permutations), 205 (MH2O).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load a modern map's terrain (Priority: P1)

The operator opens a modern build (through Spec 238) and loads a world map. Its terrain appears with
correct heights, textures, height-based texture blending, holes, liquids and area names.

**Why this priority**: Terrain is the foundation of the viewer and the most direct evolution of the
existing split-ADT path.

**Independent Test**: For one representative build per era tier, load a known outdoor zone. Tiles
stream, texture blending looks right, and the minimap tile for the same area resembles the rendered
terrain.

**Acceptance Scenarios**:

1. **Given** a build whose WDT references tiles by file id, **When** the map loads, **Then** every tile's root/obj/tex/lod files resolve by id and the tile renders.
2. **Given** a build whose WDT references tiles by name, **When** the map loads, **Then** tiles resolve by path as today.
3. **Given** a tile with height-texturing parameters, **When** it renders, **Then** layers blend by height, not by alpha alone.
4. **Given** a tile with liquid, **When** it renders, **Then** liquid type and material resolve through that build's database tables.

---

### User Story 2 - Modern doodads and world objects appear (Priority: P1)

Placed M2 doodads and WMO buildings in a modern map load with their geometry, textures and doodad
sets. Placements that reference models by file id resolve correctly.

**Why this priority**: A map without its objects isn't recognizable. The id-based placement change
is the most common reason modern maps look empty in older tools.

**Independent Test**: In a representative zone, 100% of placements resolve to a model file or are
reported as unresolved with a reason, and a city renders with its buildings.

**Acceptance Scenarios**:

1. **Given** a placement flagged as file-id-referenced, **When** it loads, **Then** the model is fetched by id.
2. **Given** a chunked M2, **When** it loads, **Then** its skin, textures and (for static display) skeleton resolve through their id tables.
3. **Given** a WMO whose groups and doodads are listed by id, **When** it loads, **Then** every group and doodad set loads.
4. **Given** a model file that is encrypted with an unavailable key, **When** it loads, **Then** a placeholder is shown and the reason is recorded.

---

### User Story 3 - Client databases drive world context (Priority: P2)

Maps, areas, lights/skies and liquid types are read from the build's own `.db2` tables, located by
file id, using the definitions that match the build version.

**Why this priority**: Area names, sky/lighting and liquid appearance all depend on era-specific
tables. Without them, rendering is correct in geometry only.

**Independent Test**: For each tier build, the map list populates from its map table, area names
appear for known zones, and a lit outdoor scene uses that build's light data.

**Acceptance Scenarios**:

1. **Given** a modern build, **When** the map list is requested, **Then** it is read from that build's map table.
2. **Given** a table whose definition doesn't match the build, **When** it is loaded, **Then** the mismatch is reported and that table's features degrade, without crashing.

---

### User Story 4 - Know what the viewer can and can't read per build (Priority: P2)

The operator can run a coverage survey over a build that reports, per format and per chunk, what was
decoded, what was skipped as unknown, and what failed. This guides the research without guesswork.

**Why this priority**: "Handles all CASC version data" is a long tail. A measured coverage report
turns it into a ranked backlog instead of anecdotes.

**Independent Test**: Run the survey on a tier build. It produces per-format counts of files read,
unknown chunks by id with occurrence counts, and failures by reason.

**Acceptance Scenarios**:

1. **Given** a build, **When** the coverage survey runs on a map, **Then** every WDT/ADT/M2/WMO touched is reported with unknown chunks and failures.

---

### User Story 5 - Older clients are unaffected (Priority: P1)

**Independent Test**: 0.5.3, 3.3.5 and 5.0.1 maps render identically before and after, and the
existing tests pass.

### Edge Cases

- The same build mixes name-referenced and id-referenced files (transitional eras): both paths must work within one map.
- A file id is present in the WDT but missing from the build (common for unused lod or minimap slots): tolerate it silently for optional slots and report it for required slots.
- An M2 references an external skeleton or animation file that is encrypted or absent: render in bind pose and note it.
- A WMO has more groups than the renderer's current admission budget (Stormwind-class): admission rules from the renderer apply; this spec does not change them.
- A chunk is new in an era and not known to any reader: it is skipped, counted in the coverage survey, and never fatal.
- A database definition is missing for an exact build: use the nearest compatible layout hash when the definitions say it matches; otherwise report it.
- Character, creature-display and item rendering: out of scope (world geometry only).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide one file-reference resolver that turns every reference form (path, FileDataID, flagged placement index) into data through the active data source.
- **FR-002**: The system MUST read modern WDTs, including per-tile file ids and companion light/occlusion/fog WDTs as far as the renderer uses them.
- **FR-003**: The system MUST read modern root/obj/tex/lod ADTs, including file-id placements, diffuse/height texture ids and texturing parameters.
- **FR-004**: The system MUST read chunked M2 models and resolve skin, texture, skeleton, bone and animation file ids needed for static world display.
- **FR-005**: The system MUST read modern WMO roots and groups, including group and doodad file ids and material texture ids.
- **FR-006**: The system MUST read `.db2` tables by file id for the database-driven features the viewer already has (maps, areas, lights, liquids), selecting definitions by build.
- **FR-007**: The renderer MUST display height-based texture blending, modern liquid materials and modern model/WMO materials to at least the fidelity it already achieves for 5.0.1. Shader-combination fidelity is delegated to Spec 198.
- **FR-008**: The system MUST provide a coverage survey command reporting decoded, unknown and failed chunks per format for a build or map.
- **FR-009**: Unknown chunks and unresolvable references MUST degrade gracefully (skip or placeholder) with diagnostics, never crash.
- **FR-010**: Each era tier claimed as supported MUST be proven on a real build of that tier, recorded in evidence.
- **FR-011**: Existing pre-6.0 behavior MUST be unchanged.
- **FR-012**: Readers MUST live in Core.IO (canonical owner). Vendored Warcraft.NET and wow.export are prior art to consult and to cross-check against, not a second owner.

### Key Entities

- **File reference**: A path or FileDataID with its origin (which chunk/field) and whether it is required.
- **Era tier**: A range of builds sharing reference and format conventions. Tier A covers 6.x–8.0 (mixed names and ids); Tier B covers 8.1–8.3 (id-first); Tier C covers 9.x and later (newest chunk revisions).
- **Coverage record**: Per build, per format: files read, chunk ids seen (known/unknown), failures by reason.
- **Database table binding**: A table name, its file id, its definition layout, and the build range it applies to.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For one real build per era tier, a chosen outdoor zone renders all tiles, and at least 99% of the tile file references resolve.
- **SC-002**: For those zones, at least 98% of M2/WMO placements load a model (the remainder are individually attributed to encryption, absence or a named failure).
- **SC-003**: Map list and area names for those builds come from the builds' own tables, and 100% of the viewer's database-driven features either work or report a definition mismatch.
- **SC-004**: The coverage survey for each tier build lists every unknown chunk id with its count, and no crash occurs across a full map survey.
- **SC-005**: The operator confirms by eye, per tier, that terrain blending, buildings and doodads look coherent against in-game reference screenshots or the build's own minimap tiles.
- **SC-006**: 0.5.3, 3.3.5 and 5.0.1 maps render unchanged, and the existing test suite passes.

## Assumptions

- Spec 238 delivers id-addressed reads before this spec's Phase 1 starts.
- The target is **world display** (terrain, doodads, WMOs, liquids, sky/light). Characters, creatures, items, UI, spells and animation playback beyond a static/idle pose are out of scope.
- The representative build per tier is chosen by the operator in Phase 0 from builds they can lawfully access.
- wow.export's `src/js/3D/loaders/*` (ADT/WDT/M2/WMO loaders, `Skin.js`, `M2Generics.js`) and renderers are a behavioral reference, and Warcraft.NET's era chunk definitions are a layout cross-check. Neither is adopted as a dependency for reading.
- Writing modern-format files is out of scope.
