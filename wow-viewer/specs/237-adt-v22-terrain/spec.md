# Feature Specification: ADT/v22 Terrain Reading and Rendering

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Depends on**: [238 CASC Data Source](../238-casc-data-source/spec.md) and [239 Modern Client Assets](../239-modern-client-assets/spec.md) for resolving referenced textures/models (US3 only; US1/US2 are independent)

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "Add support for reading and rendering the data from ADT/v22 files, because data just became available in that format for no apparent reason, and I have no tooling to read or render them, currently, but I really ought to. https://wowdev.wiki/ADT/v22"

## Context

ADT/v22 (and its sibling v23) is the experimental pre-Cataclysm terrain tile format that opens with
an `AHDR` chunk instead of `MVER`. It stores the full 129x129 outer / 128x128 inner vertex grid once
per tile (`AVTX`, `ANRM`), holds texture and model name tables per tile (`ATEX`, `ADOO`), and nests
layers, alpha maps, shadows and object placements inside per-chunk `ACNK` containers
(`ALYR`/`AMAP`/`ASHD`/`ACDO`). v23 adds whole-tile flight bounds (`AFBO`) and vertex shading (`ACVT`).

Real v22 files have just become available. The toolchain can't do anything useful with them:

- File detection recognizes `AHDR`-leading files but labels **every** one `AdtV23`, ignoring the
  `AHDR.version` field. A v22 file is currently reported as v23.
- The only reader reads the `AHDR` header and counts chunks. No chunk payload is decoded.
- Every existing test fixture is a synthetic buffer; no real `AHDR` file has ever been parsed.
- There is no terrain adapter, so the viewer can't display the terrain at all.

**First look at the real corpus (2026-09-16, [evidence/phase0-first-look-2026-09-16.md](evidence/phase0-first-look-2026-09-16.md))**: the 700 files Marlamin passed on are a
**previously undocumented revision** of this family, not the wiki's v22. Every file is `MVER` 26 followed by `AHDR`
(version 26), so the current detector's AHDR-first check never sees them. They add three chunks the wiki doesn't list
(`ALOC`, `AOCH`, `ADST`). Filenames are bare FileDataIDs that encode no position. Tile coordinates are **measured**
to come from `ALOC`: fields 1 and 2 are tile X and Y, and neighbouring tiles' height edges match exactly.

**Provenance (operator, 2026-09-16)**: these files are **public**. They ship in today's `wow_classic_beta` build, the first
WoW: Forever build on Battle.net servers and only hours old. **There is no WDT, map table entry or listfile name for
them (operator-confirmed)**, so everything must come from the tile files themselves.

The wowdev.wiki pages describe themselves as incomplete ("may not list all chunks", "do not bother
implementing until final version"). Per the project's standing rule that a named field is
unexamined until measured, every wiki layout in this spec is a **hypothesis to verify against the
real files**, not an established fact.

## User Scenarios & Testing *(mandatory)*

### User Story 0 - Fast path: see the new terrain as a wireframe (Priority: P1, first)

The operator opens `test_data/v22_adts/unknown/` in the viewer and sees the terrain of the new revision-26 tiles as a
height wireframe, each tile at its `ALOC` position. There are no textures, objects or lighting.

**Why this priority**: the files are hours old and public. A first render is worth more now than a complete decoder
later (time-to-signal). It uses **only measured facts**: detection (MVER + AHDR), `ALOC` tile X/Y, and row-major outer
heights, which are seam-proven. It does not wait on the probes for normals, alpha or placements.

**Independent Test**: open the folder; the 30 non-flat tiles appear as one continuous wireframe landscape with no
cracks at tile edges; the 669 flat tiles appear flat (or are hidden by a toggle).

**Acceptance Scenarios**:

1. **Given** the corpus folder (extensionless FileDataID names), **When** the operator opens it, **Then** 699 tiles are listed by `ALOC` X/Y and none are placed by filename.
2. **Given** two `ALOC`-adjacent tiles, **When** they render as wireframe, **Then** their shared edge has no visible crack.
3. **Given** the inner 128×128 grid is not yet proven, **When** the wireframe renders, **Then** inner vertices come from the `AVTX` second block, and the result is itself a visual check on that hypothesis (a wrong inner order shows as spikes).

---

### User Story 1 - Know exactly what the new files contain (Priority: P1)

The operator points the inspection tooling at the newly available files and gets a truthful
account: which format version each file really is, every chunk present (including ones no
documentation mentions), and whether each chunk's size agrees with the documented layout.

**Why this priority**: Nothing else can be trusted until the corpus has been measured. The wiki is
self-described as incomplete, and the version is currently mislabeled. This story also produces
the evidence that decides the layouts every later story depends on.

**Independent Test**: Run the inventory against the real corpus. It reports a per-version file
count, a chunk-occurrence table, and a list of layout disagreements, with no crashes and no
unexplained bytes.

**Acceptance Scenarios**:

1. **Given** a folder of real AHDR-family files (extensionless, named by FileDataID), **When** the operator inventories it,
   **Then** each file is reported with its actual revision (26 in the current corpus, never mislabeled v23), and header
   dimensions/grid counts and `ALOC` tile coordinates are listed.
2. **Given** a file containing a chunk the documentation does not list, **When** it is inventoried,
   **Then** the unknown chunk is reported by name, size, occurrence count and nesting position, not
   skipped silently.
3. **Given** a chunk whose payload size disagrees with the documented layout, **When** it is
   inventoried, **Then** the disagreement is reported with expected versus observed size.
4. **Given** a truncated or malformed file, **When** it is inventoried, **Then** the file is flagged
   with the reason and the rest of the corpus continues.

---

### User Story 2 - Decode a tile into usable terrain data (Priority: P1)

The operator can decode any v22 tile into structured terrain data: heights, normals, texture
names and per-chunk texture layers with alpha, shadow maps, model names with object placements, and
per-chunk area ids. Each decoded value can be dumped for examination.

**Why this priority**: Decoding is the foundation for rendering and for any later research use of
the data. It is P1 alongside the inventory because rendering can't start without it.

**Independent Test**: Decode every tile in the corpus. Heights, normals and layer counts pass
internal-consistency checks (listed under Success Criteria), and a per-tile dump can be produced
for any tile.

**Acceptance Scenarios**:

1. **Given** a real v22 tile, **When** it is decoded, **Then** the outer 129x129 and inner 128x128
   height grids are produced in the correct order, and adjacent tiles agree along shared edges.
2. **Given** a chunk with texture layers, **When** it is decoded, **Then** each layer resolves to a
   texture name from the tile's texture table, and each alpha map decodes to a full-resolution mask
   whichever encoding (8-bit uncompressed or 4-bit) was used.
3. **Given** a chunk with object definitions, **When** it is decoded, **Then** each placement
   resolves to a model name from the tile's model table and yields a position, rotation, scale and
   unique id.
4. **Given** a v23 file, **When** it is decoded, **Then** the same reader handles it and
   additionally exposes flight bounds and whole-tile vertex shading.

---

### User Story 3 - See the terrain in the viewer (Priority: P2)

The operator opens a folder of v22 tiles in the viewer and flies over the terrain with its textures
blended, its shadows, and its doodads and world objects placed, the same way a standard map is
viewed.

**Why this priority**: This is the end goal. It depends on Stories 1–2 and can only be judged
visually by the operator.

**Independent Test**: Load the corpus in the viewer. Tiles appear at the correct world positions
with seamless edges, recognizable texture blending, and objects standing on the ground.

**Acceptance Scenarios**:

1. **Given** a folder of v22 tiles, **When** the operator opens it, **Then** every tile present is
   listed and loads at its grid position.
2. **Given** a loaded tile, **When** it is rendered, **Then** terrain height, lighting normals,
   texture layers, alpha blending and shadows all display.
3. **Given** a loaded tile with object definitions, **When** it is rendered, **Then** the referenced
   models are placed at their decoded positions, or shown as a clearly marked placeholder when the
   model asset is unavailable.
4. **Given** a tile that failed to decode, **When** the map is loaded, **Then** the failure is shown
   for that tile and the rest of the map still renders.

---

### User Story 4 - Existing formats keep working (Priority: P2)

Adding v22 support does not change how any existing map format is detected, read or rendered.

**Why this priority**: The viewer has a wide set of supported eras (0.5.3 through MoP); a
detection change at the `AHDR` branch must not leak into them.

**Independent Test**: The existing test suite passes unchanged, apart from the tests that encoded
"every `AHDR` file is v23", which get replaced by version-correct tests.

**Acceptance Scenarios**:

1. **Given** an existing Alpha, standard or split-ADT map, **When** it is loaded after this change,
   **Then** detection and rendering are identical to before.

### Edge Cases

- `AHDR` appears either first (wiki v22/v23) or immediately after `MVER` (observed revision 26): both must be detected.
- `AHDR.version` is not a known revision (22, 23, 26): report it as an unknown `AHDR` revision and still inventory it; do not guess.
- Files have no extension and FileDataID names: detection must be content-based, and nothing may be inferred from the name.
- The file has a `.error` suffix (as existing detection already recognizes): keep that distinction for both versions.
- `AHDR` vertex or chunk dimensions differ from 129/129/16/16: decode using the header values when they are self-consistent with the payload sizes, and flag them otherwise.
- `ACNK` payload is 0x40 bytes or smaller (no header per the wiki): treat it as an empty chunk and record it.
- Alpha map encoding can't come from WDT settings, because **no WDT exists** for these files: infer it from the payload size, and report a size that fits neither encoding.
- `ALYR` references a texture index beyond the `ATEX` table: flag it and render that layer with a placeholder.
- `ACDO` model id beyond the `ADOO` table: flag it and skip the placement.
- The `ACDO` trailing "name/doodadsets" field is undocumented in size: measure the actual record size across the corpus before fixing a layout.
- Normal-vector triples don't normalize (magnitude far from 1): report the rate; don't silently renormalize away evidence of a wrong component order.
- A file lacks `ALOC`, or two files claim the same `ALOC` tile: flag it, and do not place that tile silently. (In the current corpus the ACNK index fields are 0, so they are not a fallback.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Detection MUST recognize AHDR-family files whether `AHDR` is the first chunk or follows `MVER`, independent of filename or extension. It MUST record the revision from `AHDR.version` (22, 23, 26 known; others flagged as unknown) and preserve the existing `.error` distinction.
- **FR-002**: The system MUST provide a corpus inventory that reports, per file: detected version, header dimensions, and every top-level and nested chunk with its size, including chunks absent from the documentation.
- **FR-003**: The inventory MUST compare each known chunk's observed size against its documented layout and report every disagreement.
- **FR-004**: The system MUST decode `AVTX` into outer (129x129) and inner (128x128) height grids, and `ANRM` into normals in the same order, honoring header dimensions.
- **FR-005**: The system MUST decode `ATEX` and `ADOO` into ordered name tables.
- **FR-006**: The system MUST decode each `ACNK` header (chunk index, area id, low-detail texture map, and for v23 flags and holes) and its nested `ALYR`, `AMAP`, `ASHD` and `ACDO` sub-chunks.
- **FR-007**: The system MUST decode alpha maps in both 8-bit uncompressed and 4-bit encodings to a full-resolution mask, and record which encoding each map used and why it was chosen.
- **FR-008**: The system MUST decode v23-only `AFBO` flight bounds and `ACVT` vertex shading when present.
- **FR-009**: Every field interpretation that the wiki leaves undocumented or ambiguous MUST be settled by a recorded measurement on the real corpus, kept as an evidence note, before it is relied on for rendering.
- **FR-010**: Decoding MUST NOT throw on a malformed tile. It MUST return a result with explicit per-channel failure diagnostics.
- **FR-011**: The inspection tooling MUST be able to dump any decoded tile's contents (header, name tables, per-chunk layers/placements, height and normal statistics) in human-readable and machine-readable form.
- **FR-012**: The viewer MUST open a folder of AHDR-family tiles as a map **from the tile files alone** (no WDT, map table or listfile exists for them) and list the tiles present.
- **FR-013**: The viewer MUST render decoded terrain with heights, normals, texture layers, alpha blending and shadows, using the same terrain rendering path as other formats.
- **FR-014**: The viewer MUST place decoded object placements, resolving model names from the tile's model table and assets from the configured data source when available.
- **FR-015**: Existing map format detection, reading and rendering MUST be unchanged.
- **FR-016**: Writing or converting to v22/v23 is out of scope.
- **FR-017**: A fast-path wireframe (US0) MUST render using only measured layout facts, and MUST be labelled in the UI as provisional (no textures/objects; inner-grid order unproven).

### Key Entities

- **AHDR tile**: One terrain tile file. Carries the format version, vertex grid dimensions and chunk grid dimensions.
- **Height/normal field**: The whole-tile outer and inner vertex grids of heights and normals.
- **Name tables**: The per-tile ordered texture names and model names that layers and placements index into.
- **Terrain chunk**: One of the chunk-grid cells (16x16 expected), with its index, area id, low-detail texture map, and (v23) flags and holes.
- **Texture layer**: A texture reference plus flags within a chunk, optionally carrying an alpha map.
- **Object placement**: A model reference with position, rotation, scale, unique id and a trailing name/doodad-set field.
- **Corpus inventory**: The measured record of versions, chunk occurrences and layout disagreements across the real files. It is the evidence base for every decoding decision.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of files in the real corpus are detected as AHDR-family with the revision from their header (699/699 unique files are revision 26 today), and 0 files are mislabeled.
- **SC-002**: 100% of bytes in every real file are accounted for by a known or explicitly inventoried unknown chunk. There are no unexplained gaps or overruns.
- **SC-003**: At least 99% of real tiles decode with no channel failures; every failure has a recorded cause.
- **SC-004**: Heights of horizontally or vertically adjacent real tiles agree along shared edges (median absolute edge difference at or near zero). This proves the outer/inner grid order is correct.
- **SC-005**: At least 95% of decoded normals have magnitude within 5% of unit length, and normals agree in direction with normals derived from the decoded heights. This proves component order and scaling.
- **SC-006**: 100% of texture-layer and object-placement references resolve to an entry in their tile's name table, or are individually flagged.
- **SC-007**: The operator can open the corpus in the viewer and confirm by eye that terrain, textures and object placements appear coherent and seamless across tile boundaries.
- **SC-009**: The US0 wireframe shows the 30 non-flat corpus tiles as one continuous surface with no crack at any of the 71 `ALOC`-adjacent tile edges (35 + 36 measured pairs).
- **SC-008**: The existing test suite passes, with only the "every AHDR file is v23" assertions replaced.

## Assumptions

- **Corpus on hand (2026-09-16)**: 700 files (699 unique) in `test_data/v22_adts/unknown/`, passed on by Marlamin. First measurements are in [evidence/phase0-first-look-2026-09-16.md](evidence/phase0-first-look-2026-09-16.md).
- **Corpus location**: `wow-viewer/test_data/v22_adts/` (operator-decided 2026-09-16). This folder is already git-ignored (`wow-viewer/test_data/*`), so the files are never committed (Data Policy). Tooling still accepts any root as an argument.
- The files are loose files on local disk. **No WDT, map table entry or listfile names exist for them** (operator-confirmed). Container/archive extraction is not part of this spec.
- The corpus is primarily v22. v23 support comes through the same reader because the formats are near-identical, but v23 is only validated as far as real v23 files are available.
- The viewer's existing terrain chunk/tile representation can carry v22 data. Whole-tile heights and normals are sliced into per-chunk grids, and whole-tile names are mapped to the existing per-tile tables.
- Referenced textures and models resolve through the viewer's normal data-source configuration. Missing assets degrade to placeholders; they do not block rendering.
- Tile grid coordinates come from `ALOC` (measured, see evidence). Filenames are FileDataIDs with no positional meaning (operator-confirmed), and the map identity is unknown. `ALOC[0]` = 2869 is unexplained and must not be named without evidence from the files themselves.
- Liquid is not documented for v22 and is out of scope unless the corpus inventory reveals a liquid chunk, in which case it becomes a follow-up.
