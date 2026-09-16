# Feature Specification: ADT v26 — First Reader and Renderer for a Brand-New Terrain Format

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-16 (retitled the same day from "ADT/v22" once the files were measured)

**Status**: Draft (fast-path wireframe next)

**Input**: User description (original, before the files were measured): "Add support for reading and rendering the data from ADT/v22 files, because data just became available in that format for no apparent reason, and I have no tooling to read or render them, currently, but I really ought to. https://wowdev.wiki/ADT/v22"

## Discovery record

These tiles are **DAT version 26** (referred to as "ADT v26" in repo paths and code names): a completely new version of the
DAT raw terrain project files, the source files the client's ADTs are built from. It had never been seen before 2026-09-16.
The operator proposed years earlier that these loose files were the raw ADT project files; the wiki's ADT/v22 page now
describes the family as DAT files as well.

| When (2026-09-16) | Event |
|---|---|
| ~8 hours before this record | The format first appears publicly, shipped in the `wow_classic_beta` build: the first WoW: Forever build on Battle.net servers, and the first build of the final WoW remaster on a **new WoW engine** |
| ~1 hour before this record | Tip-off: Marlamin glanced at the files, thought they were "v22", and passed them to the operator as the right person to look at them |
| This session | **First analysis anywhere**, by the operator with Claude. The files are measured, the format is identified as version 26, tile placement is decoded from the new `ALOC` chunk and proven by exact height seams, and a wireframe viewer path is planned in this repo's tooling ([evidence](evidence/phase0-first-look-2026-09-16.md)) |

**Why it matters**: nothing documents this format. No wiki page, no WDT, no map table entry, no listfile names and no CDN
companions exist for these files. **The tile files are the only source of truth**, and this project's viewer is the tool
positioned to visualize them first. Every fact in this spec is either measured from the files (with the evidence linked)
or explicitly marked as open. The permanent format write-up is [`docs/architecture/adt-v26-format.md`](../../docs/architecture/adt-v26-format.md).

This spec is **standalone**: it does not depend on Specs 238/239.

## Context

**What v26 is (measured)**: every file is `MVER` 26 then `AHDR` 26, followed by `ALOC AOCH AVTX ANRM [ATEX…] ADOO… ACNK×256 ACVT`.
It shares chunk names with the pre-Cataclysm experimental **ADT v22/v23** documented on wowdev.wiki (`AHDR`, `AVTX`, `ANRM`,
`ATEX`, `ADOO`, `ACNK` with nested `ALYR`/`AMAP`/`ASHD`/`ACDO`, `ACVT`). That family resemblance is why it was first taken for v22.
It differs in ways that make it a new format:

- `MVER` precedes `AHDR` (v22/v23 start with `AHDR`), and the version is **26**.
- New chunks `ALOC` (tile location, measured), `AOCH` (all zero so far) and `ADST` (unexplained).
- `ACVT` is in every file (the wiki calls it v23-only); `ATEX`/`ADOO` hold one name per chunk.
- Filenames are bare FileDataIDs that encode no position. Tile X/Y come from `ALOC[1]`/`ALOC[2]`, proven by exact height seams between neighbouring tiles.

**The toolchain before this spec** could do nothing with these files:

- File detection only recognizes `AHDR` as the *first* chunk, so v26 files (MVER first) are never recognized, and every AHDR-first file is labelled `AdtV23` regardless of version.
- The only reader reads the `AHDR` header and counts chunks. No chunk payload is decoded.
- Every existing test fixture is a synthetic buffer; no real AHDR-family file had ever been parsed.
- There is no terrain adapter, so the viewer can't display the terrain at all.

**How the wiki is used**: the v22/v23 pages are the nearest relatives, so their layouts are **starting hypotheses only**.
Per the project's standing rule that a named field is unexamined until measured, nothing from them is trusted for v26 until
the corpus confirms it. Wiki v22/v23 detection is kept (it is cheap and correct), but **v26 is the target** of every
decode and render requirement.

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

The operator can decode any v26 tile into structured terrain data: heights, normals, texture
names and per-chunk texture layers with alpha, shadow maps, model names with object placements, and
per-chunk area ids. Each decoded value can be dumped for examination.

**Why this priority**: Decoding is the foundation for rendering and for any later research use of
the data. It is P1 alongside the inventory because rendering can't start without it.

**Independent Test**: Decode every tile in the corpus. Heights, normals and layer counts pass
internal-consistency checks (listed under Success Criteria), and a per-tile dump can be produced
for any tile.

**Acceptance Scenarios**:

1. **Given** a real v26 tile, **When** it is decoded, **Then** the outer 129x129 and inner 128x128
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

The operator opens a folder of v26 tiles in the viewer and flies over the terrain with its textures
blended, its shadows, and its doodads and world objects placed, the same way a standard map is
viewed.

**Why this priority**: This is the end goal. It depends on Stories 1–2 and can only be judged
visually by the operator.

**Independent Test**: Load the corpus in the viewer. Tiles appear at the correct world positions
with seamless edges, recognizable texture blending, and objects standing on the ground.

**Acceptance Scenarios**:

1. **Given** a folder of v26 tiles, **When** the operator opens it, **Then** every tile present is
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

Adding v26 support does not change how any existing map format is detected, read or rendered.

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
- A file lacks `ALOC`, or two files claim the same `ALOC` tile: flag it, and do not place that tile silently. (ACNK index fields are chunk-local 0–15, so they cannot stand in for tile position.)

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
- **FR-014**: The viewer MUST place decoded object placements using names from the tile's own model table. Where no model can be loaded, it MUST draw a marker at the placement (bounding marker or point) so placement correctness is visible without any external assets.
- **FR-015**: Existing map format detection, reading and rendering MUST be unchanged.
- **FR-016**: Writing or converting to v26 (or v22/v23) is out of scope.
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
- **Corpus location**: `wow-viewer/test_data/v22_adts/` (operator-decided 2026-09-16; the folder name predates identifying the files as v26 and is kept as-is). This folder is already git-ignored (`wow-viewer/test_data/*`), so the files are never committed (Data Policy). Tooling still accepts any root as an argument.
- The files are loose files on local disk. **No WDT, map table entry or listfile names exist for them** (operator-confirmed). Container/archive extraction is not part of this spec.
- The corpus is entirely v26 (699 unique files). Wiki v22/v23 files are only detected; their decode paths are tested synthetically and validated only if real v22/v23 files ever appear.
- The viewer's existing terrain chunk/tile representation can carry v26 data. Whole-tile heights and normals are sliced into per-chunk grids, and whole-tile names are mapped to the existing per-tile tables.
- Referenced textures and models resolve through the viewer's normal data-source configuration. Missing assets degrade to placeholders; they do not block rendering.
- Tile grid coordinates come from `ALOC` (measured, see evidence). Filenames are FileDataIDs with no positional meaning (operator-confirmed), and the map identity is unknown. `ALOC[0]` = 2869 is unexplained and must not be named without evidence from the files themselves.
- No liquid chunk has been seen in v26 so far; liquid is out of scope unless the corpus inventory reveals a liquid chunk, in which case it becomes a follow-up.
