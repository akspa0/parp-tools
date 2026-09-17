# Feature Specification: Format Conformance Pass (WMO, ADT, M2, BLP, WDT)

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-17

**Status**: Draft

**Depends on**: [Spec 238 CASC Data Source](../238-casc-data-source/spec.md), [Spec 239 Modern Client Assets](../239-modern-client-assets/spec.md)

**Input**: "take a look at the wowdev.wiki's pages on all the stuff we are trying to read – WMO's are most obviously still broken" · "do the same pass, and write things down in a speckit plan so we can improve" · WoW.tools local (WTL) as a C# codebase to learn from.

## Context

Spec 239 made FileDataID-era clients load: maps, M2, WMO and DB2 from `wow_classic_beta` 1.60.1 render.
Loading is not the same as reading correctly. A comparison of the wowdev.wiki pages against the readers,
checked against real data, found defects that looked like rendering bugs but were reader gaps:

- **WMO batches addressed the wrong material** for 64% of batches (16-bit `material_id_large` ignored).
- **WMO LOD groups were drawn over full-detail groups** (GFID holds one group list per LOD).
- **Shader-23 WMO materials had no texture** (texture_1 empty or a placeholder).
- **Terrain drew at most 4 layers** where Azeroth uses up to 8.

Those four are fixed (commits d9fcbca1..251026d7). [research.md](research.md) lists what remains, per
format, each row marked MEASURED or CODE. This spec turns that audit into a repeatable conformance loop so
the next gap is found by a survey, not by a screenshot.

## User Scenarios & Testing

### User Story 1 — Conformance survey per format (Priority: P1)

A developer runs one command against a CASC product and gets, per format, which chunks and fields the
data uses and whether the reader handles, skips or misreads each one, with counts.

**Why this priority**: every fix above started as a count (`wmo-survey`). A survey ranks the backlog by
how much real data each gap touches and proves a fix changed the numbers.

**Independent Test**: run the survey on `wow_classic_beta` 1.60.1; the WMO section reproduces the counts
in research.md R2 (393,397 batches, 252,161 with flag 0x2) and every format lists unknown chunk ids with counts.

**Acceptance Scenarios**:
1. **Given** a CASC product, **When** the survey runs, **Then** it reports per format: files read, chunk ids seen (known / skipped / unknown) with file counts, and field-level tallies for the fields listed in research.md.
2. **Given** a fix to a reader, **When** the survey is rerun, **Then** the affected tally changes and the report states the before/after numbers.

### User Story 2 — WMO materials render as authored (Priority: P1)

Modern WMOs (Dalaran, Orgrimmar, remastered dungeons) show the right textures on each surface, including
two-layer materials blended by vertex colour alpha.

**Why this priority**: WMOs are the most visibly wrong asset after the fixes above.

**Independent Test**: side-by-side captures of `11DL_Dalaran`, `Orgrimmar2FrontGate` and `GoldshireInn`
against in-game or WTL/WebWowViewerCpp references; per-shader material counts from US1 show every shader id
with a render path or an explicit fallback.

**Acceptance Scenarios**:
1. **Given** a material with shader 6/13/21/23 (two-layer families), **When** rendered, **Then** texture_2 (or the documented second texture) blends over the base using MOCV alpha.
2. **Given** a group with TVERTS2/CVERTS2, **When** rendered, **Then** the second UV/colour set is available to the material.
3. **Given** a split-group WMO (≥9.2), **When** visibility runs, **Then** child groups are visible through their parent's portals.

### User Story 3 — Terrain reads every modern field (Priority: P2)

Terrain honours high-resolution holes, height-based texture blending (`MTXP`/`MHID`, MPHD 0x80),
texture scale, and layer animation.

**Independent Test**: survey counts chunks with `high_res_holes`; a tile known to use them shows holes
where the 64-bit mask says; a height-blended tile matches the minimap more closely than linear alpha
(scored with the existing synthetic-minimap scorecard).

### User Story 4 — Modern M2 animation and LOD (Priority: P2)

Chunked M2s load skeletons (`SKID` → `.skel`), bones (`BFID`), external animations (`AFID` → `.anim`),
and LOD skins (`LDV1`), so creatures animate and distant models use lower skins.

**Independent Test**: a creature model with `SKID` animates its stand sequence; survey shows AFID/SKID/BFID
references resolved vs. missing.

### User Story 5 — BLP and WDT companions (Priority: P3)

BLP decode covers every pixel format present in the build (including BC5 if present); WDT companion files
(`_lgt`, `_occ`, `_fogs`, `_mpv`) and `_lod.adt` are at least read and surveyed, with rendering decided per
file after measurement.

### Edge Cases

- A wiki claim contradicts measured data (e.g. MCMT "4 layers only" vs. 8-layer chunks in 1.60.1; MGI2 `lodIndex` values that do not fit the documented struct): the data wins; the contradiction is recorded in research.md and, when confirmed, proposed back to the wiki.
- A reference implementation has no license (WoWFormatLib): behaviour may be studied and cited; code is not copied.
- A field is present but its meaning is unknown (MOMX, 16 bytes per material): the survey records presence and value distributions; nothing renders from a guess.

## Requirements

### Functional Requirements

- **FR-001**: The conformance survey MUST run against any CASC product through the existing data source, local data first, CDN fill optional, and MUST NOT require the viewer.
- **FR-002**: For each format in scope (WMO root/group, ADT root/tex0/obj0, M2 chunked, BLP, WDT), the survey MUST report chunk-id inventory (known / skipped / unknown, with file counts) and the field tallies named in research.md.
- **FR-003**: Every reader change under this spec MUST cite the survey counts before and after in its evidence file.
- **FR-004**: Reader fixes MUST keep pre-6.0 behaviour unchanged, verified by the existing test suite and the Spec 239 baseline captures.
- **FR-005**: WMO rendering MUST select textures and blending per shader id; shader ids without a render path MUST fall back explicitly (logged once per shader id), never silently to texture_1.
- **FR-006**: The WMO converter MUST keep 16-bit material ids end to end and MUST NOT drop MPY2 materials above 0xFE.
- **FR-007**: ADT reading MUST honour MCNK flag `high_res_holes`.
- **FR-008**: Chunked M2 loading MUST resolve SKID, BFID and AFID file ids.
- **FR-009**: BLP decoding MUST report (not silently mis-decode) pixel formats it does not support.
- **FR-010**: research.md MUST be kept current: each row MEASURED or CODE, with the command or file.
- **FR-011**: Studying WTL/WoWFormatLib MUST record which behaviour was compared and the outcome; no code is copied from repositories without a license.

### Key Entities

- **Conformance report**: per product and format: chunk inventory, field tallies, reader outcome counts, build identity.
- **Audit row**: format, item, wiki claim, state (supported / partial / missing / fixed), evidence (MEASURED command or CODE path).

## Success Criteria

- **SC-001**: The survey reproduces the research.md WMO counts exactly on `wow_classic_beta` 1.60.1.69876.
- **SC-002**: Zero WMO materials in the survey fall into "no render path" without an explicit, logged fallback.
- **SC-003**: `11DL_Dalaran` and `Orgrimmar2FrontGate` captures show distinct textures per surface matching the reference captures (operator witness).
- **SC-004**: Every research.md row marked Missing has either a task in tasks.md or a recorded reason for deferral.
- **SC-005**: A creature M2 with SKID plays its stand animation.

## Assumptions

- `wow_classic_beta` 1.60.1.69876 is the measurement build; other products (retail, classic era) are added through the Spec 238 version picker as they are installed.
- WTL's model viewer is WebWowViewerCpp (C++/Emscripten), so its C# code informs data access and file linking more than rendering.
- The wiki is the starting inventory, not the authority.

## Out of Scope

- Writing modern formats.
- DB2 schema work beyond what WoWDBDefs already covers.
- PBR/waterfall (WFV) and light-cookie (TEXL) rendering until surveys show where they matter.
