# Superseded First Pass: WC2-First Warcraft Viewer Support

**Superseded 2026-10-14 by operator correction**: WC3 assets/tilesets in the File Browser are the priority, not WC2 or world rendering. This file preserves the earlier specification as history only; [the current specification](spec.md) is authoritative. Its original acceptance criteria are not current implementation tasks.

**Feature Branch**: 272-wc2-wc3-viewer-support

**Created**: 2026-10-14

**Status**: Specified / future backlog; ready for bounded implementation planning, not implementation approval

**Input**: Operator: “use speckit, I'd like to get some of this stuff working in the viewer, at some point. If we get WC2 support, then WC1 support becomes easy.” The operator supplied the [WC2/WC3 draft](../../docs/wc2-3-support/gemini-code-1791995880062.md), local WC2 data, and local PlayStation source for reference. The operator subsequently supplied the [integration, coordinate, and restoration addendum](operator-addendum.md) and clarified: “my wc2 cd is scratched, but even that is the original DOS version, I never had the battle.net version!”

**Scope authority**: The request, supplied draft, addendum, and DOS correction above. The draft is a requirements/research input, **not a verified binary format specification**. [Discovery evidence](evidence/discovery-2026-10-14.md) records observed contradictions and remaining proof gaps. No release commitment or runtime support is claimed.

**Operator lineage clarification (2026-10-14)**: The operator identifies WC2 as an expanded WC1 built on Blizzard's Game Development System (GDS), with WC1 behavior retained in the WC2 source. WC1 is therefore a shared-foundation reuse follow-on, not a separate-engine proposal. This source-lineage statement is operator evidence; exact WC1 data variants still need real samples before claiming compatibility.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Inspect and View Original DOS WC2 Terrain (Priority: P1)

As a researcher, I want to select my existing original DOS WC2 data and a map in the viewer, inspect the map's metadata, and view its terrain without needing Battle.net Edition, converting it to WoW terrain, or changing my original files.

**Why this priority**: This is the operator's available data and the smallest useful viewer outcome. It establishes archive, palette, and tile behavior before sprites or other games are added.

**Independent Test**: Select the supplied DOS data, inspect the eight loose maps, and preview one real map for each of the three observed terrain-era identifiers. The operator compares orientation, palette, coastlines, and tile boundaries with an original-game or independently trusted reference.

**Acceptance Scenarios**:

1. **Given** the supplied DOS data, **When** the user selects a map, **Then** the viewer identifies the source as DOS WC2 and displays its title, dimensions, terrain era, and source identity without requiring an MPQ archive or listfile.
2. **Given** a supported map and its verified terrain resources, **When** the user opens its terrain preview, **Then** a top-down view supports pan/zoom and resolves the complete map with the correct palette, orientation, and tile variations.
3. **Given** missing, damaged, truncated, or unsupported resources, **When** the user opens a map, **Then** the viewer identifies the failing map/resource and reason, does not fabricate a valid-looking terrain result, and remains usable.

---

### User Story 2 - Inspect WC2 Sprite Assets (Priority: P2)

As a researcher, I want to inspect WC2 unit/building/doodad sprite frames with the correct palette, transparency, frame offsets, and team-color interpretation.

**Why this priority**: Sprite inspection extends the archive/palette foundation without requiring gameplay simulation or making a complete unit-rendered map a condition of terrain support.

**Independent Test**: Open one real unit sprite and one building or doodad sprite without opening a map; inspect their frames against a trusted reference.

**Acceptance Scenarios**:

1. **Given** a supported sprite resource, **When** a frame is selected, **Then** its extent, placement within the frame canvas, and transparent pixels match the reference.
2. **Given** supported player-color entries, **When** the user changes the preview team color, **Then** only the verified team-color pixels change; terrain palette entries are not globally reinterpreted as transparency or team colors.

---

### User Story 3 - View Classic WC3 Assets and Maps Later (Priority: P2)

As a researcher, I want a later viewer milestone to inspect Reign of Chaos and The Frozen Throne textures/models and view their terrain, cliffs/ramps, water, and doodad placements from my own data.

**Why this priority**: It is requested by the supplied draft and addendum, but must not block the WC2-first outcome. Beta-era assets are a separate compatibility gate within this later work.

**Independent Test**: Use one identified real classic WC3 map plus representative terrain textures and a model; compare the map and assets with an independent reference without any WoW conversion step.

**Acceptance Scenarios**:

1. **Given** supported classic WC3 data, **When** a map is opened, **Then** its terrain layers, variations, cliff/ramp transitions, water heights, and doodad placements match that map's reference rather than a guessed WoW interpretation.
2. **Given** representative JPEG and palettized textures and a classic model, **When** previewed, **Then** color channels, alpha, texture replacement, geometry, and supported animation behavior are checked against real assets.
3. **Given** an unverified beta variant or missing companion data, **When** opened, **Then** it is explicitly identified as unsupported/unverified rather than accepted because it shares a signature with a supported format.

---

### User Story 4 - Compare WC3 Texture Ancestry (Priority: P3)

As a restoration researcher, I want to compare WC3 interior tile variations with WoW 0.5.3 texture quadrants and whole-tile reductions, inspect ranked matches, and retain an auditable report.

**Why this priority**: It tests the proposed art lineage without treating visual similarity or the draft's historical narrative as established provenance.

**Independent Test**: Compare an identified real WC3 terrain atlas and WoW 0.5.3 texture; inspect candidates, all four cardinal rotations, scores, and the ranked report independently of map conversion.

**Acceptance Scenarios**:

1. **Given** supported square or double-width terrain atlases, **When** candidates are extracted, **Then** the report identifies each source asset, interior variation, crop, source dimensions, and normalization used; masked transition cells are not mislabeled as solid tiles.
2. **Given** a WoW target, **When** matching runs, **Then** all four quadrants and the whole-tile reduction are compared against candidates at four cardinal rotations, with reproducible ranking and source identities in JSON and Markdown reports.
3. **Given** a constant-color candidate or ambiguous similarity, **When** scored, **Then** the result is explicitly unscorable or qualified; no score alone is presented as proof that Blizzard reused that asset.

---

### User Story 5 - Preview Restored Tile Candidates (Priority: P3)

As a researcher, I want to synthesize 256×256 candidates from WC3 interior variations, compare their repeated appearance in the viewer, and export them without overwriting my data.

**Why this priority**: This makes the draft's restoration experiment usable after source decoding and matching are trustworthy.

**Independent Test**: Synthesize one four-variation quilt and one wrap-aware single-variation upscale from real assets; inspect repeating boundaries and the exported texture through its full mip chain.

**Acceptance Scenarios**:

1. **Given** identified interior variations, **When** the user selects quilting, wrap-aware upscaling, or the draft's nearest-neighbor-plus-smoothing comparison, **Then** the viewer labels the chosen mode and shows its 256×256 output and repeated preview.
2. **Given** a synthesized candidate, **When** exported, **Then** all nine mip levels are readable by an independent supported consumer, with the selected terrain color/alpha interpretation retained.
3. **Given** an existing WoW scene, **When** the user opts into a candidate texture preview, **Then** only that temporary experiment changes; removing it returns the scene to its original state, and no source asset or map is rewritten.

---

### User Story 6 - Bridge Coordinates and Terrain Explicitly (Priority: P3)

As a researcher, I want to distinguish WC3, WoWEdit authoring, screenshot HUD, and compiled WoW coordinate conventions before jumping a camera or exporting a terrain comparison.

**Why this priority**: The addendum requests both a screenshot-HUD camera helper and WC3-to-WoW terrain conversion; its physical-unit claims must not silently become shared camera or terrain behavior.

**Independent Test**: Apply an explicitly identified coordinate profile to the two supplied screenshot examples and to independently measured terrain anchors; inspect a separate WC3-derived WoW terrain output without modifying an existing map.

**Acceptance Scenarios**:

1. **Given** a known HUD convention, validated transform, and known local origin, **When** coordinates are entered, **Then** the helper shows the interpreted horizontal/vertical axes, units, origin, and resulting viewer position before an explicit camera jump.
2. **Given** an unknown origin, unverified scale, or ambiguous HUD convention, **When** conversion is requested, **Then** the tool identifies the missing evidence and does not present a guessed camera position as verified.
3. **Given** a verified WC3 map and an explicitly selected terrain-conversion profile, **When** converted for comparison, **Then** the separate output reports its scale, axes, origin, resampling, and unsupported features, and preserves the chosen reference anchors without changing the normal WoW loading path.

### Edge Cases

- Scratched-disc provenance does not prove every copied byte is valid. Unreadable archives, out-of-bounds entries, truncated compressed streams, and excessive declared sizes fail with bounded diagnostics.
- Resource indices/names, palette precision, tile-reference bitfields, and sprite transparency may differ by game/platform/build. No silent alternate tile-index formula or “looks plausible” fallback is acceptable.
- Unknown map sections can be reported/skipped within valid bounds; missing required sections, impossible dimensions, and contradictory terrain selections must be diagnosed.
- The eight inspected loose maps cover era identifiers 0, 1, and 2, not every expansion tileset. Source-code references to another era do not supply a real test map.
- Atlas dimensions alone do not prove a particular layout. External beta mip files, channel-order differences, replacement textures, and variable placement records require variant-specific evidence.
- Matching must handle zero variance, ties, and transformed candidates deterministically. Quilting/edge blending changes pixels and must retain source provenance and settings.
- Wrap-aware filtering does not make arbitrary source art seamless. Inner quilt seams, outer repeats, and lower mip levels require visual checks.
- A height divisor does not establish a universal horizontal scale, axis mapping, screenshot origin, or WC3-to-WoW unit identity.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to select their own original DOS WC2 data and loose maps read-only. Battle.net Edition MUST NOT be required for the initial milestone.
- **FR-002**: The viewer MUST identify archive entries and diagnose invalid bounds, unsupported compression/variants, and missing resources without unbounded work or fabricated output.
- **FR-003**: Users MUST be able to inspect WC2 map identity, version, description, dimensions, terrain selection, and placement records; supported terrain preview MUST resolve the full map from verified source resources.
- **FR-004**: WC2 terrain preview MUST provide a top-down pan/zoom view with verified palette, tile orientation/variation, and map orientation, without pretending the map is WoW heightfield terrain.
- **FR-005**: Users MUST be able to inspect supported WC2 sprite frames with verified frame extents, offsets, transparency, and player-color behavior.
- **FR-006**: Compatibility findings MUST distinguish original DOS data, expansion data, and PlayStation references; source-reference claims MUST identify their source location and verification status. WC1 follow-on planning MUST examine reuse of the operator-identified shared GDS/WC1 foundation rather than assume a separate engine; actual WC1 data compatibility MUST still have WC1 evidence.
- **FR-007**: New viewing and restoration behavior MUST be explicit opt-in behavior with one authoritative UI home and MUST preserve existing WoW viewing, readers, camera behavior, and original data.
- **FR-008**: Later classic WC3 support MUST provide verified texture/model previews and terrain-map viewing including terrain transitions, variations, cliffs/ramps, water, and doodad placements.
- **FR-009**: Beta-era WC3 compatibility MUST be separately evidence-gated; shared signatures or historical similarity MUST NOT substitute for real variant validation.
- **FR-010**: Coordinate/camera and terrain-conversion tools MUST require an explicit convention, scale, axes, and origin; show unsupported/unverified assumptions; and write only separate comparison outputs, never overwrite source maps.
- **FR-011**: Texture candidate extraction MUST distinguish solid interior variations from masked transitions and retain source crop, variation, dimensions, and normalization provenance.
- **FR-012**: Texture ancestry comparison MUST rank the four target quadrants and whole-tile reduction against all four cardinal candidate rotations, emit auditable JSON/Markdown reports, and distinguish similarity from historical proof.
- **FR-013**: Users MUST be able to compare the draft's three synthesis modes, including the addendum's four-variation quilt and wrap-aware single-variation upscale, and inspect inner/outer repeat boundaries.
- **FR-014**: Synthesized 256×256 terrain exports MUST contain all nine mip levels and preserve the selected color/alpha semantics; viewer swapping MUST be temporary, reversible, and scoped to the experiment.
- **FR-015**: Each claimed compatibility outcome MUST have identified real input files and matching behavioral evidence. Builds and synthetic tests alone MUST NOT close visual, data-compatibility, camera-placement, or independent-consumer criteria.

### Key Entities *(include if feature involves data)*

- **Source profile**: Game, edition/platform, supplied root, fingerprint, and verified compatibility level.
- **Indexed resource**: Archive identity, entry identity, bounds, compression status, and verified role; not necessarily a filename.
- **Tileset and sprite**: Palette and tile definitions, or frame extents/offsets with transparency/player-color semantics.
- **Map**: Identity, version, terrain selection, grid dimensions, tile references, placement records, and unsupported sections.
- **Texture candidate and match**: Source asset/crop/variation, normalization/rotation, target region, score validity, synthesis settings, and derived output identity.
- **Coordinate/conversion profile**: Source and target conventions, units/axes/origin, reference anchors, resampling policy, and known unsupported features.
- **Compatibility evidence**: Identified inputs, source references, observed outputs, verification owner, and unresolved differences.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All eight supplied loose DOS WC2 maps yield correct inspected metadata; at least one map per observed terrain-era identifier (three total) has a complete terrain preview accepted against a trusted reference, with zero unexplained missing tiles.
- **SC-002**: At least one real WC2 unit sprite and one building/doodad sprite match trusted frame, palette, offset, and transparency references; selected team-color changes affect only verified player-color pixels.
- **SC-003**: Every case in the planned damaged/unsupported-input verification set produces an actionable diagnostic and leaves the viewer usable; source fingerprints are unchanged before/after all preview operations.
- **SC-004**: A real, identified classic WC3 map demonstrates terrain transitions, variation, cliffs/ramps, water, and doodad placements; representative JPEG/palettized textures and a classic model match independently checked references. Beta compatibility remains separately labeled until proven.
- **SC-005**: For an identified real texture pair, the comparison report accounts for all five target views and all four cardinal rotations per candidate, reproduces ranking on repeat, and records unscorable cases without false certainty.
- **SC-006**: Real-source quilting and single-variation upscale each produce a 256×256 candidate with nine independently readable mip levels; operator inspection accepts inner/outer repeats, and removing a temporary swap restores the original scene.
- **SC-007**: The coordinate helper identifies both supplied HUD conventions and either matches independently verified anchors under an explicit profile or blocks an unverified jump; a separate terrain-conversion example records its transform and limitations and agrees with its chosen anchors.
- **SC-008**: Existing WoW asset/map viewing remains functional after entering/exiting the new surfaces; no new claim of WC1, Battle.net, PlayStation-runtime, or beta support appears without separate real-data evidence.

## Assumptions

- This session specifies future work only. The next Spec Kit step is a bounded technical plan for DOS WC2 discovery/terrain preview, followed by receipt-gated tasks and one validated implementation phase at a time. No implementation plan or task-completion claim is embedded here.
- The operator's original DOS correction supersedes the draft's dual DOS/Battle.net baseline. Battle.net support is not part of initial acceptance and is not scheduled by this spec.
- PlayStation source is a read-only format/behavior reference, not code to import, build, distribute, or treat as binary-identical to DOS resources. PlayStation runtime support is out of scope.
- WC1 follows the operator-identified shared GDS/WC1 foundation in the WC2 source. Subsequent planning should trace retained engine behavior and reuse the WC2 foundation, not start a separate engine. Shared lineage is not by itself a test of every archive, compression, map, or sprite variant; real WC1 inputs are still needed. This session does not implement either game.
- WC3 viewing, ancestry comparison, texture restoration, coordinate helpers, and terrain bridging are operator-originated later scope, not prerequisites for the WC2 milestone or additions to the current release.
- The addendum's requested model/namespace alignment and algorithms are retained as [technical planning inputs](operator-addendum.md), not silently verified by inclusion in this user-facing specification.
- The operator provides local data and owns real-client visual/camera/consumer proof and any long-running comparisons or captures. Agents may perform bounded read-only probes and lightweight tests.
- Dataset roots remain runtime configuration. Operator-local roots/fingerprints belong only in scoped [discovery evidence](evidence/discovery-2026-10-14.md), never as application defaults or portable test dependencies.
- Existing shared-reader and god-class freezes, core-library ownership, shared UI/inventory rules, write containment, and receipts remain binding. No protected reader change is authorized by a draft disagreement.
- Gameplay, AI/pathfinding simulation, audio/video playback, automatic historical-authenticity claims, disc recovery, and overwriting original assets are outside this viewing/restoration request.
