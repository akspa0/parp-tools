# Feature Specification: Modern Liquid Directional Flow (WDT `MAI2` `liquidFlowTexture`)

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6 (follow-on to the v0.6 modern-data lane)

**Created**: 2026-09-18

**Status**: Draft (operator-directed 2026-09-18)

**Depends on**: [Spec 238](../238-casc-data-source/spec.md) (CASC id-addressed reads),
[Spec 239](../239-modern-client-assets/spec.md) (FileDataID-era WDT reader),
[Spec 240](../240-format-conformance/spec.md) (conformance survey),
[Spec 205](../205-mh2o-liquid-object-vertex-format/spec.md) (MH2O liquid vertex-format chain),
[Spec 243](../243-modern-to-legacy-map-conversion/spec.md) (legacy outputs this may feed)

**Input**: operator direction, 2026-09-18 — "we should also start trying to support the new water
system's directional flow … at least, to provide context to the liquids on these maps, in the
viewer's ui." Reference: [`wowdev.wiki/WDT`](https://wowdev.wiki/WDT).

## Context

The modern WDT's **`MAI2` chunk** (version ≥ `12.0.5.66330`, i.e. exactly the WoW: Forever
`wow_classic_beta` era) is a 64×64 table of 32-byte `MapFileDataIDs2` records. Its first field is:

```c
uint32_t liquidFlowTexture; // For WoW: Forever liquid flow map.
                            // R channel = +Y flows west, +G = -X flows south, 128 is 0 flow
```

The remaining seven fields are documented only as `unknown1..unknown7`. The v0.6.0-alpha release notes
already record the gap: *"The Azeroth WDT's `MAI2` chunk is not interpreted."* The viewer therefore
renders modern liquids with no awareness of the flow field the client itself uses.

The legacy side is not empty: the project's MCLQ parser already models a **flow vector** for animated
water (`MclqChunk` → `MclqFlowVector`), so a per-chunk dominant direction has a legacy home in the
Alpha target even though later `MH2O` has no flow field.

## User Scenarios & Testing

### User Story 1 — See liquid flow as context in the viewer (Priority: P1)

A user inspects a modern map's water and the viewer tells them which way the water flows there —
direction (and whether it is flowing at all) — instead of rendering a static plane.

**Why this priority**: it is the operator's stated minimum ("at least … provide context to the liquids
… in the viewer's ui").

**Independent Test**: hover or select a river tile on the tested modern map; the reported direction
matches the map's R/G values decoded per the documented encoding, and a still pool reports zero flow.

**Acceptance Scenarios**:

1. **Given** a tile whose flow texture marks a non-zero vector, **When** the user inspects a liquid
   cell in that tile, **Then** the viewer shows the decoded direction and magnitude.
2. **Given** a tile whose flow texels are all 128, **When** inspected, **Then** the viewer reports
   "no flow" rather than an arbitrary direction.
3. **Given** a map whose WDT has no `MAI2` chunk or whose `liquidFlowTexture` is 0, **When** the map
   loads, **Then** liquids render exactly as before and the absence of flow data is reported, not
   guessed.
4. **Given** the flow texture cannot be read (not on disk and CDN fill off), **When** the map loads,
   **Then** loading still succeeds and the unresolved flow reference is reported.

### User Story 2 — Flow is a first-class, reusable datum (Priority: P2)

The decoded flow is exposed as data (per tile, per liquid cell) so other consumers — the minimap
compositor, harvest, and the modern→legacy converter — can use it, not only the UI.

**Independent Test**: the same decoded vector is available to a non-UI consumer without duplicating
decode logic.

**Acceptance Scenarios**:

1. **Given** a loaded modern tile, **When** a consumer asks for flow at a cell, **Then** it receives
   the decoded vector from one shared source.

### User Story 3 — Legacy expression where a field exists, honest loss where it does not (Priority: P3)

When converting a modern map to a legacy target (Spec 243), flow is written where the target can hold
it (Alpha MCLQ's flow vector, as a per-chunk dominant direction) and explicitly reported as
unrepresentable where it cannot (LK `MH2O`).

**Independent Test**: a converted Alpha tile carries the dominant direction; the conversion report
lists flow as preserved (Alpha) or dropped (LK) rather than silently discarding it.

**Acceptance Scenarios**:

1. **Given** a modern tile with a consistent flow direction, **When** converted to Alpha, **Then** the
   emitted MCLQ carries that direction and the report says "preserved".
2. **Given** the same tile, **When** converted to LK `MH2O`, **Then** the report says flow was dropped
   with the reason (no MH2O field).

## Requirements

- **FR-001**: The WDT reader MUST locate and expose the `MAI2` chunk and its per-tile 32-byte records,
  including `liquidFlowTexture`, without treating the seven unknown fields as interpreted data.
- **FR-002**: `liquidFlowTexture` MUST be resolvable through the existing FileDataID/listfile/CASC
  path used by the other modern asset references.
- **FR-003**: Flow MUST be decoded per the documented encoding — `R` = +Y flows west, `G` = −X flows
  south, `128` = zero flow — into a normalized direction plus magnitude.
- **FR-004**: The viewer MUST surface flow as liquid context for the inspected liquid cell, and MUST
  report "no flow" for zero/absent data.
- **FR-005**: Absent, zero, or unreadable flow data MUST NOT change existing liquid rendering, and MUST
  NOT fail the map load.
- **FR-006**: The decoded flow MUST be exposed through one shared source that UI and non-UI consumers
  both use (no second decode path).
- **FR-007**: The seven `unknown` fields MUST be recorded as unknown; they MUST NOT be named or
  interpreted without evidence.
- **FR-008**: New logic MUST live in an owned service per AGENTS.md §10; a new UI surface MUST register
  an inventory row per AGENTS.md §11 / Spec 223 FR-9.
- **FR-009**: The feature MUST ship a receipt per AGENTS.md §9.2 with real-map evidence and an operator
  visual witness for the UI claim.
- **FR-010**: Conversion integration (Spec 243) MUST report flow as preserved or dropped per target,
  never silently.

## Key Entities

- **MAI2 record**: one per ADT tile; holds `liquidFlowTexture` plus seven unknown fields.
- **Flow texture**: the FileDataID-referenced image whose R/G channels encode direction.
- **Flow vector**: normalized direction plus magnitude decoded from a flow texel.
- **Flow datum**: the per-tile/per-cell published flow used by UI and non-UI consumers.

## Success Criteria

- **SC-001**: On the tested modern map, the viewer reports a flow direction for a flowing liquid area
  and "no flow" for a still area, matching the source data.
- **SC-002**: A map with no `MAI2`, a zero `liquidFlowTexture`, or an unreadable flow texture loads and
  renders identically to the current build.
- **SC-003**: The conversion report (Spec 243) states flow disposition per target: preserved (Alpha
  MCLQ flow vector) or dropped with reason (LK MH2O).
- **SC-004**: A second consumer can obtain the same vector without a duplicate decoder.
- **SC-005**: Build and focused tests pass with no new failures.

## Assumptions & open questions

- The documented encoding is treated as authoritative; magnitude scaling is assumed from the same
  note (128 = zero) and is to be confirmed against the real texture during planning.
- What the seven `unknown` `MAI2` fields carry is an open question for the completeness survey
  ([Spec 245](../245-modern-chunk-completeness-survey/spec.md)) and is out of scope here.
- Alpha MCLQ's flow vector is assumed to be a single direction per liquid instance, so a tile with
  varying flow must be reduced to a dominant direction with the reduction reported.
- Flow-aware *rendering* (animated/rippled water) is out of scope; this spec delivers data and context.