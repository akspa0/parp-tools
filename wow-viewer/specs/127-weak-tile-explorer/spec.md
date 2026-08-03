# Feature Specification: Weak-Signal & White-Plate Tile Explorer

**Feature Branch**: `127-weak-tile-explorer`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "the viewer should also do all this stuff and allow viewing of the impartial data, since it's intended as an explorer of very old data that no one else has overturned as many rocks as I have, to find and expose it... the weak signal amplifier is literally a feature in our viewer that works, but we need to make it smarter, have it analyze the tiles that aren't 'weak' but are adjacent, and scale the terrain up to around the adjacent tiles' heights."

## Context

The 0.5.3 alpha corpus contains 1756 tiles across Azeroth, Kalimdor, Kalidar and PVPZone02. 361 of them
(68 weak-signal, 293 white-plate) are excluded from every downstream consumer because their terrain relief
is negligible or absent. "Excluded from training" has silently doubled as "never looked at".

Offline measurement (2026-08-03) established that this exclusion is throwing away real data:

- **205 of the 361 carry non-zero relief.** Only 156 are bit-exact flat.
- The relief in the extreme cases is real geometry, not noise. `Azeroth_26_34` sits at world Z −501.5 with
  a total height range of 0.000519 world units, and the feature visible in it **continues across the tile
  boundary** into its neighbours — per-tile noise cannot do that.
- Normals carry relief the heightmap lost. `Kalimdor_33_12` has only 1.39 units of height range but 24% of
  its MCNR vertices are genuinely tilted.

The viewer already contains the machinery to surface this and does not use it. `WeakSignalDetector`
classifies and amplifies weak tiles, and `EstimateFactorFromRanges` accepts exactly the neighbour-derived
reference range needed to scale a weak tile toward its surroundings — but that function has **zero callers**.
The auto-factor path consults only a coarse WDL guide tile, and the manual path is a slider seeded near the
constant era compression factor. The smarter behaviour is written and unwired.

This feature is about historical preservation and exploration, not model accuracy. These tiles are artifacts
of how the world was built, and there is currently no record of them in any capacity.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Inspect a degenerate tile and see what is actually in it (Priority: P1)

An explorer selects a tile the viewer currently renders as featureless. The viewer shows what that tile
still contains: its relief stretched against its own extremes so sub-millimetre structure fills the display,
a raking-light shaded view that reveals gradient structure, its normals with weak tilt made visible, and the
tile's true height range at full precision. The explorer can tell at a glance whether the tile is genuinely
empty or merely quiet.

**Why this priority**: This is the entire point of the request — there is no record of these tiles anywhere.
Delivered alone it turns 361 invisible tiles into inspectable ones, which is a complete unit of value even if
nothing else in this spec ships.

**Independent Test**: Load a map, select `Azeroth_26_34`, and confirm the viewer displays coherent structure
and reports a range of 0.000519 rather than showing a blank tile.

**Acceptance Scenarios**:

1. **Given** a tile whose height range is 0.000519 world units, **When** the explorer selects it, **Then** the
   viewer displays its relief across the full available display range and reports the true range at a
   precision that distinguishes it from zero.
2. **Given** a tile whose heights are bit-exact identical at every vertex, **When** the explorer selects it,
   **Then** the viewer states that it is bit-exact flat and displays no invented structure.
3. **Given** a tile with 24% tilted MCNR vertices but negligible height range, **When** the explorer views its
   normals, **Then** the tilt is visibly distinguishable from a tile with no tilt at all.
4. **Given** any selected tile, **When** the explorer views it, **Then** the displayed relief is derived only
   from that tile's own stored values, with no interpolation from neighbours.

---

### User Story 2 - Amplify a weak tile toward its neighbours (Priority: P2)

An explorer enables amplification on a weak-signal tile. Instead of applying a fixed era constant, the viewer
examines the adjacent tiles that carry full-scale terrain and scales the weak tile toward their real height
range, so the restored terrain sits plausibly against the landscape around it rather than at an arbitrary
exaggeration.

**Why this priority**: The user explicitly asked for this and the supporting function already exists unused.
It depends on nothing in User Story 1, but it is less foundational: without US1 there is still no way to see
what is being amplified.

**Independent Test**: Select `Kalimdor_33_12` (range 1.39, neighbours spanning −43.06 to 332.38) with auto
amplification on, and confirm the applied factor derives from the neighbour range rather than the era constant.

**Acceptance Scenarios**:

1. **Given** a weak tile with at least one adjacent full-scale tile, **When** auto amplification is enabled,
   **Then** the factor is derived from the adjacent tiles' real height range and the source of that factor is
   stated in the interface.
2. **Given** a weak tile whose neighbours are all themselves weak or absent, **When** auto amplification is
   enabled, **Then** the viewer falls back to the coarse guide and then to the era constant, and states which
   fallback it used.
3. **Given** a weak tile at a map edge with fewer than four neighbours, **When** auto amplification is enabled,
   **Then** the available neighbours are used without error.
4. **Given** a tile with no relief to amplify, **When** auto amplification is enabled, **Then** no amplification
   is applied and the tile is reported as unamplifiable.

---

### User Story 3 - Find the degenerate tiles across a map (Priority: P3)

An explorer opens a listing of every weak-signal and white-plate tile in the loaded map, sees where they sit
relative to one another, and jumps the camera to any of them. Tiles that cluster spatially are visible as
clusters, so a run of related tiles can be followed rather than discovered one at a time.

**Why this priority**: Navigation convenience. US1 is usable without it by entering coordinates manually, but
361 tiles across four maps is impractical to explore blind.

**Independent Test**: Open the listing on Kalimdor, confirm 201 entries, select one, and confirm the camera
moves to that tile.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the explorer opens the listing, **Then** every weak-signal and white-plate
   tile in that map is listed with its coordinates, classification, and true height range.
2. **Given** the listing is open, **When** the explorer selects an entry, **Then** the camera moves to that tile
   and the tile becomes the inspection subject.
3. **Given** the listing is open, **When** the explorer sorts or filters by classification or by whether the
   tile carries any relief, **Then** the listing reflects that ordering.

---

### Edge Cases

- A tile whose height range is smaller than the floating-point spacing at its world elevation: the viewer must
  still report the measured range rather than rounding it to zero.
- A tile with heights but no normal data, or normals but no heights: each view renders from what exists and
  labels the missing one, rather than failing.
- A tile at a map boundary with fewer than four neighbours.
- A weak tile whose only adjacent full-scale tile is across a map seam.
- A tile that is bit-exact flat: amplification of any factor still yields a flat tile, and the interface must
  say so rather than appearing to have done nothing.
- Amplification factors large enough to push restored terrain outside the world's representable height band.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST classify every loaded tile as weak-signal, white-plate, or normal, using the same
  thresholds the existing weak-signal analysis already applies, so a tile is never classified one way in the
  explorer and another way by the amplifier.
- **FR-002**: The viewer MUST let the explorer select any tile and inspect it, including tiles that carry no
  relief and are currently rendered as featureless.
- **FR-003**: The viewer MUST offer a relief view normalized against the selected tile's own minimum and maximum,
  so relief of any magnitude occupies the full display range.
- **FR-004**: The viewer MUST offer a shaded relief view lit from a low raking angle. Overhead lighting flattens
  exactly the tiles this feature exists to inspect and MUST NOT be the only option.
- **FR-005**: The viewer MUST offer a normals view in which weak tilt is visually distinguishable, scaled against
  the tile's own tilt distribution rather than the theoretical maximum.
- **FR-006**: The viewer MUST display the selected tile's true minimum, maximum, and range at a precision
  sufficient to distinguish a bit-exact flat tile from one with sub-millimetre relief, and MUST state explicitly
  which of the two it is.
- **FR-007**: The viewer MUST NOT display structure on a tile that has none. Any amplification applied to a
  bit-exact flat tile MUST yield a flat result.
- **FR-008**: The amplifier's automatic factor MUST be derived from the real height ranges of adjacent tiles that
  are not themselves weak, when at least one such neighbour exists.
- **FR-009**: The amplifier MUST fall back, in order, to the coarse guide reference and then the era constant when
  no suitable neighbour exists, and MUST state which reference produced the active factor.
- **FR-010**: The viewer MUST provide a listing of every weak-signal and white-plate tile in the loaded map,
  showing coordinates, classification, and true height range, from which the explorer can jump to a tile.
- **FR-011**: The viewer MUST record the per-tile measurements it derives so a session's findings can be reviewed
  outside the viewer.
- **FR-012**: The feature MUST NOT alter the terrain data on disk. All amplification and normalization is a
  display transform over the stored values.
- **FR-013**: The classification and visualization behaviour MUST agree with the existing validated offline
  reference implementation on the same tiles, so the viewer and the offline record cannot disagree about what a
  tile contains.

### Key Entities

- **Tile classification**: The category a tile falls into — weak-signal, white-plate, or normal — together with
  the measurements that decided it: true height minimum, maximum, range, count of weak sub-chunks, and the
  fraction of normal vertices carrying real tilt.
- **Neighbour reference**: For a weak tile, the identity of its four adjacent tiles, how many of them carry
  full-scale terrain, the combined height range of those that do, and the amplification factor that range implies.
- **Inspection view**: One rendered interpretation of a selected tile — self-normalized relief, raking-light
  shading, or amplified normals — each derived solely from that tile's own stored values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 361 weak-signal and white-plate tiles in the 0.5.3 corpus are selectable and inspectable in the
  viewer; none render as an empty or unavailable tile.
- **SC-002**: For all 205 tiles measured to carry non-zero relief, the viewer displays visible structure; for all
  156 measured bit-exact flat, it displays none and labels them as flat.
- **SC-003**: The viewer's reported classification and height range match the offline reference for 100% of the
  1756 tiles in the corpus.
- **SC-004**: For every weak tile with at least one full-scale neighbour, the automatic amplification factor is
  derived from that neighbour's range, and the interface states the reference used; verified against the 45 tiles
  measured to have at least one strong neighbour.
- **SC-005**: An explorer can go from loading a map to inspecting a named degenerate tile in under 30 seconds
  without leaving the viewer or consulting an external file.
- **SC-006**: Selecting a tile from the listing and rendering all three inspection views completes fast enough to
  browse tiles one after another without a perceptible wait.

## Assumptions

- Scope is the 0.5.3 alpha corpus and the viewer only. No training-pipeline, dataset-schema, or harvest changes.
- The four maps already loadable in the viewer (Azeroth, Kalimdor, Kalidar, PVPZone02) are the target corpus.
- The viewer derives classification from terrain data it has already loaded rather than requiring a
  pre-generated inventory file. The existing offline inventory is a validation reference, not a runtime
  dependency, so the feature works on any map the viewer can open — including ones never harvested.
- Amplification remains a display-time transform; the existing behaviour of not writing terrain back to disk
  is preserved.
- The existing weak-signal classification thresholds are correct and are not being retuned by this feature.
- Adjacent tiles are the four edge-sharing neighbours; diagonals are not consulted.
- The existing manual amplification slider is retained alongside the improved automatic path.
