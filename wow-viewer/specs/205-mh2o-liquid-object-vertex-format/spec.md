# Feature Specification: MH2O LiquidObject Vertex-Format Resolution

**Feature Branch**: `205-mh2o-liquid-object-vertex-format`

**Created**: 2026-09-01

**Status**: Draft — ready to implement

**Input**: Operator observation, 2026-09-01: *"the liquid issue is that it randomly doesn't stitch
all the liquid planes as a single plane, so we get chunk boundary based issues... Not sure if this
is an issue that is 5.x-specific (assuming so, since that used to work fine for all other client
versions!)"*

## Context

It is 5.x-specific, and the cause is measured rather than inferred.

In Cataclysm and later, the second `uint16` of `SMLiquidInstance` is **`liquid_object_or_lvf`**:
below 42 it is a liquid vertex format (0–3); **at 42 and above it is a `LiquidObject.dbc` id** and
the real vertex format has to be resolved through the DBC chain. Both of this repo's MH2O decoders
cast that field straight to a vertex-format enum and `switch` on it **with no `default`**, so an id
matches no case, the height array is left null, and the mesh falls back to a flat plane at the
layer's header `minHeight`.

**Measured** — `inspect adt liquid-formats --client C:\WoW4-data\MoPBeta --map HawaiiMainLand
--limit 80`, 80 root ADTs, 17,461 liquid layers:

| `liquid_object_or_lvf` | layers | liquidType | vertex block |
|---|---|---|---|
| 42 | 17,317 | 2 (ocean) | 6,194 carry data; **6,008 read as implausible floats** → the block is depth bytes. Flat is **correct** here. |
| 2325 | 104 | 5 | **100% plausible heights**, 6 vary, spread up to 11.90 |
| 2333 | 17 | 5 | **100% plausible heights**, 10 vary, spread up to 70.15 |
| 2372 | 23 | 5 | **100% plausible heights**, 11 vary, spread up to 163.25 |

**100% of layers (17,461 / 17,461) carry a LiquidObject id.** Not one uses a vertex format in 0–3.

So the ocean was never broken — it is genuinely depth-only and flat. **The rivers and streams carry
real sloped heightmaps, with spreads up to 163 world units, and every one of them is discarded.**
The substituted flat plane is also at the *wrong* height: in every layer whose vertices vary, the
lowest vertex disagrees with the header `minHeight` by more than 0.5 units.

That is both reported symptoms from one cause. Water is flat because the heightmap is thrown away,
and adjacent chunks step against each other because each one flattens to its own incorrect header
value.

**There are two decoders and only one of them is in the render path.** `StandardTerrainAdapter`
(the viewer) calls `Mh2oChunk.Parse`; `AdtLiquidReader` serves harvest, dataset export and the
converter. Both carry the same defect independently. Fixing only `AdtLiquidReader` would change
nothing the operator can see, and fixing only `Mh2oChunk` would leave every harvested dataset
wrong. This duplication is itself in scope (Constitution II).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Waterways follow their real surface (Priority: P1)

Rivers, streams and lakes render with the sloped surface the client authored, instead of a flat
plane at the wrong height.

**Why this priority**: It is the reported defect and the measured one.

**Independent Test**: Load a MoP tile containing a river and compare the water surface against the
same location in the client.

**Acceptance Scenarios**:

1. **Given** a layer whose `liquid_object_or_lvf` is a LiquidObject id resolving to a height-bearing
   format, **When** it is decoded, **Then** its per-vertex heights are read and used.
2. **Given** a layer that resolves to a depth-only format, **When** it is decoded, **Then** it
   renders flat at its declared level, unchanged from today.

---

### User Story 2 - Liquid surfaces meet across chunk boundaries (Priority: P1)

Adjacent chunks of the same body of water join without a step.

**Why this priority**: The visible symptom the operator reported, and the reason the defect was
noticed at all.

**Independent Test**: Fly a river crossing several chunk boundaries and look for discontinuities.

**Acceptance Scenarios**:

1. **Given** two adjacent chunks carrying the same liquid body, **When** both are decoded from their
   own heightmaps, **Then** their shared edge vertices agree and no seam appears.
2. **Given** a chunk with no liquid beside one that has liquid, **When** rendered, **Then** the
   boundary is the edge of the body, not a step in its surface.

---

### User Story 3 - An unresolved format is reported, never silently flattened (Priority: P1)

A liquid layer whose vertex format cannot be determined is recorded as such rather than quietly
producing flat water.

**Why this priority**: This is what let the defect survive. A `switch` with no `default` turned an
unhandled encoding into plausible-looking output, and plausible-looking wrong output is not
reviewable. Equal priority because without it the same class of bug returns the next time Blizzard
adds a value.

**Independent Test**: Feed a layer with an unknown format id and confirm it is counted and logged.

**Acceptance Scenarios**:

1. **Given** a format id that resolves to nothing, **When** decoded, **Then** the layer is recorded
   as unresolved with its id, and the fallback used is stated.
2. **Given** any decode run, **When** it completes, **Then** the number of unresolved layers is
   available.

---

### User Story 4 - Both decoders agree (Priority: P2)

The renderer and the harvest pipeline decode MH2O identically.

**Why this priority**: Two independent decoders with the same defect is how a fix can appear to do
nothing. Lower only because US1–US3 must land first; the consolidation is the durable part.

**Independent Test**: Decode the same ADT through both paths and compare heights layer by layer.

**Acceptance Scenarios**:

1. **Given** one ADT, **When** decoded by the render path and the harvest path, **Then** the
   resulting per-vertex heights are identical.

---

### Edge Cases

- A LiquidObject id absent from the DBC.
- `LiquidObject.dbc` or `LiquidMaterial.dbc` missing entirely (pre-Cata clients, incomplete data).
- A pre-Cata client where the field genuinely *is* a vertex format 0–3 — must keep working exactly
  as it does today.
- A layer with `offset_vertex_data == 0` but a height-bearing format.
- `width`/`height` not 8×8 (all measured layers are 8×8, but the format permits sub-rectangles with
  `x_offset`/`y_offset`).
- The existing `exists` bitmap interacting with a sub-rectangle.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: A `liquid_object_or_lvf` value ≥ 42 MUST be treated as a `LiquidObject.dbc` id, not a
  vertex format.
- **FR-002**: The vertex format MUST be resolved through `LiquidObject.dbc` → `LiquidType.dbc` →
  `LiquidMaterial.dbc`.
- **FR-003**: Values below 42 MUST continue to be treated as vertex formats, preserving pre-Cata
  behaviour exactly.
- **FR-004**: A layer whose format cannot be resolved MUST be recorded and counted, with the
  fallback used stated. It MUST NOT silently produce flat water.
- **FR-005**: Height-bearing layers MUST have their per-vertex heights read and used.
- **FR-006**: Depth-only layers MUST remain flat at their declared level — ocean is not a defect.
- **FR-007**: Both MH2O decoders MUST produce identical output. Preferably there is one decoder;
  if two remain, their agreement MUST be enforced by test.
- **FR-008**: Decode MUST NOT depend on a heuristic classification of the vertex block as its
  primary path.
- **FR-009**: Missing DBCs MUST degrade to current behaviour with the degradation reported, not
  crash and not silently change output.

### Key Entities

- **LiquidObject record**: id → `LiquidTypeID`, plus flow fields not needed here.
- **LiquidType record**: id → `MaterialID` (already partially read by `DbcLiquidTypeTable`).
- **LiquidMaterial record**: id → `LVF`, the vertex format actually wanted.
- **Liquid layer**: the decoded `SMLiquidInstance` with its resolved format, heights, depths and
  existence mask.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Re-running `inspect adt liquid-formats` reports **zero** layers whose format is
  unresolved on the MoP corpus, against the current 17,461 of 17,461 unhandled.
- **SC-002**: The 144 measured river layers decode with varying per-vertex heights, and their
  decoded spread matches the measured 11.90 / 70.15 / 163.25 for their respective ids.
- **SC-003**: Ocean layers are unchanged — still flat, still at the same level.
- **SC-004**: A river crossing at least three chunk boundaries shows no step at the seams.
- **SC-005**: Both decoders return identical heights for the same ADT, enforced by test.
- **SC-006**: 0.5.3, LK and Cata liquid rendering is unchanged.

## Assumptions

- The DBC chain `LiquidObject → LiquidType → LiquidMaterial → LVF` is the client's own resolution
  order. It is documented on wowdev and is **not yet verified against this client's binaries**; the
  first implementation task is to confirm the field offsets against the MoP DBCs rather than trust
  the wiki. The measurement above stands regardless: the field is an id, and the heights exist.
- The float-plausibility probe used to *diagnose* this is **not** acceptable as the shipped decode
  path. It misclassified 18 of 6,194 ocean layers (0.3%) as height-bearing, which would put bogus
  geometry on real water. It may be retained as a reporting cross-check only.
- 5.0.1 ships `LiquidObject.dbc` and `LiquidMaterial.dbc`. If either is absent the spec degrades via
  FR-009 rather than failing.
- Ocean rendering flat is correct and is not part of the reported defect.
