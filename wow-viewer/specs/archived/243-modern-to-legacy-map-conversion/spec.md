# Feature Specification: Modern-to-Legacy Map Conversion (multi-layer alpha merge, LK + Alpha outputs)

<!-- reconciliation-2026-09-23 -->
> **ARCHIVED 2026-09-23 — open residue folded into an epic.** Plan authored; entire implementation open. Successor: [Epic 248](../../248-epic-formats-and-conversion/spec.md). Status lines and checkboxes below are historical and were audited against the code ([audit](../reconciliation-2026-09-23/audit/batch-D2.md)); they are not implementation authority.

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6 (high priority — operator: "it's important and has been overlooked for too long")

**Created**: 2026-09-18

**Status**: Draft (operator-directed 2026-09-18)

**Depends on**: [Spec 238](../238-casc-data-source/spec.md) + [Spec 239](../239-modern-client-assets/spec.md)
(reading modern FileDataID-era maps), [Spec 240](../240-format-conformance/spec.md) (layer/material
conformance), [Spec 234](../234-map-save-new-map/spec.md) (Alpha WDT / LK v18 writer targets),
[Spec 221](../221-converter-validation-harness/spec.md) (converter validation harness)

**Input**: operator direction, 2026-09-18 — "convert multiple layers of alpha masks into the older
format, by merging layers and layer/texture id data. We don't really need to ever target the
bidirectional map format … We only care about going from the new to the old, target both LK and
alphaWDT as outputs from 'modern wow'. Since we can literally render it perfectly in the viewer, we
know how to read just enough of the data to preserve it across whatever era we want. I'd love if we
could make the ui on the map converter require less user intervention and no manual labor, just
selections of which direction and what to provide for. It would be great if we could include assets
in some way, too."

## Context

The viewer can already read, render and light a modern FileDataID-era map (Specs 238/239), so the
project knows how to decode the modern terrain it needs to convert. The converter, however, only
offers routes between **Alpha 0.5.3 monolithic WDT** and **split ADT families** (`MapConversionFormat`
targets: `alpha-wdt-0.5.3`, `lk-adt-v18`, `mop-split-adt`). There is no modern→legacy route, and no
handling of the defining modern difference: **many texture layers per chunk** (1.60.1 uses up to 8,
with `AMAP` blend weights) where the legacy targets expect a small, ordered layer stack.

The operator has ruled out the reverse direction for now — with no usable real engine to consume a
modern artifact, an old→new writer would be speculative. The value is one-way: take modern terrain the
viewer can render, and re-express it faithfully as a loadable **LK v18 ADT** and an **Alpha 0.5.3
WDT**.

## User Scenarios & Testing

### User Story 1 — Convert a modern map to LK and to Alpha, with layer merge (Priority: P1)

A user points the converter at a modern map (or opens it in the viewer), picks the output family, and
gets a complete legacy map with **no manual step per tile**. Multi-layer modern chunks are merged into
the target's layer model, keeping texture ids and combining alpha masks so the result looks like the
source as closely as the target era can express.

**Why this priority**: it is the whole point of the feature; the multi-layer merge is the hard part
that has been missing.

**Independent Test**: convert one modern map to LK v18 and to Alpha 0.5.3; load each result in the
viewer; terrain, textures, liquids and placements render; layer-merge is reported per tile.

**Acceptance Scenarios**:

1. **Given** a modern chunk with more layers than the target supports, **When** it is converted,
   **Then** the layer stack is merged deterministically and the report states what was combined or
   dropped.
2. **Given** a modern chunk whose layer references a missing texture, **When** it is converted,
   **Then** the tile still converts and the missing reference is reported instead of failing the run.
3. **Given** a complete modern map, **When** it is converted to LK v18, **Then** the produced ADT/WDT
   family loads in the viewer with the source's height, texture and placement content.
4. **Given** the same map, **When** it is converted to Alpha 0.5.3 monolithic WDT, **Then** the
   produced WDT loads in the viewer.
5. **Given** an unchanged conversion request repeated, **When** it runs again, **Then** outputs are
   byte-identical (deterministic).

### User Story 2 — Batch several maps without babysitting (Priority: P1)

A user selects several maps (or a whole continent) and a target, starts the run, and walks away. Each
map succeeds or fails independently and the summary lists every outcome.

**Independent Test**: convert 3+ maps in one invocation with one deliberately broken input; the run
completes and the report isolates the failure.

**Acceptance Scenarios**:

1. **Given** multiple selected maps, **When** the run starts, **Then** it converts them all in one
   action with a single direction/target selection.
2. **Given** one map fails to convert, **When** the run finishes, **Then** the other maps are still
   written and the failure is reported with its cause.

### User Story 3 — Low-touch UI: direction + target + input only (Priority: P1)

The converter surface asks only for the direction and what to provide; every format detail (source
detection, layer merge policy, output naming, provenance) is decided for the user.

**Independent Test**: a user who has never read the docs can convert a map using only visible
selections.

**Acceptance Scenarios**:

1. **Given** the converter surface is opened, **When** the user picks a source and a target and
   presses the run control, **Then** no further per-tile or per-format input is required.
2. **Given** an output that would overwrite existing files, **When** the run starts, **Then** it
   writes under a generated project folder and never overwrites client data.

### User Story 4 — Optionally carry the referenced assets (Priority: P2)

A user can ask for the map's referenced assets (textures, models, minimaps) to be included with the
output so the exported map is self-contained for the chosen era.

**Independent Test**: run a conversion with asset inclusion on; the output folder contains the
referenced assets and a manifest stating what was included and what could not be resolved.

**Acceptance Scenarios**:

1. **Given** asset inclusion is on, **When** the conversion ends, **Then** referenced assets are
   written beside the map and a manifest lists included and unresolved assets with reasons.
2. **Given** an asset cannot be resolved, **When** the run ends, **Then** the map output still exists
   and the miss is reported.

## Requirements

- **FR-001**: The converter MUST support the direction modern → legacy for two targets: **LK v18
  ADT/WDT** and **Alpha 0.5.3 monolithic WDT**. Modern input is the FileDataID-era terrain read by the
  existing CASC/asset readers.
- **FR-002**: Multi-layer modern chunks MUST be merged into the target layer model by combining layer
  texture references and alpha masks, deterministically, to preserve appearance as far as the target
  era allows.
- **FR-003**: The conversion MUST report, per tile, which layers were merged, dropped or preserved,
  and any texture reference that could not be resolved.
- **FR-004**: A run MUST accept multiple maps and convert them in a single action, isolating per-map
  failure so one bad map does not abort the batch.
- **FR-005**: The converter surface MUST require only the direction, the target, and the input
  selection; no per-tile or per-format manual input.
- **FR-006**: Outputs MUST be written under a generated project/output folder with provenance
  metadata; client data MUST never be overwritten, and reruns MUST be reproducible.
- **FR-007**: Referenced assets MUST be optionally includable with the output, with a manifest that
  lists what was included and what could not be resolved (and why).
- **FR-008**: The route MUST be validated before writing (unsupported/lossy combinations surfaced
  rather than half-written).
- **FR-009**: New logic MUST live in owned services per AGENTS.md §10, and any new UI surface MUST
  register an inventory row per AGENTS.md §11 / Spec 223 FR-9.
- **FR-010**: Every phase MUST ship a receipt per AGENTS.md §9.2, including at least one real map
  converted to each target and loaded in the viewer.

**Out of scope**: old → modern writers; a modern "bidirectional" interchange target; speculative
engine-consumable output. The existing Alpha↔LK converter commands remain as-is.

## Key Entities

- **Source map**: a modern map (by map id / WDT) whose tiles and assets resolve through the modern
  readers.
- **Layer stack**: the ordered set of texture layers plus blend weights / alpha maps for a chunk.
- **Merge policy**: the deterministic rule mapping a source layer stack onto a target's layer capacity.
- **Conversion run**: a batch of source maps for one target, with per-map results.
- **Asset manifest**: the list of referenced assets included with, or unresolved for, an output.

## Success Criteria

- **SC-001**: A full modern map converts to LK v18 and to Alpha 0.5.3 in a single run with zero
  per-tile manual steps.
- **SC-002**: Both outputs load in the viewer and render terrain, textures, liquids and placements
  (operator visual witness on a real map).
- **SC-003**: A ≥3-map batch completes end-to-end, with one deliberately broken input isolated and
  the remainder written.
- **SC-004**: Rerunning an unchanged conversion produces byte-identical outputs.
- **SC-005**: With asset inclusion on, every referenced asset is either present in the output with a
  manifest entry or explicitly reported as unresolved with a reason.
- **SC-006**: A user unfamiliar with the tool can produce a converted map using only the visible
  selections.

## Assumptions & open questions

- The modern terrain the viewer renders is the source of truth for what must survive conversion
  (operator: "we can literally render it perfectly in the viewer").
- Legacy targets cannot express everything modern terrain carries; the merge is explicitly
  best-effort within the target era, and the per-tile report is the contract for what changed.
- Asset inclusion depends on the modern install being readable (local CASC, or CDN fill); unresolved
  assets are expected in CDN-less installs.
- Operator note (2026-09-18): refresh the vendored libraries and re-read the wowdev.wiki for new
  documentation of this build's makeup; findings here feed the modern readers, not this spec's shape.
- Open question for planning: whether the primary entry point is the CLI (`wowviewer-converter`) or
  the Editor/Data I/O surface, or both sharing one service. Default assumption: one owned conversion
  service, surfaced in both, with the CLI as the batch driver.
