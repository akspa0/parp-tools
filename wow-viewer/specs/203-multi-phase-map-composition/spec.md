# Feature Specification: Multi-Phase Map Composition

**Feature Branch**: `203-multi-phase-map-composition`

**Created**: 2026-09-01

**Status**: Draft

**Input**: Operator, 2026-09-01, on phased maps losing base-map placements and on needing more
than one phase active at once (Jade Forest, 5.0.1, three quest-progression phases).

## Context

### The merge is a replacement, not a patch — confirmed

`StandardTerrainAdapter.MergePhaseTile` does this per phase chunk:

```csharp
int parentIndex = parent.Result.Chunks.IndexOf(parentChunk);
parent.Result.Chunks[parentIndex] = phaseChunk;      // wholesale replacement
...
parent.Result.MddfPlacements.AddRange(phase.Result.MddfPlacements);   // append, never reconcile
parent.Result.ModfPlacements.AddRange(phase.Result.ModfPlacements);
```

Only one field survives: `phaseChunk.Liquid = parentChunk?.Liquid`. The log even advertises
this — `parentLiquidsPreserved=true` — which is an accurate description of a merge that
preserves *only* liquid.

So a phase chunk replaces the base chunk entirely. Anything the base chunk carried that the
phase chunk does not is gone. That is the operator's *"missing trees around objects that do
change"*, and it matches the suspicion that the logic was not revisited when `_obj0`/`_obj1`
companion loading landed: the phase's object companions now supply a *partial* placement set,
and a partial set replacing a complete one loses the difference.

Placements are meanwhile only ever **appended**. There is no reconciliation by `uniqueId` at
all, so the operator's hypothesis — that a phase may re-use a `uniqueId` to move or substitute
an existing placement — is neither implemented nor currently detectable.

### Only one phase can be active — by contract

`ITerrainAdapter.OverlayMapName` is a single `string?`. All three adapters implement it as one
value. Multiple simultaneous phases is an interface change, not a settings change.

### What 5.0.1 actually has

Confirmed present in `Wow.exe` 5.0.1.15464: `DBFilesClient\Phase.dbc`,
`DBFilesClient\PhaseXPhaseGroup.dbc`, `DBFilesClient\PhaseShiftZoneSounds.dbc`. So phasing is a
first-class client concept in this build, and phase **groups** exist — which is the shape a
"show several phases at once" feature would key off.

**Open**: whether `Map.dbc` carries a parent/child map column in 5.0.1. The operator is
explicitly unsure and suspects it may be a 7.x/8.x memory. DBC column *names* are not in the
binary, so this cannot be answered by string search — but it is cheaply answerable from the
file itself: the `WDBC` header carries `fieldCount` and `recordSize`, and comparing 5.0.1's
`Map.dbc` field count against 3.3.5's settles it. This is T001 and it gates how phase
relationships are discovered rather than configured.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A phase patches the base map instead of replacing it (Priority: P1)

Loading a phased map shows the base map's content plus the phase's changes — trees and objects
the phase does not mention stay where they were.

**Why this priority**: This is a data-loss bug with a visible symptom, and it blocks trusting
anything else about phasing.

**Independent Test**: Load a phased map and confirm base-map placements absent from the phase
are still present.

**Acceptance Scenarios**:

1. **Given** a base chunk with content the phase chunk does not carry, **When** the phase is
   applied, **Then** that content survives.
2. **Given** a phase that supplies only object companions, **When** it is applied, **Then**
   terrain from the base is retained rather than replaced by a partial chunk.
3. **Given** a phase applied and then removed, **When** the tile reloads, **Then** the result
   matches the unphased base exactly.

---

### User Story 2 - Placement identity is reconciled, and collisions are reported (Priority: P1)

Placements from a phase are merged against the base by identity, and any `uniqueId` appearing
in both is reported with what differs — position, model, scale or rotation.

**Why this priority**: Equal to US1, and it is what makes US1 correct rather than merely
additive. It also **tests the operator's hypothesis** instead of assuming it: if phases never
collide on `uniqueId`, that is a finding and the merge simplifies.

**Independent Test**: Load a phased map and read the collision report.

**Acceptance Scenarios**:

1. **Given** a `uniqueId` present in both base and phase, **When** merged, **Then** the phase's
   record wins and the collision is reported with the fields that differ.
2. **Given** a `uniqueId` only in the base, **When** merged, **Then** it is retained.
3. **Given** a corpus of phased maps, **When** loaded, **Then** the total count of `uniqueId`
   collisions is reported — including zero, which would refute the hypothesis.

---

### User Story 3 - Several phases active at once (Priority: P2)

The operator selects multiple phases for one map and sees them composed together — Jade
Forest's three quest-progression phases visible as one scene.

**Why this priority**: The operator's stated need, but it depends on a merge that composes
correctly. Stacking phases on top of a replacing merge would multiply the data loss.

**Independent Test**: Enable two then three phases on one map and confirm each contributes.

**Acceptance Scenarios**:

1. **Given** two phases selected, **When** the map loads, **Then** both are composed onto the
   base in a defined, stated order.
2. **Given** two phases that modify the same chunk or `uniqueId`, **When** composed, **Then**
   the resolution order is deterministic and reported.
3. **Given** phases are deselected, **When** the map reloads, **Then** the result matches the
   base plus exactly the remaining phases.

---

### Edge Cases

- A phase tile that exists where the base tile does not, and the reverse.
- A phase supplying `_obj0` but no root ADT, or a root but no companions.
- Two phases supplying different replacements for the same `uniqueId`.
- A phase whose chunk carries fewer texture layers than the base — must not silently drop
  layers, which is how the replacement bug presents.
- 0.5.3 phasing, which predates this mechanism entirely and must be unaffected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Phase application MUST patch the base tile field by field, not replace chunks
  wholesale. Base content the phase does not supply MUST survive.
- **FR-002**: Placements MUST be merged by `uniqueId`, with phase records overriding base
  records of the same id and base-only records retained.
- **FR-003**: `uniqueId` collisions MUST be reported with the fields that differ, so the
  substitution hypothesis is measured rather than assumed.
- **FR-004**: The system MUST support more than one active phase per map.
- **FR-005**: When multiple phases modify the same chunk or placement, resolution order MUST be
  deterministic, stated, and reported.
- **FR-006**: Removing all phases MUST reproduce the unphased base map exactly.
- **FR-007**: Phase composition MUST work with split ADT companions, patching per contributing
  file rather than assuming a phase supplies a complete tile.
- **FR-008**: 0.5.3 behaviour MUST NOT change.

### Key Entities

- **Phase layer**: One overlay map contributing to a composition, with its order.
- **Phase composition**: The ordered set of active phases over a base map.
- **Placement merge outcome**: Per `uniqueId` — base-only, phase-only, or overridden with the
  differing fields.
- **Composition report**: Per tile — chunks patched, placements added/overridden/retained, and
  collisions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a phased map that previously lost content, base placements absent from the
  phase are present after the change, confirmed by capture.
- **SC-002**: The `uniqueId` collision count across a phased-map corpus is measured and
  reported — including zero.
- **SC-003**: Three phases can be active simultaneously on Jade Forest and each contributes,
  confirmed by capture.
- **SC-004**: Deselecting every phase reproduces the unphased base exactly, verified by pixel
  comparison.
- **SC-005**: A 0.5.3 reference scene is pixel-identical to before.
- **SC-006**: Whether `Map.dbc` carries a parent/child map relationship in 5.0.1 is answered
  from the file, and recorded either way.

## Assumptions

- `Phase.dbc` and `PhaseXPhaseGroup.dbc` exist in 5.0.1 and are the natural source for which
  phases belong to a map, but this spec does not assume their schema — T001/T002 read them.
- The `uniqueId` substitution hypothesis is **unproven**. FR-003 and SC-002 are written to test
  it; a zero collision count is a legitimate outcome that simplifies the merge.
- 5.2/5.3 weekly phase cycling was partly server-driven. Only client data is in scope, and a
  phase this project cannot reconstruct from client files is out of scope rather than a gap.
- Captures and corpus runs are operator-executed.
