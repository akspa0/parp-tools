# Feature Specification: PM4-Guided Object Transfer and Museum Placement Repair

**Feature Branch**: `176-object-transfer`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [175](../175-placement-authoring/spec.md).

## Scope

Transfer selected placements from a source tile — a loaded ADT or an alpha WDT — into a target tile,
possibly in another map or another era's format. The fiddly part, and the reason this has no CLI
equivalent, is that the object-name tables (MMDX/MMID/MWMO/MWID) must be merged into the target and
IDs reconciled.

This feature also adds a **PM4-guided reconciliation mode** for Museum maps. The existing Museum
placement is treated as the visual/source candidate; PM4 is treated as a placement guide and structural
constraint, not as an output format to rewrite. The operation may propose a small transform correction,
identify a better asset from the existing game-object corpus, or clone a known donor placement when the
Museum map is missing an object. Nothing is silently changed: proposals are previewed, attributed to
their evidence, and explicitly accepted or rejected before an ADT is written.

The mode builds on the measured PM4 facts already established in Specs 184/185: the canonical PM4-to-ADT
coordinate transform is fixed, keyed PM4 groups identify WMO-class objects, geometry components can
provide per-doodad candidates, and the surface end value is a placement-height signal in the measured
corpus. These facts are inputs to matching and alignment; this spec does not reopen PM4 format decoding.

## User Story - Transfer objects between tiles and between eras (Priority: P1)

The user selects placed objects and moves them to another tile; referenced model/WMO names come along
so the target file resolves them.

**Independent Test**: Transfer a known set of objects from an alpha-WDT tile into a split-ADT tile in
another map; confirm the target renders them at the correct world positions and the written ADT loads
in an independent tool.

**Acceptance Scenarios**:

1. **Given** objects are selected in a source tile, **When** transferred, **Then** their placements
   appear in the target at the intended world positions.
2. **Given** transferred objects reference models absent from the target, **When** applied, **Then**
   the target's name tables gain the needed entries and **all index references are correct**.
3. **Given** transferred IDs collide with existing target IDs, **When** applied, **Then** new
   non-colliding IDs are assigned and **the remapping is reported**.
4. **Given** source and target are different eras, **When** applied, **Then** coordinates and rotations
   are converted correctly for the target era; **and if** the conversion cannot be performed
   faithfully, **Then** the transfer is **refused with the reason** — never approximated.
5. **Given** a transfer would place an object outside the target tile's bounds, **When** attempted,
   **Then** the user is warned with the offending objects named.
6. **Given** a transfer is applied, **When** undone, **Then** the target returns to its prior state
    **including its name tables**.
7. **Given** a loaded Museum map and its matching PM4 guide, **When** PM4-guided alignment runs, **Then**
   the viewer shows each proposed placement transform, residual/error, evidence sources, and confidence
   in the scene before any placement is mutated.
8. **Given** an existing Museum placement is a likely match for a PM4 object, **When** the user accepts the
   proposal, **Then** only that placement is corrected to the proposed PM4-guided transform and the PM4
   guide remains unchanged.
9. **Given** a PM4 object has no corresponding Museum placement, **When** corpus matching finds a uniquely
   supported game asset, **Then** the viewer offers a clone proposal naming the donor asset/placement and
   its evidence; accepting it creates a normal ADT placement with a fresh non-colliding ID.
10. **Given** multiple game assets or placements remain plausible, **When** matching runs, **Then** the
    result is shown as ambiguous and no clone or substitution is proposed as an automatic decision.
11. **Given** accepted alignment and clone proposals, **When** saved, **Then** the affected ADTs are
    written to the configured output directory and reload with the accepted placements, while rejected
    proposals leave no changes.

### Edge Cases

- A model name existing in the target under a different index.
- Source and target disagreeing on tile origin conventions.
- Objects selected across two tiles, transferred as one operation, where one target write fails.
- An alpha WDT whose per-tile data is present but whose object tables are empty.
- Undo of a transfer after the target file was modified by something else.

## Requirements

### Functional Requirements

- **FR-001**: Transfer selected placements from a source tile to a target tile, including across maps.
- **FR-002**: Merge referenced model/WMO name tables into the target and correct all index references.
- **FR-003**: Remap colliding IDs and report the remapping.
- **FR-004**: Cross-era transfer converts coordinates and rotations correctly for the target era, **or
  is refused with the reason**. Approximate transfer is prohibited.
- **FR-005**: Transfers spanning multiple target tiles apply as one operation — all targets written or
  none.
- **FR-006**: The transfer is a single undoable Editor Operation, including name-table changes.
- **FR-007**: Uses the existing core writers and converters; adds no serializer.
- **FR-008**: Declares supported eras and refuses transfers outside them.
- **FR-009**: PM4-guided alignment MUST use the established PM4/ADT coordinate service and MUST NOT
  introduce a second coordinate convention or rewrite PM4 source data.
- **FR-010**: Alignment MUST produce a preview record per candidate containing source/target identity,
  proposed transform, residual/error, confidence, and the evidence used; previewing MUST be side-effect
  free.
- **FR-011**: The system MUST distinguish accepted, rejected, and ambiguous proposals. Only explicitly
  accepted proposals may alter staged placements or create clones.
- **FR-012**: Matching MAY use the existing PM4 object-library, geometry fingerprints, confirmed-match
  records, placement-height evidence, footprint/containment, and asset metadata, but MUST report which
  signals contributed to each result and MUST NOT claim certainty from proximity alone.
- **FR-013**: A clone proposal MUST identify a resolvable existing game asset or donor placement, copy
  only a placement reference and transform, allocate a non-colliding target ID, and preserve donor
  provenance; it MUST NOT copy proprietary asset bytes or modify the donor.
- **FR-014**: A missing or ambiguous match MUST remain unresolved and be reported with the reason; the
  system MUST never fill gaps with an unlabelled best guess.
- **FR-015**: Alignment, substitution, and cloning MUST be one undoable Editor Operation per user
  decision or one atomic batch operation when the user accepts a reviewed batch.
- **FR-016**: PM4-guided changes MUST save through the existing placement/ADT writers and preserve
  unaffected chunks byte-identically, with a machine-readable operation/provenance report beside the
  output.

## Success Criteria

- **SC-001**: Objects transferred between tiles render at the intended world positions, verified
  visually **and** by reading back the written placement values.
- **SC-002**: A cross-era transfer either round-trips correctly or is refused — **no transfer produces
  silently wrong coordinates**.
- **SC-003**: Name-table merges produce targets whose every index reference resolves, verified on ≥3
  transfers involving previously-absent models.
- **SC-004**: A multi-tile transfer with an induced write failure leaves **no** target modified.
- **SC-005**: Undo of a transfer restores the target byte-identically, name tables included.
- **SC-006**: On a real Museum/PM4 pair, every accepted alignment has a read-back transform and
  evidence record, and every rejected/ambiguous proposal leaves the staged ADT unchanged.
- **SC-007**: A controlled missing-object case can be repaired by accepting a uniquely matched clone;
  the output reloads in the viewer and an independent reader, with a fresh ID and donor provenance.
- **SC-008**: A controlled ambiguous case produces no placement mutation and names the competing
  candidates and missing evidence.
- **SC-009**: At least one reviewed batch containing alignment, substitution, and clone decisions can
  be undone and redone without changing PM4 source bytes or the configured game install.

## Out of Scope

- Creating the target tile ([177](../177-adt-tile-creation/spec.md)).
- Modifying the transferred assets themselves.
- Generating or correcting PM4 files; PM4 remains a read-only guide for this workflow.
- Unattended corpus-wide auto-repair or silently replacing a Museum placement from a matcher score.
- Inventing a new object-matching model or duplicating the existing PM4/object corpus readers.

## Assumptions

- Cross-era conversion uses the existing converters. Where they cannot express a placement faithfully,
  refusing is the correct outcome — not a gap to paper over.
- ID remapping is reported rather than silent, because uniqueId is the world-layout chronology of
  record and a silent remap destroys that ordering information.
