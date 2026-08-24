# Feature Specification: PM4 field semantics and a grouping surface that tests them

**Feature Branch**: `188-pm4-field-semantics`

**Created**: 2026-08-24

**Depends on**: [185 PM4/PD4 documentation](../185-pm4-pd4-format-documentation/spec.md) owns the
naming rules and the wiki draft. This spec owns *discovering* what the remaining fields hold and
turning the viewer's grouping control into the instrument that does it.

**Input**: User description: "We know the GroupObjectID, AttrMask, and various other things we were
grouping the data by, are all wrong. We believe there are probably other bits of data being
improperly decoded and attributed to being random flags and masks instead of the actual data. I'd
like to fix that drop-down to group the pm4 data by, so it makes actual sense and provides actual
verifyable tests for the various bits we don't know WTF the data is for."

## Context

The viewer's PM4 grouping control offers ten modes. At least four group by a quantity that has since
been measured to be something else entirely, and the control gives the user no way to tell.

### What the grouping modes actually group by

| mode | field | what it really is | status |
|---|---|---|---|
| `Ck24Type` | `MSUR._0x1C` bits 24-31 | the **exponent band of a float** — groups by height octave | **falsified** |
| `Ck24ObjectId` | `_0x1C` bits 8-23 | **mantissa bytes** of that float | **falsified** |
| `Ck24TypeVsTypeFlags` | both of the above | a cross-tab of an exponent band | **falsified** |
| `AttributeMask` | `MSUR._0x02` | the **length of this surface's MSLK window** — grouping by neighbour count | **falsified** |
| `Ck24Key` | `_0x1C` whole | the placement's Z. Genuinely groups **WMO** objects | sound, WMO only |
| `MshdRegionId` | `MSHD._0x04` | `== 1` marks a tile with no surfaces; otherwise unexplained | partial |
| `GroupKey` | `MSUR._0x00` | 100% pure but only **9 values corpus-wide** and 0% distinct — a class enum that separates nothing | unmeasured |
| `TypeFlags` | `MSLK._0x00` | observed buckets, not corpus-closed | partial |
| `Height` | `MSUR._0x10` | signed plane distance | partial |
| `Tile` | filename | sound | sound |

`MSLK.GroupObjectId` — not in the dropdown but used throughout the PM4 code as an object identity —
is **near-unique per link**: 1.622 links per distinct value in the doodad population and 1.508 in the
placed one. A value that changes almost every link cannot group an object. Its recorded "99.9%
distinctness" was distinctness without cardinality, which near-unique values satisfy trivially.

### Why this keeps happening

Every one of these was adopted from a name, not a measurement. A field called `AttributeMask` invites
bitwise reading; a field called `GroupObjectId` invites grouping. Both names were invented locally.
The pattern has now repeated enough times to treat as systematic rather than unlucky: a field is
given a semantic name, downstream code trusts the name, and the name is never tested against the
data.

The remaining unnamed fields are the same hazard waiting to happen. `MSLK` alone carries a subtype, a
link id and a system flag with no evidence behind any of them, `MPRL` carries six, and `MSUR._0x00`
and `MSHD._0x00`/`_0x08` are open.

### The idea

Make the grouping control the instrument that settles these. When a user groups by a field, the
viewer should show, from the loaded data, **whether that field groups anything at all** — how many
distinct values, how big the groups are, and how well they line up with the one grouping that is
derived from geometry rather than from a guess: connected components of the adjacency graph.

A field that separates nothing then looks like it separates nothing, immediately and visibly, instead
of producing a colourful overlay that implies structure.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See whether a grouping field groups anything (Priority: P1)

A user selects a field to group PM4 objects by and immediately sees how that field behaves on the
loaded data: how many distinct values it takes, how large its groups are, and how closely it matches
a grouping derived from geometry.

**Why this priority**: This is what turns the control from a colouring toy into evidence, and it is
what would have caught every mistake listed above.

**Independent Test**: Select a field already known to be meaningless (neighbour count) and one known
to be sound (placement Z) and confirm the readout distinguishes them without the user knowing which
is which in advance.

**Acceptance Scenarios**:

1. **Given** a loaded PM4 scene, **When** a grouping field is selected, **Then** the view reports its
   distinct-value count, its group-size distribution, and its agreement with geometric components.
2. **Given** a field whose values are near-unique per record, **When** it is selected, **Then** the
   readout makes that visible rather than presenting one colour per record as structure.
3. **Given** a field with very few values spread across all objects, **When** it is selected,
   **Then** the readout shows high purity **and** near-zero distinctness together, never purity
   alone.
4. **Given** any grouping mode, **When** it is displayed, **Then** it carries the confidence of the
   underlying field's decode, and a falsified field is labelled as such.

---

### User Story 2 - Retire or relabel the falsified grouping modes (Priority: P1)

A user no longer sees grouping modes that group by a float's exponent band or by a window length
presented as if they were object attributes.

**Why this priority**: These actively mislead. A mode that colours by height octave under the name
"type" will keep generating false structure for as long as it exists.

**Acceptance Scenarios**:

1. **Given** the grouping control, **When** it is opened, **Then** no mode presents a slice of the
   placement-Z float as a type or an object id.
2. **Given** a mode kept for continuity with older reports, **When** it is shown, **Then** it is
   labelled with what the field actually holds.
3. **Given** the placement-Z grouping, **When** it is used, **Then** it is scoped to the population
   it works on and says so, because doodad surfaces all carry zero there.

---

### User Story 3 - Group the doodad population by something real (Priority: P2)

A user working with doodad collision — the surfaces carrying no placement height — can group it by a
key that actually separates objects.

**Why this priority**: That population is 186,060 surfaces across 283 of 309 files, the majority of
the corpus by file coverage, and it currently has **no** working grouping: its placement key is
uniformly zero and the field long used as its identity does not group.

**Independent Test**: Group the doodad population, then check the resulting groups against MDDF
doodad placements, which prior work matched geometrically at 95.1%.

**Acceptance Scenarios**:

1. **Given** the doodad population, **When** it is grouped by connectivity, **Then** the resulting
   objects are reported with their agreement against nearby doodad placements.
2. **Given** a candidate identity field for that population, **When** it is offered, **Then** it is
   accompanied by its cardinality so a near-unique field cannot masquerade as an identity.

---

### User Story 4 - Audit every remaining field against its name (Priority: P2)

A researcher gets, for every field in these formats, a measurement of what it behaves like —
constant, near-unique, enumerated, an index, a window, a float — independent of what it is called.

**Why this priority**: The fields already burned were all caught one at a time, by accident. Several
remain unexamined and carry semantic names. A single sweep is cheaper than four more accidents.

**Independent Test**: Run the sweep and confirm it independently re-derives the fields already
settled — the window length, the float, the near-unique link id — without being told.

**Acceptance Scenarios**:

1. **Given** any record field, **When** the sweep runs, **Then** it reports value cardinality, range,
   whether values are plausible indices into each chunk, whether the field is constant, and whether
   consecutive values form a running window.
2. **Given** a field whose name asserts a bitfield, **When** the sweep runs, **Then** it reports
   whether the value population is consistent with independent bits or with small integers.
3. **Given** a field the sweep cannot characterise, **When** results are published, **Then** it is
   listed as uncharacterised rather than omitted.

---

### Edge Cases

- A field that is genuinely a bitfield in one population and a small integer in another.
- A field that groups well on one tile and not corpus-wide; per-file and corpus figures can disagree
  and both are needed.
- Fields that are near-unique because the corpus is one map — cardinality alone cannot separate "an
  id" from "a counter".
- Grouping modes referenced by saved reports or older exports, which must remain interpretable after
  a rename.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every grouping mode MUST display the decode confidence of the field it groups by.
- **FR-002**: The system MUST NOT offer a grouping mode built on a reading that has been falsified,
  except when explicitly labelled with what the field actually holds.
- **FR-003**: Selecting a grouping mode MUST report, from the loaded data, its distinct-value count
  and group-size distribution.
- **FR-004**: Grouping quality MUST be reported as purity **and** distinctness together; neither may
  be shown alone.
- **FR-005**: Grouping modes MUST be comparable against a geometry-derived grouping that uses no
  guessed field.
- **FR-006**: A grouping scoped to one population MUST state its scope wherever it is offered.
- **FR-007**: The doodad population MUST have at least one grouping that separates objects, with its
  agreement against doodad placements reported.
- **FR-008**: The field sweep MUST characterise every record field in both formats by behaviour, not
  by name.
- **FR-009**: The sweep MUST test each field for index-like, window-like, constant, enumerated and
  float-like behaviour and report which fit.
- **FR-010**: Any field the sweep cannot characterise MUST be listed as uncharacterised.
- **FR-011**: Field findings MUST flow into the terminology catalog that spec 185 owns rather than
  into a second vocabulary.
- **FR-012**: Renaming or retiring a mode MUST NOT change the geometry rendered, and the corpus
  figures published by existing analyzers MUST be identical before and after.
- **FR-013**: No new PM4 reader or chunk parser may be introduced.

### Key Entities

- **Grouping mode**: a rule assigning objects to groups, with its source field and confidence.
- **Grouping quality**: cardinality, group sizes, purity and distinctness against a reference.
- **Reference grouping**: the geometry-derived grouping, used as the yardstick because it depends on
  no guessed field.
- **Field characterisation**: what a field behaves like, independent of its name.
- **Population**: a subset of surfaces sharing a structural property, such as carrying a placement
  height or not.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of offered grouping modes display the confidence of their underlying field.
- **SC-002**: Zero modes present a slice of the placement-Z float as a type or an object id.
- **SC-003**: Selecting any mode reports cardinality and group-size distribution without further
  action.
- **SC-004**: Purity and distinctness appear together in 100% of quality readouts.
- **SC-005**: A field that is near-unique per record is visibly identifiable as such from the readout
  alone, confirmed by a user who was not told which field it is.
- **SC-006**: The doodad population has a grouping whose agreement with doodad placements is
  reported, and that figure is reproducible from the CLI.
- **SC-007**: The sweep covers 100% of record fields in both formats, and independently re-derives
  the window length, the placement float and the near-unique link id without being given them.
- **SC-008**: Every field is either characterised or explicitly listed as uncharacterised; none are
  silently omitted.
- **SC-009**: Corpus figures from the existing analyzers are identical before and after this work.

## Assumptions

- The geometry-derived grouping is the reference because it uses no guessed field; it is not assumed
  correct in an absolute sense, only independent.
- Doodad-population identity may not exist in the file. Delivering a working grouping for it does not
  require finding a stored key.
- The sweep is a characterisation instrument, not a decoder; naming what a field means stays with
  spec 185 and its evidence rules.
- Long corpus sweeps and any real-session viewer checks are user-run.

## Out of Scope

- Renaming code fields — spec 185 FR-003 owns that, and this spec supplies evidence to it.
- Decoding MPRR, which spec 185 tracks under its own structural constraint.
- Changing how PM4 geometry is built or rendered.
- Object-to-asset matching, which the placement-Z key already handles for the placed population.
