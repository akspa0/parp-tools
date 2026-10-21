# Feature Specification: Offline WoWMuseum ADT Patcher

**Feature Branch**: `281-museum-adt-patcher`

**Created**: 2026-10-21

**Status**: Draft

**Input**: Operator description: "There's lots of old tooling in there for pm4 object matching, which should be done as an offline tool - patch the adt's with new data, sort of thing, since that data is a pain to pull together properly. We should produce a tool that simply asks for the Museum ADT's and the original pm4's, and goes from there, to produce properly patched/restored tiles for 3.3.5 lk adt format outputs. Basically, our tooling produces WoWMuseum patches, to fill in the missing bits from the data you give the tooling. We don't include any data in our tooling, nor do we distribute patched ADTs."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Batch-patch a map from two input folders (Priority: P1)

An operator points the tool at a folder of WoWMuseum ADTs and a folder of original PM4 files for the same map. The tool pairs the files by tile, matches the PM4 object segments against the Museum placements, and writes a patched 3.3.5 LK ADT for each tile, plus a provenance report. The operator does not open the viewer.

**Why this priority**: This is the whole feature. The matching data is expensive to assemble, so it must run as a repeatable batch job, not as an interactive viewer session.

**Independent Test**: Given a Museum ADT folder and a PM4 folder for one map, the tool writes one patched LK ADT per paired tile and one report per tile, and the reports name every accepted, rejected, and unmatched object.

**Acceptance Scenarios**:

1. **Given** a Museum ADT folder and a PM4 folder, **When** the operator runs the tool, **Then** it discovers the tile pairs and reports the pair count before writing anything.
2. **Given** a discovered pair, **When** the tool processes it, **Then** it writes a patched LK v18 ADT to the output folder and a provenance report beside it.
3. **Given** a tile with no PM4 or no Museum ADT, **When** the tool runs, **Then** it reports the tile as unpaired and does not write a partial output.

---

### User Story 2 - Review before writing (Priority: P1)

The operator runs the tool in dry-run mode first, reads the proposal report, and decides which proposals to accept. The tool then applies only the accepted proposals.

**Why this priority**: The matching is heuristic. The operator must be able to inspect and correct it before any ADT is written.

**Independent Test**: A dry run writes a review file and no ADTs; a second run with the review file writes only the accepted proposals.

**Acceptance Scenarios**:

1. **Given** dry-run mode, **When** the tool runs, **Then** it writes a review file and no ADT output.
2. **Given** a review file with accepted and rejected proposals, **When** the tool runs in apply mode, **Then** only the accepted proposals are written.
3. **Given** a proposal with no decision, **When** the tool runs in apply mode, **Then** the proposal is treated as rejected and reported.

---

### User Story 3 - Produce 3.3.5 LK ADT output (Priority: P1)

The operator receives a monolithic 3.3.5 LK v18 ADT per tile that other 3.3.5 tooling can read. The output preserves the Museum terrain and adds the restored placements.

**Why this priority**: The output format is the deliverable. A patched file that no 3.3.5 tool can read is useless.

**Independent Test**: The output ADT opens in the viewer's 3.3.5 adapter and in an independent 3.3.5 ADT reader, and the restored placements are present in MDDF/MODF.

**Acceptance Scenarios**:

1. **Given** a patched tile, **When** it is read by the 3.3.5 adapter, **Then** the terrain chunks and the restored placements are present.
2. **Given** a patched tile, **When** it is compared to the input Museum ADT, **Then** the terrain bytes are unchanged except where a proposal explicitly edits them.
3. **Given** a patched tile, **When** the provenance report is read, **Then** every added placement names its source PM4 object and its match score.

---

### User Story 4 - No bundled data, no distributed patches (Priority: P1)

The tool ships with no Museum ADTs, no PM4 files, and no patched ADTs. It reads only the paths the operator supplies and writes only to the output folder the operator supplies.

**Why this priority**: This is a distribution and legal constraint, not a preference. The tool must never carry or redistribute client data.

**Independent Test**: The repository and the release archive contain no `.adt` or `.pm4` data files, and the tool refuses to write outside its output folder.

**Acceptance Scenarios**:

1. **Given** the release archive, **When** it is inspected, **Then** it contains no `.adt`, `.pm4`, or `.pd4` data files.
2. **Given** an output path outside the operator-supplied output folder, **When** the tool tries to write, **Then** it refuses and reports the refusal.
3. **Given** an input path that is also the output path, **When** the tool runs, **Then** it refuses to overwrite the input.

---

### User Story 5 - Deterministic, auditable output (Priority: P2)

The operator can re-run the tool on the same inputs and get byte-identical output, and can audit every decision from the provenance report.

**Why this priority**: Reproducibility is what makes the patches reviewable and safe to share as a process rather than as files.

**Independent Test**: Two runs on the same inputs produce identical output hashes; the report lists every proposal with its inputs, score, decision, and output hash.

**Acceptance Scenarios**:

1. **Given** the same inputs and review file, **When** the tool runs twice, **Then** the output hashes match.
2. **Given** a provenance report, **When** the operator reads it, **Then** every proposal has a source, a score, a decision, and an output hash.

---

### Edge Cases

- A PM4 file with no Museum ADT: report as unpaired, write nothing.
- A Museum ADT with no PM4: report as unpaired, write nothing.
- A PM4 file that fails to parse: report the failure, continue with the other tiles, exit non-zero.
- A Museum ADT that is not 3.3.5 LK: report the format mismatch and skip the tile.
- A proposal that would duplicate an existing placement: report as already aligned, do not duplicate.
- A proposal that conflicts with an existing placement: report as conflict, require an explicit decision.
- An output folder that already contains a file: refuse unless `--overwrite` is passed.
- A review file that names a proposal that no longer exists: report the stale entry and continue.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The tool MUST accept a Museum ADT folder and a PM4 folder as inputs.
- **FR-002**: The tool MUST pair inputs by tile coordinates and report paired and unpaired tiles.
- **FR-003**: The tool MUST build PM4 object segments from each PM4 file.
- **FR-004**: The tool MUST read Museum placements from each ADT.
- **FR-005**: The tool MUST score PM4 segments against Museum placements and produce proposals.
- **FR-006**: The tool MUST support a dry-run mode that writes a review file and no ADTs.
- **FR-007**: The tool MUST support an apply mode that writes only accepted proposals.
- **FR-008**: The tool MUST write a monolithic 3.3.5 LK v18 ADT per patched tile.
- **FR-009**: The tool MUST write a provenance report per patched tile naming every proposal, its source, score, decision, and output hash.
- **FR-010**: The tool MUST NOT bundle or distribute any ADT, PM4, or PD4 data.
- **FR-011**: The tool MUST refuse to write outside the operator-supplied output folder.
- **FR-012**: The tool MUST refuse to overwrite an input file.
- **FR-013**: The tool MUST produce deterministic output for the same inputs and review file.
- **FR-014**: The tool MUST report unmatched, ambiguous, and conflicting proposals explicitly.
- **FR-015**: The tool MUST return a non-zero exit code when any tile fails.
- **FR-016**: The tool MUST be usable without the viewer and without a GPU.

### Key Entities

- **Tile Pair**: A Museum ADT and a PM4 file that share tile coordinates.
- **Proposal**: A candidate placement derived from a PM4 object segment and a Museum placement, with a score and a decision.
- **Review File**: A machine-readable list of proposal decisions produced by dry-run and consumed by apply.
- **Provenance Report**: A per-tile record of every proposal, its inputs, score, decision, and output hash.
- **Patched ADT**: A 3.3.5 LK v18 ADT written to the output folder.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A one-map run pairs every tile that has both inputs and reports every tile that does not.
- **SC-002**: Dry-run writes a review file and zero ADTs.
- **SC-003**: Apply writes one patched ADT per accepted tile and zero ADTs for rejected tiles.
- **SC-004**: Every patched ADT opens in the 3.3.5 adapter with its terrain and restored placements present.
- **SC-005**: Two runs on the same inputs produce identical output hashes.
- **SC-006**: The release archive contains zero `.adt`, `.pm4`, or `.pd4` data files.
- **SC-007**: The tool refuses to write outside the output folder and refuses to overwrite an input.
- **SC-008**: Every proposal in the provenance report has a source, score, decision, and output hash.

## Assumptions

- The Museum ADTs are 3.3.5 LK v18 monolithic ADTs.
- The PM4 files are the original client files for the same map and tile grid.
- The operator supplies both folders; the tool does not search a client install.
- The matching heuristics are the existing `Pm4AssetMatchScorer` and `Pm4ReconciliationEngine`; this spec moves them offline, it does not invent a new matcher.
- The output is a patch process, not a data distribution. The operator is responsible for the inputs and for how the outputs are used.
- The viewer keeps PM4 inspection (overlay, selection, info, outliner, object groups) but loses the matching, alignment, and reconciliation tabs.
