# Feature Specification: Asset Repair Patterns

**Feature Branch**: `174-asset-repair-patterns`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [173](../173-asset-integrity-gate/spec.md) — its census defines which patterns are
worth writing.

## Scope

For specific, named defects, the user can apply a repair. Refusal alone (173) would block real work on
assets the user needs — the >384-group downport exists precisely because these assets must be usable.

**Stated risk, and the reason every requirement below is narrow**: repair logic layered on formats
that are only partly understood is itself a source of new corruption. This was raised in session; the
user weighed it and chose repair. The risk is contained by requirement, not by argument.

## User Story - Known-bad patterns can be repaired, provably (Priority: P2)

The user repairs a quarantined asset. The repair never touches the source, is verified by re-reading
its own output, and records exactly what it changed and lost.

**Independent Test**: Take a real WMO exceeding 384 groups, repair it, and confirm the result loads in
the target client **and renders correctly** — the check the existing merge path has never been held
to.

**Acceptance Scenarios**:

1. **Given** a quarantined asset matching a known pattern, **When** the user views it, **Then** the
   defect, the proposed repair, and **what the repair will lose** are stated before anything is
   applied.
2. **Given** a repair is applied, **When** it completes, **Then** the source asset is unmodified and
   the repaired asset is a new artifact.
3. **Given** a repair produces an artifact, **When** verified, **Then** it is re-read through the same
   validation the original failed. **A repair whose output does not validate is reported as failed and
   is not offered for use.**
4. **Given** a repaired asset is used in a write, **When** provenance is inspected, **Then** the
   repair, its pattern, and its losses are recorded.
5. **Given** a defect matching no known pattern, **When** the user asks, **Then** they are told no
   repair exists — the system does not attempt a general fix.
6. **Given** a pattern applied to an asset it was not designed for, **When** the mismatch is detected,
   **Then** it is refused rather than applied approximately.

### Edge Cases

- A repair that succeeds structurally but produces a visually wrong asset — caught only by the render
  check, which is why one is required.
- Two patterns both matching one asset.
- A repair whose output triggers a different defect.
- An asset repaired against one build, used against another.

## Requirements

### Functional Requirements

- **FR-001**: Repairs are **per-named-pattern**. No general-purpose repair.
- **FR-002**: Repairs are **opt-in**, with defect, proposed change, and expected losses stated before
  application.
- **FR-003**: Repairs **never modify the source**; output is a new artifact.
- **FR-004**: A repair's output must pass the same validation the original failed. A repair that
  cannot prove itself is reported as failed and is not offered.
- **FR-005**: A pattern applied to a non-matching asset is refused, never approximated.
- **FR-006**: Repairs applied are recorded in output provenance with their losses.
- **FR-007**: Repair coverage follows the 173 census — patterns are written for defects actually
  present in the corpus, not for hypothetical ones.
- **FR-008**: A repair intended to make an asset renderable must be verified by an actual render, not
  only a structural re-read.

## Success Criteria

- **SC-001**: Every repair's output passes the validation its input failed; a repair that cannot is
  never offered.
- **SC-002**: No source asset or game-install file is modified by any repair — verified by hashing
  before and after.
- **SC-003**: A real >384-group WMO is repaired and the result both loads and **renders** in the
  target client.
- **SC-004**: Every repaired artifact's provenance names the pattern applied and what it lost.
- **SC-005**: The number of repair patterns equals the number of defect classes from the 173 census
  that were deliberately addressed — with the unaddressed ones listed, not silently omitted.

## Out of Scope

- General-purpose or heuristic repair of unknown defects. Breadth here is a liability, not a feature.
- Rewriting the existing converters.

## Assumptions

- Repair coverage is deliberately narrow and grows only as specific defects are understood.
- If a defect class turns out to have no safe repair, saying so is a valid outcome of this spec.
