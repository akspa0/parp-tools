# Feature Specification: Viewer — Load Zarr Datastore

**Feature Branch**: `183-datastore-viewer-load`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first**.
**Depends on**: [180](../180-multi-build-datastore/spec.md).

## Scope

A **Load Zarr Datastore** entry point in the viewer: pick a datastore, pick a build, and the viewer
works as if that client were installed.

**Deliberately last.** The datastore's primary surface is ML/AI; viewer loading is a convenience that
falls out of a correct store, not the reason to build one. It is only worth doing once 179/180 have
made the store trustworthy.

## User Story - Load a datastore build in the viewer (Priority: P3)

The user picks a build from a datastore and the viewer behaves exactly as it would with the original
client — one artifact instead of a directory of clients.

**Independent Test**: Load the same map from a datastore build and from the original client; renders
and loaded data must match.

**Acceptance Scenarios**:

1. **Given** a datastore, **When** the user chooses Load Zarr Datastore, **Then** its builds are listed
   with identity and contents summarized.
2. **Given** a build is selected, **When** the viewer loads, **Then** behavior matches loading the
   original client — same maps, models, tables.
3. **Given** a datastore build is loaded, **When** the user switches builds, **Then** it works without
   restarting **and without extracting either build to disk**.
4. **Given** the datastore lacks data a request needs, **When** the request is made, **Then** the gap is
   reported specifically — **never silently rendered as absent content**.
5. **Given** a build is loaded, **When** the user checks provenance, **Then** the datastore, build, and
   original client identity are all visible.
6. **Given** a build is loaded, **When** disk activity is observed, **Then** no copy of build content is
   written anywhere.

### Edge Cases

- A datastore built by a newer version of the tooling than the viewer.
- Switching builds while assets from the previous build are still streaming.
- A datastore on slow or removable storage.
- Both a datastore and a live client configured at once.

## Requirements

### Functional Requirements

- **FR-001**: Offer **Load Zarr Datastore**, listing builds with identity and contents.
- **FR-002**: A loaded datastore build behaves equivalently to the original client.
- **FR-003**: Switching builds does not require a restart.
- **FR-004**: Reading a build must not extract or copy it to disk.
- **FR-005**: Missing data is reported specifically, never silently rendered as absent content.
- **FR-006**: Provenance — datastore, build, original client identity — is visible while loaded.
- **FR-007**: Datastore access goes through the existing `IDataSource` abstraction, not a bypass.
- **FR-008**: The datastore is an **additional** source; the live MPQ path remains supported.

## Success Criteria

- **SC-001**: Loading a build from a datastore reaches the same rendered result as loading the original
  client, verified on ≥3 maps.
- **SC-002**: Switching builds writes no copy of build content to disk — verified by observing writes
  during a switch.
- **SC-003**: A deliberately incomplete datastore reports its gaps specifically rather than rendering
  empty content.
- **SC-004**: No new data-source bypass is introduced — datastore reads go through the same abstraction
  as MPQ and loose files.

## Out of Scope

- Editing datastore contents. The datastore is built and read; the Editor writes loose output.
- Replacing the live MPQ path.

## Assumptions

- The datastore is correct by the time this lands — 180's verification is the gate, and this spec does
  not re-verify content.
- Read performance is expected to be comparable to or better than MPQ; if it is not, that is a finding
  to report rather than a blocker for this spec.
