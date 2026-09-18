# Feature Specification: Modern M2 Camera Paths and Modern-Data Renderer Benchmarking

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6

**Created**: 2026-09-18

**Status**: Draft (operator-directed 2026-09-18)

**Depends on**: [Spec 239](../239-modern-client-assets/spec.md) (modern M2 / `MD21` reading),
[Spec 238](../238-casc-data-source/spec.md) (CASC id reads),
[Spec 236](../236-scene-lighting-doodad-performance/spec.md) (scene lighting changes the modern frame
cost), [Spec 242](../242-wmo-instancing-performance/spec.md) (measures its fix with this benchmark),
[Spec 233](../233-marketing-capture-automation/spec.md) (existing path-driven capture + receipt)

**Input**: operator direction, 2026-09-18 — "we gotta fix it so we can load the m2's from the modern
wow assets as camera paths, too. I'd like to be able to benchmark the renderer with the new client
data like we do for older client data."

## Context

Two capabilities that exist for legacy data do not hold for modern data.

**Camera paths.** The viewer loads M2 camera tracks and turns them into a playable path
(`M2CameraPathImporter`, `M2CameraPathOverlayBuilder`, camera-path authoring UI, path warmup, key-still
captures). The overlay builder will only build when the model document actually carries cameras
(`CanBuild` requires `CameraCount > 0`). The standalone camera-path load path reads through the
non-chunked M2 reader, while modern WoW: Forever models arrive as **chunked `MD21`** and are converted
before use. The M2 reader does contain a modern camera branch (a `0x74` camera stride with a
field-of-view track), so the question is *where* modern cameras are lost — in the era dispatch, in the
chunked conversion, or nowhere — and that has not been established.

**Benchmarking.** Frame-time mean/p99, hitch counts and frame history already exist
(`WorldRenderFrameHistory`), and the marketing-tour capture flow already produces a path-driven video
plus a receipt for legacy data. `inspect casc bench` measures **data reads**, not renderer frames, so
modern client data has no equivalent renderer benchmark.

This matters now: [Spec 242](../242-wmo-instancing-performance/spec.md) exists because modern-data
framing measured ~5.5 FPS, and the only way to prove its fix is a repeatable modern-data benchmark.

## User Scenarios & Testing

### User Story 1 — Load a modern asset's M2 camera path (Priority: P1)

A user points the viewer at a camera-carrying M2 from a modern CASC install and flies the authored
path exactly as they can with legacy-client camera assets.

**Why this priority**: it is the stated defect; without it the benchmark below has no repeatable path.

**Independent Test**: import a modern camera M2 from the CASC install and get a path with keys and a
playable timeline, equal in behaviour to the same operation on a legacy asset.

**Acceptance Scenarios**:

1. **Given** a modern camera-carrying M2 resolved from CASC, **When** it is imported as a camera path,
   **Then** the path loads with keyframes and plays.
2. **Given** the same asset opened as a standalone model, **When** the viewer inspects it, **Then** its
   cameras are visible as path overlays.
3. **Given** a modern model that carries no cameras, **When** it is loaded, **Then** the viewer says so
   and does not present it as a camera path.
4. **Given** camera tracks whose coordinate space differs from the legacy convention, **When** the path
   is played, **Then** the camera lands on the right world position for the modern map (or the offset
   is reported rather than silently wrong).

### User Story 2 — Benchmark the renderer with modern client data (Priority: P1)

A user runs a path-driven flyby on a modern map and gets the same kind of receipt they get for legacy
data: frame-time samples, hitches, draw-call/submission counters, and the map/build identity.

**Why this priority**: it is how renderer changes are judged, and the operator asked for parity.

**Independent Test**: run the documented modern benchmark; receive a receipt that identifies the build,
the map, the camera path, and the frame-time distribution.

**Acceptance Scenarios**:

1. **Given** a loaded modern map with a bound camera path, **When** the benchmark is run, **Then** a
   receipt is produced with mean/p99 frame time, hitch count and submission counters.
2. **Given** the same command run twice on the same map, path and settings, **When** the receipts are
   compared, **Then** the methodology (frames sampled, warmup, counters) is identical and the numbers
   are comparable.
3. **Given** a legacy map, **When** the same benchmark is run, **Then** it uses the same procedure so
   the two eras are directly comparable.
4. **Given** a run on a modern map, **When** the receipt is read, **Then** it states the build, map,
   path identity and whether CDN fill was used.

### User Story 3 — Eras are comparable on equal footing (Priority: P2)

A user can answer "is the renderer slower on modern data, and by how much?" from two receipts rather
than from two different tools.

**Independent Test**: two receipts (one legacy, one modern) share the same fields and can be placed
side by side.

**Acceptance Scenarios**:

1. **Given** a legacy receipt and a modern receipt produced the same way, **When** compared, **Then**
   the frame-time and counter fields are the same shape and units.

## Requirements

- **FR-001**: The viewer MUST load camera tracks from M2 models resolved through the **modern** asset
  path (chunked `MD21` from a CASC install) into the same camera-path document used for legacy assets.
- **FR-002**: Before any change, the actual loss point MUST be identified and recorded with evidence:
  modern era dispatch, chunked conversion, or elsewhere. Assumed causes are not accepted as findings.
- **FR-003**: A modern model with cameras MUST expose those cameras to the overlay builder and to
  camera-path import; a modern model without cameras MUST be reported as such.
- **FR-004**: Camera coordinate space MUST be resolved (or its unresolved state reported explicitly);
  a path that is silently offset is a defect, not a limitation.
- **FR-005**: A path-driven renderer benchmark MUST be runnable on modern client data and produce a
  receipt containing: build, map, camera-path identity, warmup policy, frames sampled, frame-time
  mean and p99, hitch count, and the existing submission/draw-call counters.
- **FR-006**: The benchmark MUST use the same procedure and receipt fields for legacy and modern data
  so the eras can be compared directly.
- **FR-007**: The benchmark MUST be deterministic in methodology (same inputs ⇒ same procedure), even
  where absolute timings vary by machine.
- **FR-008**: The benchmark MUST NOT be gated on the test suite, and MUST NOT claim proof from
  compilation. Receipts are runtime artifacts and remain operator-owned runs.
- **FR-009**: New logic MUST live in an owned service per AGENTS.md §10; a new UI surface MUST register
  an inventory row per AGENTS.md §11 / Spec 223 FR-9.
- **FR-010**: Receipts MUST follow AGENTS.md §9.2 with files, exact commands, exit status and a
  criterion→evidence table; visual/FPS claims require the real run, not a build.

## Key Entities

- **Modern camera model**: a `MD21` model from a modern install whose document carries camera
  definitions.
- **Camera path**: the imported, playable path document (keys, timing, coordinate space, provenance).
- **Benchmark run**: a path-driven capture with a fixed warmup and sampling policy for one map/build.
- **Benchmark receipt**: the recorded identity, methodology and frame-time/counter results.

## Success Criteria

- **SC-001**: A modern camera-carrying M2 from the tested CASC install imports as a playable camera
  path, matching legacy behaviour (operator visual witness).
- **SC-002**: The loss point for modern cameras is documented with evidence before implementation.
- **SC-003**: A modern-map benchmark run produces a receipt with the fields in FR-005, on a real
  `wow_classic_beta` map.
- **SC-004**: A legacy-map run and a modern-map run produce same-shape receipts that can be compared
  side by side.
- **SC-005**: Re-running the same benchmark reproduces the same methodology fields.
- **SC-006**: Build and focused tests pass with no new failures.

## Assumptions & open questions

- The camera defect is assumed to be a plumbing loss (era dispatch / chunked conversion) rather than a
  format misunderstanding; FR-002 requires this to be proven before code.
- Benchmark comparability across machines is out of scope; the contract is same-shape receipts and a
  fixed procedure, with absolute numbers treated as machine-local.
- Several `MD21` assets on the tested install may not carry cameras at all; the spec requires reporting
  that rather than synthesising a path.
- The benchmark will be the measurement vehicle for [Spec 242](../242-wmo-instancing-performance/spec.md);
  whichever lands first, the other depends on it.