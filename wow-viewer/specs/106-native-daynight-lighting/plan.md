# Implementation Plan: Native Day/Night Lighting Fidelity

**Branch**: `106-native-daynight-lighting` | **Date**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Replace the viewer's rotating authored sun with build-scoped native world-light direction models. The first model is exact 0.5.3.3368 evidence: a constant 225-degree native light-ray azimuth with time-interpolated polar angles. Convert it through one evidence-bound native-to-viewer transform, then compose it with one coherent LIT or DBC color/fog/sky source. Keep the fixed 45-degree dynamic-shadow projection outside this pipeline. Client-exact capture fails closed until a user-owned native/viewer comparison proves the coordinate transform.

## Technical Context

**Language/Version**: C#/.NET 10; Python 3.11+ only for existing synthetic-store validation.

**Primary Dependencies**: `WowViewer.Core.Renderer`, shared LIT/DBC readers, active `WoWViewer` terrain renderer, `WowViewer.Tool.Capture`, existing Spec 103 synthetic-store validators.

**Storage**: Versioned in-repo profile definitions and JSON capture/comparison evidence; existing hash-bound PNG sidecars and Parquet synthetic indices.

**Testing**: Focused xUnit unit/contract tests, Capture/Viewer build, user-owned native/client image comparisons.

**Target Platform**: Windows desktop viewer and headless Capture tool.

**Project Type**: Shared renderer library plus desktop/headless consumers.

**Performance Goals**: Lighting evaluation is allocation-free per frame and does not add a GPU pass; time stepping remains interactive.

**Constraints**: One color source per profile; no duplicate readers; no guessed client-exact direction/transform/MCSH attenuation; user owns native capture and all heavy dataset runs.

**Scale/Scope**: Initial exact direction support is 0.5.3.3368 global outdoor lighting. LIT/DBC color paths and Spec 103 provenance are consumers, not reimplemented formats.

## Constitution Check

- **Repo independence**: Pass. All changes remain under `wow-viewer` and client roots stay configuration inputs.
- **Library first**: Pass. Direction model, coordinate transform, and profile contract belong in `WowViewer.Core.Renderer`; viewer and Capture only consume them.
- **Real-data validation**: Pass with a pending user-owned proof. Static exact-build/PDB evidence is recorded; native screenshot comparison is the remaining runtime signoff.
- **Residual/image-only contract**: Pass. Lighting provenance changes synthetic RGB generation only; it adds no deployment-time model input.
- **One phase at a time**: Pass. Phase 0 evidence and Phase 1 deterministic contract precede capture integration and empirical calibration.

## Project Structure

```text
src/core/WowViewer.Core.Renderer/Terrain/
  NativeWorldLightDirectionModel.cs
  NativeWorldLightCoordinateTransform.cs
  TerrainLightingProfile.cs
  LitTerrainDayNightProfile.cs

src/viewer/WoWViewer/Terrain/
  TerrainLighting.cs

tools/capture/WowViewer.Tool.Capture/
  Program.cs

tests/WowViewer.Core.Tests/
  NativeWorldLightDirectionModelTests.cs
  TerrainLightingProfileTests.cs

tests/WowViewer.Tool.Capture.Tests/
  LightingProvenanceContractTests.cs

specs/106-native-daynight-lighting/
  research.md
  data-model.md
  contracts/lighting-profile-and-capture.md
  quickstart.md
```

**Structure Decision**: The renderer library owns pure direction/profile evaluation; the viewer applies it each frame; Capture serializes it. Existing LIT/DBC readers and Spec 103 store code retain their present ownership.

## Phase 0 — Evidence lock (complete)

1. Record exact 0.5.3 PDB/live-client evidence, including the vector formula, table values, and the distinction between world light and shadow projection.
2. Correct the architecture research document so future work cannot relitigate the false fixed-45-degree world sun.

**Proof**: [research.md](research.md) and the architecture document identify `DayNightUpdateLighting`, `SetColors`, and `SetDirection` separately and preserve the captured vector.

## Phase 1 — Deterministic world-light contract

1. Add an immutable build-scoped direction-model contract with normalized-time validation, periodic table interpolation, native light-ray semantics, and source-direction conversion.
2. Encode the exact 0.5.3.3368 theta/phi model from the recorded evidence; do not derive it from color data.
3. Add a versioned native-to-viewer coordinate-transform contract that is explicitly `unproven` until calibration evidence exists.
4. Replace `LitTerrainDayNightProfile`'s dependency on `AuthoredTerrainDayNightProfile` for direction with a supplied direction profile; retain authored fallback only with its existing authored label.
5. Make the interactive `TerrainLighting` consume the same profile contract instead of generating its own orbital direction when an exact profile is selected.
6. Add focused tests for interpolation, wrapping, ray/source inversion, non-finite input rejection, transform application, and separation from fixed shadow projection.

**Exit criteria**: Exact 0.5.3 direction evaluates deterministically for declared times; uncalibrated profiles cannot claim client-exact output; existing authored profiles remain visibly authored.

## Phase 2 — Coherent profile and capture provenance

1. Introduce a `TerrainLightingProfile` that joins one direction model, one transform, one color/fog/sky source, MCSH evidence, and revision identifiers.
2. Make Capture accept a named/serialized profile rather than separate loosely-coupled `--lighting-source` and authored direction behavior.
3. Extend the v2 capture sidecar with direction-model revision, transform revision/state, native and viewer vectors, and explicit shadow-system identity.
4. Reject client-exact Capture requests when the direction model, transform, color source, or provenance hash is absent, stale, mixed, or non-finite.
5. Update the Spec 103 synthetic store contract so client-exact rows require those fields and source-group split rules remain enforced.
6. Add sidecar/store contract tests for accept and reject paths, including a double-lit input.

**Exit criteria**: A client-exact profile cannot be constructed accidentally from LIT/DBC colors plus the authored sun; provenance is sufficient to audit every generated RGB row.

## Phase 3 — Controlled calibration and native comparison (user-owned)

1. Prepare a single declared-time capture manifest: exact client build, map/tile, camera/top-down framing, time, LIT/DBC source, expected native vector, and output hashes.
2. The user captures the native game and viewer/Capture images for a terrain tile with directional relief and MCSH.
3. Compare orientation and image/color statistics; choose the one fixed native-to-viewer axis/sign transform that matches the lock time.
4. The user captures at two held-out times; validate that the locked transform predicts their orientations without per-time adjustment.
5. Store comparison artifacts and promote the transform from `unproven` to `calibrated` only if all three comparisons pass.
6. Separately measure MCSH attenuation and sky-band placement; do not block direction promotion on those measurements, but do not label either client-exact until independently proven.

**Exit criteria**: 0.5.3 global outdoors has a calibrated transform or remains deliberately unavailable as client-exact. No image capture or data harvest is run by the agent.

## Phase 4 — Dataset integration and regression protection

1. Generate a small user-owned clean-synthetic source group at declared times using the calibrated profile.
2. Validate sidecars, source-group split assignment, rights class, source hashes, and prohibition of re-lighting captured minimaps.
3. Add a regression matrix covering authored fallback, LIT+exact-direction, DBC+exact-direction, and rejected uncalibrated/mixed profiles.
4. Update Spec 103 T040 from an open-ended calibration task to concrete evidence references and only retain unresolved MCSH/sky/local-zone work.
5. Update memory-bank context/progress and the architecture research record with the calibration result.

**Exit criteria**: The synthetic path can represent time diversity truthfully, and no model/training claim is made until the user separately runs the authorized corpus/training work.

## Deferred, explicitly out of scope

- Recovering direction models for builds other than 0.5.3.3368.
- Local-zone LIT/Light* spatial blending until its coordinate transform is proven. The interactive
  viewer keeps exact local Light* evaluation available for diagnostics, but does not apply that
  overlay to terrain by default; the global viewer light is the identity case.
- Five-sky-band altitude thresholds and native MCSH attenuation until captured comparison evidence exists.
- WMO interior lighting, dynamic local lights, liquid shading, unit-shadow implementation, and any training run.

## Bounded runtime correction — 2026-08-11

- A 4.0.0.11927 live probe showed local Light* values can be strongly orange/dim at noon. Because
  the local-zone spatial contract is not yet proven, `WorldScene` now leaves that overlay opt-in
  instead of allowing it to darken ordinary outdoor terrain.
- MH2O liquid families are resolved from the exact-build `LiquidType` DBC records. The loader uses
  the table's actual ID field and DBC row names/class fields; numeric family guesses are not used
  for the active runtime path. A missing DBC row follows the documented safe water default.
- Focused liquid/lighting tests and the viewer build are the validation gate for this slice; a
  user-run real-scene screenshot remains required for visual signoff.

## Complexity Tracking

No constitution violations. This adds small immutable contracts to an existing renderer library rather than a new subsystem or parser.
