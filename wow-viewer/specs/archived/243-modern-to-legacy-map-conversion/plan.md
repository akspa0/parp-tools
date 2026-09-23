# Implementation Plan: Modern-to-Legacy Map Conversion (multi-layer alpha merge, LK + Alpha outputs)

**Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch) | **Date**: 2026-09-18 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `wow-viewer/specs/archived/243-modern-to-legacy-map-conversion/spec.md`

## Summary

Add a one-way **modern → legacy** conversion route to the existing Map Converter: read a modern
FileDataID-era map through the modern readers (Specs 238/239/240), merge its multi-layer chunks
(up to 8 layers + `AMAP` weights) into the target's layer model, and write **LK v18 ADT/WDT** and
**Alpha 0.5.3 monolithic WDT**. The route is batch-capable, low-touch (direction + target + input
only), writes under a generated project folder with provenance, and optionally carries referenced
assets with a manifest. All new logic lives in an owned service (AGENTS.md §10); the existing
Alpha↔LK converter commands are untouched.

## Technical Context

**Language/Version**: C# / .NET 10 (repo `net10.0`), PowerShell 7 for operator commands.

**Primary Dependencies**: existing `WowViewer.Core.IO` writers (`LkAdtWriter`, `AlphaWdtWriter`,
`LkToAlphaConverter`, `AlphaToLkConverter`), `MapConversionFormats` (source/target enums +
validation), the modern readers from Specs 238/239/240, and the viewer's `IDataSource`/CASC path.

**Storage**: loose files only — LK v18 ADT/WDT and Alpha 0.5.3 WDT under a generated project output
folder. No MPQ/CASC is ever written (Constitution VII).

**Testing**: xUnit in `wow-viewer/tests/WowViewer.Core.Tests` (unit + round-trip), plus an
operator-owned real-client load witness per target (Constitution III).

**Target Platform**: Windows desktop viewer + CLI (`wowviewer-converter` / harvest tool).

**Project Type**: library-first (Constitution II) — one owned conversion service in
`WowViewer.Core.IO`, thin CLI + Editor surfaces.

**Performance Goals**: a full modern map converts in one run with no per-tile manual step; batch of
≥3 maps completes end-to-end. No hard latency target; determinism (byte-identical reruns) is required.

**Constraints**: deterministic output (SC-004); never overwrite client data (FR-006); unsupported
routes surfaced before writing (FR-008); no new `ViewerApp`/`WorldScene` members (AGENTS.md §10).

**Scale/Scope**: one map to a whole continent per run; two targets (LK v18, Alpha 0.5.3); optional
asset inclusion.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|---|---|---|
| I. Repo Independence | All new code under `wow-viewer/`; no external `.csproj` refs. | PASS |
| II. Library-First | Conversion logic in `WowViewer.Core.IO`; CLI/Editor are thin wrappers. | PASS |
| III. Real-Data Validation | Receipts require a real modern map converted to each target and loaded in the viewer. | PASS (planned) |
| IV. Model Architecture | N/A (no ML model in this feature). | PASS |
| V. Streaming-First Dataset | N/A (no dataset pipeline change). | PASS |
| VI. No Client Path Assumptions | Client root is configuration; no hardcoded path. | PASS |
| VII. Containers Are Inputs | Writes loose ADT/WDT only; never MPQ/CASC. | PASS |
| AGENTS.md §10 (god-class freeze) | New logic in an owned service; no new `ViewerApp`/`WorldScene` members. | PASS (planned) |
| AGENTS.md §11 (UI inventory) | New UI surface registers an inventory row (Spec 223 FR-9). | PASS (planned) |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```text
specs/archived/243-modern-to-legacy-map-conversion/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.IO/Maps/
├── ModernToLegacyMapConversionService.cs   # NEW owned service (orchestration, batch, provenance)
├── ModernLayerMergePolicy.cs               # NEW deterministic layer-stack merge
├── ConversionAssetManifest.cs              # NEW asset inclusion + manifest
├── MapConversionFormat.cs                  # EXTEND: add ModernFileDataId source format
├── LkAdtWriter.cs                          # reuse (LK v18 target)
├── AlphaWdtWriter.cs                       # reuse (Alpha 0.5.3 target)
└── LkToAlphaConverter.cs / AlphaToLkConverter.cs  # reuse; untouched

wow-viewer/src/viewer/WoWViewer/
├── ViewerApp_MapConverter.cs               # EXTEND existing dialog (no new members beyond the
│                                           #   existing converter state; delegate to the service)
└── (UI inventory row registered per Spec 223 FR-9)

wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs
└── EXTEND: `convert-map` command (batch driver) — thin wrapper over the service

wow-viewer/tests/WowViewer.Core.Tests/
├── ModernLayerMergePolicyTests.cs          # NEW
├── ModernToLegacyMapConversionServiceTests.cs  # NEW (round-trip + determinism)
└── MapConversionFormatTests.cs             # EXTEND (modern source validation)
```

**Structure Decision**: single library-first project. The conversion service is the one canonical
owner of the modern→legacy route; the CLI and the Editor dialog are thin surfaces over it. The
existing Alpha↔LK converter commands and writers are reused unchanged.

## Phased Implementation Breakdown

### Phase 0 — Research & route validation (no code)
- Confirm the modern reader surface (Specs 238/239/240) exposes per-chunk layer stacks + `AMAP`
  weights and the FileDataID→path resolution needed for texture references.
- Decide the merge policy shape (see `research.md`): how N modern layers map onto LK v18's layer
  capacity and Alpha 0.5.3's 4-layer model, and how alpha masks combine.
- Confirm the LK v18 and Alpha 0.5.3 writer entry points and their layer/alpha contracts.
- Resolve the open question: one owned service surfaced in both CLI and Editor (default assumption).

### Phase 1 — Design & contracts
- `data-model.md`: SourceMap, LayerStack, MergePolicy, ConversionRun, AssetManifest.
- `contracts/`: the service API + the CLI command contract + the per-tile report schema.
- `quickstart.md`: the operator command for a real map → both targets.

### Phase 2 — Core service (owned, no UI)
- `ModernLayerMergePolicy`: deterministic merge of a modern layer stack onto a target capacity,
  combining texture ids and alpha masks; emits a per-tile merge record.
- `ModernToLegacyMapConversionService`: source detection, route validation (FR-008), per-map
  conversion, batch isolation (FR-004), provenance + project output folder (FR-006).
- Reuse `LkAdtWriter` / `AlphaWdtWriter` for the two targets.

### Phase 3 — Asset inclusion
- `ConversionAssetManifest`: resolve referenced textures/models/minimaps through the modern readers;
  copy resolvable assets beside the output; record unresolved ones with reasons (FR-007).

### Phase 4 — Surfaces
- CLI `convert-map` batch driver (thin wrapper).
- Editor Map Converter dialog: add the modern source direction + target selection, delegate to the
  service, show the per-tile report; register the UI inventory row (Spec 223 FR-9).

### Phase 5 — Validation & receipts
- Unit + round-trip + determinism tests.
- Operator-owned: convert one real modern map to LK v18 and to Alpha 0.5.3, load each in the viewer
  (SC-002), and a ≥3-map batch with one broken input (SC-003). Receipt per AGENTS.md §9.2.

## Complexity Tracking

> No Constitution Check violations. Section intentionally empty.
