# Implementation Plan: M2 Render Parity Recovery

**Branch**: `035-m2-render-parity-recovery` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `wow-viewer/specs/035-m2-render-parity-recovery/spec.md`

## Summary

Recover stable world M2 rendering parity for 3.3.5-era content (especially trees and cutout/transparent doodads) by introducing a deterministic route contract, explicit material-pass classification evidence, and a bounded compatibility-first refactor path in `wow-viewer`.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, `Warcraft.NET` (reference parsing source)

**Storage**: Filesystem logs/evidence under `wow-viewer/output/tmp/`

**Testing**: `dotnet build`, existing probe commands (`--probe-m2-adapter`, `--probe-m2-runtime`), targeted viewer runtime checks

**Target Platform**: Windows desktop runtime with staged WoW client data

**Project Type**: Desktop viewer + shared runtime libraries

**Performance Goals**: No visible regression in world frame-time behavior for equivalent visibility sets during parity runs

**Constraints**:
- Must keep ownership in `wow-viewer` surfaces
- Must use staged clients only (`output/tmp/wowarchive-clients/`)
- No parser rewrites of already-complete client file tooling

**Scale/Scope**:
- Initial closure scope: 3.3.5 world M2 parity samples
- Primary files: `WorldAssetManager`, `WmoRenderer`, `M2Renderer`, `ModelRenderer`, route/probe diagnostics

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo Independence**: PASS — all planned changes stay inside `wow-viewer/`.
- **Library-First**: PASS — route contract and diagnostics are shared/runtime-owned, not tool-local forks.
- **Real-Data Validation**: PASS — parity proof uses staged client roots and concrete model samples.
- **No Untrusted Client Paths**: PASS — only staged client roots are in scope.
- **Phase Discipline**: PASS — recovery is decomposed into bounded phases with explicit proof gates.

## Project Structure

### Documentation (this feature)

```text
specs/035-m2-render-parity-recovery/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── README.md
│   ├── m2-route-decision.schema.json
│   └── m2-material-pass-profile.schema.json
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
src/viewer/WoWViewer/
├── Rendering/
│   ├── M2Renderer.cs
│   ├── ModelRenderer.cs
│   ├── WarcraftNetM2Adapter.cs
│   └── WmoRenderer.cs
├── Terrain/
│   ├── WorldAssetManager.cs
│   └── WorldScene.cs
└── AssetProbe.cs

src/core/WowViewer.Core.Runtime/
└── World/
   └── Passes/
```

**Structure Decision**: Single-project viewer/runtime refactor within existing `wow-viewer` solution layout.

## Phase Plan

### Phase 0: Route Contract & Evidence Design

- Define canonical world M2 route decision model.
- Define material-pass profile evidence schema.
- Define parity sample set and evidence naming contract.

### Phase 1: Deterministic Routing Refactor

- Centralize M2 route selection and fallback reason capture.
- Ensure probe output and runtime route use same decision source.
- Preserve compatibility fallback while making fallback explicit.

### Phase 2: Material/Pass Parity Refactor

- Normalize cutout/transparent/opaque world pass classification.
- Eliminate route-dependent pass drift between adapter/runtime paths.
- Add diagnostic output for per-layer pass semantics.

### Phase 3: Parity Validation and Guardrails

- Execute parity sample runs on staged 3.3.5 content.
- Record evidence bundles for probe/runtime comparisons.
- Add regression guard checks that fail on route drift.

### Phase 4: Documentation and Handoff

- Update M2 architecture notes with recovered contract.
- Sync spec/task evidence and known limits.
- Publish operator quickstart and parity checklist.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| None | N/A | N/A |
