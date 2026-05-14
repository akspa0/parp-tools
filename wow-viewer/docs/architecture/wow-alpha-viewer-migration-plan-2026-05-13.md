# WoWAlphaViewer Migration Plan

## Status

- status: active
- date: 2026-05-13
- owner: `wow-viewer`
- strategy: strict layer-by-layer port from legacy `MdxViewer` behavior into a clean `WoWAlphaViewer` implementation

## Outcome Target

Build an open-source, durable, modern viewer implementation that:

1. matches `0.5.3` engine behavior as the reference baseline,
2. supports multi-era data (`0.5.3` through `4.0.0`) through explicit compatibility layers,
3. includes first-class GUI workflow for conversion tooling,
4. avoids the monolithic, mixed-responsibility architecture that made legacy maintenance brittle.

## Non-Negotiable Rules

1. `gillijimproject_refactor` remains read-only reference input.
2. Every migrated behavior must be assigned to one architecture layer.
3. No feature may bypass layer boundaries for convenience.
4. Each layer requires proof before the next layer begins.
5. Performance policy must follow LOD/visibility discipline, not brute-force submission.

## Architecture Layers

### Layer 0 — Foundation Contracts (current executable slice)

Owner surface: `wow-viewer/src/viewer/WowViewer.App`

Scope:
- app composition contracts
- module registration and lifecycle contracts
- diagnostics/reporting contracts for startup readiness

Proof:
- app builds cleanly
- CLI can report registered layers and readiness

### Layer 1 — Host Shell + Session State

Owner surface: `wow-viewer/src/viewer/WowViewer.App`

Scope:
- deterministic app host boot path
- workspace/session state persistence
- command routing and user-safe failure boundaries

Proof:
- session bootstrap and reload reproducible in CLI/desktop host

### Layer 2 — World Session Bootstrap

Owner surface: `WowViewer.App` + `WowViewer.Core.IO`

Scope:
- attach staged client root
- map resolution and WDT bootstrap
- tile-selection entry path with explicit diagnostics

Proof:
- staged `0.5.3` and one later-era map bootstrap with deterministic logs

### Layer 3 — World Runtime Data Graph

Owner surface: `wow-viewer/src/core/WowViewer.Core.Runtime/World`

Scope:
- terrain/object/liquid/sky data pipelines
- frame-stage contracts
- pass ownership and metrics

Proof:
- bounded runtime frame output with stable stage counters

### Layer 4 — Visibility, Culling, and LOD Discipline

Owner surface: `WowViewer.Core.Runtime`

Scope:
- `0.5.3`-style visibility and LOD policy baseline
- object family range gates
- budget-aware scheduling hooks

Proof:
- measurable reduction in submitted counts with same camera scenes

### Layer 5 — Terrain + Liquid Shader Baseline

Owner surface: `WowViewer.App` renderer consumer + runtime seam contracts

Scope:
- terrain shader path
- liquid shader path
- pass cost instrumentation

Proof:
- shader-active captures and pass counters on staged real data

### Layer 6 — Skybox + Lighting Baseline

Owner surface: runtime + app consumer

Scope:
- real skybox rendering
- baseline world lighting matched to `0.5.3` expectations
- compatibility seam for later-era lighting differences

Proof:
- reproducible world captures proving skybox and lighting path active

### Layer 7 — Model Consumers (M2/MDX/WMO)

Owner surface: `WowViewer.Core.Runtime` + app consumers

Scope:
- standalone and world-integrated model consumer parity
- pass routing and correctness checks

Proof:
- deterministic model/world captures and runtime summaries

### Layer 8 — Converter UX Integration

Owner surface: `WoWAlphaViewer` GUI + converter tool adapters

Scope:
- bulletproof, user-friendly conversion workflows in GUI
- safe defaults, explicit provenance, actionable errors

Proof:
- guided conversion flows validated on staged clients without manual command reconstruction

### Layer 9 — Parity Closure + Legacy Decoupling

Owner surface: docs + validation workflow

Scope:
- close parity matrix rows
- remove legacy runtime dependency for normal verification

Proof:
- `WoWAlphaViewer`-first validation workflow documented and sufficient

## Reference Behavior and Research Policy

- reference runtime behavior: native `0.5.3` client
- reverse-engineering support: x64dbg/Ghidra evidence when behavior is ambiguous
- later-era support policy: additive compatibility, never baseline drift

## Parity Matrix Requirement

Feature porting must be tracked in the companion matrix file:

- `wow-viewer/docs/architecture/wow-alpha-viewer-parity-matrix-template-2026-05-13.md`

No layer is complete until all scoped matrix rows are proven.

## Validation Language Rule

- `wow-viewer` build/test/runtime proof is primary.
- legacy `MdxViewer` runs are compatibility evidence only.
- no parity claims without matrix closure and proof artifacts.
