# Quickstart: Renderer Improvements Convergence

## Purpose

Use this feature pack as the active owner plan for renderer modernization work that previously lived in specs 030, 031, and 032.

## Read Order

1. Read [spec.md](./spec.md)
2. Read [plan.md](./plan.md)
3. Read [research.md](./research.md)
4. Use specs `030`, `031`, and `032` only as source-slice references after the convergence documents

## Current Boundaries

- `036` is the active owner plan for terrain/WMO/lighting/sky/fog/liquid/viewer integration convergence
- `035` remains the active owner for M2 parity recovery
- `gillijimproject_refactor` remains read-only reference only

## Validation Roots

- Trusted client root prefix: `I:/parp/parp-tools/output/tmp/wowarchive-clients/`
- Primary solution build:

```powershell
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## Suggested Phase Proof Surfaces

### Phase 1

- Lighting-state evaluation snapshots at multiple times of day
- Sky/fog parameter dumps sourced from staged clients

### Phase 2

- Terrain-cell counts, face-plane counts, hole-mask checks
- Known-position world-to-cell addressing checks

### Phase 3

- Interior/exterior WMO sample renders
- Batch-flag and lightmap-pass diagnostics

### Phase 4

- Close/mid/far terrain comparisons
- Water and magma routing comparisons
- Shadow overlay and fog consistency checks

### Phase 5

- Viewer time-of-day controls
- Runtime diagnostics surfaced in the app host
- Screenshot/capture proof runs for renderer review

## Evidence Convention

- Keep evidence under feature-scoped output roots when execution begins, for example:

```text
wow-viewer/output/tmp/renderer-improvements/
  phase-1/
  phase-2/
  phase-3/
  phase-4/
  phase-5/
```

## Next Step

Generate `tasks.md` from this convergence plan once the user wants to start implementation slicing.
