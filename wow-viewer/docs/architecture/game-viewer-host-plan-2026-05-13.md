# game-viewer Host Plan

Superseded as the top-level direction by:

- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`

This file remains active as the **app-host and viewer migration sub-plan** inside the broader engine program.

## Status

- status: active
- date: 2026-05-13
- owner: `wow-viewer`
- intent: port legacy `MdxViewer` capability into a standalone `game-viewer` host backed by `wow-viewer`, `WowViewer.Core`, `WowViewer.Core.IO`, and `WowViewer.Core.Runtime`

## Goal

Deliver a `wow-viewer`-backed app-host product surface named **game-viewer** with the same practical world-viewing functionality as legacy `MdxViewer`, while keeping all new ownership in `wow-viewer` and treating `gillijimproject_refactor` as read-only reference plus compatibility evidence.

This plan no longer defines the entire program. It defines the viewer/app slice of the larger engine direction.

The editor/import-export shell direction is now also constrained by:

- `wow-viewer/docs/architecture/wow-engine-editor-and-interop-plan-2026-05-14.md`
- `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/README.md`

## Reference Behavior Target

- Primary runtime reference is the native `0.5.3` client behavior.
- Feature direction is **0.5.3-first correctness**, then multi-era compatibility (`0.5.3` through `4.0.0`) on top of that baseline.
- Performance direction is **engine-style LOD and visibility discipline**, not brute-force scene submission.
- Reverse-engineering evidence should use current x64dbg/Ghidra workflows when behavior conflicts are discovered.

## Hard Boundaries

1. Do not add new viewer architecture to `gillijimproject_refactor/src/MdxViewer`.
2. Keep file-format ownership in shared libraries (`WowViewer.Core` and `WowViewer.Core.IO`).
3. Keep runtime ownership in `WowViewer.Core.Runtime`.
4. Use staged client roots under `output/tmp/wowarchive-clients/` for real-data validation.
5. Land one bounded slice at a time with explicit proof and explicit remaining boundary.

## Scope Matrix (High-Level)

- shell + workspace UX owner: `wow-viewer/src/viewer/WowViewer.App`
- game/data root manager owner: `wow-viewer/src/viewer/WowViewer.App` + future shared profile services
- world bootstrap and map/session opening owner: `wow-viewer/src/viewer/WowViewer.App` + `WowViewer.Core.IO`
- world runtime frame composition owner: `wow-viewer/src/core/WowViewer.Core.Runtime/World`
- model runtime consumption owner: `wow-viewer/src/core/WowViewer.Core.Runtime/M2` and `.../Mdx`
- compatibility validation harness only: `gillijimproject_refactor/src/MdxViewer`

## Ordered Slices

### Slice 0 — Product Branding Surface (Start Here)

**Intent**
- Introduce `game-viewer` as the user-facing app identity without changing architectural ownership seams.

**In Scope**
- User-visible app title strings in `WowViewer.App` desktop and capture windows.
- CLI banner/log prefixes/help text in `WowViewer.App` command outputs.
- New migration plan doc (this file) and bounded status notes.

**Out of Scope**
- No namespace/package/project renames.
- No runtime behavior changes.
- No feature-parity claims.

**Proof**
- `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug`

### Slice 1 — Explicit Legacy-to-New Parity Matrix

**Intent**
- Build a deterministic feature matrix from legacy `ViewerApp_*` surfaces to `wow-viewer` workspaces/panels/commands.

**Proof**
- Matrix committed in `wow-viewer/docs/architecture/` with each row marked: done / partial / missing / intentionally dropped.

### Slice 2 — Game Manager And Import/Export Shell

**Intent**
- Make the app start as a managed data-root and asset-interop tool, not only as a world-view shell.

**Must Include**
- Managed game/data roots
- active-root switching
- capability/profile summary
- first import/export/conversion entrypoints

**Proof**
- app shell can register/select a root and launch at least one bounded import/export workflow from the managed root context

### Slice 3 — World Session Closure Pass

**Intent**
- Close highest-value world-session parity gaps in `wow-viewer` runtime + app consumer path.

**Must Include**
- Terrain and object visibility policy aligned to `0.5.3` reference behavior.
- LOD/range/priority controls that reduce overdraw and unnecessary submissions.

**Proof**
- Real-data `world-frame` runs with staged clients plus deterministic output summaries/captures.

### Slice 4 — Terrain/Liquid Shader Baseline

**Intent**
- Replace flat preview-style world composition with a real shader baseline for terrain and liquids.

**Must Include**
- Dedicated terrain shader path (layering, alpha, and lighting-ready inputs).
- Dedicated liquid shader path (surface type-aware rendering baseline).
- Explicit performance instrumentation for pass cost before/after shader integration.

**Proof**
- Real-data captures and runtime counters showing shader path active and stable on staged `0.5.3` plus one later-era map.

### Slice 5 — Skybox And Lighting Parity Baseline

**Intent**
- Land real skybox rendering and lighting behavior that matches `0.5.3` expectations as the baseline.

**Must Include**
- Runtime skybox consumer path in `wow-viewer` app/world renderer.
- World-lighting baseline that can be extended for non-`0.5.3` data.
- Clear seam between baseline lighting and later-era compatibility extensions.

**Proof**
- Bounded capture workflow proving skybox + lighting active in world session, with reproducible settings.

### Slice 6 — Standalone Asset Consumer Closure

**Intent**
- Raise M2/MDX/WMO standalone workspaces to practical parity targets needed for normal inspection workflows.

**Must Include**
- explicit renderer layers per supported asset family
- compatibility-aware asset workspaces
- early Warcraft 3-compatible model/archive preview boundary where profile support exists

**Proof**
- Focused CLI capture paths (`m2-gpu-frame`, `mdx-gpu-frame`) and desktop workspace acceptance notes.

### Slice 7 — DBC/DB2 Editor Baseline

**Intent**
- Move the app toward real metadata-table ownership rather than read-only lookup support.

**Proof**
- one bounded schema-driven DBC/DB2 edit workflow on a supported active profile

### Slice 8 — Legacy Utility Triage

**Intent**
- Classify legacy editor/utility surfaces into: migrate now, migrate later, keep external, or retire.

**Proof**
- Documented triage with bounded follow-up slices.

### Slice 9 — Decoupling Checkpoint

**Intent**
- Reach a point where routine viewer validation no longer depends on launching legacy `MdxViewer`.

**Proof**
- `game-viewer`-first validation workflow documented and runnable from `wow-viewer` only.

## Validation Language Rule

- Build/test/CLI proof in `wow-viewer` is primary.
- Legacy `MdxViewer` runtime checks are compatibility evidence, not ownership evidence.
- Avoid claiming full parity until the parity matrix rows are closed.
