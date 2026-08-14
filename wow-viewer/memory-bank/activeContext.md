# Active Context — wow-viewer

Last updated: 2026-08-14

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** Spec 147 Phase 1: make the fullscreen minimap a single-owned
  surface and implement/test deterministic drag versus same-target triple-click teleport behavior.
- **Proof owner:** Spec 147 focused interaction tests and a Debug build first; the user owns the
  fullscreen drag/teleport runtime check, then the fog-residency and dense-doodad captures.
- **Completed slice:** Spec 147 planning artifacts define minimap interaction, active-fog-bounded
  tile/object coverage, compatibility-aware doodad instancing, diagnostics, tasks, and proof gates.
  The read-only audit identified duplicate fullscreen draw ownership as the first concrete input
  risk.
- **Main unproven gap:** no production code has changed for Spec 147. `TerrainManager` still
  discards its supplied `fogEnd` for streaming targets, and dense WMO-internal doodad batching is
  still placement-local until its phases are implemented and tested.
- **Explicitly out of scope for the next slice:** WDL/sky/stars, audio, shader reconstruction,
  format readers, whole-map loading, and runtime FPS/visual claims from source tests alone.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| 147 Minimap/fog/doodad instancing | Draft/planned | Review the plan, then implement the Phase 1 fullscreen minimap ownership and pure interaction contract. |
| 146 Audio/camera playback | Draft/planned | Parked while the active renderer baseline gate is validated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 143 World context and lighting | LIT source/fallback slice implemented with user gate | Validate variant switching and no-Light-DBC fallback, then continue WMO area and lighting evidence. |
| 142 World scene graph | In progress | Validate the Phase 8M near-field selector correction, then continue retained-radius and WMO movement proof. |
| 139–141 Terrain/minimap reconstruction | Active/parked ML lanes | Reopen only for the named spec and user-run training/validation. |
| 138 Cross-era renderer research | Evidence/planning | Do not generalize one client build to every era. |
| 128–131 PM4 | Established research lane | Use the PM4 spec pack and `workstream-pm4-decode.md`. |

## Stable boundaries

- New code, tests, tools, and viewer docs go in `wow-viewer/`; the legacy tree is read-only
  reference unless a bounded compatibility fix is explicitly requested.
- Keep format readers library-first and tools thin. Do not duplicate or rewrite working client-file
  readers. Keep the Alpha/standard terrain split. `AlphaWdtWriter.cs` is frozen unless explicitly
  reopened with focused proof.
- Client roots are runtime configuration. `H:\CLIENTS` is approved; never hardcode a local client
  path. Record root, build identity, and fingerprint for client-backed proof.
- Training, GPU work, broad harvests, long captures, and real-client/runtime testing are user-run.
- The default source proof is:
  `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  and
  `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Continuity rules

- Update this dashboard and `progress.md` only when the implementation handoff changes.
- Put durable technical findings in the owning workstream or architecture note, not here.
- Preserve negative results and open gates, but remove superseded narrative from the default path.
- End every handoff with: current target, proof owner, completed slice, unproven gap, next bounded
  action, and explicit out-of-scope items.
