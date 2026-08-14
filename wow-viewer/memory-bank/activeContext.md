# Active Context — wow-viewer

Last updated: 2026-08-13

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** Spec 142 Phase 8M/8O: user-run camera movement proof that the
  nearest terrain footprint remains selected, resident, and rendered while the detail and retained
  radius controls change.
- **Proof owner:** focused selector tests and an isolated viewer build prove source contracts; the
  user owns real-client movement, camera-path, visual, and FPS proof.
- **Completed slice:** `DirectionalTileSelector` now protects the largest complete camera-centered
  square supported by the detail budget before forward-cone expansion: 3×3 for 9–24 tiles and 5×5
  at 25. The regression suite covers the 25-tile near-field case.
- **Main unproven gap:** the user must confirm that nearby ADTs no longer unload or disappear in the
  viewer. If holes remain after this selector correction, inspect detailed terrain bounds/fog culling
  and renderer-space tile mapping separately from admission.
- **Explicitly out of scope for the next slice:** FOV radiation redesign, whole-map loading, new
  scene-graph traversal, audio playback work, and FPS claims from source tests alone.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| 146 Audio/camera playback | Draft/planned | Parked while the active renderer baseline gate is validated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 143 World context and lighting | First area slice landed | Resolve evidence/setup tasks before WMO area or lighting claims. |
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
