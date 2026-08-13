# Active Context — wow-viewer

Last updated: 2026-08-12

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** Spec 142 Phase 8P user-run validation of WMO near-field residency ordering.
- **Proof owner:** focused C# tests and a Debug build for source-level contracts; the user owns real
  client, OpenGL, audio-device, capture, and performance proof.
- **Main unproven gap:** source build proves selected GPU uploads and immediate-neighbor WMO prioritization compile, but no user-run movement test has yet confirmed that WMO flash-in is gone.
- **Explicitly out of scope for the next slice:** FOV radiation, whole-window object submission, unique-model submission redesign, broad renderer rewrites, audio, and FPS claims without production capture evidence.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| 146 Audio/camera playback | Draft/planned | Parked while the active renderer baseline gate is validated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window and placeholder audit tasks. |
| 143 World context and lighting | First area slice landed | Resolve evidence/setup tasks before WMO area or lighting claims. |
| 142 World scene graph | In progress | Validate Phase 8P near-field WMO readiness ordering, then continue retained-radius movement proof. |
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
