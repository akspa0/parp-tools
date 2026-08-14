# Active Context — wow-viewer

Last updated: 2026-08-14

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** Spec 148 Phase 1 audio truth surface: finish source/read/decode
  diagnostics, then have the user run a configured client and inspect current-tile MCSE rows.
- **Proof owner:** Focused audio/coordinate tests and cross-platform viewer build pass; the user owns
  real-client archive provenance, audible playback, and FPS/near-field visual proof.
- **Completed slice:** Spec 148 now has a provenance-first world-simulator spec/plan/tasks pack;
  MCSE emitters preserve raw and transformed positions; the runtime exposes non-playing diagnostic
  rows; the audio panel shows IDs, coordinates, path/source, decode/backend state, and terminal reason.
- **Main unproven gap:** The current slice does not yet prove exact `sounds.mpq` provenance, does not
  enumerate WMO-area triggers, and has not changed any MCSE coordinate transform or MIDI/DLS backend.
- **Explicitly out of scope for the next slice:** Camera model rendering, whole-map loading,
  renderer rewrite, shader reconstruction, fake audio conversion, and claims of audible/FPS gains.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| 148 Artifact world simulator runtime | Phase 1 diagnostics in progress; user gate open | Finish read/decode/source-stage coverage, then inspect a real client before actor/residency work. |
| 147 Minimap/fog/doodad instancing | Phase 2 implemented; Phase 3/4 open | User-run minimap proof, then implement fog coverage and structured batching diagnostics. |
| 146 Audio/camera playback | Area music resolution and master mute control implemented; playback gates open | User-run mute/unmute and DBC ZoneMusic proof; MIDI/DLS remains unsupported. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 143 World context and lighting | LIT source/fallback and pre-alpha v2 parser implemented with user gate | Validate the v2 viewer path, then continue WMO area and lighting evidence. |
| 142 World scene graph | In progress | User-run dense-WMO capture to compare internal-doodad batching against the prior placement-local path. |
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
