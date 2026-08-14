# Active Context — wow-viewer

Last updated: 2026-08-14

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

## Current handoff

- **Next implementation target:** Spec 143 Phase 4: validate the pre-alpha version-2 LIT profile
  in the viewer, then continue only with evidence-backed WMO/M2 lighting inputs.
- **Proof owner:** Focused LIT tests and inspect-tool archive proof are complete; the user owns the
  configured viewer runtime/visual lighting check against `H:\\053-client`.
- **Completed slice:** The shared reader now recognizes the observed negative-count v2
  `areatest.lit` shape: embedded Global Light header, legacy prefix, two nine-track data sets, and
  float bands. Modern LIT track validation remains strict; the alternate v2 set is retained for
  inspection and the primary set drives partial-light selection.
- **Main unproven gap:** The exact pre-alpha legacy prefix/secondary-set semantics remain
  unassigned, and viewer visual parity plus WMO/M2 light consumption are not proven.
- **Explicitly out of scope for the next slice:** WMOAreaID archaeology, shader reconstruction,
  whole-map loading, audio, and runtime FPS/visual claims beyond the user-run LIT check.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| 147 Minimap/fog/doodad instancing | Draft/planned | Review the plan, then implement the Phase 1 fullscreen minimap ownership and pure interaction contract. |
| 146 Audio/camera playback | Draft/planned | Parked while the active renderer baseline gate is validated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 143 World context and lighting | LIT source/fallback and pre-alpha v2 parser implemented with user gate | Validate the v2 viewer path, then continue WMO area and lighting evidence. |
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
