# Quickstart: PM4-Guided Museum Placement Repair

This is the planned validation path. It does not modify a game install, PM4 file, or source Museum map.
Use a configured client root and a separate output directory. The example paths are placeholders; do not
hardcode a machine-local client path into source or portable configuration.

## Prerequisites

- The editor host/bridge/session dependencies from Specs 166–168 are available.
- The placement-authoring and asset-integrity dependencies from Specs 173 and 175 are available.
- A known PM4/Museum pair with the correctly named `_obj0.adt` companion.
- A configured client build containing the model/WMO corpus and a recorded build fingerprint.
- An output directory outside the client installation.

## Source validation

From PowerShell 7 at the repository root:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --no-restore
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore
```

The focused tests must cover canonical PM4 coordinate conversion, candidate status/ambiguity,
placement transaction identity checks, ID allocation, name-table remapping, and atomic failure. Existing
baseline failures must be recorded rather than attributed to this feature.

## Planned operator flow

1. Configure the client root, build fingerprint, Museum source map, PM4 guide directory, and an empty
   output directory in the viewer.
2. Load the Museum map and the matching PM4 guide. Confirm the PM4 companion path is the intended
   unpadded `_obj0.adt`; missing companions are reported instead of falling back to another tile.
3. Open the PM4/Museum reconciliation panel and run **Preview**. The viewport overlays current Museum
   placements, PM4 guide geometry, and proposed corrected transforms without staging a file write.
4. Inspect proposals individually or as a filtered group. For each proposal, review action (`Align`,
   `Substitute`, or `Clone`), current/proposed transform, residuals, score breakdown, evidence, source
   identity, and status.
5. Accept only proposals that are visually and evidentially correct. Reject false candidates. Leave
   ambiguous/unresolved proposals untouched or choose a different explicitly identified candidate.
6. Run **Validate batch**. Resolve all reported tile-bound, era, source-hash, ID, name-table, and output
   path failures before applying.
7. Run **Apply**. The viewer writes loose ADT/WDT output files and a provenance sidecar; it never writes
   to the source map, PM4 guide, or game-client container.
8. Reload the output in a fresh viewer session and inspect the corrected placements with PM4 overlay still
   enabled. Compare the read-back placements and hashes to the report.
9. Use **Undo** before any unrelated output edit, then confirm the pre-operation placement/name-table
   state is restored. Test **Redo** if the editor session exposes it.

## Evidence record for a real-client checkpoint

Record the following in the owning spec's research or validation note:

- Configured client root, exact build identity, and content fingerprint.
- PM4 guide paths and hashes; Museum ADT/WDT source paths and hashes.
- Output directory and every written output path/hash.
- Counts of guide objects, existing placements, align/substitute/clone proposals, accepted/rejected/
  ambiguous/unresolved results.
- One example of each accepted action, including before/after transforms and evidence.
- One intentionally ambiguous case that produced no mutation.
- Independent-tool read-back result and user-owned visual result. Build/test success alone is not visual
  or runtime proof.

## PowerShell handoff for user-owned visual proof

After implementation, the user should launch the active `WoWViewer` project with the configured client
root and output directory, load one small Museum/PM4 pair, capture the before/preview/after states, and
confirm that the accepted object moves while rejected and unresolved objects remain unchanged. Heavy
corpus sweeps, long captures, and real-client proof remain user-owned and must not be launched by the
implementation agent.

