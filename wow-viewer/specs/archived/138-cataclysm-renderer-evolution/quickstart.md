# Spec 138 Planning Quickstart

This is a planning and validation route. It does not launch broad extraction or training.

## 1. Confirm reference state

The local Marlamin reference checkout is `wow-viewer/libs/Marlamin/WoWTools.Minimaps`. Its
`CascLib`, `TACT.NET`, and `Warcraft.NET` submodule directories are currently empty, so do not
assume that checkout provides buildable archive dependencies. DBCD, WoWDBDefs, and wow-listfile
are already present and used by the viewer/tooling; the new work must reuse those paths.

The GitHub references are:

- https://github.com/ladislav-zezula/CascLib
- https://github.com/WoW-Tools/CascLib
- https://github.com/wowdev/TACTSharp
- https://github.com/wowdev/TACT.Net
- https://github.com/Marlamin/WoWTools.Minimaps
- https://github.com/Kruithne/wow.export

## 2. Prepare configured fixtures

Select one known-good, user-approved client root for each checkpoint: 0.5.3, 1.x, 3.3.5,
4.0.0, 6.x, 7.x, and 11.x. Keep roots outside the repository and record the exact build identity,
locale, adapter, listfile, and fingerprint in the probe report.

## 3. Planned probe command

The implementation phase should expose a thin inspect command similar to:

```powershell
dotnet "I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll" archive-probe --client-root "H:/CLIENTS/<build>" --profile "<profile-id>" --listfile "I:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt" --output "I:/parp/parp-tools/wow-viewer/output/probes/spec-138/<profile-id>.json"
```

This command is a planned interface, not an assertion that the command exists yet.

## 4. Validation gates

The first implementation gate is not a minimap or renderer screenshot. It is a source probe that
can read a known terrain file through the selected adapter, report the selected build/profile, and
fail closed when the requested capability is unavailable. Terrain rendering begins only after the
6.x CascLib probe and the later-CASC probe both produce reviewable provenance.
