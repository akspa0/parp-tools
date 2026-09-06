# Verification Guide: Source Decomposition — God-Class Split

Run this guide for one completed extraction phase. It deliberately separates source proof from
operator-owned interactive proof.

## Before modifying source

```powershell
Set-Location I:/parp/parp-tools/wow-viewer
(Get-Content src/viewer/WoWViewer/Terrain/WorldScene.cs).Count
(Get-Content src/viewer/WoWViewer/ViewerApp.cs).Count
Get-ChildItem src/viewer/WoWViewer -Filter 'ViewerApp_*.cs' |
  ForEach-Object { "{0}: {1}" -f $_.Name, (Get-Content $_.FullName).Count }
```

Record the result in the owning Spec 228 receipt. Confirm the UI feature's Spec 227 inventory row
and T004 gate are complete before touching an interactive selection route.

## Source and test proof

```powershell
Set-Location I:/parp/parp-tools/wow-viewer
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneSelection"
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

The phase receipt must name every moved file, the exact commands and exit status, before/after
line counts, and the criterion-to-evidence table. A failed or unavailable focused test is a
blocking result, not a reason to omit the receipt.

## Operator smoke (interactive phases)

With a configured client root and current build, test the exact input route that was moved:

1. Hover overlapping WMO/MDX/liquid and confirm the same target identity and distance policy.
2. Click a scene object and confirm the same selection reaches its already-authoritative Inspector
   destination.
3. Confirm an object behind terrain is not selected when the current route previously rejects it.
4. Record client root fingerprint, build identity, map, inputs, observed result, and any capture in
   the owning receipt.

This smoke is operator-owned. Build and test output do not establish visual, input, rendering, or
performance acceptance.
