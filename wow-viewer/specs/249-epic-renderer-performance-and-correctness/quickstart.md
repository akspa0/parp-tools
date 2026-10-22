# Evidence and verification quickstart — renderer refresh

Prepared commands only. From wow-viewer in PowerShell 7, after a bounded implementation:

```powershell
dotnet build ./WowViewer.slnx -c Debug
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~LiquidVertexFormatChain|FullyQualifiedName~DbcLiquidTypeTable|FullyQualifiedName~AdtLiquidReader|FullyQualifiedName~WmoGroupLiquidSummaryReader|FullyQualifiedName~WmoLiquidLayoutResolver'
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~BuildScopedLightDbcProfileResolver|FullyQualifiedName~SkyDomeVertexBuilder|FullyQualifiedName~TerrainViewerLightingComposer'
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~GroundEffectPlacement|FullyQualifiedName~GroundEffectDetailDoodad'
```

No renderer/client/GPU job is executed by this plan. Add newly authored pure-policy tests to the
relevant filter once named. Existing tests alone cannot prove live material or scene parity.

## Gate E — operator/reference input card
Provide revision, configured client/build/fingerprint, map and asset/group/chunk identifiers,
camera position/orientation/time, density/distance/lighting/sky settings, exact complaint and
baseline capture. Record configured reference location/revision and cited symbol/build if available;
otherwise mark reference unavailable. Never install or assume an external decompiler implicitly.

## Gate G — ground effects (Spec 259)
Use the same terrain/WMO sample and camera stations before fade, inside fade and beyond radius;
record generated/submitted counts, opacity, material route and explicit density/radius. Change
one setting at a time. Compare unchanged lighting before any separate tint/lighting proposal.

## Gate L / LP — liquid
For each reported family, identify one WMO and MH2O example where present in the client. Record
raw fields/rows, missing-row behavior, layout resolution, mixed-layer reduction and selected
material. Diagnose before correction; after correction capture matched geometry/color/coverage
and unaffected controls. A reader mismatch requires explicit separate approval.

## Gate S / SP — sky
Same exact-build/map/camera/time: blank/simple mode, proven database choice, day/night and existing
fallback. Record database/model resolution separately from night-visibility/pass gate. Verify
known-good assets actually exist; record missing choices honestly. Operator judges appearance.

## Receipts and stop rule
Receipts live under the owning spec's evidence directory and include changed paths, exact command
exit statuses, criterion→real-output table, matched baseline/after outputs, source provenance and
operator verdict. Keep captures inside repository/project output or operator-supplied path; do
not bundle game data. Generate unchecked tasks next; implement E0 only before downstream gates.
