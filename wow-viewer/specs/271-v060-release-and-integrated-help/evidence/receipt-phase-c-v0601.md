# Spec 271 Phase C Receipt — v0.6.0.1 Corrective Release

Date: 2026-10-14 · Tag: `v0.6.0.1` · Release: https://github.com/akspa0/parp-tools/releases/tag/v0.6.0.1

## Files changed

- `.github/workflows/wowviewer-build.yml`, `.github/workflows/wowviewer-release.yml`
- `wow-viewer/WowViewer.slnx`, `wow-viewer/eng/Version.props`
- `wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs`
- `wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementModels.cs`
- `wow-viewer/src/viewer/WoWViewer/Rendering/IModelRenderer.cs`
- `wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs`
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs`
- `wow-viewer/tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectPlacementTests.cs`
- `wow-viewer/tools/README.md`, viewer README, `docs/CLI-TOOLS.md`, `docs/WoWViewer/USERGUIDE.md`, `CHANGELOG.md`, `docs/releases/v0.6.0.1.md`
- Spec 259 and 271 plan/spec/task documents, `specs/STATUS.md`, `memory-bank/activeContext.md`
- `memory-bank/progress.md`, `evidence/receipt-phase-c-v0601.md`, and Spec 259's
  `evidence/receipt-lighting-propagation-2026-10-14.md`
- Pre-existing `imgui.ini` and `wow-viewer/imgui.ini` changes were not staged or included.

## Verification commands and results

- `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:q` — exit 0; 0 errors. Existing analyzer/package warnings remain.
- After restoring the development branch version, `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:q` — exit 0; 0 errors, 826 warnings. `eng/Version.props` now evaluates to `0.6.1-dev`; the release tag retains `0.6.0.1`.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~DetailDoodads" -v:q` — exit 0; 8 passed, 0 failed.
- Release-tag CI: `gh run watch 38508425730 --exit-status` — exit 0; all four RIDs published the viewer and all nine tools, uploaded both artifacts, and the release job succeeded.
- Branch CI: `gh run view 38506799386 --json status,conclusion,jobs,url` — conclusion `success`; Windows x64, Linux x64, macOS arm64, macOS x64 all passed the nine-project count guard and uploaded viewer + tools artifacts.
- `gh api repos/akspa0/parp-tools/releases/tags/v0.6.0.1 --jq ".assets[] | {name,size,download_count,browser_download_url}"` — exit 0; enumerated eight release ZIPs (four viewer, four CLI-tools). Windows assets: viewer 67,959,826 bytes; tools 56,827,966 bytes.
- `gh release download v0.6.0.1 --pattern 'ParpToolsWoWViewer-*win-x64.zip' --dir wow-viewer/output/v0.6.0.1-verified` — exit 0; both Windows ZIPs inspected from repository-managed output.
- `pwsh -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; $root='wow-viewer/output/v0.6.0.1-verified'; Get-ChildItem $root -Filter '*.zip' | ForEach-Object {Expand-Archive -LiteralPath $_.FullName -DestinationPath (Join-Path $root $_.BaseName) -Force}; $tools=Join-Path $root 'ParpToolsWoWViewer-tools-win-x64'; $viewer=Join-Path $root 'ParpToolsWoWViewer-viewer-win-x64'; $apps=(Get-ChildItem $tools -Filter 'WowViewer.Tool.*.exe').Count; $dbd=(Get-ChildItem $tools -Filter '*.dbd' -Recurse).Count; $core=(Get-ChildItem $tools -Filter 'WowViewer.Core.dll' -Recurse).Count; $viewerTool=(Get-ChildItem $viewer -Filter 'WowViewer.Tool.*' -Recurse).Count; & (Join-Path $tools 'WowViewer.Tool.Inspect.exe') --help *> $null; $helpExit=$LASTEXITCODE; Write-Output ('tool apphosts='+$apps+'; DBD files='+$dbd+'; shared Core copies='+$core+'; viewer tool binaries='+$viewerTool+'; Inspector exit='+$helpExit)"` — exit 0: CLI ZIP contains 9 apphosts, 1,342 DBD files, one `WowViewer.Core.dll`; viewer ZIP contains 0 CLI apphosts; Inspector `--help` exits 0.
- Local shared-output Windows smoke: all nine apphosts started; raw shared CLI output was 138.8 MiB, compressed tools ZIP 55.8 MiB. The two CLI programs without a `--help` option returned their expected usage output.

## Acceptance evidence

| Criterion | Result | Evidence |
|---|---|---|
| AC-C1 — four viewer ZIPs plus four CLI-tools ZIPs; all nine tools and shared dependencies | **PASS** | Release asset API listed all eight files. Downloaded Windows tools ZIP contains nine apphosts, one copy of `WowViewer.Core.dll`, and 1,342 DBD definitions; four-RID CI passed the same publish/count checks. |
| AC-C2 — tag release workflow succeeds and publishes the downloads | **PASS** | Release workflow run `38508425730` completed successfully; release is public at the URL above with eight ZIP assets. |
| AC-C3 — MCCV/MCSH tint reaches the rendered doodad material | **Code/test portion PASS; visual proof OPEN** | Focused tests verify neutral, MCCV-lit, and shadow-attenuated tint values; Debug solution build passes and the per-instance tint is passed into MDX material RGB. Real-client visual smoke remains operator-owned and is tracked as Spec 259 T015 / Spec 271 C4. |
