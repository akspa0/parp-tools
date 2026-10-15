# WC3 File Browser Implementation Receipt — 2026-10-14

**Follow-up:** The operator's real beta report exposed BLP0 and MDX gaps. The [2026-10-15 UTC receipt](receipt-beta-blp0-and-mdx-2026-10-15.md) records their fixes and real-file evidence. The original BLP0/model limitations below describe this earlier pass, not current support.

**Implemented**: File Browser WC3 mode, loose/MPQ source selection with local supplemental listfile, full-path search/filter, image fit/zoom/checkerboard and build-text previews, exact-byte single-file extraction with no overwrite/path containment and cancellation, stale-selection invalidation and owned resource cleanup. No world/model loading, reader rewrite, container writing, or source renaming.

## Files

- Core: [file source](../../../src/core/WowViewer.Core.IO/Wc3/Wc3FileSource.cs), [preview reader](../../../src/core/WowViewer.Core.IO/Wc3/Wc3AssetPreviewReader.cs), [path validation](../../../src/core/WowViewer.Core.IO/Wc3/Wc3PathSafety.cs), [path pinning](../../../src/core/WowViewer.Core.IO/Wc3/Wc3PathGuard.cs).
- Viewer: [owned browser](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3FileBrowserService.cs), [GL preview texture](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3PreviewTexture.cs), [Navigator ownership](../../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.cs), [existing File Browser route](../../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs:337).
- Tests: [source](../../../tests/WowViewer.Core.Tests/Wc3FileSourceTests.cs), [preview](../../../tests/WowViewer.Core.Tests/Wc3AssetPreviewTests.cs), [extraction/cancellation](../../../tests/WowViewer.Core.Tests/Wc3AssetExtractionTests.cs).
- Docs: current spec/plan/tasks/quickstart, status/dashboard/progress, and existing UI inventory row updated alongside implementation. No new shell/scene members; source readers and existing settings preserved.

## Verification

Commands run from workspace root (PowerShell 7); actual build/test output retained in [build log](../../../output/spec272/validation/build.log) and [test log](../../../output/spec272/validation/tests.log):

```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore
dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v quiet
dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter 'FullyQualifiedName~Wc3|FullyQualifiedName~MpqArchiveCatalog|FullyQualifiedName~ArchiveCatalogBootstrapper|FullyQualifiedName~BlpSummaryReader' -v minimal
git diff --check
git diff --exit-code -- wow-viewer/src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs wow-viewer/src/viewer/WoWViewer/ViewerApp.cs wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs
```

- Viewer build: exit **0**, 0 errors. Solution build: exit **0**, 0 errors (826 warnings, including existing package-vulnerability/analyzer warnings; dependencies were not upgraded here).
- Final incremental viewer build after the preview guard: exit **0**, 0 errors (314 warnings), [final log](../../../output/spec272/validation/viewer-final.log). Repeated focused tests with `--no-build --no-restore`: exit **0**, 77/77 passed. Final whitespace check: exit **0**.
- Focused tests: exit **0**, **77 passed / 0 failed / 0 skipped**; includes the added pre-cancelled-extraction case. These use repository-contained scratch files/synthetic format fixtures, not the real beta corpus.
- Protected-reader/shell/scene diff: exit **0**, no changes. Whitespace check passed before final documentation sync.
- Review caught and corrected window-lifetime registration before the window exists; registration is now lazy when the WC3 browser is first opened. File rows draw stored paths literally rather than interpreting embedded label markers.

## Criterion → Evidence / Remaining Proof

| Criterion | Available evidence | Still open |
|---|---|---|
| SC-001 beta file/text workflow | File Browser route + compiled source/preview controls; tests read actual scratch files | Operator opens real beta root and checks usable image/text flow |
| SC-002 texture fidelity | Synthetic palette/PNG/image-bounds tests; direct existing RGBA decoder, no Alpha normalization | Actual beta/JPEG dimensions/colors/alpha. BLP0 external mips explicitly unsupported; shared palette 4-bit alpha expands to 240 maximum, visibly labeled |
| SC-003 MPQ/listfile | Existing unchanged catalog/bootstrap; source and existing MPQ regression tests | Real WC3 MPQ/local listfile proof; no verified Zezula download is claimed |
| SC-004 extraction | Scratch source/output byte equality, source unchanged, no overwrite/traversal/reparse escapes, cancelled request creates no output | Real beta payload/hash and operator extraction UI checks; mid-operation UI cancellation behavior needs smoke |
| SC-005 isolation/lifetime | Separate owned source, generation invalidation, render-thread texture cleanup; no world-source calls/changes | Operator repeated switching/close and existing-world/camera verification |

No real-client/GPU/image/audio/runtime proof is inferred from build or unit tests. No denied beta inventory was repeated. No feature acceptance criterion is closed wholesale. T001/T003 are code/test receipts; T002/T004 remain open for actual UI/real-data proof.

## Try It

Open **File Browser**, enable **WC3 files (loose / MPQ)**, select the supplied extracted beta root and click **Open WC3 data**. Select a texture or text/table entry. Choose an output outside the source and use **Extract / copy selected**. See [quickstart](../quickstart.md) for limits and the bounded operator smoke.
