# Beta BLP0 and MDX Follow-Up — 2026-10-15 UTC

**Operator repro:** screenshot showed a successful 27,635-entry WC3 file source but a BLP0 unsupported message; operator also reported that MDX files were not attempted. This is an explicit extension of the file-preview slice, not world rendering.

## Changes

- [BLP0 adapter](../../../src/core/WowViewer.Core.IO/Wc3/Wc3Blp0Decoder.cs): parses the 28-byte header; resolves numbered companions from the same source; joins shared JPEG tables with external scan data; preserves four raw components rather than applying CMYK conversion; supports bounded palette/alpha data and direct mip selection.
- [Preview reader](../../../src/core/WowViewer.Core.IO/Wc3/Wc3AssetPreviewReader.cs): companion resolver and actual MDX parse route. [WC3 MDX framing adapter](../../../src/core/WowViewer.Core.IO/Wc3/Wc3MdxAdapter.cs): adapts v800 MODL/MTLS framing to the existing reader without changing geometry/source bytes or that reader.
- [File browser](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3FileBrowserService.cs), [owned MDX preview](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3ModelPreview.cs), [GL state guard](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3ModelGlState.cs), and [borrowed WC3 model source](../../../src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/Wc3ModelDataSource.cs): existing MDX renderer in an isolated target, with texture names isolated from its shared WoW cache. BLP0 model textures become lossless in-memory BGRA payloads for the unchanged renderer; no source/export mutation.
- [BLP0 tests](../../../tests/WowViewer.Core.Tests/Wc3Blp0Tests.cs), [MDX adapter tests](../../../tests/WowViewer.Core.Tests/Wc3MdxAdapterTests.cs). Managed raw-JPEG dependency BitMiracle.LibJpeg.NET 1.5.324 added to [core I/O](../../../src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj) and [central versions](../../../Directory.Packages.props). No shared MPQ/ADT/WMO/M2/MDX reader or model/world renderer was edited.
- Spec/plan/tasks/quickstart, status/dashboard/progress, and authoritative UI inventory synchronized. Existing implementation receipt retained as history, not rewritten into visual proof.

## Real Inputs and Format Findings

Operator-local root: **I:\WC3BetaData** (runtime probe argument only). Focused source subdirectory: Abilities/Spells/Human/InnerFire. No broad corpus decoding/harvest occurred.

| Input | Observed data | SHA-256 |
|---|---|---|
| Crown1.blp | 1052 bytes; BLP0; JPEG; 128×32; shared JPEG length 624 at byte 28, shared bytes start at 32 | 553D0BD0556DDF9D25AC177E995D607DB54549FF1A2174E2EA6F30A32AD9CBC1 |
| Crown1.b00 | 7360 bytes; external mip scan data | 7B66A146B2A0299392596E6D3CD1A26EA07F7E25B97213A66C2490D434CCBC3B |
| Crown1.b03 | 284 bytes; mip level 3 | F1F988C00BDD93F0CBED49EB463562D406974BA9D5F4046EF49AF47100E634A2 |
| InnerFireTarget.mdx | 12082 bytes; MDLX v800; WC3 LAYS material framing, not the shared reader's counted material framing | 6B8E426B0247580EA59102EA804E292DD419C389EF817502F62D522F199DFB11 |

The first guessed basename omitted the final digit and found no files; the focused directory check corrected it. Earlier failed NuGet/compile attempts were corrected (published libjpeg version and managed lifetime API). The unadapted MDX parse failed at MTLS; the opt-in framing adapter corrected that actual source failure. No claim that every beta or retail variant matches these findings is made.

## Commands and Actual Results

From workspace root, PowerShell 7:

```powershell
dotnet run --project wow-viewer/output/spec272/beta-probe/BetaProbe.csproj -c Debug -- 'I:\WC3BetaData' 'wow-viewer/output/spec272/beta-decoded' 'Abilities\Spells\Human\InnerFire\Crown1.blp' 'Abilities\Spells\Human\InnerFire\Crown1.b03' 'Abilities\Spells\Human\InnerFire\InnerFireTarget.mdx'
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v quiet
dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter 'FullyQualifiedName~Wc3|FullyQualifiedName~MdxGeometryReader|FullyQualifiedName~MpqArchiveCatalog|FullyQualifiedName~ArchiveCatalogBootstrapper|FullyQualifiedName~BlpSummaryReader' -v minimal
```

The tiny probe project/source are repository-managed output utilities, not a new shipped tool. Paths are explicit operator arguments; no hardcoded client root was added to source or portable tests. [Probe output](../../../output/spec272/validation/beta-probe.log), [final viewer build](../../../output/spec272/validation/beta-viewer-final.log), [final tests](../../../output/spec272/validation/beta-tests-final.log).

All final commands exited **0**. Viewer: **0 errors** (314 warnings). Tests: **86 passed / 0 failed / 0 skipped**. Real probe:

| Output | Actual result | RGBA SHA-256 |
|---|---|---|
| Crown1 mip 0 | 128×32; [decoded PNG](../../../output/spec272/beta-decoded/Crown1.blp.png) | D77E458F6CB6B0A1352430FB6274477608F64E9FD96CE0B7B328CEDFBDE84582 |
| Crown1 mip 3 | 16×4; [decoded PNG](../../../output/spec272/beta-decoded/Crown1.b03.png) | 356DE5F2533AC1FF8A012D94018FDB44BAEBBDBC69F637A82197AD3F5E64FE0F |
| Adjacent MDX | v800; 6 geosets, 90 vertices, 4 textures, 3 sequences | Not an image/render result |

The decoded crown PNG was inspected as a static 128×32 black/white patterned image. This is actual CPU output, **not** in-viewer/GPU or independently calibrated source-color proof. Generated proprietary images remain under ignored output, not source/distributable assets.

## Criterion → Evidence / Open Proof

| Criterion | Evidence now | Remaining operator proof |
|---|---|---|
| SC-001 source/preview workflow | Operator screenshot proves source browse; real texture decodes through new adapter | Reopen selected texture in rebuilt UI |
| SC-002 image and new MDX preview | Real texture/mip PNGs and hashes; real MDX parse; raw-channel and framing regression tests; existing renderer wiring | Image alpha/color against reference; actual MDX appearance/textures/animation |
| SC-003 archive/listfiles | Existing MPQ regressions pass, readers unchanged | Real WC3 MPQ/listfile proof remains open |
| SC-004 source/output fidelity | Raw extraction path is unchanged; existing extraction regressions pass | Operator payload/hash smoke remains open |
| SC-005 isolation/lifetime | Off-screen target, GL-state save/restore, borrowed source, source-specific texture keys; shared renderer/world code unchanged | Actual switching/close behavior and existing-world isolation |

T005 covers actual CPU beta decoding plus tests, not GPU signoff. MDX preview code/CPU parse are implemented; T006 remains open for operator visual proof. No game/viewer was launched by the agent.
