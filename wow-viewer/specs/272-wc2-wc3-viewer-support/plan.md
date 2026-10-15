# Implementation Plan: WC3 File Browser and Tileset Preview

**Branch**: 272-wc2-wc3-viewer-support | **Date**: 2026-10-14 | **Spec**: [Current operator-corrected specification](spec.md)

**Input**: WC3 asset/tileset support in the existing File Browser, not world rendering; operator-supplied extracted beta data and Zezula supplemental listfiles; reuse existing MPQ readers.

## Summary

Add an owned file-only WC3 session beneath the existing Navigator File Browser. It reads loose extracted beta files directly and archive files through the existing shared archive catalog/bootstrapper. It supplies full-path filtering, bounded text and image preview, and byte-preserving selected-file extraction. It never calls the WoW build picker, world-source swap, model loader, terrain loader, or restoration converters.

**Implementation update (operator-directed 2026-10-14):** The file-only core and Navigator UI are now implemented; see [tasks](tasks.md) and [receipt](evidence/receipt-wc3-implementation.md). The operator requested implementation rather than further planning, so real beta/listfile checks below remain explicit proof gates, not blockers on writing/testing the bounded slice. Shared preview records live beside the I/O adapter instead of adding an unnecessary core-model project/file. Safe path helpers and a cancellation-token extraction overload implement the same contract. No beta/visual claim is closed by compilation or synthetic tests.

**Beta bug-fix amendment (2026-10-15 UTC):** The operator demonstrated BLP0 failure and requested MDX loading. Add the opt-in BLP0 header/companion decoder (BitMiracle.LibJpeg.NET 1.5.324 for unchanged four-channel samples), WC3 MODL/MTLS framing adapter above the shared MDX reader, and isolated off-screen use of the existing MDX renderer. A borrowed WC3 source supplies in-memory decoded textures with preview-specific cache names; source bytes, world session and protected readers remain unchanged. [Real-file evidence and remaining visual gap](evidence/receipt-beta-blp0-and-mdx-2026-10-15.md) supersede the earlier explicit BLP0/model exclusions below.

## Technical Context

**Language/Version**: C# with the repository's preview language setting; core targets .NET 10, viewer .NET 10 Windows. No Python or ML work.

**Primary Dependencies**: Existing archive contracts/catalog/bootstrapper, SereniaBLPLib, ImageSharp, ImGui.NET, Silk.NET.OpenGL, shared UI widgets. No new MPQ parser, renderer backend, HTTP service, or mandatory network downloader.

**Storage**: Read-only configured local roots and archives; explicitly selected loose-file extraction outputs within repository/project-managed or operator-supplied output roots. No container writes, new dataset store, or automatic source rewrites.

**Testing**: Existing xUnit core test project; focused source/listfile/preview/extraction tests; lightweight real-input probes after operator approval; operator-owned UI/pixel/world-isolation proof.

**Target Platform**: Existing Windows desktop viewer. Shared core remains platform-independent where filesystem policy permits.

**Project Type**: Desktop application with shared library-first file operations. Contracts are in-process operations, not REST endpoints; adding a network server solely to satisfy a generic planning template would be unrelated scope.

**Performance Goals**: Open/index/preview/copy work must not run unbounded on the UI thread. One source session, one current preview, bounded text/image buffers, incremental/cancellable file enumeration, and selected-only extraction. No FPS/performance claim is made without operator measurements.

**Constraints**: Preserve all protected readers, shared rendering, frozen shell/scene membership, original data, current WoW session, and unrelated dirty settings. All new source files remain under approximately 2,000 lines. WC3 source paths/listfiles are runtime configuration, not machine defaults.

**Scale/Scope**: One authoritative File Browser surface; loose beta root and MPQ-backed roots. Actual beta file types/count/build remain unverified because the requested inventory was declined. That is a real-data verification gate, not permission to guess support.

## Operator Intent Check

- **Operator's own words**: WC3 tilesets in the file viewer, “not the world renderer”; unpack/view files from a selected game/data directory. The supplied beta build text is valuable for original/short filenames.
- **Smallest useful delivery**: Open the beta directory in the File Browser; select and inspect a real tileset image and build text; copy those selected files unchanged. Add archive mode by passing WC3 lists to the already-working archive boundary.
- **Explicit exclusions**: WC2-first work, world/terrain map loading, new MPQ decoding, runtime model support, original-name reconstruction/renaming, de-atlasing, texture synthesis, camera/coordinate bridging, CDN fill, Reforged promises, or a new independent browser window.
- **Pre-design check**: PASS. Every operation maps to current FR-001–FR-010; older WC2/world/restoration requirements are historical/future, not implicit dependencies.
- **Post-design check**: PASS. Separate session state is confined to the existing File Browser; the current WoW data source and world never receive the WC3 source. Archive reads remain shared. Discovery failures are surfaced rather than “fixed” by base-reader changes.

## Constitution Check

| Gate | Pre-research | Post-design | Evidence/constraint |
|---|---|---|---|
| Library-first / repository independence | PASS | PASS | Shared models/I/O in existing core projects; no external source references |
| Existing readers preserved | PASS | PASS | Reuse [archive boundary](../../src/core/WowViewer.Core.IO/Files/IArchiveReader.cs:3) and [bootstrapper](../../src/core/WowViewer.Core.IO/Files/ArchiveCatalogBootstrapper.cs:66), not another parser |
| Real-data claims | PASS with proof pending | PASS with proof pending | Exact beta root is operator supplied but uninspected; no pixel/MPQ/extraction success claimed |
| Container read-only / write containment | PASS | PASS | Only loose selected-file outputs; no MPQ/CASC writing, OS-temp staging, or source rename |
| No client-path assumptions | PASS | PASS | Folder/listfile/output selection; local roots only in evidence |
| No god-class growth | PASS | PASS | Existing Navigator delegates to a new owned service; existing host window/GL access supplies lifetime/display |
| Single UI home / shared styling | PASS | PASS | Navigator File Browser; register its new WC3 route in the authoritative inventory during implementation |
| One validated phase at a time | PASS | PASS | Gates below stop progression until receipts exist; no phase marked implemented here |

The constitution's old architecture-document location is superseded by the workspace's current Spec Kit location rule. There is no constitution exception requested. Missing real inputs prevent implementation signoff, not documentation of an honest design.

## Project Structure

### Documentation (this feature)

- [Specification](spec.md), [prior scope](prior-scope.md), and [operator addendum](operator-addendum.md).
- [Research](research.md), [data model](data-model.md), [in-process contracts](contracts/file-browser.md), and [operator quickstart](quickstart.md).
- [Discovery/limits](evidence/wc3-file-browser-discovery-2026-10-14.md) and [planning receipt](evidence/receipt-wc3-planning-2026-10-14.md).
- Task breakdown is the next Spec Kit output; no implementation tasks are checked in this planning pass.

### Source Code (repository root)

Proposed files below are **not yet implemented**:

```text
wow-viewer/src/core/WowViewer.Core/Wc3/Assets/
  Wc3FileBrowserModels.cs
wow-viewer/src/core/WowViewer.Core.IO/Wc3/
  Wc3FileSource.cs                # filesystem + existing IArchiveCatalog adapter
  Wc3AssetPreviewReader.cs        # bounded metadata/text + existing image decoder
  Wc3AssetExtractionService.cs    # safe selected-file copy, no archive writing
wow-viewer/src/viewer/WoWViewer/Workbench/Services/Wc3FileBrowser/
  Wc3FileBrowserService.cs        # owns source/selection/jobs, no world state
  Wc3FileBrowserPanel.cs          # controls and file-only preview
  Wc3PreviewTexture.cs            # render-thread upload/dispose only
wow-viewer/tests/WowViewer.Core.Tests/
  Wc3FileSourceTests.cs
  Wc3AssetPreviewTests.cs
  Wc3AssetExtractionTests.cs
```

Existing bounded edits: [Navigator service](../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.cs:52) (ownership/delegation), [File Browser body](../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs:337) (WC3 route/control delegation), and the [UI inventory](../archived/227-ui-reaudit/surface-inventory-v2.md:99) / linked reachability record (same authoritative surface).

No change is planned to the MPQ readers, BLP library, standalone model loader, current data-source session, CASC service, terrain/world renderer, camera, or frozen shell/scene classes. If real beta decoding needs a new format adapter, stop for a bounded evidence-backed plan amendment; do not rewrite the shared decoder to make the corpus pass.

**Structure Decision**: New WC3 file operations live above proven shared I/O. A loose source enumerates its actual filesystem names (including text/table extensions); an archive source bootstraps the existing catalog with internal plus explicit WC3 listfile names and validates availability. The viewer service receives only read-only source/operation dependencies and GL/window lifetime inputs; it does not use shared world data-source fields.

## Research and Design Decisions

See [research](research.md) for alternatives. Key choices:

1. Direct loose-file support is first-class, not a fake installed game or a prerequisite re-archive operation.
2. For MPQs, use existing catalog ordering/provenance and report it explicitly. Do not claim emulation of WC3 patch/load-order semantics. No per-archive-selection or game-engine patch resolver is required for this initial effective-file browser.
3. A supplemental listfile is an explicit local selection; union valid names with internal lists through the bootstrapper. No WoW community listfile default for WC3.
4. Use the existing BLP decoder for variants it proves it can handle; preserve RGBA and avoid the RGB-only/Alpha-normalizing path. Supported alternate image types are selected only after real corpus inventory. Beta BLP0/external mips are not assumed from a signature.
5. Preview the complete source atlas. Tile cell selection, bitmask rendering, historical matching, and output re-encoding are deferred.
6. Use a monotonically increasing source/selection generation plus cancellation. Stale CPU results are discarded; GL texture upload/delete remains on the UI/context thread. Subscribe to the existing host window's closing event without adding shell members; close explicitly also cancels/releases the session.
7. Extraction checks canonical destination containment, refuses source-tree output, rejects reparse-point traversal and case collisions, defaults to no overwrite, and uses same-output-directory partial files renamed only after success. No OS-temp paths.

## Phase Roadmap and Verification Gates

### Phase 0 — Evidence and Names (next implementation prerequisite)

1. With operator approval, inventory a bounded sample of the supplied beta root, identify a real texture and build text/table file, and record build/relative paths/hashes.
2. Obtain/import a local Zezula WC3 listfile; record package/member provenance/hash. Automatic download is not a prerequisite: HTTPS currently failed certificate-name validation and no verified package was acquired.
3. Reuse the existing catalog on an available real MPQ fixture, or explicitly keep archive acceptance pending if the operator only supplies extracted files. Do not repackage loose files into an MPQ.

**Gate**: Receipt identifies actual sample variants and listfile provenance; no runtime proof is inferred from listfile names. If every beta image requires an unsupported codec, propose that adapter before moving into preview implementation.

### Phase 1 — Shared File Operations

1. Add source/entry/preview/extraction models and the in-process contract; keep game-root configuration injectable.
2. Implement loose enumeration and archive bootstrap adapters, full-path search, source identity, and truthful incomplete coverage/errors.
3. Add bounded metadata/text and supported RGBA image decoding through existing libraries.
4. Add safe exact-byte selected-file extraction with cancellation and outcomes.
5. Validate unit tests, real sample metadata/pixels/bytes, hashes, and filesystem escape/collision cases.

**Gate**: Focused tests plus real-input/output receipts for file operations; unsupported image variants remain open rather than synthesized. This gate does not claim UI/world isolation.

### Phase 2 — File Browser Integration and Operator Proof

1. Own/delegate the WC3 service from the existing Navigator File Browser; no call to world-source initialization.
2. Add root/listfile selection, full-path list/search/filter, source image/text preview, and selected-file extraction controls using shared UI widgets.
3. Implement preview generation/lifetime and render-thread texture release; register the route in the UI inventory/reachability record.
4. Run the solution build and focused core tests; provide the operator's bounded UI checklist and output locations.
5. Operator verifies real beta texture/text browsing, extraction fidelity, unsupported-codec messaging, repeated close/switch behavior, and unchanged existing WoW world/session.

**Gate**: SC-001–SC-005 each have their named evidence. SC-003 stays open without a real MPQ fixture. No completed implementation claim while only build/unit evidence exists.

## Validation Commands

From the repository root, using PowerShell 7:

```powershell
dotnet build ./wow-viewer/WowViewer.slnx -c Debug
dotnet test ./wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Wc3"
dotnet test ./wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MpqArchiveCatalog|FullyQualifiedName~ArchiveCatalogBootstrapper|FullyQualifiedName~BlpSummaryReader"
```

These are future implementation checks, **not commands run by this planning pass**. Real-input probes take runtime root/listfile/output arguments; no fictitious CLI flags are published before such a probe exists. UI/pixel proof is operator-owned.

## Agent Context Update

No update-agent-context script exists in this checkout's Spec Kit tree (bounded search returned none). Do not invent or modify a global agent-policy file to simulate it. Update the existing continuity dashboard/progress manually with the scope correction and next gate. No new technology was introduced.

## Complexity Tracking

No constitution violations or new packages/projects are proposed. The separate owned file session is necessary to avoid the existing world-coupled open-game path; it is not a second authoritative browser or a replacement archive stack.
