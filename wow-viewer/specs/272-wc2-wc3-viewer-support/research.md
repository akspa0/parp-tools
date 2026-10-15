# Research — WC3 File Browser Slice

**Date**: 2026-10-14. **Scope**: Current [specification](spec.md), not the superseded WC2/world/restoration plan.

## R1 — Reuse Archive I/O, Not the World-Open Route

**Decision**: Wrap the existing shared [archive contracts](../../src/core/WowViewer.Core.IO/Files/IArchiveReader.cs:3) and [bootstrapper](../../src/core/WowViewer.Core.IO/Files/ArchiveCatalogBootstrapper.cs:66) in a separate file-only session. Reuse MPQ/listfile decoding unchanged.

**Rationale**: The operator explicitly identified existing 0.5.3 support as sufficient base tooling. [Current game-folder loading](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs:191) initializes/replaces WoW world/session state; it is not the right file-only integration seam.

**Alternatives considered**: New MPQ parser (rejected: duplicate/protected functionality); extending the normal WoW session (rejected: world side effects); reusing the existing catalog beneath a small owned adapter (chosen).

## R2 — Extracted Beta Data Is a Real Source, Not an Archive

**Decision**: Support loose source-relative files directly, including build text/tables; do not require MPQs/listfiles for the supplied extracted root.

**Rationale**: The operator supplied extracted beta data, not an executable install. A requested read-only inventory was declined, so actual files/signatures/build identity remain unobserved. No beta codec or content count is assumed.

**Alternatives considered**: Rebuild an MPQ (rejected by read-only-container policy); use the existing loose-file index unchanged (its [extension allowlist](../../src/viewer/WoWViewer/DataSources/MpqDataSource.cs:57) omits build text); enumerate actual filesystem entries in the new shared source adapter (chosen).

## R3 — Warcraft Listfiles Are Explicit Inputs

**Decision**: Use internal names plus a selected local WC3 supplemental listfile with provenance; never apply the WoW community listfile by default. Validate names against the loaded archive.

**Rationale**: [Catalog listfile ingestion](../../src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs:32) and [internal extraction](../../src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs:73) already exist. The publisher's [download page](https://www.zezula.net/en/mpq/download.html) lists a 9.83 MB [ZIP](https://www.zezula.net/download/listfiles.zip). Browsing confirmed the public link, but command-line retrieval failed TLS hostname validation for both hostnames; no package/member/hash is verified. No TLS bypass is proposed.

**Alternatives considered**: Mandatory startup download (rejected: offline/TLS dependency); commit a guessed filename list or unverified third-party package (rejected); allow a local list and record acquisition as the first evidence gate (chosen). WC1/2 lists can remain available future profiles; they do not prove those games use the same containers.

## R4 — Full Atlas and Text Preview Only

**Decision**: Use an opt-in RGBA image adapter above existing libraries and bounded text preview; display source atlases, not reconstructed terrain.

**Rationale**: The [current BLP selection route](../../src/viewer/WoWViewer/Workbench/Services/ModelLoading/StandaloneModelLoaderService.cs:1204) is unsupported, despite the browser filter. [Existing BLP image output](../../libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/BlpFile.cs:339) is a reuse candidate, not real beta proof. [RGB-only reading](../../src/core/WowViewer.Core.IO/Blp/BlpRgbReader.cs:19) normalizes for WoW Alpha and drops alpha, so it is unsuitable for truthful WC3 source preview.

**Alternatives considered**: Change shared texture/render paths (rejected); assume all BLP0/JPEG variants work (rejected); add a bounded preview adapter with explicit unsupported results and a real-data gate (chosen). Do not add de-atlasing, synthesis, alias inference, or map support.

## R5 — UI Home and Lifetime

**Decision**: The existing [Navigator File Browser](../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs:337) owns a WC3 mode via a dedicated service; keep its state separate from the shared WoW host data source. Use existing [GL/window host access](../../src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs:232) and closing notification for cleanup, without new shell/scene members.

**Rationale**: [Navigator ownership](../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.cs:52) is already small and delegated. Generation-checked CPU jobs and render-thread GL cleanup prevent stale previews and disposed-context work. Existing model/camera selection helpers must continue to refer only to their original WoW list, not WC3 selections.

**Alternatives considered**: A new top-level browser window (rejected: duplicate authoritative home); global data-source swapping (rejected: world coupling); a new ViewerApp partial (prohibited).

## R6 — Exact-Byte Extraction with Containment

**Decision**: Copy selected source payloads as loose files to an explicit destination, refusing source-tree writes and unsafe relative paths; default no-overwrite, validate canonical/reparse containment, finalize same-directory partials only after success.

**Rationale**: “Unpack” does not authorize archive writing, speculative original-name restoration, broad corpus extraction, or OS-temp staging. Preview decode must not alter extraction bytes.

**Alternatives considered**: Whole-install automatic unpack (rejected: unnecessary/broad); source-side renamed exports (rejected); selected-only raw copy with per-file statuses (chosen).

## Resolved Design Questions / Open Evidence Gates

Design choices above are resolved; no hidden clarification placeholders remain. Actual beta variants, listfile contents/provenance, real archive availability, exact pixel fidelity, and UI isolation are explicit execution/verification gates, not asserted facts. The missing agent-context script is handled by manual continuity sync, not an invented command. No source or test implementation was made by this research pass.
