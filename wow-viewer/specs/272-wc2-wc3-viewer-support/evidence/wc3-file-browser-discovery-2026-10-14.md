# WC3 File Browser Discovery and Operator Corrections — 2026-10-14

## Scope Authority

1. Operator corrected the initial WC2-first interpretation: WC3 assets/tilesets in the **file viewer**, explicitly not the world renderer; select an installed directory and unpack/view files.
2. Operator supplied **I:\WC3BetaData** as already-extracted WC3 beta data with build-process text/original-to-short filename clues. This is an operator-local evidence path, never an application/test default.
3. Operator directed reuse of existing 0.5.3 MPQ/listfile support and supplied **https://www.zezula.net/download/listfiles.zip**. WC1/2/3 list availability was operator-reported; individual package contents were not independently inspected.

The bounded inventory command for the beta root was **denied** with the archive-reuse direction. It did not execute. No file count, build identity, texture signature, sample filename, or hash from that root is claimed. Do not repeat the denied inventory without approval.

An earlier shallow approved candidate-root check found an empty WC3 research directory, not the subsequently supplied beta root. It is not evidence that the beta corpus is empty.

## Existing-Code Evidence

- [Core archive interface](../../../src/core/WowViewer.Core.IO/Files/IArchiveReader.cs:3), [name ingestion/internal lists](../../../src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs:32), and [bootstrapper](../../../src/core/WowViewer.Core.IO/Files/ArchiveCatalogBootstrapper.cs:66): reuse, not replacement.
- [Current MPQ viewer source](../../../src/viewer/WoWViewer/DataSources/MpqDataSource.cs:205) already bootstraps shared catalogs. Its [loose-file extension index](../../../src/viewer/WoWViewer/DataSources/MpqDataSource.cs:57) omits build text.
- [Current open-game source session](../../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs:191) couples source loading to WoW/world initialization; a separate file-only session avoids changing that base path.
- [Current File Browser](../../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs:337) is the authoritative UI home. The [model-loader selection default](../../../src/viewer/WoWViewer/Workbench/Services/ModelLoading/StandaloneModelLoaderService.cs:1204) does not preview BLP images.
- [Existing decoder](../../../libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/BlpFile.cs:339) exposes RGBA image output; beta correctness is not proven by that API. The [RGB-only helper](../../../src/core/WowViewer.Core.IO/Blp/BlpRgbReader.cs:19) normalizes/drops alpha and is not the planned source-fidelity route.
- Navigator service/body measured 156/446 lines; standalone model loader 1,678 lines. New behavior belongs in an owned service, not that loader or a god-class extension.

The primary agent inspected the archive contracts/catalog, current MPQ source, decoder, and Navigator ownership. A read-only sub-agent inspected UI/session/preview/extraction seams and existing tests. Neither changed runtime code or ran clients.

## Public Listfile Acquisition Evidence

The publisher's [download page](https://www.zezula.net/en/mpq/download.html) was inspected with the browser tools. Actual page text: “Listfiles for MPQ Archives (9.83 MB)”; actual link: [listfiles.zip](https://www.zezula.net/download/listfiles.zip). It describes older-archive filename lists, not new content codecs.

Attempts/outcomes:

| Attempt | Actual outcome |
|---|---|
| Browser HTTPS www download page | Certificate-name error |
| Browser HTTPS non-www download page | Aborted navigation |
| Browser HTTP page navigation, then DOM inspection | Page/link text inspected; final DOM location reported HTTPS |
| PowerShell HTTPS www ZIP retrieval | Exit 1: RemoteCertificateNameMismatch |
| PowerShell HTTPS non-www ZIP retrieval | Exit 1: RemoteCertificateNameMismatch |

No certificate check was disabled. No verified package, archive member, extracted listfile, byte count, digest, or redistribution grant was acquired. Only an output-directory creation was requested under repository-managed output. Do not bundle a nonexistent or unverified download; local import remains the supported design fallback.

## Tooling and Validation Limits

Spec Kit setup-plan exited 0 and created the plan scaffold for the existing branch/feature pin. A bounded script search found no agent-context update script in this checkout, so dashboard/progress sync is manual. No new technology/agent policy was introduced.

No build, unit test, source decode, unpack/copy, real image display, or world-isolation run occurred in this planning pass. [The plan](../plan.md) records the corresponding verification gates rather than marking them complete.
