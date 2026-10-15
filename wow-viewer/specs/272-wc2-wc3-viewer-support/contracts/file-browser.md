# In-Process Service Contracts — WC3 File Browser

Design-only contracts for [FR-001–FR-010](../spec.md). This desktop feature adds no HTTP endpoint/server; the API boundary is an injectable in-process shared-core service. Proposed operation names below are not implementation claims.

| User action / operation | Input | Output / error contract |
|---|---|---|
| Open source | Configured root, loose/MPQ/auto choice, optional local supplemental listfile, cancellation | Source identity/kind, confirmed entries, coverage diagnostics; unsupported layout or failed open retains no half-published session |
| Filter entries | Current source ID, substring/extension/category, result page | Stable full-path entries and total/continuation state; no silent 5,000-item completeness claim |
| Read original bytes | Current source ID and entry path | Exact stored payload; missing/unreadable reported independently from preview support |
| Preview entry | Current source/entry/selection generation, bounded limits | RGBA image or bounded text/metadata; explicit unsupported/missing-companion/failed status; stale generations never publish |
| Extract selected | Current source, selected paths, explicit output root, no-overwrite policy, cancellation | Per-file exact-byte outcomes; reject unsafe destinations before writing; no partial-success label |
| Close source | Current source ID | Cancel/invalidate work, release owned handles and CPU buffers; render thread releases viewer texture resources |

## Archive Adapter Requirements

- Delegates reads/name bootstrap to [existing archive contracts](../../../src/core/WowViewer.Core.IO/Files/IArchiveReader.cs:3); no raw MPQ parsing or altered ordering in this feature.
- Internal plus optional supplemental names are availability-checked; a listfile is not a list of guaranteed real entries.
- Effective duplicate/provenance policy is displayed honestly and is not labeled WC3 game-engine patch emulation.

## File/Text/Image Boundaries

- Source-relative path resolution cannot escape a loose root. An image's companion lookup cannot access arbitrary external paths.
- Stored short filenames and text content are retained exactly. No implicit original-name substitution.
- Original-byte reads/extraction never pass through a texture converter or text re-encoder.
- New shared code has no access to the current world, camera, model renderer, UI GL objects, or machine-local default paths.

## Error Categories

InvalidRoot, UnsupportedLayout, MissingListfile, IncompleteNames, MissingEntry, ReadFailure, UnsupportedPreview, MissingCompanion, InvalidImage, TextTruncated, UnsafeDestination, ExistingDestination, Cancelled, DisposedSource. UI presents actionable descriptions without reporting unsupported formats as corrupt or absent without evidence.
