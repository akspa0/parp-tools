# Data Model — WC3 File Browser

Design-only entities for the [current specification](spec.md); proposed types are not existing APIs. No map/world models are needed for file browsing.

## Source Session

- Stable session ID/generation; configured root; kind (loose, MPQ, unsupported); display identity/build if actually known; local listfile selection/provenance; name-coverage status; diagnostics; disposal/cancellation owner.
- Read-only. Do not scan parent drives, follow directory reparse points, or replace the host's WoW data source.
- States: Closed → Opening → Ready or Failed; Ready → Closing → Closed. A new open invalidates all earlier source/selection results.

## File Entry

- Source/session ID; exact stored relative/virtual path; display path; extension; size when known; content availability; source provenance when exposed by the shared reader; supported preview kind.
- Stable identity is source ID plus normalized comparison key; preserve stored path for display and extraction. Case-insensitive duplicates receive one explicit effective-file policy rather than conflicting output destinations.
- Listfile-only names are candidates until actual source availability is confirmed. Unknown archive provenance stays unknown; do not fabricate a base/patch claim.

## Preview Request / Result

- Source ID, entry ID, selection generation, requested preview kind/mip, cancellation token, and bounded limits.
- Result: image RGBA bytes/dimensions/format or bounded text/encoding/truncation; alternatively metadata plus actionable unsupported/missing/error status.
- Validate image dimension/pixel byte counts, payload/mip bounds, and text limits before allocation/display. Companion files resolve only inside the current source. Specific limits become named tested implementation constants, not claims about corpus sizes.
- States: None → Loading → Ready, Unsupported, Failed, or Cancelled. Only the current source/selection generation may publish. GL handles remain viewer-owned and are never part of core results.

## Extraction Request / Per-File Result

- Source ID; selected entry IDs; explicit output root; no-overwrite default; cancellation state.
- Each result: stored source path; final relative/output path; status (copied, skipped-existing, rejected-path, cancelled, failed); byte count and optional SHA-256 proof; error text.
- Validate every destination before writing: no rooted/drive/UNC/traversal forms, no source-tree output, no case-collision output, no reparse-point escape. Use partial files within the selected output root; rename only on complete success; remove/mark partials on failure.
- Extraction consumes original source bytes independently of preview decode. No archive writer or automatic filename remapping exists.

## Listfile Provenance

- Original publisher URL/local import; retrieval date/hash if acquired; chosen ZIP member name/hash if imported; coverage diagnostics; no assumed per-game list filename until package contents are inspected.
- A failed download is a failure record, not an empty successful list. Local-file import and loose browsing remain usable offline.
