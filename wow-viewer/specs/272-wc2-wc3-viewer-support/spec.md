# Feature Specification: WC3 File Browser and Tileset Preview

**Feature Branch**: 272-wc2-wc3-viewer-support (existing branch retained)

**Created**: 2026-10-14

**Status**: File-browser code implemented; solution build and focused tests pass; real beta/UI proof remains operator-owned

**Input**: “I'm most interested in WC3 asset support, not WC2, so let's focus on the tilesets for WC3 data, to display in the file viewer section of the program, not the world renderer. Can we get that going, where I try to load a WC3 installed game directory, and then get to unpack and view the data files in there?”

**Dated operator correction — 2026-10-14**: This replaces the [earlier WC2-first scope](prior-scope.md). The operator supplied an extracted WC3 beta directory containing build-process text and historical original-to-short filename clues. The operator further directed reuse of existing 0.5.3 MPQ/listfile support and the [Zezula listfiles](https://www.zezula.net/download/listfiles.zip), not another MPQ reader. Exact local inputs and acquisition limitations belong in [evidence](evidence/wc3-file-browser-discovery-2026-10-14.md).

**Operator bug-fix amendment — 2026-10-14 local / 2026-10-15 UTC:** The operator's screenshot proves source browsing but failed BLP0 preview; the operator also explicitly requests MDX loading through existing support. This slice now includes beta BLP0 shared-header/external-mip resolution and an isolated MDX file preview using the existing reader/renderer, never the world loader. Raw four-component JPEG decoding is WC3-only (managed libjpeg); shared WoW decoders/readers stay unchanged. Real source bytes, not the draft, determine beta layout. Models are no longer excluded from the current file-viewer slice.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Open WC3 Data in the File Browser (Priority: P1)

As a researcher, I want to select a WC3 installed or already-extracted data directory and browse its actual files without choosing a WoW build or loading a world.

**Why this priority**: It is the entry point for the operator's beta corpus and the requested archive workflow.

**Independent Test**: Select the supplied extracted beta root; separately select an available MPQ-backed root and a WC3 listfile. Inspect source identity, full relative paths, and search/filter results.

**Acceptance Scenarios**:

1. **Given** extracted WC3 beta files, **When** the user opens that directory in the File Browser, **Then** its supported files are discoverable using their stored relative paths, including terrain textures and build text, without a WoW build prompt or world transition.
2. **Given** WC3 MPQs and internal or user-selected supplemental listfiles, **When** the user opens the directory, **Then** known names are resolved using the existing archive reader, absent listfile names are not shown as real files, and the UI explains incomplete name coverage.
3. **Given** a directory without a supported file/archive layout, **When** it is selected, **Then** the viewer reports the unsupported layout without silently treating it as WoW data or replacing the active world session.

### User Story 2 - Preview WC3 Tileset Textures (Priority: P1)

As a researcher, I want selecting a terrain texture to show its original image in the File Browser, not a terrain map or world render.

**Why this priority**: A real texture preview is the smallest useful visual outcome requested by the operator.

**Independent Test**: Open representative real beta tileset assets; compare colors, alpha, dimensions, orientation, and stored image content with an independent decoder/reference.

**Acceptance Scenarios**:

1. **Given** a supported tileset texture, **When** selected, **Then** the File Browser shows its full source atlas, virtual/relative path, dimensions, and format, with fit/zoom and a visible transparency background.
2. **Given** a beta texture requiring companion data or an unsupported encoding, **When** selected, **Then** the browser reports that exact limitation and keeps the original bytes available for extraction rather than inventing a successful preview.
3. **Given** repeated selection or closing/reopening a source, **When** a pending preview finishes, **Then** it cannot replace a newer selection or retain the previous source's image as if it were current.

### User Story 3 - Read Build-Process Text (Priority: P1)

As a researcher, I want to open build-process text and tables alongside textures so that I can inspect original filenames and short-name references without renaming the corpus.

**Why this priority**: The operator explicitly values the beta build text and its historical filename evidence.

**Independent Test**: Select one real build text/table file, read the displayed bounded text, and compare its bytes and stored name with the source.

**Acceptance Scenarios**:

1. **Given** a supported text/table file, **When** selected, **Then** the File Browser shows readable, scrollable content with its stored path and any encoding/truncation notice.
2. **Given** source text referring to an original name or 8.3 alias, **When** viewed or copied out, **Then** the original text and stored filename remain unchanged; guessed automatic alias resolution or renaming is not performed.

### User Story 4 - Unpack Selected Files Safely (Priority: P1)

As a researcher, I want to unpack selected archive files, or copy selected already-extracted files, into an explicit output directory with their source-relative layout intact.

**Why this priority**: The operator requested unpacking, not merely listing filenames.

**Independent Test**: Extract one texture and one build text/table file and compare their hashes with the exact bytes read from the selected source; exercise invalid paths and existing destination files.

**Acceptance Scenarios**:

1. **Given** selected files and an output directory, **When** extraction completes, **Then** each successful output is byte-identical to its source payload and retains a safe relative path; the UI reports successes, failures, and output locations.
2. **Given** an escaping/rooted path, a source-directory destination, or an existing output, **When** extraction is requested, **Then** unsafe writes are rejected and overwrite is never silent.
3. **Given** cancellation or a file failure, **When** an operation stops, **Then** the viewer remains usable and does not report a partial file as successful.

### Edge Cases

- Extracted beta data is not a modern installed game and may not contain any archives or executable version information.
- Archive filename discovery and content decoding are separate capabilities. A listfile supplies names, not missing data or new codecs.
- Supplemental listfiles may include names absent from a particular beta; internal names may be incomplete. Absence from a listfile does not prove absence from an archive.
- Beta texture signatures, external mip companions, palettized alpha, and four-channel JPEG behavior require real input validation; shared BLP signatures alone do not establish support.
- Short names must not be silently expanded, normalized into unrelated paths, or overwritten using speculative historical mappings.
- Path traversal, absolute paths, UNC/drive-relative names, duplicate case-insensitive destinations, symbolic/reparse-point escapes, and cancelled writes require safe extraction behavior.
- Unknown binary files need truthful metadata/unsupported-preview messages, not text decoding or model/world loading by default.
- A modern CASC root must be detected/reported separately; this phase does not silently promise Reforged compatibility or enable network/CDN fetching.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to open WC3 extracted data and supported installed/MPQ directories from the existing File Browser without a WoW build picker or world initialization.
- **FR-002**: Archive access MUST reuse the existing MPQ/listfile implementation. The workflow MUST accept internal listfiles and a user-selected WC3 supplemental listfile, including the operator-named Zezula source; names MUST be checked against the opened source before being listed as files.
- **FR-003**: The browser MUST expose full stored relative/virtual paths, source identity, search, a tileset/texture filter, and an all-files/text route; it MUST distinguish incomplete name coverage from an empty source.
- **FR-004**: Supported WC3 tileset images MUST preview as source images/atlases within the file-viewer surface, showing format/dimensions, fit/zoom, and transparency without world-renderer changes or atlas-to-map composition. The operator's beta follow-up adds BLP0 external mip companions and isolated MDX file preview through the existing reader/renderer, not the world loader.
- **FR-005**: The browser MUST display bounded readable text/table content with encoding/truncation status and preserve original filenames and bytes; automatic historical filename reconstruction is not required.
- **FR-006**: Users MUST be able to unpack/copy selected files to an explicit output directory, preserving safe relative paths and exact source bytes, with no silent overwrite and per-file outcomes.
- **FR-007**: Extraction MUST prevent writes outside the selected output root and into the source data, including traversal/reparse-point escapes, and MUST not label partial/cancelled writes successful.
- **FR-008**: Unsupported layouts/codecs/companions and missing listfiles MUST produce actionable messages. Raw extraction MUST remain independent of whether an image/text preview is supported.
- **FR-009**: Source/selection changes MUST cancel or invalidate stale work and release source handles/preview resources. WC3 browsing MUST not replace or mutate the current WoW source, camera, or world scene.
- **FR-010**: Real compatibility, pixel fidelity, extraction fidelity, and UI isolation MUST have identified real input/output evidence; synthetic tests/builds alone MUST not close these claims.

### Key Entities *(include if feature involves data)*

- **WC3 file source**: Configured root, loose/archive kind, dataset identity, listfile provenance/coverage, and read-only lifetime.
- **File entry**: Stable relative/virtual path, byte availability, source provenance if available, and preview capability.
- **Preview**: Selected source/entry identity, image dimensions/pixels or bounded text, decode status, and selection generation.
- **Extraction request/result**: Selected entries, explicit destination, collision policy, per-file output/hash/status, and cancellation outcome.
- **Listfile provenance**: Original URL/local selection, retrieval hash/date, and compatibility limits; not game content.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The operator can open the supplied extracted WC3 beta directory, locate at least one real terrain texture and one real build text/table file, and inspect both in the File Browser without a world load or WoW build prompt.
- **SC-002**: Representative real texture previews match independently checked dimensions, orientation, colors, and alpha; unsupported beta variants are explicitly labeled instead of falsely accepted.
- **SC-003**: For an available real WC3 MPQ input, the browser lists only confirmed names from internal/supplemental listfiles and can read at least one texture payload through the unchanged MPQ reader; missing-name coverage is reported truthfully.
- **SC-004**: At least one real texture and one real text/table file unpack/copy with hashes matching their source payloads; every planned malicious-path/collision/cancellation case is rejected or reported without an escaped, silently overwritten, or falsely successful file.
- **SC-005**: Repeated source/selection changes never show a stale preview as current; opening/closing the WC3 browser leaves the existing WoW session and world unchanged during operator verification.

## Assumptions

- The supplied extracted beta corpus is the first real-data target. Classic WC3 MPQ browsing reuses the existing reader with Warcraft-specific supplemental names; no new archive parser is authorized.
- Listfile retrieval is user-visible/configurable, not a startup network dependency. A download failure does not prevent loose-file browsing or local listfile selection. Redistribution permissions must be established before bundling third-party lists in releases.
- All client roots and output roots are runtime selections. Exact operator-local paths are recorded only in scoped evidence, never hardcoded in application/test defaults.
- The current slice is file browsing, texture/text and isolated MDX preview, and byte-preserving unpacking. Map/world rendering, terrain conversion, texture ancestry/synthesis, coordinate teleport helpers, WC2, and WC1 remain deferred; they do not gate this slice.
- Prior GDS/WC1 lineage input and the [restoration addendum](operator-addendum.md) remain historical/future references, not current implementation requirements.
- Existing reader freezes, owned-service boundaries, shared UI primitives/inventory registration, no god-class growth, read-only game containers, and operator-owned real visual proof remain binding.
- Technical choices are in the [plan](plan.md), with implementation in [tasks](tasks.md), the [initial receipt](evidence/receipt-wc3-implementation.md), and [beta follow-up](evidence/receipt-beta-blp0-and-mdx-2026-10-15.md). BLP0 header/mip decoding now works on the selected real sample, and the MDX framing adapter parses its adjacent model. Isolated MDX rendering is wired; actual in-viewer appearance remains operator-owned. Shared WoW readers/renderers are unchanged.
