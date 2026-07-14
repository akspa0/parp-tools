# Feature Specification: Legacy M2 model rendering (client 0.11 – 2.4.3)

**Feature Branch**: `104-legacy-m2-rendering` (work lands on `v0.5.0-prerelease`, per project convention)

**Created**: 2026-07-14

**Status**: Draft

**Input**: User description: "M2 model rendering support for legacy client versions 0.11 through 2.4.3 (WoW alpha through end of The Burning Crusade). The viewer parses these M2 files enough to render bounding boxes, but the actual geometry (mesh) and materials do NOT render. Root cause already identified: `M2ModelReader` hardcodes `embeddedSkinProfileCount`/`embeddedSkinProfileOffset` to 0; for M2 version ≤ 263 the skin/geometry profiles are embedded inside the .m2 itself (nViews/ofsViews), not in external .skin files. Investigate per format-version boundary, well-documented versions first (2.4.3, 1.12.1), then mid-range, then the sparsely-documented early alphas (1.0.0, 0.12, 0.11) using dynamic runtime tracing. x64dbg MCP is available; Ghidra is not installed."

## Overview

Models from WoW client builds **0.11 through 2.4.3** currently load as empty bounding boxes in
the viewer: the header and bounds parse, but no mesh and no materials appear. This feature is the
investigation and implementation needed to render these legacy models correctly. It is
research-heavy: the concrete deliverable is a per-format-version understanding of the M2 header
layout and the **embedded skin profiles** (geometry submeshes, triangle indices, texture-unit /
material bindings) that these older formats store inside the `.m2` file itself, plus the reader
changes that consume them.

The scope discriminator is the **M2 format version** (the `uint32` at header offset `0x04`), not
the client build string — several builds share a format version, and the format changed at known
boundaries. WotLK (version 264+) externalized skin profiles to `.skin` files; every version in
scope (≤ 263) keeps them embedded via `nViews` / `ofsViews`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Render mesh + materials for late-format legacy models (2.4.3 and 1.12.1) (Priority: P1)

Someone opens a character, creature, or doodad M2 from a staged 2.4.3 (TBC) or 1.12.1 (Vanilla)
client and sees the actual textured model — not an empty box. These two versions are chosen first
because their M2 format is well-documented on wowdev.wiki and has multiple known-good reference
implementations (WoW Model Viewer, others) to validate byte offsets and rendering against without
needing a debugger.

**Why this priority**: This is the MVP and the highest-leverage, lowest-risk slice. The embedded-
skin reading path built and proven here is the same mechanism every other in-scope version reuses;
getting it right against documented formats de-risks everything downstream. It also covers the two
most-used legacy eras (Vanilla and TBC), which is the bulk of the value.

**Independent Test**: Load a set of known 2.4.3 and 1.12.1 M2s (character, creature, WMO doodad,
simple static prop) in the viewer and confirm each renders visible geometry with correctly bound
textures, matching a reference implementation's output for the same file. Fully testable and
valuable on its own without touching any earlier version.

**Acceptance Scenarios**:

1. **Given** a 2.4.3 creature M2 with embedded skin profiles, **When** it is loaded, **Then** the
   viewer renders its mesh triangles with the correct textures bound per submesh, not a bounding box.
2. **Given** a 1.12.1 character M2, **When** it is loaded, **Then** submeshes render and the model's
   silhouette and texturing visually match a known-good reference render of the same file.
3. **Given** an M2 whose embedded skin data is malformed or truncated, **When** it is loaded,
   **Then** the viewer degrades gracefully (renders what it can or the bounding box) without crashing.

---

### User Story 2 - Extend rendering to the mid-range in-scope versions (2.0.0 alpha, 2.1, 2.2, 2.3) (Priority: P2)

Models from the intermediate TBC-era and 2.0 alpha builds render correctly, having confirmed the
embedded-skin layout established in Story 1 holds (or documenting the deltas where it does not).

**Why this priority**: These sit in the same format family as 2.4.3 with at most minor field drift,
so they should largely "come for free" once Story 1 works — but they must be explicitly verified,
not assumed, because alpha builds are notorious for undocumented field shifts.

**Independent Test**: Load representative M2s from each of 2.0.0 alpha, 2.1, 2.2, and 2.3 and
confirm mesh + materials render; record any per-version offset deltas discovered.

**Acceptance Scenarios**:

1. **Given** an M2 from each mid-range version, **When** it is loaded, **Then** mesh and materials
   render correctly using the Story 1 reading path, or the specific format delta that prevents it is
   documented and handled.

---

### User Story 3 - Crack the sparsely-documented early alphas (1.0.0, 0.12, 0.11) (Priority: P3)

Models from the earliest alpha builds — where wowdev.wiki documentation is thin or absent — render
correctly, with their header and embedded-skin layouts recovered through dynamic runtime tracing of
the actual game client (x64dbg) where static documentation runs out.

**Why this priority**: Highest effort and uncertainty, lowest volume of assets, and depends on the
reading infrastructure from Stories 1–2 being solid first. This is the genuine reverse-engineering
frontier where the debugger earns its keep.

**Independent Test**: Load M2s from 1.0.0, 0.12, and 0.11 staged clients and confirm they render;
the recovered header/skin layout for each is captured in the research notes with the evidence
(documented offsets or a traced client load path) that established it.

**Acceptance Scenarios**:

1. **Given** the header layout for an early-alpha M2 version is unknown, **When** the actual client
   loading that version is traced, **Then** the field offsets and embedded-skin structure are
   recovered and recorded well enough to parse the file.
2. **Given** the recovered layout for 1.0.0 / 0.12 / 0.11, **When** an M2 of that version is loaded,
   **Then** mesh and materials render, or the remaining unknown is precisely documented.

---

### Edge Cases

- A model reports view/skin-profile counts that exceed the file size or point outside the file
  (corrupt or misidentified format) — must be detected and rejected without a crash.
- A version-boundary model (e.g. an early 264 that still embeds, or a late 263 that references
  externally) is misclassified by the version number alone — the reader must validate the embedded
  offsets are sane before trusting them.
- Vertex structure differences across eras (bone weights/indices packing) cause geometry to render
  but deform incorrectly — distinct from the "empty box" failure and tracked separately.
- A staged client for a target version is missing or incomplete under `output/tmp/wowarchive-clients/`
  — the investigation for that version is blocked, not failed, and reported as such.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The M2 reader MUST detect the M2 format version from the header and select the
  correct header/skin layout for that version rather than assuming a single (WotLK-era) layout.
- **FR-002**: For every in-scope version (M2 format version ≤ 263, client 0.11–2.4.3), the reader
  MUST read the **embedded** skin profiles (via the file's own view/skin count and offset fields)
  instead of hardcoding the embedded-skin count and offset to zero.
- **FR-003**: The reader MUST extract, from each embedded skin profile, the geometry needed to
  render: submesh definitions, the triangle index set, and the texture-unit / material bindings that
  associate submeshes with textures.
- **FR-004**: The viewer MUST render the extracted mesh with its materials/textures for in-scope
  models, replacing the current bounding-box-only output.
- **FR-005**: The reader MUST validate embedded offsets and counts against the file bounds and fail
  safe (graceful degradation, no crash) on malformed or truncated data.
- **FR-006**: The investigation MUST proceed in the documented priority order — well-documented
  versions (2.4.3, 1.12.1) first, then mid-range, then early alphas — and MUST record, per format
  version, the confirmed header field offsets and embedded-skin structure with the evidence that
  established them (documentation reference or traced client behavior).
- **FR-007**: For versions where documentation is insufficient, the investigation MUST use dynamic
  runtime tracing of the actual client (x64dbg) to recover the layout, and MUST capture the findings
  in a durable research artifact usable by the reader implementation.
- **FR-008**: Rendering correctness for the well-documented versions MUST be validated against at
  least one independent known-good reference implementation's output for the same source files.
- **FR-009**: Existing WotLK+ (version 264+) M2 rendering, which already works, MUST NOT regress.

### Key Entities *(include if feature involves data)*

- **M2 header**: The fixed-layout front of the .m2 file; carries the format version, bounds (already
  parsed correctly), and the count/offset fields for all sub-structures including views/skin profiles.
  Field positions shift across format versions.
- **Embedded skin profile (view)**: A per-file geometry description (present inside the .m2 for
  in-scope versions) containing submesh definitions, the triangle index list, and texture-unit bindings.
  This is the data the current reader never reads and whose absence produces the empty-box symptom.
- **Submesh (geoset)**: A contiguous slice of the mesh sharing a material/texture; the unit of
  rendering and material binding.
- **Texture-unit / material binding**: The association from a submesh to the texture(s) and render
  state used to draw it.
- **Format-version profile**: The per-M2-version record of confirmed header offsets and embedded-skin
  structure — the research deliverable that the reader consumes and that each version tier populates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 2.4.3 and 1.12.1 M2 models render visible mesh and correctly-bound materials in the
  viewer instead of an empty bounding box (Story 1 done).
- **SC-002**: For the well-documented versions, the viewer's render of a representative sample of
  models visually matches a known-good reference implementation's render of the same files.
- **SC-003**: Mesh + materials render for all mid-range in-scope versions (2.0.0 alpha, 2.1, 2.2,
  2.3), or every version that does not is accompanied by a documented, specific format delta.
- **SC-004**: At least the early-alpha versions with staged clients (1.0.0, 0.12, 0.11) either render
  mesh + materials, or have their remaining unknowns precisely documented with the evidence gathered.
- **SC-005**: Every in-scope format version has a recorded profile of its confirmed header offsets and
  embedded-skin structure, each entry traceable to documentation or a client trace.
- **SC-006**: No regression: WotLK+ (264+) models that render today still render after the changes.
- **SC-007**: No malformed or misidentified legacy M2 causes the viewer to crash; worst case is a
  graceful fallback to bounding-box or partial render.

## Assumptions

- The single already-identified root cause (embedded skin profiles hardcoded to zero) is necessary
  but may not be sufficient for the earliest alphas; additional header-offset differences are expected
  and are part of the investigation, not a surprise.
- Staged clients for the in-scope versions are available under `output/tmp/wowarchive-clients/`; where
  a specific version's client is missing, that version's investigation is blocked and reported, not failed.
- x64dbg (with the automate MCP bridge, already configured and responding) is the available dynamic-
  analysis tool; Ghidra is not installed, so static disassembly is out of scope unless Ghidra is
  installed as a separate setup step first.
- wowdev.wiki and existing open reference implementations are accepted sources of truth for the
  well-documented versions and as validation baselines.
- Animation, particles, ribbons, attachments, and bone-driven deformation correctness are out of scope
  for this feature — the goal is static mesh + material rendering. Vertex-packing/deformation issues, if
  they surface, are tracked separately from the empty-box problem.
- This is deliberately investigation-first: understanding and documenting each version's layout is a
  first-class deliverable, not just the code changes.
