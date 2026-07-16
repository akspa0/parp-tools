# Feature Specification: Viewer Stabilization

**Feature Branch**: `110-viewer-stabilization`

**Created**: 2026-07-16

**Status**: Draft

**Input**: User description: "Stabilize the WoW viewer: expose usable fog start and end overrides so maps without LIT remain visible, prevent time-of-day terrain disappearance, provide LIT-entry overlays and navigation in both minimap views, export terrain-derived minimaps at a selected time of day, restore native M2 rendering for 1.0.0 and later clients, remove obsolete dataset tools, repair modern converter and inspect tool entry points, and document bounded WMO and M2 conversion contracts separately from rendering."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Keep every loaded map visible (Priority: P1)

A viewer user can inspect any loaded map regardless of whether it has lighting data. They can adjust fog start and fog end directly, and a selected time of day never makes terrain disappear.

**Why this priority**: Invisible terrain makes the viewer unusable and blocks all other work.

**Independent Test**: Load one map with lighting data and one without it; set several times of day and fog ranges; verify that terrain remains visible and that the displayed range changes take effect immediately.

**Acceptance Scenarios**:

1. **Given** a map without a LIT file or other usable lighting profile, **When** it loads, **Then** it receives a visible fallback fog range rather than a zero-width fog range.
2. **Given** any loaded map, **When** the user changes fog start or fog end, **Then** the requested valid range takes effect without being silently overwritten by lighting evaluation.
3. **Given** a time-of-day sample with missing, invalid, or degenerate lighting values, **When** the scene renders, **Then** terrain remains visible with a safe fallback rather than becoming fully clipped, fully fogged, or non-drawn.
4. **Given** a user enters an invalid fog range, **When** the setting is applied, **Then** the viewer shows a valid ordered range and explains any adjustment it made.

---

### User Story 5 - Inspect and navigate LIT entries (Priority: P1)

A viewer user can see positional LIT entries over the current minimap and full-screen minimap, select an entry, and double-click it in a list to move the 3D camera to a safe viewing point above that entry.

**Why this priority**: Lighting data must be inspectable spatially before it can be trusted to explain a map's appearance.

**Independent Test**: Load a map with a positional LIT file, enable the overlay in the Lighting surface, verify the same markers and selected state appear in both minimap views, then double-click a list row and verify the 3D camera focuses that entry.

**Acceptance Scenarios**:

1. **Given** a loaded LIT file with positional entries, **When** the user enables LIT minimap markers, **Then** both minimap views show the same in-bounds markers without changing lighting or fog behavior.
2. **Given** an LIT marker or entry-list row, **When** the user selects it, **Then** its selected state is visible in the Lighting list and both minimap views.
3. **Given** a positional LIT entry in the list, **When** the user double-clicks it, **Then** the 3D camera moves to a safe, downward-looking point above the entry and reports the selected entry.
4. **Given** a default or invalid-position LIT entry, **When** the user views the list, **Then** its non-navigable state is explicit and it does not produce a misleading minimap marker.

---

### User Story 6 - Export terrain-derived minimaps (Priority: P1)

A viewer user can export aligned terrain-only and liquid-bearing minimaps for any on-disk map at a selected time of day,
even when the client does not ship a minimap asset for that terrain. They can request individual
tile PNGs, one combined map PNG, or both from the Tools menu.

**Why this priority**: Missing or stale authored minimaps should not make on-disk terrain
uninspectable, particularly in Alpha-era clients such as 0.5.3.

**Independent Test**: Use a small fixture pack to prove MCAL/MCLY compositing and whole-map
stitching, then have the user run one real map export with and without a usable LIT profile.

**Acceptance Scenarios**:

1. **Given** a client map with terrain tiles but no minimap assets, **When** the user exports a
   synthesized minimap, **Then** each terrain baseline PNG is composed directly from the tile's MTEX/MCLY/
   MCAL data and BLP pixels rather than from an authored minimap image.
2. **Given** a selected minute-precise time of day and a readable supported LIT profile, **When** the export runs,
   **Then** the PNGs use the profile's global clear-weather colors and record the source, selected
   time, and remaining authored lighting assumptions in a manifest.
3. **Given** no usable LIT profile, **When** the export runs, **Then** it still emits visible
   terrain using a clearly labeled authored day/night fallback; it does not claim client-exact
   lighting.
4. **Given** an export configured for tiles and a whole map, **When** it completes, **Then** it
   writes one PNG per occupied terrain tile and a single stitched PNG covering their map-coordinate
   bounds, leaving missing tiles transparent instead of inventing terrain.
5. **Given** an authored minimap tile and a complete decoded terrain-material baseline, **When** the
   tile enters a dataset stream, **Then** its metadata records tint/shadow evidence and an optional
   LIT-chroma time bucket without claiming an exact historical capture time.
5. **Given** overlapping MCAL overlays or high-frequency tiled BLP inputs, **When** a terrain tile
   is synthesized, **Then** its layer ordering matches the terrain renderer, while each minimap
   material colour is a phase-independent BLP average so the
   result contains neither normalized-layer blends nor aliasing/interpolation artifacts.
6. **Given** a bounded export whose first occupied WDT slot cannot be decoded, **When** the user
   requests one output tile, **Then** the skipped slot is recorded but does not consume the output
   limit before a synthesizable terrain tile is reached.
7. **Given** an Alpha WDT, **When** the exporter enumerates occupied MAIN cells, **Then** it uses
   the same row-major `(tileX, tileY)` coordinates as the Alpha tile reader and does not transpose
   valid sparse-map entries into false decode failures.
8. **Given** synthesized terrain lighting uses the authored solar fallback, **When** time is noon,
   **Then** its direction is vertical; immediately after noon its source projects from the raster's
   top-left rather than carrying a permanent diagonal bias.
9. **Given** a synthesized terrain tile, **When** it is emitted, **Then** the exporter also emits an
   aligned `_liquid` PNG using decoded unified liquid coverage and resolved basic liquid types,
   while retaining the terrain-only baseline separately. The liquid target records its rendered
   pixel count and does not claim native water-material or animation parity. Alpha MCLQ coverage
   MUST honor its 8×8 per-chunk cell visibility flags, and a rendered liquid pixel MUST represent
   a complete covered source cell rather than a single wet vertex on a dry-terrain boundary.
10. **Given** an otherwise readable tile whose MTEX path is absent, undecodable, or lacks a usable
   MCLY/MTEX table, **When** narrow same-name and related-texture recovery cannot resolve it,
   **Then** the exporter uses a successfully decoded deterministic catalog RGB proxy rather than
   dropping the tile. The manifest records the original reference and `catalog_rgb_last_resort_proxy`.
11. **Given** a placement lies on a tile edge, **When** its Alpha roof/object-mask footprint is
   rasterized into a 256² minimap buffer, **Then** the footprint is clipped to that buffer and
   cannot make terrain decode fail from a 257² object-mask assumption.

---

### User Story 7 - Keep interactive controls visible and reachable (Priority: P1)

A viewer user can see a real grab on each fog slider and can always reach the UniqueId playback
transport from Tools > Archeology without a nested-tab change switching them to another tool.

**Why this priority**: A control that responds to blind dragging, or a playing range with no visible
stop action, is functionally unavailable and makes the viewer state unsafe to inspect.

**Independent Test**: Open Lighting and confirm each active fog range field has a visible track and
grab. Start UniqueId playback, move between Range/Layers/Playback/Capture, and pause or stop it
from both the Playback tab and the active-playback status strip.

**Acceptance Scenarios**:

1. **Given** an active world, **When** the user opens Lighting, **Then** Fog Start and Fog End are
   visible slider controls with a rendered grab, not drag-only numeric fields.
2. **Given** an active UniqueId playback run, **When** the user is on any Archeology subtab,
   **Then** pause and stop remain visible and do not require returning to World.
3. **Given** the user selects Range, Layers, Playback, or Capture under Tools > Archeology,
   **When** the nested tab changes, **Then** the parent Tools selection remains Archeology.
4. **Given** the legacy interface is active, **When** the user invokes UniqueId Archeology from the
   menu, **Then** its dedicated Archaeology window opens with playback transport instead of leaving
   controls embedded in World.

---

### User Story 2 - Render M2 assets through their native path (Priority: P1)

A viewer user opening an M2 from a 1.0.0-or-later client sees a visible, native M2 render or a precise diagnostic explaining the unsupported format capability. The viewer never treats format conversion as its rendering path.

**Why this priority**: M2 is the primary world-object format; invisible objects make world inspection incomplete.

**Independent Test**: Open representative 1.0.0, 1.12.1, 2.4.3, and WotLK-or-later M2 assets and verify either visible geometry/materials or a specific version-bound failure; verify the displayed renderer is native M2 for every successful case.

**Acceptance Scenarios**:

1. **Given** a valid 1.0.0 M2, **When** it loads, **Then** the viewer submits its decoded division to the native M2 rendering path and does not construct an MDX compatibility render model.
2. **Given** a valid supported post-1.0.0 M2, **When** it loads, **Then** it uses the appropriate native M2 reader and rendering route for its version.
3. **Given** an M2 whose required render data cannot be decoded, **When** it loads, **Then** the viewer reports the exact format/version/failed capability instead of showing a silent empty scene.
4. **Given** a conversion command exists for an M2, **When** a user runs it, **Then** its output is an explicit file-export artifact and it is never used as an implicit renderer fallback.

---

### User Story 3 - Use only working modern tool surfaces (Priority: P2)

A viewer user sees a compact Tools menu containing only supported current workflows. Inspect and conversion operations open their modern entry points and clearly report missing dependencies before the user attempts a non-working operation.

**Why this priority**: Dead dataset/ML menu entries and missing tool binaries make the application look functional when it is not.

**Independent Test**: Enumerate every Tools menu item in both UI modes, invoke each item, and verify it either opens a working current surface or is absent by design.

**Acceptance Scenarios**:

1. **Given** the main Tools menu, **When** it is displayed, **Then** obsolete MK Dataset and VLM Dataset entries are absent.
2. **Given** an inspect or conversion action, **When** the user invokes it, **Then** the application reaches its modern in-repository tool path or presents an actionable dependency diagnostic.
3. **Given** a legacy-only tool has no supported modern replacement, **When** the menu is shown, **Then** the dead entry is removed instead of left as a broken launcher.

---

### User Story 4 - Export conversion is explicit and bounded (Priority: P2)

A user can determine which WMO v14/v17 and M2-to-MDX conversions are supported, what each conversion preserves, and which result is suitable for Alpha export. Conversion capability is verified independently from viewer rendering.

**Why this priority**: Reliable export is useful, but it must not become a hidden substitute for native asset support.

**Independent Test**: Run supported fixture conversions, inspect the output summaries, and verify the documented preservation and failure behavior without loading converted output as a renderer workaround.

**Acceptance Scenarios**:

1. **Given** a supported WMO v14 or v17 fixture, **When** the user selects a direction, **Then** the tool produces the corresponding supported output or a precise unsupported-feature report.
2. **Given** a supported M2-to-MDX export request for Alpha, **When** conversion completes, **Then** the output has a recorded source identity, conversion profile, and validation result.
3. **Given** a feature that cannot be preserved in a target format, **When** conversion is requested, **Then** the tool fails clearly or records the loss; it does not claim reliable conversion.

### Edge Cases

- Lighting values are non-finite, negative, or collapse fog start and end to the same value.
- A map provides no LIT file, or a LIT file is present but lacks a usable sample for the chosen time.
- A M2 shares a format number with a different layout or has truncated embedded render data.
- A tool executable or optional dependency is absent from an otherwise valid checkout.
- A WMO or M2 contains target-format features that have no faithful conversion representation.
- A terrain tile has no readable texture layer or one referenced BLP fails to decode.
- A selected export time is outside a 24-hour day, or a LIT file is present but has no globally
  evaluable clear-weather color tracks.
- Playback is active when the user changes Archeology subtabs, unloads the world, or changes a
  visible UniqueId endpoint manually.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST expose fog start and fog end as user-adjustable controls whenever a world is loaded.
- **FR-002**: The viewer MUST keep user-selected fog overrides separate from lighting-derived recommendations and MUST make the active source visible to the user.
- **FR-003**: The viewer MUST maintain a finite, ordered, non-zero active fog range; maps without usable lighting data MUST receive a visible fallback range.
- **FR-004**: A time-of-day or lighting-evaluation failure MUST NOT make terrain non-drawn, fully clipped, or fully obscured.
- **FR-005**: The viewer MUST provide a concise diagnostic when it substitutes safe lighting or fog values.
- **FR-006**: Supported 1.0.0-or-later M2 assets MUST render through native M2 readers and a native M2 renderer.
- **FR-007**: The viewer MUST distinguish a format-reader failure, unavailable render capability, and a rendering failure in its M2 diagnostics.
- **FR-008**: The viewer MUST NOT use M2-to-MDX conversion as a renderer fallback.
- **FR-009**: The main Tools menu MUST exclude obsolete MK Dataset and VLM Dataset workflows.
- **FR-010**: Every remaining inspect and conversion command MUST resolve to a supported in-repository implementation or an actionable dependency diagnostic.
- **FR-011**: The project MUST publish a versioned WMO v14/v17 conversion capability table, including supported directions, fidelity limits, and validation evidence.
- **FR-012**: The project MUST publish an M2-to-MDX Alpha-export contract that identifies supported source profiles, preserved data, known loss, and validation evidence.
- **FR-013**: Conversion proof and renderer proof MUST remain separate in user-facing diagnostics, tests, and documentation.
- **FR-014**: The viewer MUST expose an opt-in LIT marker overlay that uses the loaded LIT entry positions and is shared by the regular and full-screen minimap views.
- **FR-015**: Enabling, disabling, selecting, or navigating LIT markers MUST NOT modify active lighting, fog selection, terrain loading, or renderer routing.
- **FR-016**: The viewer MUST show a navigable list of LIT entries with name, index, position, and spatial extent, while clearly identifying default or invalid-position entries as non-navigable.
- **FR-017**: Selecting a positional LIT entry from the list or minimap MUST highlight the same entry in the list and both minimap views.
- **FR-018**: Double-clicking a navigable LIT list entry MUST move the 3D camera to a safe viewing point above that entry and preserve a usable downward-looking orientation.
- **FR-019**: The Tools menu MUST expose a terrain-derived minimap export that accepts a configured
  client root, map name, minute-precise time of day, output location, and independent
  per-tile/whole-map targets. It MUST accept compact `HHmm` (for example `1215`), `HH:mm`, and
  existing decimal-hour values.
- **FR-020**: The export MUST compose minimap pixels from decoded terrain texture layers and alpha
  masks; it MUST NOT require or substitute a client-authored minimap image.
- **FR-021**: The export MUST use the global clear-weather LIT profile when it can be evaluated at
  the selected time and otherwise use a visible labeled authored fallback. It MUST record which
  source it used and MUST NOT claim unproven local-zone lighting as client-exact.
- **FR-022**: A whole-map export MUST stitch exactly the emitted terrain-tile outputs into one PNG
  with explicit tile-coordinate bounds and transparent missing-tile regions.
- **FR-023**: The export MUST write a machine-readable manifest with client build identity, source
  map, selected time, lighting evidence state, output dimensions, tile bounds, and per-tile result.
- **FR-023k**: The authored direction used with LIT color tracks or no-LIT fallback MUST share the
  terrain world/minimap-raster coordinate contract: it MUST be vertical at noon, vary with time,
  and record its authored evidence state because early LIT profiles do not provide a client sun
  vector.
- **FR-023l**: Every synthesized terrain tile MUST have an aligned liquid-bearing companion PNG
  whose filename ends in `_liquid.png`; whole-map export MUST stitch an equivalent `_liquid` map.
  The companion MUST use decoded unified liquid coverage and resolved basic types when available,
  record its render profile and liquid-pixel count in the manifest, and preserve the liquid-free
  terrain baseline as a separate output. Its coverage MUST be cell-complete: Alpha MCLQ 8×8 cell
  flags are authoritative where available, and isolated coverage vertices MUST NOT produce a
  liquid strip along dry terrain-cell boundaries. It MUST NOT claim parity with client water
  textures, animation, reflection, or shader effects.
- **FR-023m**: A missing/undecodable terrain BLP or absent usable MTEX table MUST NOT skip an
  otherwise readable tile solely for lack of material RGB. Recovery MUST try same-stem and related
  candidates first, then a successfully decoded deterministic catalog BLP selected by source-folder
  and terrain-family affinity. The original reference, resolved path, and
  `catalog_rgb_last_resort_proxy` kind MUST remain in the per-tile metadata.
- **FR-023n**: Alpha object/roof-mask rasterization MUST validate against each destination buffer's
  actual dimensions, not the 257² terrain height grid. An edge placement MUST not abort terrain
  decoding merely because roof masks are 256².
- **FR-023a**: The compositor MUST compose MCAL overlays in terrain-layer order. At minimap scale,
  it MUST use each decoded BLP's phase-independent material average rather than sampling the
  terrain renderer's repeated diffuse UVs; output MUST be invariant under diffuse-repeat phase and
  must not introduce moire, static-like aliasing, or bilinear/trilinear interpolation artifacts.
- **FR-023c**: Alpha-era MCLY data MUST be normalized from its native `[chunkX, chunkY, layer]`
  layout to the tensor-pack `[chunkY, chunkX, layer]` layout before composition. The compositor
  MUST honor `MclyLayerMask` and MUST NOT blend a missing MCLY layer merely because its default
  texture-id slot is zero.
- **FR-023d**: When MCNR arrives as Alpha's sparse staggered terrain-vertex lattice, the compositor
  MUST evaluate the Lambert term at real vertices and interpolate it over the terrain triangles. It
  MUST NOT treat the alternating dense-raster gaps as up-facing normals. The MCSH 256² occupancy
  signal remains a separate model target and is not rewritten by this visual interpolation.
- **FR-023e**: Default synthesized minimap RGB MUST omit MCSH. MCSH remains a decoded static-shadow
  signal for diagnostics and training; an MCSH-baked RGB preview is permitted only through explicit
  opt-in and MUST be labeled as an exceptional historical-minimap diagnostic, not the normal export
  contract.
- **FR-023f**: When an authored minimap RGB and complete decoded terrain-material baseline are
  available, dataset metadata MUST record a versioned minimap-lighting inference: status, tint RGB,
  tint strength/fit, MCSH darkening correlation, and an optional LIT-global-clear chroma time bucket.
  The time bucket MUST carry explicit inference evidence and MUST NOT claim an exact capture time.
  Missing RGB, incomplete terrain textures, or unavailable LIT candidates MUST yield an explicit
  not-evaluated/low-evidence status rather than fabricated fields.
- **FR-023g**: Terrain texture pixel sidecars MUST be emitted only when they remain index-aligned
  with the full MTEX name table. If any texture cannot decode, serializers MUST omit the shifted
  sidecar table and record its incomplete state in metadata.
- **FR-023h**: A readable terrain tile with MCLY base-layer material but no MCAL payload MUST remain
  exportable as base-layer-only terrain. The compositor MUST NOT invent overlay alpha; it continues
  to apply available normal/LIT lighting and records the normal tile result.
- **FR-023i**: A malformed or differently sized MCNR validity mask MUST NOT abort minimap export.
  Normal samples outside that mask are treated as unknown/neutral while valid normal data continues
  to contribute; the compositor MUST not index a mask beyond its declared bounds.
- **FR-023j**: When a referenced terrain diffuse BLP cannot decode, terrain consumers MUST try the
  original path first and MAY then use a successfully decoded RGB proxy in deterministic order:
  (1) the same-stem `_s.blp` companion, then (2) no more than sixteen ordinary `.blp` assets from
  the loaded archive/listfile catalog. The second tier MUST scan moved paths and rank exact or
  strongly similar basenames ahead of shared directory-theme tokens; it MUST exclude material-only
  suffixes such as `_s`, `_n`, and `_h` and reject candidates without a strong basename relation.
  Derived minimap/dataset paths MUST retain the original MTEX path/ID and
  record requested/resolved paths plus either `specular_companion_rgb_proxy` or
  `related_diffuse_rgb_proxy`. The terrain viewer MUST apply the same recovery order and log its
  selected proxy. Neither consumer may claim that specular, alpha, blend, or native engine material
  semantics were reproduced.
- **FR-023b**: The synthesis `--limit` option MUST limit emitted terrain-tile PNGs rather than
  WDT candidates attempted; skipped or failed slots MUST remain in the manifest and processing MUST
  continue until the requested number of outputs is emitted or candidates are exhausted.
- **FR-024**: Active fog Start and End controls MUST use visible slider tracks and grabs; drag-only
  numeric controls are not an acceptable replacement.
- **FR-025**: UniqueId range/layer/playback/capture controls MUST be owned by Tools > Archeology
  and MUST NOT be duplicated in the World surface.
- **FR-026**: Archeology nested-tab selection MUST be independent from the parent Tools selection.
  While playback is active, pause and stop MUST remain available on every Archeology subtab; world
  unload or an unavailable range MUST stop playback safely.

### Key Entities *(include if feature involves data)*

- **Fog override**: A user-selected start/end range, its active/inactive state, and the visible source of the range.
- **Lighting sample**: The time-specific light values obtained from client data or a fallback policy.
- **M2 render capability**: The version-bound reader result, decoded render data availability, native render-route result, and diagnostic outcome.
- **Tool surface**: A visible launcher, its current owner, its availability status, and its replacement or removal decision.
- **Conversion capability profile**: A source/target format direction, preserved data categories, known loss, fixture evidence, and failure behavior.
- **LIT map inspection state**: Whether markers are enabled, the selected LIT entry index, and the navigability/position/extent shown consistently in both minimap views and the Lighting list.
- **Synthesized minimap export**: The selected client/map/time/output targets, lighting provenance,
  paired terrain/liquid artifacts, per-tile results, and stitched-map bounds.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On maps with and without lighting data, 100% of tested valid fog ranges produce a finite start less than end and visible terrain at each supported time-of-day sample.
- **SC-002**: Changing either fog endpoint changes the active range within the next rendered frame and the selection survives an unload/reload cycle according to its documented scope.
- **SC-003**: Representative 1.0.0, 1.12.1, 2.4.3, and WotLK-or-later M2 samples each produce either visible native geometry or a specific, version-bound diagnostic; no sample fails silently.
- **SC-004**: The Tools menu audit has zero obsolete dataset entries and zero launchers that fail solely because an unreported binary is missing.
- **SC-005**: Each advertised WMO v14/v17 and M2-to-MDX conversion direction has fixture-based evidence and a documented fidelity status before it is described as reliable.
- **SC-006**: On a map with positional LIT data, enabling the LIT overlay yields matching markers in both minimap views, and a double-click on any navigable list row focuses the corresponding 3D location without changing the active fog range.
- **SC-007**: The compositor fixture test proves renderer-ordered four-layer texture composition,
  phase-independent material averaging, MCNR lighting application without default MCSH bake, and
  transparent-hole stitching; a user-run real-client export records its build fingerprint and
  selected lighting provenance.
- **SC-008**: In both UI modes, fog controls render visible slider grabs and UniqueId playback can
  be paused or stopped without changing the parent Tools tab; selecting its inner Playback tab
  leaves Tools > Archeology selected.

## Assumptions

- Existing native-renderer, legacy-M2, and UI-consolidation specifications remain the detailed owners of their already-defined behavior; this feature coordinates the user-visible stabilization slices.
- A safe visible fallback is preferable to pretending that unavailable lighting data is client-exact.
- M2-to-MDX is an opt-in export workflow for Alpha compatibility, not a runtime dependency.
- LIT marker overlays are an opt-in diagnostic layer; the current loaded LIT source remains the sole data source and map markers do not imply client-exact lighting coverage.
- A synthesized minimap is a derived terrain artifact, not a replacement for raw client minimap
  pixels or proof of visual parity.
- Conversion reliability is claimed only for profiles with preservation and fixture evidence; unsupported cases stay explicit.

## Out of Scope

- Reintroducing ML/dataset authoring workflows into the main viewer Tools menu.
- Using converted MDX assets to render M2 models.
- Claiming visual parity for a client/build without representative asset proof.
- Broad animation, particle, or material-feature reconstruction beyond what is required to make the reported M2 render path visible and diagnosable.
- Editing LIT data, deriving new lighting values from the overlay, or using a minimap marker as a substitute for a lighting or terrain visibility fix.
- Object, sky, fog, or WMO/M2 rendering in synthesized minimaps. Liquid output is a separately
  labeled analytic overlay from decoded coverage/type evidence; full native water-material
  reconstruction remains out of scope. Terrain RGB remains a separate baseline with MCSH preserved
  as its own evidence channel.
