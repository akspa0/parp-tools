# Spec 271 — v0.6.0 Release & Integrated Help / Image Guide

**Owner**: Epic 251 (viewer UX) · **Opened**: 2026-10-11 (operator-directed) · **Branch**: `v0.6.0-dev` → `main`, continuing on `v0.6.1-dev`

## Operator brief (verbatim intent)

- Ship `v0.6.0` proper: the 1.60 (WoW: Forever) support "works with every new build, with no issues", the
  development-map restoration works, minimap-only terrain can be regenerated, and PM4 objects can be
  partially restored to the development map. This is a celebratory milestone and the stable base for all
  future work.
- Rewrite README data, user guides and anything else that pertains to the release.
- Documentation must state plainly: **bring your own data**; the viewer supports **0.5.3–0.11 completely**
  and **0.12–5.x partially**; the dataset/ML research (rebuilding maps from images alone) is
  **unexplored / unstable** and documented minimally.
- Phase maps live in the **Archaeology** tab: for each base map they show the DBC/DB2-derived phase maps if the
  loaded client has them; otherwise the user can load **any map on top of any other map**. This tooling has
  **no documentation today** — fix that.
- Provide an **image guide** of all features, surfaced as **integrated help inside the tool**.

## Phase A — Release v0.6.0 (this session)

- A1. `eng/Version.props` → `0.6.0` (no prerelease suffix); title bar / About read it.
- A2. `docs/releases/v0.6.0.md` with `<!-- release: full -->`; `CHANGELOG.md`; root + `wow-viewer` READMEs;
  `docs/WoWViewer/README.md`; `USERGUIDE.md` (version, support matrix, Archaeology phase/any-on-any map
  guide, restored development map, BYO-data, experimental-ML warning).
- A3. Merge `v0.6.0-dev` → `main`, tag `v0.6.0`, push (triggers `wowviewer-release.yml`).
- A4. Create and push `v0.6.1-dev`; memory bank + STATUS updated so a fresh chat can resume.

## Phase B — Integrated help & image guide (v0.6.1-dev, NOT in this session)

- B1. In-app **Help** surface (Help menu + per-panel "?"), a guide index covering every workbench
  destination (Quick, Inspector, Editor, Archaeology, Scene, Utilities) and every sidebar.
- B2. Image guide: one captured screenshot per feature stored under `docs/WoWViewer/images/`, embedded in
  `USERGUIDE.md` and rendered in the in-app help. **Screenshot capture is operator-owned** (AGENTS §5/§3):
  the agent prepares a shot list; the operator (or the existing capture automation, run by the operator)
  produces the images.
- B3. Archaeology tab docs: phase-map detection from `Map.dbc`/`Map.db2`/`PhaseXAreaID`-family tables,
  and the "load any map over any map" flow, each with a captured walkthrough.
- Phase B scope is the operator's wording above only; further help-system features need an amendment.

## Operator amendment — corrective v0.6.0.1 release (2026-10-14)

Operator request: include the `wow-viewer/tools/` tooling in compiled platform builds and ship it as
`v0.6.0.1`; also apply lighting to ground-effect doodads. This corrective release is limited to:

- Publish every executable C# project under `wow-viewer/tools/` into each of the four platform
  archives, keep each project's output isolated, and include the tool README/helper scripts.
- Register all tool projects in the solution and trigger build CI on the active `v0.6.1-dev` branch.
- Preserve terrain ground-effect MCCV tint and MCSH shadow attenuation when submitting each doodad
  model for rendering.
- Include the already-committed 1.60.1 taxi mount lookup fixes present on the release commit.

### Corrective release acceptance criteria

- AC-C1: each of the four release archives contains the viewer and all nine tool apphosts, DLLs,
  dependency manifests, and runtime configuration files under `tools/`.
- AC-C2: the release job runs successfully from tag `v0.6.0.1` and creates the four downloadable archives.
- AC-C3: terrain ground-effect model submissions preserve the interpolated MCCV RGB multiplier and
  MCSH attenuation; focused tests cover neutral, lit, and shadowed tint values. Operator visual smoke
  on client terrain with MCCV and MCSH remains a separate proof gate.

## Known limitations carried into v0.6.0 (documented, not hidden)

- PM4 object restoration is partial: objects on tiles *adjacent* to PM4-covered tiles that exist in the
  museum data are still missing from restored tiles (operator observation 2026-10-11).
- `LkAdtReader` adds 8 bytes to MCAL/MCSH offsets (MCNK-start-relative); the restore tool repairs this locally.
  Fixing the shared reader is a separate, operator-approved item.
- Terrain ML / dataset reconstruction is experimental and not covered by the user guide beyond a stub.

## Acceptance criteria

- AC-A1: viewer title bar and About read `0.6.0`. (Operator visual.)
- AC-A2: `v0.6.0` tag exists on `main`; release workflow run succeeds and publishes a full release with 4 platform zips.
- AC-A3: `v0.6.1-dev` exists on origin, branched from the `v0.6.0` commit.
- AC-B1..B3: tracked in `tasks.md`; unchecked until delivered with receipts.
