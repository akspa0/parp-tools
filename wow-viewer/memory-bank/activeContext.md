# Active Context — wow-viewer

Last updated: 2026-09-10 (compacted during a Spec 224 `speckit-cleanup` pass)

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, and `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md),
[progress ledger](progress.md) (dated session history), and [archives](archive/README.md) are
on-demand context, never default reading. Landed-work narrative lives in `progress.md`, not here —
this file states only the current lane and what's still open.

## Spec 224 governance audit — COMPLETE (2026-09-11)

224-T201's receipt/symbol audit now covers all 5 active specs carrying checked tasks. 227 clean
(2026-09-10); 232 had 8 of 12 checks corrected to `[ ]` (2026-09-10); **233 clean** (3/3 receipts,
14 checks); **231 clean** (0 un-checked — T001 resolved to PASS via `inventory-v3-baseline.md`;
T074 kept checked but flagged "receipt not in evidence/"); **223 had 16 receipt-less pre-§9.2 checks
un-checked** (T101–T107, Gate A, T201, T202, T301–T302, Gate B, T401, T501–T502) and is listed for
operator decision. Reports:
[cleanup-2026-09-11.md](../specs/224-speckit-governance/evidence/cleanup-2026-09-11.md) (this pass)
and [cleanup-2026-09-10.md](../specs/224-speckit-governance/evidence/cleanup-2026-09-10.md).
224-T201 closes on operator acknowledgment; 224-T202 (archival) and Gate 2 remain open.

## New operator-reported gaps (2026-09-10, not yet investigated)

Filed as new tasks, not fixed: Spec 232 gained Phase 7 (T067 remove "Bake MCSH shadows" minimap
option permanently; T068 "Include WMO geometry" minimap checkbox broken; T069 no-water minimap
shading glitches; T070 cell fine-tune needs true 1-cell X/Y granularity — Hellfire Ramparts vs
Expansion01 won't align closer than 1–3/1–2 cells with current tooling; T071 offset counters too
small to show a signed two-digit value, must scale with UI text). Spec 231 gained Phase 8 (T080
hovered-WMO doodad-set combo disappears on mouse-leave). Operator also re-stated the save-pipeline
gap directly — that's Spec 234 below, already the whole point of the spec.

## Active lane — Spec 235 Legacy MDX/M2 Rendering (1.0.0-3.0.1) — Phases 0–4 Implemented & Receipted (2026-09-11)

[235-legacy-mdx-m2-rendering/spec.md](../specs/235-legacy-mdx-m2-rendering/spec.md) — Implementation complete for Phases 0 through 4 on branch `235-legacy-mdx-m2-rendering`.
- **Core Reader Unification (`M2ModelReaderDispatcher.cs`, `M2Era100ModelReader.cs`)**:
  - Dropped the `0x102`–`0x107` `NotSupportedException` refusal wall.
  - Supports MD20 versions `<= 0x107` with classic division layout in `M2Era100ModelReader`.
  - Implemented 108-byte (`0x6C`, version `0x100`) and 112-byte (`0x70`, version `0x104`–`0x107` with `boneNameCrc` at `+0x0C`) bone parsing with track normalization.
  - Populates `EmbeddedSkinDocuments` from embedded division records; `M2SkinProfileRuntime` initializes directly without looking for non-existent external `.skin` files.
- **World Placement & Fallback (`WorldAssetManager.cs`, `WowViewerM2RuntimeBridge.cs`)**:
  - Added explicit handling for `M2Era1121EraTag.Md20_1X_V100_Era100` routing directly to the runtime bridge.
  - Implemented FR-005 Bounding-Box Fallback (`BuildBoundingBoxFallbackModel`, 8-vertex, 12-triangle unit box with `usesCompatibilityFallback: true`) so models with no drawable geometry render bounds rather than remaining invisible.
- **Tooling (`WowViewer.Tool.Inspect`)**:
  - `RunM2Inspect` generates `M2GeometryDocument` for `Md20_1X_V100_Era100` using `GlobalVertices`.
- **Verification Across Staged Clients (All Pass Exit 0)**:
  - 1.0.0.3980 (`xyz.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=1, geometry available=true, vertices=72.
  - 2.0.0.5610 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=138, geometry available=true, vertices=4864.
  - 2.4.3.8606 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=143, geometry available=true, vertices=5712.
  - 3.0.1.8303 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=143, geometry available=true, vertices=5712.
  - 3.3.0.10958 (`BloodElfMale.m2`): `ERA: 3.3.5 (MD20 v0x108)`, bones=151, geometry available=true, vertices=6778.
- **Unit Tests & Build**:
  - `M2Era100ModelReaderTests`: 11 passed, 0 failed.
  - `ModelRouteClassifierTests`: 7 passed, 0 failed.
  - Full solution build: 0 errors.
- **Receipt**: [phase1-reader-unification.md](../specs/235-legacy-mdx-m2-rendering/evidence/phase1-reader-unification.md).

## Current lane — Spec 234 Map Save & New Map Creator

[234-map-save-new-map/spec.md](../specs/234-map-save-new-map/spec.md) — Draft, **not planned**.
Save merged/composed maps to Alpha 0.5.3 WDT and LK v18 ADT from both Archaeology and the Editor's
Data I/O page (one shared pipeline), plus a New Map creator in the Editor. Multi-map explicitly out
of scope. Supersedes Spec 230 US2/US3. **Next: speckit-plan.**

## Other open lanes (all implemented-with-operator-gates; see each spec's tasks.md for detail)

- **Spec 232** (cartography composition) — T056/T057/T058/T059/T064/T066 source-landed; operator
  visual witnesses owed for all. T051 (lock badge), T053 (WL inspector), T054 (default page), T015d
  (rotated-seam screenshot) also await witnesses. Next after witnesses: T055 (UniqueId color-coding),
  T065 (chunk off-by-one re-audit).
- **Spec 233** (marketing capture automation) — P1 source landed; **T015 operator tour witness**
  (`FlybyUndead`, Warm Path, Feature Tour + Video) is the next concrete action, then Phase 4 receipts.
- **Spec 231** (Editor/Archaeology UI overhaul) — Phases 0–4, 6-batch-1, and 7 landed; Phase 6
  batch 2 (T061–T064) and Phase 5 (navigation smoke, inventory v3 final) open.
- **Spec 227** (UI re-audit) — only T001/T002 done; **T004 gate genuinely still open** (confirmed
  by 2026-09-10 audit), which is why Spec 228 stays blocked.
- **Spec 228** (source decomposition) — blocked on 227 T004, 0 tasks landed.
- **Spec 223** (UI consolidation) — operator retest owed (fog, WMO-only global WMO, playback/
  capture, ffmpeg, spatial UI) with a real client; video-capture release hardening also needs a
  real with/without-UI recording + playback + Play+Video witness before shipping.
- **Spec 226** (renderer polish) — wireframe root causes fixed; captures still owed.
- v0.5.3-rc1 — tag push + GitHub Actions release are operator-owned; visual pass on wireframe/
  selection/toolbar changes still owed.

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video, and client-data proof are operator-owned.
- Preserve unrelated dirty work; stage named files only.
- New UI work follows Spec 228 (owned service classes, no new `ViewerApp_*`/`WorldScene` members)
  and AGENTS.md §9 receipts.

## Handoff

**Immediate:** the Spec 224 audit is done and its report says so — the next bounded action is
`speckit-plan` for Spec 234. Operator decisions owed: the 16 un-checked Spec 223 checks (accept or
supply retroactive receipts) and the Spec 231 T074 receipt gap.

**Do not claim:** any Spec 232/233/223/226 visual/runtime acceptance; the Spec 223 Phase 1–5 source
tasks are now un-checked pending receipts; and the Spec 235 plan rewrite stays blocked on a real-file
inspection (read `tools/inspect` usage + the `--client`/`--game-path` invocations in specs 104/154/205
first).
