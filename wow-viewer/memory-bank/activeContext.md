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

## Spec 224 governance audit — PARTIAL (2026-09-10), retry owed

First real run of the repo's own receipt audit (224-T201, previously never executed). 227 audited
clean. **232 had 8 of 12 checked tasks corrected to `[ ]`** (T050, T052, T056, T057, T058, T059,
T064, T066 — checked against acceptance criteria their own receipts admit are still open; see its
tasks.md for detail). **233, 231, 223 were NOT audited** (agents failed on a session rate limit) —
**their checked-task counts are unverified; do not cite them as proof of completion.** Report:
[cleanup-2026-09-10.md](../specs/224-speckit-governance/evidence/cleanup-2026-09-10.md). Retry
those 3 audits next session before trusting their tasks.md state.

## New operator-reported gaps (2026-09-10, not yet investigated)

Filed as new tasks, not fixed: Spec 232 gained Phase 7 (T067 remove "Bake MCSH shadows" minimap
option permanently; T068 "Include WMO geometry" minimap checkbox broken; T069 no-water minimap
shading glitches; T070 cell fine-tune needs true 1-cell X/Y granularity — Hellfire Ramparts vs
Expansion01 won't align closer than 1–3/1–2 cells with current tooling; T071 offset counters too
small to show a signed two-digit value, must scale with UI text). Spec 231 gained Phase 8 (T080
hovered-WMO doodad-set combo disappears on mouse-leave). Operator also re-stated the save-pipeline
gap directly — that's Spec 234 below, already the whole point of the spec.

## New lane — Spec 235 Legacy MDX/M2 Rendering (1.0.0-3.0.1)

[235-legacy-mdx-m2-rendering/spec.md](../specs/235-legacy-mdx-m2-rendering/spec.md) — Draft, not
planned. Operator directive 2026-09-10: MDX/M2 support for 1.0.0-3.0.1 is "very much
non-functional" (no objects render, no bounding boxes); fuckported (non-standard-rewritten) assets
must render if Warcraft.NET can read them; MDX torch/light-emitter effects are missing.
**Mid-authoring this spec discovered it would have duplicated two existing, untracked specs**:
104 (legacy-m2-rendering, 7/27 tasks checked, unaudited) and 154 (m2-era-reader-parity, planned but
never task-broken) — both had real measured evidence (154: the broken range is exactly `0x100`
through `0x107`, not an approximate "1.x-3.0.1"; `3.3.0`/`0x108` is the real known-good reference,
not "3.3.5") but neither was listed in `epics/active-epics.md`, which is how they went invisible
despite 104 claiming "Status: Active." 235 now explicitly supersedes both (dated notes added to
each; content preserved, not archived) and incorporates their findings.

**Planned 2026-09-10**, then **Phase 0 reconciliation partially executed the same day** — findings
are in [235's research.md](../specs/235-legacy-mdx-m2-rendering/research.md) and they **invalidate
the shape of the current plan.md**. Read research.md before plan.md; plan.md's six-phase per-era
structure predates the findings and needs rewriting.

**What Phase 0 established (code-evidenced):**

1. `FormatProfileRegistry.ResolveMdxProfile` is **dead code** (zero callers) — that is why "MDX is
   fine". `ResolveModelProfile` (M2) is live, and its `major == 1` → `null` return disables both
   embedded-profile routes for the entire Vanilla era (`WorldAssetManager.cs:1340`, `:1375`).
2. **The world-placement loader never reaches the 1.0.0 reader.** `WorldAssetManager`'s no-skin
   chain branches on `Md20_1X_V100`/`V101` but has **no `Md20_1X_V100_Era100` case**. Fixing
   `M2Era100ModelReader`'s bone bug alone would render nothing.
3. **No bounding-box fallback exists** — every failure returns `null`, so a failed load is invisible.
   That is the "no bounding boxes either" symptom; FR-005 must be built, not reconnected.
4. The `0x102`–`0x107` refusal is a hardcoded `NotSupportedException` in
   `M2ModelReaderDispatcher.DetectEra`, citing "spec 049" — a **stale citation** (049 is an archived
   UI spec).
5. **Warcraft.NET's `MD21` already reads this whole era generically** (annotated
   `VersionBeforeLegion`→`VersionAfterWoD`, one flag-keyed conditional, no per-version branching) and
   is already wrapped by `WarcraftNetM2Adapter`. It captures `ViewCount` but **never walks the
   embedded skin/view table** — the single real gap, matching Spec 104's independent diagnosis.
6. `M2Era100Constants`' Ghidra-derived offsets are **+8 shifted** vs Warcraft.NET's layout, and the
   Era100 tests are self-admittedly synthetic ("run without a staged client") so they cannot detect
   a wrong offset. Unverified.

**Operator direction**: stop per-build special-casing; use Warcraft.NET's M2 implementation (already
wrapped) plus wowdev.wiki as reference. "MDX is fine, M2 is not fully there." "1.0.0+ uses .MDX as
the extension, but M2 as the format, with the MD20 chunk" — verified the routing already honours
this (magic bytes decide, extension is diagnostic only).

**BLOCKED / next action**: no real file was ever read. `m2 inspect --archive-root … --virtual-path
World\ArtTest\BoxTest\XYZ.mdx` failed against the 1.0.0.3980 client, whose `Data/` uses the older
content-segmented archives (`base.MPQ`, `model.MPQ`, …). **This is a wrong invocation, not a missing
capability — the repo already has tooling that inspects everything.** Next session: read
`tools/inspect`'s usage and the real client-read invocations in specs 104/154/205 (several take
`--client`/`--game-path`, not `--archive-root`) **before** improvising flags or guessing asset paths.
Then settle finding 6 against one real file, and rewrite plan.md around the reframe at the end of
research.md.

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

**Immediate:** finish the Spec 224 audit (un-check anything lacking a real receipt, write the
dated report, update the AGENTS.md cleanup ledger), then run `speckit-plan` for Spec 234.

**Do not claim:** any Spec 232/233/223/226 visual/runtime acceptance, or Spec 224 audit completion,
until this pass's report says so.
