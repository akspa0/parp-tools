# Spec Status Router

**Consolidated 2026-09-06** (first full consolidation pass; cleanup routing updated later that
day). The old per-spec detail table is preserved at
[archived/status-history/2026-09-06-status-pre-consolidation.md](../status-history/2026-09-06-status-pre-consolidation.md).
Active work is organized as **epics**: [epics/active-epics.md](../status-history/2026-09-06-active-epics.md). The
[routing registry](2026-09-23-registry-pre-reconciliation.md) classifies cold and superseded records so this page remains the
small fresh-chat entry point. Specs 226–230 supersede older unimplemented/partial work only where
the current owner or registry names a forward pointer; do not infer closure from age alone.

## This week's implementation order

| # | Spec | Task | Done when |
|---|---|---|---|
| 0 | [233 Renderer Marketing Capture Automation](../233-marketing-capture-automation/spec.md) | P1 source implementation landed: path warmup + direct framebuffer capture now has a Feature Tour action and timed clean-scene callouts; receipt/transport still open | Operator records a real Flyby tour/video and then verifies its benchmark receipt, handoff, and reviewed README assets |
| 1 | [231 Editor/Archaeology UI Overhaul](../231-editor-archaeology-ui-overhaul/spec.md) | speckit spec+plan+tasks authored 2026-09-07 (operator-directed); 4-page Editor IA, Archaeology de-hosting, dedupe D1–D6 | Implemented in a fresh session per plan.md phases; operator navigation smoke |
| 2 | [227 UI Re-Audit](../227-ui-reaudit/spec.md) | Inventory v2 with screenshots; dropdown/collapsible sidebar standard; dedupe weak-signal amplifiers, minimaps, Inspector repetition; fix Archaeology styling; sane Editor tabs | Operator walks the build against inventory v2 and finds no duplicates |
| 3 | [228 Source Decomposition](../228-source-decomposition/spec.md) | Planned first behavior-preserving hover/click selection extraction, blocked by Spec 227 T004 | Core characterization + build/tests + operator smoke; no god-class members or partials added |
| 4 | [226 Renderer Polish](../226-renderer-polish/spec.md) | Verify the terrain-wireframe shader hypothesis (multi-textured tiles only) and the model-wireframe gating (dead on MDX/M2/WMO) | Captures name both root causes. UPDATE 2026-09-07: both root causes fixed (textured-line invisibility → flat-color passes); operator visual verification still owed |
| 5 | [229 WoW Shell + Keybind Profiles](../229-wow-shell-keybind-profiles/spec.md) | speckit-plan from 227's per-context action inventory | Plan authored; first contextual keybind profile compiles |
| 6 | [230 Reconstruction Editor](../230-reconstruction-editor/spec.md) | speckit-plan (Rosetta placement, New Map Generator UI, Alpha/LK save targets) | Plan authored |
| 7 | [223 residual](../223-ui-consolidation-audit/tasks.md) | Operator gate retest (fog/overlays/capture) + T609/T610 | Operator acceptance recorded with receipts |
| 8 | [232 Cartography Composition](../232-cartography-composition-project/spec.md) | Cell-level layer alignment, layer-stack project persistence (save/lock/autoload), full-map client-format export | Operator: rotated layer alignment fine-tune works, saved project reloads, exported map loads in client |
| 9 | [234 Map Save & New Map Creator](../234-map-save-new-map/spec.md) | speckit spec authored 2026-09-09 (operator-directed): save merged/composed maps to Alpha WDT + LK ADT from Archaeology AND Editor Data I/O; New Map creator in Editor; multi-map explicitly out of scope | Plan authored; save pipeline round-trips both targets with receipts; operator client-load witness |
| 10 | [235 Legacy MDX/M2 Rendering](../235-legacy-mdx-m2-rendering/spec.md) | spec + plan authored 2026-09-10 (operator-directed): MDX/M2 rendering for 1.0.0-3.0.1 non-functional; supersedes unimplemented residue of 104 + 154 (which had real measured evidence — exact `0x100`-`0x107` broken range — but were untracked in any epic); adds fuckported-asset/Warcraft.NET parity + MDX light-emitter effects; 6-phase plan, Phase 0 must reconcile `FormatProfileRegistry` against the real `M2ModelReaderDispatcher` mechanism before any reader change | speckit-tasks, then Phase 0 reconciliation; build-by-build survey (US1) complete; objects render + bbox fallback across the range; operator visual witness |
| 11 | [236 Unified Scene Lighting & Doodad Performance](../236-scene-lighting-doodad-performance/spec.md) | spec + plan + tasks authored 2026-09-15 (operator-directed): v0.5.4-dev epic for dark MDX shading fix, Half-Lambert model diffuse, multi-surface light casting (torches/WMO lights), doodad batching performance overhaul, client-constrained map generator, and save/export pipeline (incorporating Spec 234) | Phase 1 shading fix, multi-surface lighting, doodad instancing, and save pipeline verified with receipts |
| 12 | [237 ADT v26 — brand-new format](../237-adt-v26-terrain/spec.md) | **v0.6** · spec + plan + tasks authored 2026-09-16 (operator-directed): **first reader/renderer anywhere for ADT v26**, a completely new terrain format that first shipped ~8h earlier in the first WoW: Forever (`wow_classic_beta`) build; tip-off from Marlamin ~1h before analysis; this project did the first analysis. 699 tiles on hand; MVER 26 + AHDR 26, new ALOC/AOCH/ADST chunks; tile X/Y decoded from ALOC and seam-proven. Format doc: `docs/architecture/adt-v26-format.md`. **2026-09-20: all three revisions now measured on real files** — first **v22** ever loaded (Expansion01 / Terokkar-BoneWastes, 4 files, renders; evidence/first-v22-dat-render-2026-09-20.md) and **v23** (IcecrownCitadel; evidence/real-v23-dat-icecrown-2026-09-20.md). v22 omits empty ACNK and its AMAP is an **unidentified encoding** (128-3474 B, MCAL RLE refuted), so v22 currently renders **layer 0 only**. | Phase 0 inventory tooling must reproduce evidence/phase0-first-look-2026-09-16.md; fast-path wireframe (US0, T045–T050) first. Terrain from tile files alone; **textures/models resolve name → FileDataID (33/33, 285/285) → local CASC install via 238** |
| 12b | [247 DAT Capture & LK ADT Export](../247-dat-capture-and-adt-export/spec.md) | **v0.6** · spec authored 2026-09-20 (operator-directed): captured (not synthesized) top-down PNG per tile + stitched overview, and one-way **DAT → LK v18 ADT** export so the recovered v22/v23 tiles become usable files. Reuses `LkAdtWriter`/`LkWdtWriter`, `AdtAhdrTileSlicer`, `AdtAhdrAlpha`, and the existing offscreen capture plumbing — no new format writer. 241 stays deferred; this is the narrow one-way case. | **US3 DELIVERED 2026-09-20**: `adt-ahdr export-lk` converts DAT → LK v18 ADT + WDT + loss manifest. Real runs: v22 Expansion01 4 tiles / 999 chunks / 289 MCSH / 997 area ids / 849 objects; v23 IcecrownCitadel 3 tiles / 48 MCAL / 768 MCCV. All outputs structurally valid, 0 unaccounted bytes. **Not loaded yet** (operator proof). **US5 CODE COMPLETE 2026-09-20**: DAT folders usable as Cartography layer donors via a `dat:<folder>` locator (`DatLayerSource` + 4 seam points in `StandardTerrainAdapter` + an "Add DAT folder..." button); 10 tests, **unwitnessed in the viewer**, Standard base maps only. US1 v22 `AMAP` codec still OPEN — oracle for a scored search confirmed (`ACNK` +0x12, 0 violations / 997 chunks); US2 capture not started. Oracle for a scored search is confirmed (`ACNK` +0x12, 0 violations / 997 chunks). Then capture, then export. |
| 13 | [238 CASC Data Source](../238-casc-data-source/spec.md) | **v0.6** · spec + plan + tasks authored 2026-09-16 (operator-directed; Marlamin in contact): local install + remote CDN via vendored TACTSharp (Warcraft.NET has no CASC; vendored CascLib/TACT.NET folders are empty), id-addressed reads, keys, verified cache | Phase 0 era survey over CDN + library decision; **local-first**: install in progress at `I:\wow12\World of Warcraft`; first consumer is 237's asset resolution; byte-identical verification vs independent CDN extraction |
| 14 | [239 Modern Client Assets](../239-modern-client-assets/spec.md) | **v0.6** · spec + plan + tasks authored 2026-09-16 (operator-directed): post-5.0.1 FileDataID era — WDT MAID, ADT id placements/MTXP, chunked M2, WMO GFID/MODI, DB2 by id; wow.export as behavioral reference; survey-first | Needs 238 remote gate (T027); tier builds not chosen yet, so tier C defaults to live retail via CDN; coverage survey first |
| 15 | [240 Format Conformance](../240-format-conformance/spec.md) | **v0.6** · spec + plan + tasks authored 2026-09-17: wowdev.wiki × readers audit against `wow_classic_beta` 1.60.1 ([research.md](../240-format-conformance/research.md)); already fixed: WMO MOBA 16-bit material ids (64% of batches), GFID per-LOD groups, shader-23 base texture, 8 terrain layers. WTL (MIT) and WoWFormatLib (no license: behaviour only) as references | Phase 2 conformance survey reproduces research.md WMO counts; then WMO per-shader materials (MVP) |
| 16 | [242 WMO Instancing Performance](../242-wmo-instancing-performance/spec.md) | **v0.6** · spec authored 2026-09-18 (operator-directed): v0.6.0-alpha regression — the scene-light gate disables WMO shell instancing globally, so modern data renders one draw call per placement (~5.5 FPS, 16,431 WMO draw calls on `wow_classic_beta` 1.60.1 `Azeroth`). Fix: decide instancing per placement from whether a light actually reaches it | speckit-plan (FR-006 god-class check first), then implement; receipt must carry a real before/after FPS + draw-call pair |
| 17 | [243 Modern-to-Legacy Map Conversion](../243-modern-to-legacy-map-conversion/spec.md) | **v0.6** · spec authored 2026-09-18 (operator-directed, "important and overlooked too long"): one-way **modern → legacy** conversion producing **LK v18 ADT/WDT** and **Alpha 0.5.3 WDT**, merging multi-layer alpha/texture-id stacks into the target's layer model; batch many maps in one run; near-zero-touch UI (direction + target + input only); optional asset inclusion with a manifest. Old → modern writers explicitly out of scope. **Plan authored 2026-09-18** (`plan.md` + `research.md` + `data-model.md` + `contracts/` + `quickstart.md`): coverage-ranked layer merge, one owned `ModernToLegacyMapConversionService` surfaced in CLI + Editor | speckit-tasks, then implement; receipt must include a real modern map converted to both targets and loaded in the viewer |
| 18 | [244 Modern Liquid Directional Flow](../244-modern-liquid-flow/spec.md) | **v0.6** · spec authored 2026-09-18 (operator-directed): read the modern WDT **`MAI2`** chunk (`MapFileDataIDs2[4096]`, ≥ 12.0.5.66330) and its `liquidFlowTexture`, decoded as R = +Y flows west / G = −X flows south / 128 = zero; surface flow as liquid **context in the viewer UI**; expose one shared flow datum for other consumers; state legacy disposition (Alpha MCLQ flow vector preserved, LK MH2O dropped). Flow-aware *rendering* out of scope. `MAI2` is currently a documented v0.6.0-alpha limitation | speckit-plan (confirm magnitude scaling against the real texture), then implement; receipt needs a real-map UI witness |
| 19 | [245 Modern Chunk Completeness Survey](../245-modern-chunk-completeness-survey/spec.md) | **v0.6** · spec authored 2026-09-18 (operator-directed): inventory **every** chunk in the modern WDT/`_occ`/`_lgt`, root ADT, `_tex0`, `_obj0`, `_lod` families with counts + current handling + code reference; per ignored chunk recorded meaning, confidence and disposition; per candidate state **legacy build-in feasibility** for LK v18 and Alpha 0.5.3 including alpha-mask and texture-id re-expression, with the loss stated. Research deliverable; no runtime change | speckit-plan; receipt is a reproducible corpus walk (counts + command), and findings feed 243/244 |
| 20 | [246 Modern M2 Camera Paths & Modern-Data Benchmarking](../246-modern-m2-camera-paths-and-benchmarking/spec.md) | **v0.6** · spec authored 2026-09-18 (operator-directed): load camera tracks from **modern `MD21` CASC M2s** as playable camera paths (same document/overlay as legacy), and make a **path-driven renderer benchmark** runnable on modern maps with the same receipt shape as legacy data (build/map/path identity, warmup, frames, mean+p99, hitches, submission counters). Camera loss point must be evidenced first (FR-002); this benchmark is the measurement vehicle for 242 | speckit-plan (diagnose where modern cameras are dropped), then implement; receipt needs a real `wow_classic_beta` run and an operator visual witness for the path |

## v0.6 release scope (pinned 2026-09-16)

Worked on the current `v0.5.4-dev` branch. Release theme: **modern data access + experimental terrain formats**.

| Spec | Role | Order |
|---|---|---|
| 238 CASC Data Source | foundation: bytes from local/remote CASC | 1 |
| 239 Modern Client Assets | FileDataID-era readers + renderer hookups | 2 (after 238 Phase 1) |
| 237 DAT v22/v23/v26 (brand-new format, first seen 2026-09-16; v22+v23 real files 2026-09-19/20) | AHDR reader + terrain render from the tile files alone; all three revisions load | **now**: v22 AMAP encoding unidentified — v22 blends layer 0 only; fully independent of 238/239 |
| 240 Format Conformance | per-format survey + reader/renderer gaps from the wowdev.wiki audit | 3 (after 239 loads real builds; survey first) |
| 242 WMO Instancing Performance | fix the v0.6.0-alpha per-placement regression so modern data is interactive again | 4 (blocking modern-data usability; opened 2026-09-18) |
| 243 Modern-to-Legacy Map Conversion | make modern terrain usable in the legacy eras: merge multi-layer chunks into LK v18 and Alpha 0.5.3 outputs, batched, low-touch | 5 (operator priority, 2026-09-18) |
| 244 Modern Liquid Directional Flow | stop ignoring the WDT `MAI2` liquid flow map; give liquids their flow context in the viewer, and state its legacy disposition | 6 (after 243's service shape; independent of 245) |
| 245 Modern Chunk Completeness Survey | one authoritative inventory of everything the modern readers discard, and which of it can be re-expressed in legacy targets | 7 (research; feeds 243/244) |
| 246 Modern M2 Camera Paths & Benchmarking | close the modern-data gap in camera-path import + renderer benchmarking so modern frames are measurable like legacy frames | 8 (measurement vehicle for 242; operator-directed 2026-09-18) |

Carried in from v0.5.4-dev: 236 (active) remains the current owner until its gates close. 242 was
opened from the v0.6.0-alpha build and is not implemented in it; 243, 244, 245 and 246 are new,
unimplemented, and operator-directed.

## Standing gates (operator-owned)

- **224 cleanup**: the operator has directed the current context-reduction pass. The literal Gate 1
  and the full task-receipt audit remain open until their own receipt says otherwise; this cleanup
  does not silently close either gate.
- **v0.5.2.3 runtime verification**: slider order, overlay balance, hover occlusion, About credits.
- Interactive visual gates still owed on shipped specs (210, 211, 205, 203, 152-phase-6) — tracked
  in [epics/active-epics.md](../status-history/2026-09-06-active-epics.md).

## Epics (watch-list)

See [epics/active-epics.md](../status-history/2026-09-06-active-epics.md) for members, supersessions and next actions:

1. **UI & Approachability** — 231 · 227 · 229 · 225 · 212 · 223 (145/080 superseded into it)
2. **Reconstruction & Editing** — 232 · 230 · 222 · 219 · 220 · 208 · 203 · 196 · 194 · 192 (195 superseded)
3. **Renderer Performance & Correctness** — 226 · 153 · 152 · 151 · 150 · 204 · 207 · 206 · 202 · 201 · 200 · 199 · 198 · 160
4. **Formats, Readers & Writers** — 238 · 239 · 237 (v0.6) · 197 · 221 · 220 · 193 · 105 · 205
5. **World Simulation, Audio & Environment** — 214 · 215 · 216 · 217 · 218 · 148 · 147 · 146 · 144 · 143 · 142
6. **PM4 Decoding** — 128–131 · 149 · 184 · 188 · 186 · 187
7. **ML & Dataset Tooling** (eventually continue) — 139–141 · 138 · `data-harvester/`
8. **Infrastructure & Governance** — 224 · 228 · 213 · 190

## Status rules

- `Draft` means design exists; it is not implementation proof.
- `Implementing` means code may exist but required validation is open.
- `Implemented with user gates` means focused source/build proof exists and real-client proof remains.
- `Complete` requires the owning spec's validation gates to pass; do not infer it from task counts.
- New work: register the spec here AND in the epic, follow [AGENTS.md](../../../AGENTS.md) §9–11
  (receipts, inventory row, owned service class), and keep one authoritative plan per feature —
  the newest spec supersedes older unimplemented ones.
- Old directories not named in the weekly order are **cold**, not default context; see
  [registry.md](2026-09-23-registry-pre-reconciliation.md) before reviving or archiving one.
