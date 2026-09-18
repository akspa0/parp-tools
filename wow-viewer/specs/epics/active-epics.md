# Active Epics — wow-viewer (consolidated 2026-09-06)

Supersession rule (operator directive 2026-09-06): **Specs 226–230 supersede anything older that is
not implemented or only partially implemented.** Old specs below are watch-list members, not
independent plans; their unimplemented residue is folded into the epic's next actions. The old
per-spec detail table lives at [archive/2026-09-06-status-pre-consolidation.md](../archive/2026-09-06-status-pre-consolidation.md).

---

## Epic 1 — UI & Approachability (TOP PRIORITY)
**Goal:** the viewer is approachable — a friendly tool, not a pile of tangled wires.
**Members:** 227 (re-audit, OWNS), 229 (WoW shell + keybind profiles), 225 (overhead ortho view),
212 (spatial HUD, decorative-only today), 223 (four-tab structure; residual operator gate).
**Superseded:** 145 (WoW UI overhaul) and 080 (UI consolidation Phase 2E) → their unimplemented
residue is absorbed by 227/229.
**Next:** 227 inventory v2 with screenshots → dropdown sidebar standard → 229 plan → 225 Phase 1.
**Gates:** 223 operator retest; every new surface needs inventory row + owned service + receipt.

## Epic 2 — Reconstruction & Editing
**Goal:** reconstruction-first editing: existing data → new maps, through the Editor tab.
**Members:** 234 (map save — merged ADT/alphaWDT from Archaeology + Editor — and New Map creator;
OWNS save targets + map creation; supersedes 230 US2/US3), 232 (cartography composition, active),
230 (reconstruction editor; retains Rosetta placement US1), 222 (cartography workbench),
219 (transform seam), 220 (WMO doodad editing), 208 (cross-map transplant), 203 (multi-phase
composition, implemented), 196 (WDL magnetizer, complete), 194 (stratigraphy, complete),
192 (terrain generator, complete), 195 (superseded by 219/222).
**Next:** 230 speckit-plan after 227/229 shape the Editor surfaces; 222-T101 occupied-tile queries.
**Note:** editing model is reconstruction from existing data — not sculpting/hand-painting.

## Epic 3 — Renderer Performance & Correctness
**Goal:** correct images at high FPS with measured, attributable improvements.
**Members:** 236 (v0.5.4-dev: unified scene lighting, multi-surface light casting, doodad batching performance, OWNS active execution), 242 (**v0.6** per-placement WMO shell instancing under scene lights — fixes the 236 Phase 2 whole-scene gate that stalls modern data; operator-directed 2026-09-18), 246 (**v0.6** modern M2 camera paths + modern-data renderer benchmarking — closes the camera-import gap for `MD21` CASC assets and gives modern maps the same path-driven benchmark + receipt shape as legacy data; the measurement vehicle for 242), 226 (wireframe + MDX lighting defects, absorbed into 236), 153/152/151 (hitch,
stability, admission — mostly shipped/measuring), 204 (off-thread decode), 207 (draw-call
reduction), 206 (Zarr residency), 202 (batching), 201 (metric attribution), 200 (portal fallback),
199 (MCAL decode), 198 (shader permutations), 160 (skybox).
**Next:** 246 first if benchmarking must precede the fix (it is 242's measurement vehicle), otherwise 242 speckit-plan → per-placement instancing fix (modern-data FPS regression); then 236 Phase 3 doodad instancing and the remaining Phase 1/2 operator visual gates.

## Epic 4 — Formats, Readers & Writers
**Goal:** proven, era-faithful read/write for every format we touch.
**Members:** 238 (**v0.6** CASC data source — local + remote CDN via TACTSharp), 239 (**v0.6** modern client assets — FileDataID-era WDT/ADT/M2/WMO/DB2, depends on 238), 240 (**v0.6** format conformance — wowdev.wiki × reader audit, per-format survey, WMO shader/LOD/material and terrain/M2 field gaps, depends on 238/239), 243 (**v0.6** modern → legacy map conversion — merges multi-layer modern chunks into LK v18 ADT/WDT and Alpha 0.5.3 WDT, batched and low-touch, optional asset bundling; operator-prioritised 2026-09-18, consumes 234 writers + 238/239 readers), 244 (**v0.6** modern liquid directional flow — WDT `MAI2` `liquidFlowTexture` decode, viewer liquid context, legacy disposition per target; operator-directed 2026-09-18), 245 (**v0.6** modern chunk completeness survey — inventory of every discarded modern chunk plus per-chunk legacy build-in feasibility via alpha-mask/texture-id re-expression; feeds 243/244), 237 (**v0.6** ADT v26 — brand-new terrain format first seen 2026-09-16 in the first WoW: Forever build; first analysis is ours — AHDR family, read + render from tile files alone; corpus on hand; wireframe fast path first; independent of 238/239), 197 (split ADT/targets, active), 221 (converter validation harness, active),
220 (WMO writing), 235 (legacy MDX/M2 rendering 1.0.0-3.0.1 + fuckported-asset compatibility +
light effects, Draft, not planned — supersedes 104 and 154's unimplemented residue), 193 (1.x M2
parity, Benilla reference — prior art for 235), 105 (format profiles, prior art for 235's 1.0.0
pillar), 205 (MH2O, implemented — operator visual proof owed).
**Superseded:** 104 (legacy M2 rendering) and 154 (M2 reader era parity) → unimplemented residue
absorbed into 235; both were previously missing from this epic's tracking entirely, which is how
they went untracked despite 104 claiming "Status: Active."
**Next:** 243 speckit-plan → plan the modern→legacy conversion service (layer-merge policy, LK + Alpha
targets, batch driver, asset manifest) — operator priority; 244 speckit-plan (WDT `MAI2` flow decode +
viewer liquid context); 245 survey first if it would change 243/244's chunk plan; then speckit-plan for
235, starting with reconciling `FormatProfileRegistry` against whatever era-resolution mechanism 104/154
already built (235 FR-014); 221-T101 object corpus validator; 197 slot-aware MoP writer research (gates
Spec 230 US3).

## Epic 5 — World Simulation, Audio & Environment
**Goal:** the world behaves: audio, weather, physics, lighting.
**Members:** 214 (physics, policy landed), 215 (weather), 216 (cursor light), 217 (audio
lifecycle), 218 (creature staging), 148 (world simulator), 147 (minimap/fog), 146 (audio/camera),
145 (→ moved to Epic 1), 144 (capture paths), 143 (lighting), 142 (scene graph).
**Next:** on hold pending Epic 1; drafts (215–218) need speckit-plan when revived.

## Epic 6 — PM4 Decoding
**Goal:** decode PM4/PD4 completely and provably.
**Members:** 128–131 (core research lane), 149 (region navigation/audio), 184 (generation),
188 (field semantics), 186/187 (server data/museum sim drafts).
**Next:** dormant; reopen via the PM4 spec pack before code.

## Epic 7 — ML & Dataset Tooling (eventually continue)
**Goal:** model-assisted terrain reconstruction keeps advancing when viewer stability allows.
**Members:** 139–141 (terrain/minimap reconstruction ML), 138 (cross-era renderer research),
`data-harvester/` (v16–v60 dataset pipelines, training configs).
**Next:** nothing scheduled; revisit after Epics 1–2 make the tooling approachable. Operator note:
"ML work with models needs to eventually continue, too."

## Epic 8 — Infrastructure & Governance
**Goal:** the project can be worked on safely.
**Members:** 224 (governance — Gate 1 open: approve AGENTS.md §9–11), 228 (source decomposition —
rule binding, extraction queued), 213 (MCP tooling harness draft), 190 (Rosetta corpus, complete —
consumed by Epic 2).
**Next:** operator approves §9–11 → first `speckit-cleanup` monthly run; 228 first extraction.
