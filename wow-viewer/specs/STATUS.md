# Spec Status Router

**Consolidated 2026-09-06** (first full consolidation pass). The old per-spec detail table is
archived at [archive/2026-09-06-status-pre-consolidation.md](archive/2026-09-06-status-pre-consolidation.md).
Active work is organized as **epics**: [epics/active-epics.md](epics/active-epics.md). Supersession
rule: **Specs 226–230 supersede any older spec that is not implemented or only partially
implemented**; old specs are watch-list members of an epic, not independent plans.

## This week's implementation order

| # | Spec | Task | Done when |
|---|---|---|---|
| 1 | [227 UI Re-Audit](227-ui-reaudit/spec.md) | Inventory v2 with screenshots; dropdown/collapsible sidebar standard; dedupe weak-signal amplifiers, minimaps, Inspector repetition; fix Archaeology styling; sane Editor tabs | Operator walks the build against inventory v2 and finds no duplicates |
| 2 | [228 Source Decomposition](228-source-decomposition/spec.md) | First behavior-preserving extraction (hover/pick or capture service) under the god-class freeze | Build + focused tests green; no god-class members added; file budget respected |
| 3 | [226 Renderer Polish](226-renderer-polish/spec.md) | Verify the terrain-wireframe shader hypothesis (multi-textured tiles only) and the model-wireframe gating (dead on MDX/M2/WMO) | Captures name both root causes |
| 4 | [229 WoW Shell + Keybind Profiles](229-wow-shell-keybind-profiles/spec.md) | speckit-plan from 227's per-context action inventory | Plan authored; first contextual keybind profile compiles |
| 5 | [230 Reconstruction Editor](230-reconstruction-editor/spec.md) | speckit-plan (Rosetta placement, New Map Generator UI, Alpha/LK save targets) | Plan authored |
| 6 | [223 residual](223-ui-consolidation-audit/tasks.md) | Operator gate retest (fog/overlays/capture) + T609/T610 | Operator acceptance recorded with receipts |

## Standing gates (operator-owned)

- **224 Gate 1**: approve [AGENTS.md](../AGENTS.md) §9–11 rules → unlocks the monthly
  `speckit-cleanup` cadence (ledger: last 2026-09-06, next due 2026-10-01).
- **v0.5.2.3 runtime verification**: slider order, overlay balance, hover occlusion, About credits.
- Interactive visual gates still owed on shipped specs (210, 211, 205, 203, 152-phase-6) — tracked
  in [epics/active-epics.md](epics/active-epics.md).

## Epics (watch-list)

See [epics/active-epics.md](epics/active-epics.md) for members, supersessions and next actions:

1. **UI & Approachability** — 227 · 229 · 225 · 212 · 223 (145/080 superseded into it)
2. **Reconstruction & Editing** — 230 · 222 · 219 · 220 · 208 · 203 · 196 · 194 · 192 (195 superseded)
3. **Renderer Performance & Correctness** — 226 · 153 · 152 · 151 · 150 · 204 · 207 · 206 · 202 · 201 · 200 · 199 · 198 · 160
4. **Formats, Readers & Writers** — 197 · 221 · 220 · 193 · 105 · 205
5. **World Simulation, Audio & Environment** — 214 · 215 · 216 · 217 · 218 · 148 · 147 · 146 · 144 · 143 · 142
6. **PM4 Decoding** — 128–131 · 149 · 184 · 188 · 186 · 187
7. **ML & Dataset Tooling** (eventually continue) — 139–141 · 138 · `data-harvester/`
8. **Infrastructure & Governance** — 224 · 228 · 213 · 190

## Status rules

- `Draft` means design exists; it is not implementation proof.
- `Implementing` means code may exist but required validation is open.
- `Implemented with user gates` means focused source/build proof exists and real-client proof remains.
- `Complete` requires the owning spec's validation gates to pass; do not infer it from task counts.
- New work: register the spec here AND in the epic, follow [AGENTS.md](../AGENTS.md) §9–11
  (receipts, inventory row, owned service class), and keep one authoritative plan per feature —
  the newest spec supersedes older unimplemented ones.
