# Spec Status Router

**Consolidated 2026-09-06** (first full consolidation pass; cleanup routing updated later that
day). The old per-spec detail table is preserved at
[archived/status-history/2026-09-06-status-pre-consolidation.md](archived/status-history/2026-09-06-status-pre-consolidation.md).
Active work is organized as **epics**: [epics/active-epics.md](epics/active-epics.md). The
[routing registry](registry.md) classifies cold and superseded records so this page remains the
small fresh-chat entry point. Specs 226–230 supersede older unimplemented/partial work only where
the current owner or registry names a forward pointer; do not infer closure from age alone.

## This week's implementation order

| # | Spec | Task | Done when |
|---|---|---|---|
| 0 | [233 Renderer Marketing Capture Automation](233-marketing-capture-automation/spec.md) | P1 source implementation landed: path warmup + direct framebuffer capture now has a Feature Tour action and timed clean-scene callouts; receipt/transport still open | Operator records a real Flyby tour/video and then verifies its benchmark receipt, handoff, and reviewed README assets |
| 1 | [231 Editor/Archaeology UI Overhaul](231-editor-archaeology-ui-overhaul/spec.md) | speckit spec+plan+tasks authored 2026-09-07 (operator-directed); 4-page Editor IA, Archaeology de-hosting, dedupe D1–D6 | Implemented in a fresh session per plan.md phases; operator navigation smoke |
| 2 | [227 UI Re-Audit](227-ui-reaudit/spec.md) | Inventory v2 with screenshots; dropdown/collapsible sidebar standard; dedupe weak-signal amplifiers, minimaps, Inspector repetition; fix Archaeology styling; sane Editor tabs | Operator walks the build against inventory v2 and finds no duplicates |
| 3 | [228 Source Decomposition](228-source-decomposition/spec.md) | Planned first behavior-preserving hover/click selection extraction, blocked by Spec 227 T004 | Core characterization + build/tests + operator smoke; no god-class members or partials added |
| 4 | [226 Renderer Polish](226-renderer-polish/spec.md) | Verify the terrain-wireframe shader hypothesis (multi-textured tiles only) and the model-wireframe gating (dead on MDX/M2/WMO) | Captures name both root causes. UPDATE 2026-09-07: both root causes fixed (textured-line invisibility → flat-color passes); operator visual verification still owed |
| 5 | [229 WoW Shell + Keybind Profiles](229-wow-shell-keybind-profiles/spec.md) | speckit-plan from 227's per-context action inventory | Plan authored; first contextual keybind profile compiles |
| 6 | [230 Reconstruction Editor](230-reconstruction-editor/spec.md) | speckit-plan (Rosetta placement, New Map Generator UI, Alpha/LK save targets) | Plan authored |
| 7 | [223 residual](223-ui-consolidation-audit/tasks.md) | Operator gate retest (fog/overlays/capture) + T609/T610 | Operator acceptance recorded with receipts |
| 8 | [232 Cartography Composition](232-cartography-composition-project/spec.md) | Cell-level layer alignment, layer-stack project persistence (save/lock/autoload), full-map client-format export | Operator: rotated layer alignment fine-tune works, saved project reloads, exported map loads in client |

## Standing gates (operator-owned)

- **224 cleanup**: the operator has directed the current context-reduction pass. The literal Gate 1
  and the full task-receipt audit remain open until their own receipt says otherwise; this cleanup
  does not silently close either gate.
- **v0.5.2.3 runtime verification**: slider order, overlay balance, hover occlusion, About credits.
- Interactive visual gates still owed on shipped specs (210, 211, 205, 203, 152-phase-6) — tracked
  in [epics/active-epics.md](epics/active-epics.md).

## Epics (watch-list)

See [epics/active-epics.md](epics/active-epics.md) for members, supersessions and next actions:

1. **UI & Approachability** — 231 · 227 · 229 · 225 · 212 · 223 (145/080 superseded into it)
2. **Reconstruction & Editing** — 232 · 230 · 222 · 219 · 220 · 208 · 203 · 196 · 194 · 192 (195 superseded)
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
- Old directories not named in the weekly order are **cold**, not default context; see
  [registry.md](registry.md) before reviving or archiving one.
