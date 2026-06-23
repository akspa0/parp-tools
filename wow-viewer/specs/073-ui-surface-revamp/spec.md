# Spec 073 — UI Surface Revamp

**Goal:** Clean up duplicated UI controls, fix alignment issues, and surface the converter tools in the viewer — without removing functionality or breaking legacy mode.

**Scope (surface only):**
- No code deletion, no architecture rewrites.
- Legacy shell-panel mode stays untouched behind `View > Legacy UI`.
- Tab-mode UI gets polished: deduplicated controls, consistent alignment, complete Tools tab.

**Sub-plans:**
1. `073a` — Toolbar / left sidebar dedup and alignment.
2. `073b` — Tools tab converter integration.
3. `073c` — Tab/sub-tab alignment and polish pass.
4. `073d` — Model/World/Terrain panel alignment polish.

**Success criteria:**
- Every visible control has one obvious home (no duplicates).
- Toolbars, sidebars, and tab panels align to the viewport grid without overlap or sprawl.
- Tools tab exposes the major converter commands the user can run.
- Legacy mode still works.
- Build clean after each sub-plan.
