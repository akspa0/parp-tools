# Active Context — wow-viewer

Last updated: 2026-09-06

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md), and
[archives](archive/README.md) are on-demand context, never default reading.

## Current lane — Spec 227 UI inventory gate

- **Source evidence complete:** T001 source audit and T002 Spec 223 reconciliation have receipts.
  No viewer behavior changed by either task.
- **Next action — operator-owned:** T003 current-build screenshot/input matrix, including minimap
  teleport and Inspector per-object mapping. T004 then records the inventory gate.
- **Stop condition:** do not consolidate a UI surface or start its source extraction until T004
  names its authoritative home and has the required evidence.

## Planned next lane — Spec 228 source decomposition

- Plan, research, design model, task list, and verification guide now exist.
- First candidate is the UI-neutral hover/click selection algorithm. It is blocked by Spec 227 T004;
  it must preserve current routes, add no `WorldScene`/`ViewerApp` feature members or partials, and
  obtain build/test plus an operator input smoke receipt.
- Capture, camera paths, Sidebars, Editor, and PM4 remain future candidates, not authorized
  extraction work.

## Separate operator gate — Spec 223

- Retest fog, WMO-only global WMO rendering, playback/capture controls, ffmpeg output, and spatial
  UI behavior with a real client. Source/build evidence is not runtime acceptance.

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video, and client-data proof are operator-owned.
- Preserve unrelated dirty work; stage named files only.
- Keep new plans/tasks in `specs/`; keep this file as a compact handoff only.

## Handoff

**Current target:** Spec 227 T003 operator matrix.

**Then:** Spec 227 T004 → Spec 228 T001/T002 baseline receipts → one bounded selection extraction.
**Do not claim:** UI acceptance, real-client acceptance, or a completed monthly Spec 224 audit.
