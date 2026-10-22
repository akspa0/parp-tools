# Research — Spec 286 (2026-10-22)

Read-only source evidence, not runtime/screenshot proof. Implements [operator scope](spec.md).

## Footer
- **Observed:** [manager](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs:212)
  explicitly enables bottom horizontal scrolling. [Status content](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs:523)
  combines long coordinates and diagnostic text positioned using global display width.
- **Decision:** actual-region fit policy; essential FPS/actions visible; pending/CPU/chunk/model/audio
  detail on FPS hover from existing snapshots. Optional popup only for interactive long detail.
- **Rejected:** scrollbar suppression alone, which hides overflow rather than fixes it.

## Scale/reset
- **Observed:** [manager](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs:64)
  composes user × display scale, but [Settings](../../src/viewer/WoWViewer/Workbench/Services/Settings/SettingsWindowService.cs:159)
  directly writes global font scale (range 0.85–2.20); [load](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs:229)
  accepts 0.75–2.50. [Layout reset](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs:86)
  correctly does not mutate user font scale.
- **Decision:** one preference service normalizes 0.85–2.20/default 1.0 and persisted outliers once;
  retain manager display composition and three distinct reset meanings.
- **Rejected:** changing layout-only reset silently or adding another scale store.

## Maps
- **Observed:** [discovery](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs:253)
  only opens after old count zero/new nonzero; this cannot represent replacement or successful-empty.
- **Decision:** committed activation identity + once-consumed current discovery result. Audit all
  success paths during implementation; ignore stale completion and ordinary refresh.
- **Rejected:** reopening from the frame loop or treating nonempty map count as activation success.

## Selection/category
- **Observed:** existing [hover](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs:467)
  enlarges names, while [nameplate](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs:520)
  is a projected world-space pill requiring display name. [Bounds](../../src/viewer/WoWViewer/Terrain/WorldScene.cs:3929)
  already have submission/toggles/spawn colors.
- **Decision:** presentation adapter, viewport-centered identity header, existing selection authority;
  one category policy shared with enabled bounds. Explicit metadata or bounded known-asset mapping
  supports tree classification; otherwise Unknown. This resolves the design without fabricating data.
- **Rejected:** another picker/selection database, fake health, or a broad semantic/ML project.

## Proof and ownership
- **Decision:** renderer plans remain with [249](../249-epic-renderer-performance-and-correctness/plan.md)
  and [259](../259-ground-effects-and-wmo-detail-doodads/plan.md). No renderer rewrite is bundled
  into this shell follow-up. Real-client, DPI/input and screenshot gates stay operator-owned.
- **Setup:** guarded PowerShell setup selected Spec 286. No agent-context update script exists in
  this checkout; explicit continuity updates replace it. No new stack/dependency is introduced.
- **Unknowns:** actual visual behavior and available client-specific category evidence are gates,
  not assumed successes; no unresolved design clarification is required to generate tasks.
