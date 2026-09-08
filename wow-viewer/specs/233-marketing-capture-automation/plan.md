# Implementation Plan: Renderer Marketing Capture Automation

**Branch**: `233-marketing-capture-automation` | **Date**: 2026-09-08 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/233-marketing-capture-automation/spec.md`

## Summary

Layer a reproducible feature-tour session over the existing camera-path warmup and direct framebuffer video path. A library-owned model validates recipes, tracks beats, writes a performance-bearing receipt, and produces an external authoring handoff. The viewer composes that service at its existing playback/capture seams and renders only the active callout while ordinary chrome is hidden. A future MCP/ComfyUI adapter consumes the handoff; it is not implemented or assumed available in this plan.

## Technical Context

**Language/Version**: C# / .NET 10 (`net10.0` core; `net10.0-windows` viewer).

**Primary Dependencies**: Existing Silk.NET + ImGui viewer, `System.Text.Json`, existing ffmpeg raw-frame encoder route, `WorldRenderFrameHistory`.

**Storage**: Versioned JSON recipes, receipts, and authoring-handoff descriptors inside the viewer-managed `output/captures/` root; no client-root or desktop-path persistence.

**Testing**: xUnit in `tests/WowViewer.Core.Tests`; focused pure-model tests plus solution build. Real client/video/UI/FPS/Comfy validation remains an operator gate.

**Target Platform**: Windows desktop viewer; transport-neutral JSON consumable by an external MCP host.

**Project Type**: Desktop application with shared runtime library.

**Performance Goals**: No allocations from the tour model on the frame path; reuse the bounded existing frame history and do receipt serialization only at terminal state. Preserve existing camera-path preload behavior.

**Constraints**: No new `ViewerApp`/`WorldScene` fields or partial classes; no hardcoded client roots; no external HTTP/Comfy calls, uploading, publishing, or fabricated assets; preserve existing readers and encoder route.

**Scale/Scope**: One built-in camera-path tour and versioned extensibility for additional beats/recipes; this phase does not register every UI control, create a Comfy workflow, alter README media, or set up Patreon.

## Constitution Check

| Gate | Result | Evidence / design response |
|---|---|---|
| Repo independence | Pass | All source/contracts/artifacts live below `wow-viewer`; client identities are metadata, never paths. |
| Library-first | Pass | Recipe/attempt/receipt/handoff policy belongs in `WowViewer.Core.Runtime/Marketing`; the viewer adapter only composes and draws. |
| Real-data validation | Pending operator gate | Unit tests establish contracts only. Real client, playback, visual/UI, video, and FPS evidence require the operator quickstart. |
| No client-path assumptions | Pass | Capture identity uses map/build/path label; any client path stays transient in existing loading routes. |
| Blizzard-container policy | Pass | No archive/container write is introduced. |
| God-class freeze | Pass by design | New state lives in `MarketingTourAttempt` attached to the existing active recording lifecycle; no new `ViewerApp` field or partial file. |
| External safety | Pass | Handoff is a local validated descriptor; no localhost HTTP, MCP invocation, upload, or publishing is in scope. |

Re-check after Phase 1 design: **pass**; the data model and contract constrain all path-bearing handoff fields to relative managed-output references.

## Project Structure

### Documentation (this feature)

```text
specs/233-marketing-capture-automation/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── feature-tour-recipe.schema.json
│   └── authoring-handoff.schema.json
├── evidence/
└── tasks.md
```

### Source Code

```text
src/core/WowViewer.Core.Runtime/
└── Marketing/
    ├── FeatureTourRecipe.cs
    ├── FeatureTourAttempt.cs
    ├── TourAttemptReceipt.cs
    ├── MarketingCaptureOutputPolicy.cs
    └── AuthoringHandoff.cs

src/viewer/WoWViewer/
├── Capture/
│   └── MarketingTourOverlayRenderer.cs
├── ViewerApp_CameraPaths.cs             # existing composition seam only
└── ViewerApp_CaptureAutomation.cs       # existing capture terminal seam only

tests/WowViewer.Core.Tests/
└── MarketingCapture/
    ├── FeatureTourRecipeTests.cs
    ├── TourAttemptReceiptTests.cs
    └── AuthoringHandoffTests.cs
```

**Structure Decision**: Keep deterministic policy in the shared runtime library so external adapters and tests share one contract. The viewer does not own duplicate JSON or performance logic; it merely supplies existing map/build/path, lifecycle notifications, and `FrameHistory` snapshots to the service.

## Phase Roadmap

1. **Foundation** — Implement/validate recipe, beat, attempt, outcome, output-containment, and handoff models in Core Runtime. This is independently testable without a GPU.
2. **Receipt evidence** — Serialize receipts and handoffs atomically adjacent to a managed capture; validate typed rejection/failure paths.
3. **Viewer composition** — Extend the existing `Play + Video` seam with an explicit `Feature Tour + Video` action. Carry a `MarketingTourAttempt` through the existing warmup and active-recording objects, reset/snapshot existing frame history at actual record start/stop, and write a terminal receipt. No new `ViewerApp` field is allowed.
4. **Callout presentation** — Render the active beat through the owned overlay adapter during the existing full-frame UI capture tap, with normal chrome hidden. Start with valid callout beats; introduce actual named-control registry adapters only in a later bounded task.
5. **External and publishing gates** — Expose a local validated handoff through the selected future MCP contract after its callable schema exists; operator runs ComfyUI workflow, reviews real assets, and then adds README embeds/links. Patreon remains separately scoped.

## Complexity Tracking

No constitution violations or new projects are required.
