---
description: "Build the new thin wow-viewer viewer host on top of library-owned source, scene, streaming, and GPU rendering services. Use when earlier slices already moved ownership out of WowViewer.App and the next step is user-facing cutover without reintroducing app-local preview architecture."
name: "wow-viewer Thin Viewer Host Cutover"
argument-hint: "Optional focus such as host project bootstrap, input dispatch, workspace/session wiring, temporary feature gating, or parity verification order"
agent: "agent"
---

# wow-viewer Thin Viewer Host Cutover

Target repo file: .github/prompts/wow-viewer-gpu-viewer/09-thin-viewer-host-cutover.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. results/artifacts from prompts 01 through 08
4. wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs
5. wow-viewer/README.md
6. .github/copilot-instructions.md

## Problem

Even after library extraction, the current user-facing host path can drift back into temporary app-local ownership unless a dedicated thin host is created with strict boundaries.

## Goal

Create a new thin viewer host that owns only windowing/session/input/layout concerns and consumes library-owned source, scene, streaming, and GPU render services.

## Required Constraints

1. Do not touch gillijimproject_refactor implementation code.
2. Do not rebuild app-local preview architecture in the new host.
3. Keep WowViewer.App as temporary/legacy proof harness until parity is proven.
4. Keep the host thin: no deep rendering/runtime ownership in shell.

## Concrete Scope

1. create new viewer host project under wow-viewer/src/viewer
2. wire host to library-owned source path and scene services
3. wire host to library-owned GPU render services
4. implement minimal workspace/session/state flow needed for parity checks
5. add feature gates so unfinished slices remain explicit

## Out Of Scope

1. no editor feature migration
2. no broad panel redesign beyond required host usability
3. no reopening of library ownership work already completed

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. live GUI proof on fixed real-data roots for:
   - faster world open than current branch
   - viewport input ownership (WASD/QE/wheel)
   - minimap/tile selection
   - multi-tile textured terrain
   - in-world object visibility
3. explicit statement of remaining parity gaps after cutover attempt

## Deliverables

1. new thin host project and wiring
2. parity proof captures/metrics
3. explicit cutover status and remaining gaps
4. migration note that clarifies WowViewer.App post-cutover role
