---
description: "Port working camera and minimap coordinate behavior from MdxViewer into wow-viewer-owned non-UI services. Use when the next slice is movement/input basis parity, tile/world/screen coordinate parity, or minimizing app-local camera/minimap logic before deeper scene cutover."
name: "wow-viewer Camera and Minimap Foundations"
argument-hint: "Optional focus such as camera movement basis, yaw/pitch parity, tile indexing, minimap mapping, or UI-neutral geometry contracts"
agent: "agent"
---

# wow-viewer Camera and Minimap Foundations

Target repo file: .github/prompts/wow-viewer-gpu-viewer/03-camera-and-minimap-foundations.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/src/MdxViewer/Rendering/Camera.cs
4. gillijimproject_refactor/src/MdxViewer/MinimapHelpers.cs
5. wow-viewer/src/viewer/WowViewer.App/WorldViewCamera.cs
6. wow-viewer/src/viewer/WowViewer.App/WorldMinimapRenderer.cs
7. wow-viewer/README.md

## Problem

Camera and minimap behavior still depends on app-local implementations, which makes parity fragile and prevents clean library ownership.

## Goal

Port camera and minimap geometry behavior into wow-viewer-owned UI-neutral services.

- preserve working movement basis and coordinate mapping behavior
- avoid introducing new app-local logic

## Required Constraints

1. Do not touch gillijimproject_refactor implementation code.
2. Keep this slice non-UI and service-focused.
3. Keep WowViewer.App as a temporary consumer only.

## Concrete Scope

1. port free-camera movement and view-state math into shared service/contracts
2. port minimap tile/world/screen coordinate mapping into shared geometry helpers
3. wire temporary app consumers to shared services
4. add focused tests for movement and mapping parity

## Out Of Scope

1. no terrain adapter extraction yet
2. no renderer parity work yet
3. no WorldScene breakup yet
4. no shell UI redesign

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for camera basis math and minimap coordinate mapping
3. explicit statement of which app-local camera/minimap paths were replaced by shared services

## Deliverables

1. camera service/contracts in wow-viewer libraries
2. minimap geometry service/contracts in wow-viewer libraries
3. temporary app consumer wiring
4. focused parity validation
