# 06: Viewer Refactor Implementation Plan

**Author**: Cascade
**Status**: Ready for Implementation

---

## 1. Objective

This document provides a detailed, step-by-step implementation plan for refactoring the `WoWRollback` viewer into a modern, plugin-based architecture. This plan is designed to be executed in a fresh chat session to ensure focus and avoid context window limitations.

## 2. Phased Implementation Checklist

We will proceed through these phases methodically. Each item represents a distinct, verifiable step.

### Phase 1: Project Setup & Foundation

- [ ] **Create `WoWRollback.Viewer` Project**
    - [ ] Create a new, empty C# project named `WoWRollback.Viewer` in the `WoWRollback.sln`.
    - [ ] Configure the `.csproj` file to ensure all non-C# files (HTML, CSS, JS) are included and copied to the build output directory on build.
- [ ] **Migrate Assets**
    - [ ] Move all files and folders from `WoWRollback/ViewerAssets/` into the new `WoWRollback/WoWRollback.Viewer/` project directory.
    - [ ] Verify that the application still builds and the viewer runs correctly after the file migration.
- [ ] **Cleanup**
    - [ ] Delete the now-empty `WoWRollback/ViewerAssets/` directory.

### Phase 2: Backend Manifest Generation

- [ ] **Modify `WoWRollback.Cli`**
    - [ ] Create a new service responsible for generating the `overlay_manifest.json`.
    - [ ] This service will scan the output directory for generated overlay data and build the manifest accordingly.
    - [ ] Add a new CLI flag, `--viewer-next-objects`, to conditionally add the `objects-next` plugin to the manifest for A/B testing.

### Phase 3: Frontend Runtime Core

- [ ] **Create Runtime Directory Structure**
    - [ ] In `WoWRollback.Viewer/js/`, create a new `runtime/` directory.
    - [ ] In `WoWRollback.Viewer/js/`, create a new `plugins/` directory.
- [ ] **Implement Runtime Core**
    - [ ] Create `js/runtime/runtime.js`. This module will be responsible for fetching `overlay_manifest.json`, loading plugins, and managing their lifecycle.
    - [ ] Create `js/runtime/plugin-interface.js` to define the common interface (`initialize`, `loadTile`, `render`, `teardown`) that all plugins will implement.
    - [ ] Create `js/runtime/resource-loader.js`, a centralized, cache-aware utility for fetching all plugin data.

### Phase 4: Iterative Plugin Migration

*This process will be repeated for each simple overlay.*

- [ ] **Migrate Terrain Properties Plugin**
    - [ ] Create `js/plugins/terrain.js` implementing the plugin interface.
    - [ ] Move the rendering logic from `js/overlays/terrainPropertiesLayer.js` into the new plugin.
    - [ ] Update `main.js` to initialize the new runtime instead of the old `overlayManager` for this specific layer.
    - [ ] Verify the terrain overlay works correctly via the new plugin system.
    - [ ] Delete `js/overlays/terrainPropertiesLayer.js`.
- [ ] **Migrate Remaining Simple Layers**
    - [ ] Repeat the process for `areaId`, `holes`, and `liquids`.

### Phase 5: Parallel Refactor of Complex Overlays

- [ ] **Shadow Map Layer**
    - [ ] Create `js/plugins/shadow.js` and migrate the logic from `js/overlays/shadowMapLayer.js`.
- [ ] **M2/WMO Object Layer (A/B Implementation)**
    - [ ] **Do not touch the existing object overlay logic.**
    - [ ] Create `js/plugins/objects-next.js`.
    - [ ] Begin implementation of the new, high-performance WebGL-based renderer within this plugin.
    - [ ] The viewer will load this plugin **only** when it is present in the manifest (driven by the `--viewer-next-objects` CLI flag).

### Phase 6: Deprecation

- [ ] Once all overlays are migrated and the `objects-next.js` plugin is deemed stable and superior:
    - [ ] Delete the `js/overlays/` directory entirely.
    - [ ] Delete `js/overlays/overlayManager.js`.
    - [ ] Remove all related legacy code from `main.js`.
    - [ ] Remove the `--viewer-next-objects` CLI flag and make the new object plugin the default.

## 3. Next Steps

This plan is now ready. Please start a new chat session, and we can begin executing **Phase 1** of this plan.
