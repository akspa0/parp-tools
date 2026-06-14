# Spec Analysis: ViewerApp Refactor + GPU Acceleration + LOD Modernization

- status: analysis
- date: 2026-06-10
- working-label: `viewerapp-gpu-lod-modernization`
- proposed feature id: `056`
- owner: `wow-viewer`
- branch: `v0.5.0-dev`
- parents: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (replaced 2026-06-14 — viewer-first, UE bridge; this spec is viewer-internal OpenGL modernization), `wow-engine-editor-and-interop-plan-2026-05-14.md`, `game-viewer-host-plan-2026-05-13.md`, `wow-viewer-full-porting-roadmap.md`, `wow-viewer-library-completeness-plan-2026-05-06.md`

> Tiny context check (RULE 11A):
> - target surface: `wow-viewer/src/viewer/WoWViewer/ViewerApp*.cs` size + render pipeline modernization (GPU/LOD)
> - proof owner: not yet chosen (Vulkan vs OpenGL fallback vs Silk.NET instancing; see Open Questions)
> - main unproven gap: current renderer is single-tile CPU-mesh builder with no LOD, no instancing, no compute, and the active `ViewerApp.cs` is 621k bytes
> - explicitly out of scope: the Unreal Engine bridge (post-v0.5.0-dev), rewriting format readers, full M2 parity recovery, gameplay, browser/embed delivery

---

## 1. User Intent (parsed)

The user asked for one combined effort with three sub-goals:

1. **Make `ViewerApp.cs` smaller in LoC** by extracting a separate Viewer and Renderer library.
2. **Refactor the renderer to use GPU acceleration**.
3. **Add Level-of-Detail optimizations** to that renderer.

The user also said:

- "We have some old plans talking about this, but we haven't committed to it in a single focused plan yet."
- "Let's audit what we've got, and figure out the best route forwards."
- The Unreal Engine bridge plan is for **beyond v0.5.0-dev**, so it is out of scope for this feature.

This is the perfect entry point for `speckit-analyze`: the user explicitly asked for an audit and route-finding, not implementation.

---

## 2. Existing State (audit findings)

### 2.1 `ViewerApp.cs` size in `wow-viewer/`

| File | Bytes | Notes |
|---|---|---|
| `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` | **621,301** | The "thin" host entry class. Most of the code has already been split out into partials. |
| `ViewerApp_Pm4Utilities.cs` | 177,557 | PM4 toolbar / overlay code |
| `ViewerApp_Sidebars.cs` | 136,664 | Left/right sidebar panels |
| `ViewerApp_CaptureAutomation.cs` | 89,373 | Headless capture driver |
| `ViewerApp_Investigation.cs` | 57,838 | PM4 investigation panel |
| `ViewerApp_MlTraining.cs` | 35,198 | ML training side-pane |
| `ViewerApp_TerrainAnalysis.cs` | 33,530 | Terrain analysis |
| `ViewerApp_MinimapAndStatus.cs` | 21,974 | Minimap + status |
| `ViewerApp_StartupAutomation.cs` | 21,761 | Startup automation |
| `ViewerApp_Workspaces.cs` | 16,692 | Workspace registration |
| `ViewerApp_ClickSelection.cs` | 16,061 | Click selection |
| `ViewerApp_WmoGroups.cs` | 12,797 | WMO group UI |
| `ViewerApp_Themes.cs` | 9,765 | ImGui theming |
| `ViewerApp_ClientDialogs.cs` | 10,615 | File dialogs |
| `ViewerApp_WdlPreview.cs` | 7,132 | WDL preview |
| `MinimapHelpers.cs` | 8,255 | |
| `ViewerApp_RenderQuality.cs` | 4,824 | |
| `ViewerApp_LogViewer.cs` | 4,121 | |

`ViewerApp.cs` is already a **partial class**. The legacy `MdxViewer` had a similar pattern with even larger files (legacy `ViewerApp.cs` was 628,913 bytes, `ViewerApp_Pm4Utilities.cs` was 145k). The wow-viewer port has more partials but the same growth pattern.

The user's "split into a Viewer and Renderer library" intent is real: there is no clean "viewer host" vs "renderer engine" split. The viewer app owns its own `Rendering/` namespace with 28+ files (`M2Renderer.cs`, `MdxAnimator.cs`, `ModelRenderer.cs`, `WmoRenderer.cs`, `TerrainRenderer.cs`, `LiquidRenderer.cs`, `SkyDomeRenderer.cs`, `ShaderProgram.cs`, `FrustumCuller.cs`, etc.) **and** it lives inside `WowViewer` (the viewer app project), not in a shared library.

### 2.2 Existing renderer library (already partially there)

`wow-viewer/src/core/WowViewer.Core.Renderer/` exists, but is small and minimal:

```
Headless/   FrameCapture, HeadlessContext, PngWriter, RenderSurface  (CPU/headless)
Output/     FrameCapture, PngWriter                                  (duplicate of Headless/)
Liquid/     LiquidRenderer, LiquidShader
Scene/      FrustumCuller, RenderVariant, SceneCamera, SceneRenderer
Sky/        SkyRenderer
Terrain/    TerrainConstants, TerrainMesh, TerrainMeshBuilder, TerrainRenderer, TerrainShader
Texture/    TextureCache
Validation/ NativeValidationWorldSceneAdapter
Wmo/        WmoMesh, WmoRenderer, WmoShader
```

`SceneRenderer.RenderTile(...)` only handles **one tile at a time**, with a single `Dictionary<string, MeshCacheEntry>` cache, no instancing, no LOD, no frustum culling, no compute.

So we have two parallel renderers:

- The **new shared** `WowViewer.Core.Renderer` (sparse, single-tile, no LOD, no instancing, no compute)
- The **old viewer-local** `wow-viewer/src/viewer/WoWViewer/Rendering/*` (28+ files, full feature set, but bound to the active viewer app project and bound to a single game format adapter)

The current state is essentially "we have the foundation files for a renderer library, but they are a stub next to the real renderer code that still lives in the viewer app."

### 2.3 Existing plans and specs that overlap

| Doc / Spec | Scope overlap | Decision |
|---|---|---|
| `wow-engine-modernization-plan-2026-05-14.md` | Replaced 2026-06-14 — viewer-first, UE bridge. OpenGL is the viewer rendering path; no Vulkan. | This spec IS the OpenGL modernization effort for the viewer. |
| `wow-engine-editor-and-interop-plan-2026-05-14.md` | Defines the editor + interop shell | Indirectly affected (the viewer app will become a host of the engine). |
| `game-viewer-host-plan-2026-05-13.md` (slices 3, 4, 5, 6) | World session closure, terrain/liquid shader baseline, skybox + lighting, standalone asset consumer | **Direct overlap** with the "renderer modernization" half. Slices 3-6 are basically the same effort framed differently. |
| `wow-viewer-full-porting-roadmap.md` (Phase I, Priority 5) | Lists "OpenGL renderer port" + "Vulkan renderer primary backend" + "Map editor" as a long-range Priority 5 | The user is now asking for this lane to be pulled forward. |
| `wow-viewer-library-completeness-plan-2026-05-06.md` (Section 2.3, Section 3 Phase F) | "Rendering System" table marks `MdxRenderer`, `TerrainRenderer`, `WmoRenderer`, `LiquidRenderer`, `ShaderProgram`, `RenderQueue`, `FrustumCuller`, `Material` as **Missing**. Notes Phase F is "out of scope and requires its own architecture spec." | **This is exactly the architecture spec being requested now.** |
| `specs/009-full-project-reimplementation-spec/` (2,650 lines) | The master design reference. Has full rendering pipeline + GLSL shader source. | Should be referenced for the GPU/LOD section. |
| `specs/020-renderer-culling-and-tile-capture` (P1/P2) | Frustum and tile-level capture culling fix in the current `WowViewerWorldRuntimeBridge` | Must remain compatible; this new spec cannot regress the tile capture culling fix. |
| `specs/030-wmo-render-pass-architecture` | WMO-specific render pass architecture | WMO pass is a sub-concern. Should be referenced, not duplicated. |
| `specs/031-terrain-cell-awareness` | Terrain cell awareness | Sub-concern. Reference. |
| `specs/032-native-renderer-parity` | Native renderer parity | Sub-concern. Reference. |
| `specs/036-renderer-improvements` | "Convergence" spec that pulls 030-032 into a single owner plan, plus a User Story 3 about live terrain/world frame pacing on 3.3.5.12340 | **Direct overlap**. This new spec should either **superset** 036 (so 036 is folded in) or explicitly defer to it for the parts that are already covered. |
| `specs/044-viewer-shell-usability` | Viewer shell UX (dockable panels, menu cleanup, cursor as model) | Not a renderer concern; **out of scope** for the GPU/LOD spec. |
| `specs/045-scene-graph-workbench` | Scene graph tree in the right sidebar | Viewer-app concern, not a renderer-library concern; **out of scope** for the GPU/LOD spec. |
| `specs/049-viewer-ui-consolidation` | UI consolidation | **Out of scope** for the GPU/LOD spec. |
| `specs/055-unreal-engine-bridge` | Unreal Engine bridge | User explicitly said: beyond v0.5.0-dev, **out of scope**. |
| `gv-17-backend-bridge-vulkan-opengl.md` (in `game-viewer-plan-pack-2026-05-14/`) | Vulkan+OpenGL backend bridge design | **Direct overlap**. This is the micro-plan that should be promoted into the new spec. |
| `gv-14-render-layer-contracts.md`, `gv-15`, `gv-16`, `gv-17` | Render layer contracts + terrain/liquid/object packets + backend bridge | This is the engine-side framing. The new spec should consume these, not redefine them. |
| `wow-viewer/AGENTS.md` RULE 8: "one phase at a time" + the engine plan's "one phase at a time" rule | Strict phasing | New spec must be phased, not a single "modernize everything" push. |

### 2.4 Existing `Core.Runtime.World` (already has LOD hooks)

`wow-viewer/src/core/WowViewer.Core.Runtime/World/` already contains:

- `WorldTerrainLodSelector.cs` (LOD selection logic)
- `WorldObjectVisibilityCollector.cs` + `WorldObjectVisibilityContext.cs` (visibility culling)
- `WorldObjectInstance.cs` (per-instance world object)
- `WorldRenderCompositionBuilder.cs` + `WorldRenderCompositionFrame.cs` (frame packet assembly)
- `WorldFramePassCoordinator.cs` + `WorldObjectPassCoordinator.cs` (pass routing)
- `M2RuntimeFramePipeline.cs` (M2 frame pipeline)
- `Validation/HeadlessValidationCaptureSession.cs` (validation session harness)

This is **good news**: the engine-runtime LOD/visibility/pass-routing surface already exists, and the new spec can integrate with it instead of inventing a parallel one.

### 2.5 What the user is **not** asking for

Reaffirmed by the user message and RULE 8:

- Unreal Engine bridge (out of v0.5.0-dev, per user)
- Full M2 parity recovery (tracked in 037/038)
- New file format readers (format readers are complete per RULE 3)
- Browser/embed delivery (long-range; engine plan Pillar C says "WebGL component" is a delivery surface, not a renderer shape)
- M2 animation farm (tracked in 053)
- DBC/DB2 editor (engine plan slice I5)
- V14 ML training (separate lane per RULE 7)

---

## 3. Spec Analysis

### 3.1 Completeness

| Dimension | Finding |
|---|---|
| User story coverage | The user gave 3 sub-goals but no per-story detail. We need to author stories for: (a) splitting ViewerApp into host vs renderer, (b) the new shared renderer library, (c) GPU acceleration, (d) LOD. |
| Acceptance criteria | None yet. Must be staged with real-data validation per AGENTS.md. |
| Edge cases | Many: CPU mesh cache invalidation, GPU resource lifetimes across map switch, tile streaming vs. static tile set, no-graphics-headless capture, format-specific terrain shader variants, frustum culling correctness across the tile seam. |
| Error states | Renderer resource loss, OpenGL context loss, Vulkan device lost, missing shader compilation, NaN camera, etc. Not yet defined. |

**Verdict: Spec must be authored; this analysis is a prerequisite.**

### 3.2 Dependencies

**Upstream dependencies (must exist or be already in place):**

- `WowViewer.Core.Runtime.World` LOD/visibility/pass-routing — **exists** in stub form (`WorldTerrainLodSelector.cs`, `WorldObjectVisibilityCollector.cs`).
- `WowViewer.Core.Runtime` world frame composition — **exists**.
- Staged clients under `output/tmp/wowarchive-clients/` — **exists** per RULE 9 and the data-paths doc.
- `WowViewer.Core.Renderer` skeleton — **exists** (this is where new renderer library slices will live).
- Specs 020, 030, 031, 032, 036 — **exist** as overlapping partial plans. Must be reconciled.

**Downstream blockers (this spec blocks these):**

- The engine plan Phase E2 (Vulkan baseline) requires a host that can drive it; this spec produces the host surface.
- Spec 045 (scene graph workbench) and any future host UX will need a stable, host-thin renderer.
- Future Unreal Engine bridge (post-v0.5.0-dev) consumes the engine/renderer contracts this spec stabilizes.

**Cross-spec dependency check:**

- This spec must **not** regress `020-renderer-culling-and-tile-capture` (P1 culling fix is a hard prerequisite for capture correctness).
- This spec must **not** duplicate or rewrite the existing `WowViewer.Core.Renderer/*` skeleton — that would be a re-rebuild, violating RULE 3 ("tooling for reading game client files is COMPLETE. DO NOT REWRITE IT.") and the spirit of RULE 8.
- The viewer-app files under `wow-viewer/src/viewer/WoWViewer/Rendering/*` are **legacy viewer-local renderer code**, not "game client file reading tooling." The user is explicitly asking to refactor *out of* the viewer app. That is allowed and intended, but should be done by **moving** the rendering code into `WowViewer.Core.Renderer` (and possibly `WowViewer.Core.Runtime`), not by **rewriting** it.

### 3.3 Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Scope explosion**: a "make ViewerApp smaller + add GPU + add LOD" spec is huge. Without strict phasing, it can absorb 6 months of work. | Hard cap at 10 phases. Each phase has one concern. Use the bite-sized rule. |
| 2 | **Backend choice paralysis**: Vulkan-first is named in the engine plan, but no Vulkan code exists. OpenGL fallback exists. A new "GPU acceleration" spec could spend its whole budget arguing about the backend. | Force a single backend decision **in the analyze/spec step, before implementation**. The engine plan already names Vulkan-primary / OpenGL-fallback. Default to that unless the user picks something else. |
| 3 | **LOD semantics ambiguity**: "Level of Detail" can mean (a) terrain mesh LOD (per-tile chunk simplification), (b) object/doodad LOD (M2/WMO distance-based detail), (c) shader LOD (cheaper shaders at distance), (d) texture LOD (mip selection), (e) WMO interior LOD. | Spec must enumerate the LOD *kinds* explicitly. |
| 4 | **Behavior regression on real data**: any change to the rendering path can regress terrain alpha mask, WMO placement, M2 animation, or WDL preview parity. | Every phase must have a real-data validation step on staged clients per the engine plan and AGENTS.md RULE 8. |
| 5 | **Silk.NET instancing/draw-call batching is already GPU acceleration in the loose sense**; the user may mean compute/instanced/multi-tile, or they may mean "just don't hammer GL calls one at a time." | The analyze doc must surface this ambiguity before writing the spec. |
| 6 | **Format-specific shader code is bound to Alpha vs LK vs Cata**: any refactor that ignores this will regress LK 3.3.5 or Cata 4.0.0 parity. | The render layer contract from `gv-14` / `gv-15` / `gv-16` must be honored. |
| 7 | **ViewerApp split is mechanical but tempting to overreach**: a refactor that touches every partial can absorb infinite time. | Phase the split: only split partials that contain renderer-coupling code. UI partials (`Sidebars`, `Pm4Utilities`, `MlTraining`, `Themes`, `ClickSelection`, `LogViewer`, `ClientDialogs`, `Workspaces`, `WdlPreview`, `MinimapAndStatus`) can stay in viewer-app and don't need to be touched by this spec. |
| 8 | **Existing MdxRenderer.cs is 2,866 lines (per the library completeness plan)**: porting it is a major effort. Spec must not silently commit to that. | Treat legacy renderer code as a "read-only reference" and only port what this spec needs. |
| 9 | **GPU resource lifetimes** across map switch + tile streaming + ImGui docking can leak or stutter. | Phases must include resource-lifecycle tests. |
| 10 | **Headless capture must keep working** (used by `WowViewer.Tool.ValidationCapture` and the V16.2 dataset). | The renderer library must continue to expose a `HeadlessContext` style entry point. The existing `WowViewer.Core.Renderer.Headless/*` must remain valid. |

### 3.4 Constitution compliance

| Principle | Compliant? | Note |
|---|---|---|
| I. Repo independence | Yes | All work is inside `wow-viewer/`. No cross-repo references. |
| II. Library-first | **Required by design** | The user is explicitly asking to move renderer code from viewer-app into a shared library. |
| III. Real-data validation | **Required by design** | All phases must validate against staged clients. |
| IV. Residual model chain | N/A | This is a renderer/host refactor, not ML. |
| V. Streaming-first dataset pipeline | N/A | Same. |
| VI. No game client path assumptions | Yes | Use `output/tmp/wowarchive-clients/` only. |

**Safety constraints:**

- `gillijimproject_refactor` is read-only — we treat it as reference input, not code we modify. Compliant.
- Format reader/writer ownership — we do not modify any `WowViewer.Core.IO` parsers. Compliant.
- Terrain alpha risk area — any terrain rendering change must validate against both Alpha and LK. Required.
- `AlphaWdtWriter.cs` is frozen — not touched. Compliant.

**Development workflow:**

- One phase at a time — required.
- Spec docs are source of truth — this analysis becomes the spec once we have a `spec.md`.
- Bite-sized plans — required, max 10 steps per phase.

**Constitution verdict: Compliant, with the Library-First and Real-Data-Validation principles being the load-bearing ones for this spec.**

### 3.5 Gaps (vs. existing code + plans)

1. **No unified renderer-improvements owner plan for the GPU+host-split work.** Spec 036 is the closest, but it scopes to "renderer improvements convergence" on live 3.3.5.12340 maps, not the ViewerApp split or the library promotion.
2. **No spec for promoting the viewer-app `Rendering/` namespace into a shared library.** The 28+ files there are still inside the viewer project.
3. **No LOD spec.** `WorldTerrainLodSelector.cs` exists as a class but no spec drives it into the new renderer.
4. **No GPU backend decision recorded as a spec.** The engine plan says "Vulkan primary, OpenGL fallback" but no plan names the spec that lands the first GPU-accelerated path.
5. **No concrete ViewerApp-reduction LoC target** that the user can sign off on.
6. **The viewer-app `Rendering/*.cs` files duplicate some logic from the new shared `WowViewer.Core.Renderer`**. Need an explicit "use shared" vs "keep local" matrix per file.

### 3.6 Recommendation

**Approve as the basis for a new spec, with these constraints:**

1. Promote this analysis into `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/spec.md` (Spec Kit).
2. Force the user to answer **3-5 scope questions** before writing the spec — see Open Questions below.
3. After spec is written, use `speckit-plan` + `speckit-tasks` to break it into phases of ≤ 10 steps.
4. **Reconcile with `specs/036-renderer-improvements`** before plan generation: either supersede 036, or hand off 036's User Story 3 (live 3.3.5 frame pacing) to a separate track.
5. **Treat the existing `WowViewer.Core.Renderer` skeleton as the canonical owner of the new renderer library code.** Add to it; do not fork.
6. **Treat the existing `wow-viewer/src/viewer/WoWViewer/Rendering/*` files as code to be gradually migrated**, not rewritten. Each migration phase is one bounded concern.
7. **Do not touch format readers, format writers, the ML training pipeline, the Unreal Engine bridge, the VLM/PM4 workbench, the dataset harvester, or the audio engine.** Out of scope.

---

## 4. Open Questions (need user answer before spec)

These are the choices that materially change the spec. The user said "let's figure out the best route forwards," so we should not assume.

### Q1. What does "split into a Viewer and Renderer library" mean concretely?

The user said this is one of the sub-goals. The options are:

- **A. Move existing `wow-viewer/src/viewer/WoWViewer/Rendering/*` into `WowViewer.Core.Renderer`** (one-time move, no rewrite). The Viewer app becomes a thin host that calls into the shared renderer. This is the cleanest interpretation.
- **B. Build a new shared renderer from scratch and port the legacy code into it** (the "long, careful" interpretation). This is what `wow-viewer-library-completeness-plan-2026-05-06.md` Phase F implies.
- **C. Keep the existing viewer-app renderer; add a separate, smaller shared library for new GPU/LOD work** (the "additive" interpretation).

**Recommendation: A** (move, don't rewrite). Move is consistent with RULE 3 spirit (don't rewrite working code) and gets us a real library without rebuilding.

### Q2. What does "GPU acceleration" mean here?

The current `WowViewer.Core.Renderer` already uses Silk.NET.OpenGL (which is GPU). So "GPU acceleration" must mean one or more of:

- **a.** Multi-tile / multi-draw-call batching (reduce GL state changes; use instanced rendering for repeated geometry)
- **b.** Compute shaders (procedural terrain, GPU culling)
- **c.** Async resource streaming (double/triple-buffered mesh upload)
- **d.** Move from immediate-mode `Begin/End`-style to retained-mode VBO/IBO + uniform-buffer pipeline
- **e.** All of the above

**Recommendation: a + d** for v0.5.0-dev. Compute (b) and async streaming (c) are post-v0.5.0 unless the user wants them pulled in.

### Q3. What does "LOD" cover?

The engine plan calls out `terrainLOD`, `mapObjLightLOD`, `terrainAlphaBitDepth`, `MaxLights`, `projectedTextures`, `waterLOD`, M2 optimization flags (per `specs/036-renderer-improvements/spec.md` User Story 2). The user said "Level of Detail optimizations" plural.

Options:

- **A. Terrain mesh LOD only** (per-tile chunk simplification based on camera distance)
- **B. Object LOD only** (M2/WMO distance-based detail switching, draw-distance culling)
- **C. Full LOD matrix** (terrain + objects + water + lights + alpha bit depth + projected textures)
- **D. Whatever `WorldTerrainLodSelector` already provides + object LOD**

**Recommendation: A + B for v0.5.0-dev**, with C deferred. The full LOD matrix is too big for one spec.

### Q4. What is the GPU backend for v0.5.0-dev?

The engine plan says Vulkan-primary / OpenGL-fallback. There is no Vulkan code yet. Options:

- **A. Land Vulkan primary in v0.5.0-dev** (the engine plan E2 phase). Highest risk, highest payoff.
- **B. Land OpenGL modernized (Silk.NET retained-mode, instanced, multi-tile) in v0.5.0-dev, defer Vulkan to a follow-on** (the engine plan Phase E3 is the OpenGL fallback, but we treat it as primary for now).
- **C. Stay on current Silk.NET immediate-mode, add multi-tile batching and LOD only, no backend change** (lowest risk, smallest "GPU" claim).

**Recommendation: B** for v0.5.0-dev. Vulkan primary is too big a phase-zero commitment. OpenGL modernized through Silk.NET with instancing, UBO, and retained-mode VBO/IBO is real GPU work and unblocks the renderer split.

### Q5. What is the LoC-reduction target for `ViewerApp.cs`?

The user said "a bit smaller." This is a numeric claim. Options:

- **A. Cut `ViewerApp.cs` in half** (~310k bytes, from 621k) by moving renderer-coupled partials out.
- **B. Cut `ViewerApp.cs` to < 200k bytes** by also moving non-renderer partials (e.g. `Sidebars`, `Pm4Utilities`, `MlTraining`).
- **C. No numeric target — just "substantially smaller."**

**Recommendation: A for v0.5.0-dev.** The renderer-coupled partials are the right slice to extract. UI partials are viewer-app concerns and don't need to leave.

### Q6. Should this spec supersede `specs/036-renderer-improvements`?

`036` is the existing convergence owner for renderer work. Two live owner plans will collide.

- **A. Yes, supersede 036.** This new spec becomes the single owner. `036` is archived.
- **B. No, defer to 036.** This new spec is a ViewerApp-reduction sub-effort; 036 owns the renderer improvements and live 3.3.5 frame pacing.

**Recommendation: A** but only if the user's intent is "one focused plan" (which they said). If the user wants to keep 036 as a separate owner, this spec must explicitly hand off "live 3.3.5 frame pacing" to 036.

### Q7. v0.5.0-dev scope boundary?

The user said the Unreal Engine bridge is for **beyond v0.5.0-dev**. They did not say what else is in or out for v0.5.0-dev.

Options for "in v0.5.0-dev":

- **A. ViewerApp split + shared renderer library + OpenGL modernized (instanced, multi-tile) + terrain/object LOD** (the recommended core)
- **B. A minus the LOD work** (smaller, leaves LOD for a follow-on spec)
- **C. A plus Vulkan primary** (biggest possible v0.5.0-dev lane; very risky)

**Recommendation: A** with **B as the fallback** if the user wants to keep v0.5.0-dev small.

---

## 5. Proposed spec framing (preview, not yet written)

Once the user answers the Open Questions, the spec body should follow this shape:

```
# Spec 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization

## Status
- branch: v0.5.0-dev
- owner: wow-viewer
- parents: wow-engine-modernization-plan, game-viewer-host-plan (slices 3-6),
           wow-viewer-full-porting-roadmap (Phase I), wow-viewer-library-completeness-plan (Phase F),
           specs/036-renderer-improvements (superseded by this spec, per Q6)
- intent: shrink ViewerApp.cs by moving renderer code into WowViewer.Core.Renderer,
          modernize the shared renderer for multi-tile GPU work, add terrain + object LOD

## Out of scope
- Unreal Engine bridge (per user, beyond v0.5.0-dev)
- Format readers / writers (complete per RULE 3)
- ML training pipeline (RULE 7)
- Audio engine (separate plan)
- VLM/PM4 workbench
- M2 parity recovery (tracked in 037/038)
- Browser/embed delivery (engine plan Pillar C, long-range)

## User stories
1. (P1) Shrink ViewerApp.cs substantially via shared renderer library
2. (P1) Land a modernized shared renderer that handles multi-tile worlds
3. (P1) Add terrain mesh LOD
4. (P2) Add object LOD
5. (P2) Real-data validation suite that survives the refactor
6. (P3) Live 3.3.5 frame pacing ← handed to/from 036

## Requirements
- per-question answer above drives FR-001..FR-NNN

## Success criteria
- ViewerApp.cs LoC reduction target
- shared renderer library compile + tests
- multi-tile render parity
- LOD correctness on staged 0.5.3, 3.3.5, 4.0.0
- headless capture still works

## Phases
- 10 or fewer phases, each ≤ 10 steps
```

The actual spec, plan, and tasks files are produced via `speckit-specify` / `speckit-plan` / `speckit-tasks` after the user answers the Open Questions.

---

## 6. Validation language rule (per AGENTS.md)

- Library compile + tests in `wow-viewer` are primary proof.
- Real-data captures on staged `output/tmp/wowarchive-clients/0_5_3_3368` (Alpha) and `3_3_5_12340` (LK) are required for any change touching terrain, liquid, WMO, or M2 rendering.
- The legacy `MdxViewer` is compatibility evidence, not ownership evidence.
- Do not claim "modern replacement engine" until the new renderer can load and render real worlds through the new shared renderer without legacy ownership seams.

---

---

## 7. Locked Decisions (from user answers, 2026-06-10)

| Question | Decision |
|---|---|
| Q1. Library split | **B. Build a new shared renderer from scratch.** Port `MdxRenderer` (and friends) carefully into `WowViewer.Core.Renderer`. The viewer-app `Rendering/*` becomes a parallel path during cutover, then is retired. |
| Q2. GPU acceleration | **Multi-tile batching + retained-mode VBO/IBO + UBO + instanced rendering.** Compute shaders and async streaming are explicitly out of v0.5.0-dev scope. |
| Q3. LOD coverage | **Full LOD matrix: terrain mesh LOD + object LOD (M2/WMO distance + draw-distance culling) + water LOD (`waterLOD`) + light LOD (`mapObjLightLOD`, `MaxLights`) + WDL for far horizon (1/16th far-distance LOD) + BLP mipmap selection.** |
| Q4. GPU backend | **OpenGL modernized via Silk.NET.** Vulkan primary is deferred to a follow-on spec. |
| Q5. LoC target | **No numeric target; "substantially smaller."** |
| Q6. Reconcile with 036 | **Yes, supersede `specs/036-renderer-improvements`.** 036 is archived. This spec becomes the single owner. |
| Q7. v0.5.0-dev scope | **ViewerApp split + shared renderer library + OpenGL modernized + full LOD matrix** (the recommended core). |

These locked decisions become the spec's "Decisions" section.

## 8. Locked Out-of-Scope

Confirmed in the user's message and locked by Q1/Q2/Q4/Q7:

- Unreal Engine bridge (post-v0.5.0-dev, per user)
- Compute shaders (Q2)
- Async resource streaming (Q2)
- Vulkan primary backend (Q4)
- Format readers / writers (RULE 3)
- ML training pipeline (RULE 7)
- Audio engine (separate plan)
- VLM / PM4 workbench
- M2 parity recovery (tracked in 037/038)
- Browser/embed delivery (engine plan Pillar C, long-range)

## 9. Next Step

Load `speckit-specify` and write `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/spec.md` from this analysis + the locked decisions.

After the spec is approved, run `speckit-plan` to produce `plan.md` (phases, max 10 steps each, dependency-ordered) and `speckit-tasks` to break each phase into concrete bite-sized steps. Then `speckit-implement` to land one phase at a time with real-data validation.

*End of analysis. Spec authoring next.*
