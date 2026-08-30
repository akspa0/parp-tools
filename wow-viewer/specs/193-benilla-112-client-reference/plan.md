# Implementation Plan: Benilla 1.12.1 Reference Integration & 1.x M2 Parity

**Feature Spec**: [Spec 193](spec.md)  
**Target Codebase**: `wow-viewer/src/core/WowViewer.Core.IO/`, `wow-viewer/src/viewer/WoWViewer/Rendering/`

---

## 1. Technical Architecture & Strategy

We will use Benilla (`https://github.com/samwhosung/benilla`) as an architectural and layout oracle to enhance our C# engine in three distinct phases:

```
[Benilla 1.12.1 Rust Reference]
          │
          ├──> 1. M2 Header & Embedded Views Layout  ───> [M2ModelReader100.cs (C#)]
          │                                                    │
          ├──> 2. Material Passes & Blend Modes      ───> [ModelRenderer.cs (C#)]
          │                                                    │
          └──> 3. GPU Batching & Draw Call Pipeline  ───> [WorldScene.cs / Instancing (C#)]
```

---

## 2. Phased Roadmap

### Phase 1: M2 1.x Embedded View & Submesh Layout Verification
- Cross-reference `M2ModelReader100.cs` layout offsets against Benilla's `M2` parsing logic.
- Verify `nViews` / `ofsViews` unpacking, index lookups, and triangle list reconstruction.
- Validate submesh bounding boxes and center of mass points.

### Phase 2: Material Flags & Texture Unit Mapping
- Map `M2TextureUnit` flags (two-sided, unlit, unfogged, depth-test, depth-write) to OpenGL/Vulkan pipeline states in `ModelRenderer.cs`.
- Refine animated UV texture transformations using Benilla's shader uniform calculation.

### Phase 3: Renderer Optimization & Batching Insights
- Study Benilla's WGPU buffer allocation and indirect draw dispatch strategies.
- Apply batching optimizations to `WorldScene.cs` for smooth rendering of dense 1.12.1 doodad scenes.

---

## 3. Verification & Guardrails

- **Unit Tests**: Retain and expand `WowViewer.Core.Tests` for 1.12.1 M2 decoding and embedded view parsing.
- **Frozen Format Guardrail**: Do not modify working readers for other eras (e.g. 0.5.3 MDX or 3.3.5 M2). Preserve clear format dispatching in `M2ModelReaderDispatcher.cs`.
