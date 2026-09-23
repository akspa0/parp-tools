# Tasks: Benilla 1.12.1 Client Reference & 1.x M2 Parity

**Feature Spec**: [Spec 193](spec.md) | **Plan**: [Plan](plan.md) | **Research**: [Research](research.md)  
**External Resource**: [Benilla (`samwhosung/benilla`)](https://github.com/samwhosung/benilla) — Modern Rust-based World of Warcraft 1.12.1 client implementation.

---

## Phase 1: 1.x M2 Embedded View & Submesh Layout Verification (US1)

- [ ] **T101**: Inspect Benilla's `M2` model parsing struct definitions and compare byte offsets against `M2ModelReader100.cs` in `WowViewer.Core.IO`.
- [ ] **T102**: Cross-reference `nViews` / `ofsViews` unpacking, index lookups, and triangle list reconstruction between Benilla and `M2ModelReader100.cs`.
- [ ] **T103**: Verify submesh bounding box interpretations and center coordinates (`M2Submesh` / `M2SkinSection`).
- [ ] **T104**: Author unit test in `WowViewer.Core.Tests` asserting exact submesh and index decoding for representative 1.12.1 M2 models.

---

## Phase 2: Material Flags, Texture Units & Blend Combiners (US1 & US2)

- [ ] **T201**: Review Benilla's texture combiner mapping (Modulate, Modulate2X, Decal, Add, ModulateAdd) against OpenGL state in `ModelRenderer.cs`.
- [ ] **T202**: Cross-check `M2TextureUnit` render flags (two-sided, unlit, unfogged, depth-test, depth-write) with `WowViewer.Core.Runtime` material flags.
- [ ] **T203**: Validate animated UV matrix generation and shader uniform passing.
- [ ] **T204**: Verify forward kinematics bone transformation compounding against Benilla's quaternion evaluation.

---

## Phase 3: Modern Renderer Batching & Scene Pipeline Optimization (US3)

- [ ] **T301**: Analyze Benilla's GPU buffer management, vertex buffer suballocation, and multi-instance submission patterns.
- [ ] **T302**: Incorporate draw call reduction strategies into `WorldScene.cs` / `ModelRenderer.cs` for dense 1.12.1 doodad scenes.
- [ ] **T303**: Profile and verify 60+ FPS framerate stability on complex Vanilla scenes.
