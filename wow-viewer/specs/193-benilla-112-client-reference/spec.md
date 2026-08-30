# Feature Specification: Benilla 1.12.1 Client Reference & 1.x M2 Rendering Parity

**Feature Branch**: `193-benilla-112-client-reference`  
**Created**: 2026-08-30  
**Status**: Active / Reference Architecture  
**External Resource**: [Benilla (`samwhosung/benilla`)](https://github.com/samwhosung/benilla) — Modern Rust-based World of Warcraft 1.12.1 client implementation.

---

## 1. Overview & Purpose

The **Benilla** project (`https://github.com/samwhosung/benilla`) is an actively maintained, clean-room, modern Rust implementation of a World of Warcraft 1.12.1 (Vanilla) client. It provides a complete, modern reference implementation for:
1. **1.x Era M2 (`MD20 0x100`) Model Parsing**: Header structure, embedded skin profile offsets (`nViews` / `ofsViews`), vertex formats, bone definitions, animation sequences, texture lookups, and material transparency flags.
2. **Submesh & Material Binding**: Texture coordinate transforms, blend modes (`Combiners` / `TextureUnits`), and multi-pass rendering logic.
3. **Skeleton Animation & Bone Hierarchy**: Forward kinematics, quaternion interpolation, and bone transformation pipelines.
4. **Modern Renderer Architecture**: Efficient geometry batching, modern GPU rendering pipelines (WGPU/Vulkan/Metal/DX12), and scene graph management.

### Architectural Boundary & Constitution
- **Tooling Ownership**: Our codebase remains 100% native C# (`net10.0`) within `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Renderer`, and `WoWViewer`.
- **Reference Role**: Benilla serves as an external oracle, architectural reference, and validation baseline to resolve ambiguities in legacy 1.x M2 reading and rendering, without adopting Rust dependencies or copying code wholesale.

---

## 2. User Stories & Acceptance Criteria

### User Story 1 — 1.x M2 Reading & Embedded Skin Parity (Priority: P1)
**As a** tools developer working on legacy World of Warcraft model loading,  
**I want** to cross-reference our C# 1.x M2 reader (`M2ModelReader100.cs`, `M2ModelDocument.cs`) against Benilla's Rust 1.12.1 M2 parser,  
**So that** all 1.12.1 character, creature, and doodad M2 files parse embedded skin profiles, submeshes, and texture units with 100% fidelity.

**Acceptance Criteria**:
1. **Given** a 1.12.1 M2 asset with embedded skin data, **When** parsed by `M2ModelReader100`, **Then** submesh vertex indices, triangle lists, and texture unit bindings match the structures decoded by Benilla.
2. **Given** a 1.12.1 model with animated texture coordinates or multiple material layers, **When** inspected in `WoWViewer`, **Then** material blend modes and texture indices resolve correctly without missing textures or inverted normals.

---

### User Story 2 — Skeletal Bone Hierarchy & Animation Interpolation (Priority: P2)
**As a** viewer developer,  
**I want** to verify our skeletal transformation and keyframe interpolation algorithms against Benilla's animation system,  
**So that** 1.x model animations (walk, run, attack, idle) evaluate smoothly and accurately without bone distortion.

**Acceptance Criteria**:
1. **Given** a 1.12.1 skeletal animation track, **When** evaluated at time $t$, **Then** bone matrices match the forward kinematics and quaternion slerp behavior evidenced in 1.12.1 client disassemblies and Benilla's runtime.

---

### User Story 3 — Modern Renderer Ergonomics & Performance (Priority: P3)
**As a** renderer engineer,  
**I want** to study Benilla's batching, draw call submission, and shader pipeline strategies,  
**So that** we can incorporate architectural insights into our C# OpenGL/Vulkan renderer (`ModelRenderer.cs`, `WorldScene.cs`) to minimize CPU-GPU overhead and render thousands of doodads smoothly.

**Acceptance Criteria**:
1. **Given** dense 1.12.1 scenes with thousands of M2 doodads and WMO structures, **When** rendered in `WoWViewer`, **Then** draw call batching and state transitions achieve stable 60+ FPS framerates.

---

## 3. Related Specifications

- [Spec 104: Legacy M2 Model Rendering (1.0.0 – 2.4.3)](../104-legacy-m2-rendering/spec.md)
- [Spec 136: M2 Doodad Rendering Performance Optimization](../136-m2-doodad-rendering-performance-optimization/spec.md)
- [Spec 154: M2 Era Reader Parity](../154-m2-era-reader-parity/spec.md)
- [Spec 190: Rosetta Calibration Corpus](../190-rosetta-calibration-corpus/spec.md)
