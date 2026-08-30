# Phase 0 Research: Benilla 1.12.1 Rust Client Reference Architecture

**Resource URL**: `https://github.com/samwhosung/benilla`  
**Target Era**: World of Warcraft 1.12.1 (Build 5875)  
**Implementation Language**: Rust (using WGPU / modern graphics pipelines)

---

## 1. Key Architectural Subsystems in Benilla

### 1.1 M2 Model Parsing & Embedded Skin Profiles
In Vanilla 1.12.1 (`MD20 0x100`), M2 models store their skin profiles inline within the model file rather than in external `.skin` files (which were introduced in Wrath of the Lich King 3.x).

Benilla implements clean Rust structs mapping the 1.12.1 M2 layout:
- **Header Structure**: Magic `MD20`, Version `0x100`, Name, Global Flags, Global Sequences, Animations, Animation Lookups, Bones, Key Bone Lookups, Vertices, Views (`nViews`, `ofsViews`), Colors, Textures, Transparency, Texture Animations, Tex Replace, Render Flags, Bone Lookups, Texture Lookups, Tex Unit Lookups, Transparency Lookups, Tex Anim Lookups.
- **Embedded View Profiles (`M2View` / `M2Skin`)**:
  - `nIndices`, `ofsIndices`: Lookup indices into the master vertex buffer.
  - `nTriangles`, `ofsTriangles`: Triangle list referencing the vertex index lookup table.
  - `nProperties`, `ofsProperties`: Vertex bone weighting properties.
  - `nSubmeshes`, `ofsSubmeshes`: Mesh partitions (`M2Submesh` / `M2SkinSection`) with bounding boxes, center coordinates, vertex starts, vertex counts, triangle starts, and triangle counts.
  - `nTextureUnits`, `ofsTextureUnits`: Material pass definitions (`M2TextureUnit` / `M2Batch`) linking submeshes to texture lookups, render flags, color animations, and shader IDs.

---

### 1.2 Texture Coordinate Transformations & Shader Combiners
1.12.1 clients use fixed-function or early pixel shader combiners for multi-pass texture blending. Benilla maps these to modern shader uniforms:
- Combiner mode 0: Modulate (Color * Texture)
- Combiner mode 1: Modulate2X
- Combiner mode 2: Decal
- Combiner mode 3: Add
- Combiner mode 4: ModulateAdd

---

### 1.3 Skeletal Transformation Pipeline
- **Bone Hierarchy**: Tree of parent-indexed bones where local transformations $T_{local} = \text{Translate}(t) \cdot \text{Rotate}(q) \cdot \text{Scale}(s)$ are compounded down the tree:
  $$T_{global} = T_{parent} \cdot T_{local}$$
- **Vertex Skinning**: Up to 4 bone influences per vertex:
  $$P_{skinned} = \sum_{i=0}^{3} w_i \cdot (T_{bone[i]} \cdot P_{bind})$$

---

## 2. Comparison with WowViewer C# Implementation

| Subsystem | Benilla (Rust) | WowViewer (C#) | Alignment Status & Opportunities |
|---|---|---|---|
| **1.12.1 M2 Reading** | `M2` decoder in Rust | `M2ModelReader100.cs` in `WowViewer.Core.IO` | **High**. Use Benilla to cross-check submesh offsets and texture unit flag interpretations. |
| **Material / Render Flags** | WGPU pipeline descriptors | `ModelRenderer.cs` / OpenGL in `WoWViewer` | **Medium**. Study Benilla's blend state mapping for complex layered doodads and translucent particles. |
| **Skeletal Animation** | Rust forward kinematics | `M2AnimationTrack.cs` in `WowViewer.Core.Runtime` | **High**. Validate quaternion interpolation and looping edge cases. |
| **Batching & Instancing** | Modern WGPU indirect drawing | `ModelRenderer` / `WorldScene` MDX batching | **High**. Learn buffer management and instance submission patterns for high doodad counts. |

---

## 3. Reference Principles for WowViewer

1. **Native C# Implementation**: All production tooling remains pure C# (`net10.0`).
2. **Oracle / Ground Truth Verification**: Use Benilla as a trusted third-party reference alongside Ghidra disassemblies and official client captures.
3. **Continuous Cross-Era Compatibility**: Ensure fixes for 1.12.1 M2s in `M2ModelReader100` do not regress Alpha 0.5.3 (`MDLX`), TBC 2.4.3 (`MD20 0x100`), or LK 3.3.5 (`MD20 0x108`).
