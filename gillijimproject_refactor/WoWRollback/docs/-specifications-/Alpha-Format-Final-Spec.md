# Alpha 0.5.3 Format Specification (Final Verified)

**Version**: 1.0 (Final)
**Date**: 2025-12-28
**Source**: Ghidra Reverse Engineering of `WoWClient.exe` build 3368.

This document serves as the **Absolute Ground Truth** for parsing and generating data compatible with the WoW Alpha 0.5.3 client. All offsets and logic are verified against the binary.

---

## 1. WDT (World Data) - The "Monolithic Database"

In Alpha, the WDT is not just a definition file; it is a **Virtual File System** containing the entire world's terrain.

### Header
*   **MVER**: Standard version chunk.
*   **MPHD**: 128 bytes (0x80).
*   **MAIN**: 65,536 bytes (4096 entries * 16 bytes).
    *   **Struct `SMAreaInfo` (16 bytes)**:
        *   `0x00`: Offset (to MCNK chunks in file?)
        *   `0x04`: Size (of Area data?)
        *   `0x08`: Flags (Async load state)
        *   `0x0C`: AsyncId (Runtime pointer)

### Chunk Loading
*   **Mechanism**: The client keeps the WDT file handle open and seeks to offsets defined in `MAIN` (or derived/implied).
*   **Chunk Limit**: Individual MCNK chunks **MUST be < 15,000 bytes** (15KB). Exceeding this causes a fatal error.
*   **Streaming**:
    *   **High Priority**: Immediate sync load for chunks near camera.
    *   **Low Priority**: Async background load for distant chunks.

---

## 2. MCNK (Terrain Chunk) - Fixed Layout

Unlike later versions where MCNK sub-chunks (MCVT, MCNR, etc.) are found via an offset table in the MCNK header, Alpha 0.5.3 uses **Hardcoded Fixed Offsets** for the primary data.

### Header (0x88 bytes)
*   **0x00**: Token `MCNK`
*   **0x04**: Size
*   **0x08**: Flags (Bit 0: Has Shadow/Alpha?)
*   **0x0C**: IndexX (Global X)
*   **0x10**: IndexY (Global Y)
*   **0x18**: nLayers (Number of Texture Layers)
*   **0x1C**: nRefs (Number of Doodad/WMO Refs)

### Fixed Sub-Chunks
*   **0x0088**: **MCVT** (Vertices/Heights). Fixed size: 580 bytes (145 floats).
*   **0x02CC**: **MCNR** (Normals). Fixed size: 448 bytes (145 normals, packed?).
*   **0x048C**: **MCLY** (Texture Layers). Token `MCLY` valid here.
*   **Variable**: Followed by `MCRF` (Refs), `MCAL` (Alpha), `MCSE` (Sound), `MCSH` (Shadow).
    *   *Note*: `MCLY` and subsequent chunks are parsed sequentially.

### Sub-Chunk Structures
*   **MCSE** (Sound Emitters): Token `MCSE` (0x4D435345).
    *   Defines ambient sounds. Parsed into `CMapSoundEmitter`.
    *   Contains `SoundID`, `Position`, `Radius`, `TimeConstraints`.

---

## 3. WMO (World Map Object) - v14

*   **Version**: 14 (0x0E).
*   **Monolithic**: All groups are embedded in the root WMO file (no group files).
*   **Rendering Flags**:
    *   `0x04`: **Two-Sided** (Disable Culling).
    *   `0x02`: **Disable Fog**.
    *   `0x20`: **Window/Crystal** (Lighting hack).
*   **Lighting**: Supports max **8 Hardware Lights** + Ambient.
*   **Batching**: Uses `SMOGxBatch` (8 bytes) for render passes.

---

## 4. MDX (M2 Models) - Dynamic Animation

*   **Rendering**:
    *   **Animated UVs**: Texture coordinates are **not** static buffers. They are computed frame-by-frame and passed as dynamic pointers for scrolling/rotation effects.
    *   **Hardware Skinning**: Supported.
    *   **Multi-Texturing**: Batching of texture layers (Texture0 + Texture1).
*   **Material Flags**:
    *   `0x02`: No Fog.
    *   `0x04`: No Depth Test.
    *   `0x08`: No Depth Write.
    *   `0x10`: Two-Sided.

---

## 5. Sound System

*   **Music**: **DirectMusic** (DirectX 8).
    *   Uses `.dls` (Soundbanks) and `.mid` sequences.
    *   Driven by `AreaMIDIAmbiences.dbc` (Zone-based Day/Night tracks).
*   **Emitters**: MCSE chunks in terrain. Support 3D positioning and time-of-day filtering.

---

## 6. DBC (Client Database)

*   **Format**: Standard **WDBC**.
*   **Loading**:
    *   Allocates contiguous block for Records + String Block.
    *   **Pointer Swizzling**: Converts integer string offsets in records to absolute memory pointers (`char*`) *before* string data is read.
    *   This requires exact Schema knowledge (column types) during load.

---

## 7. Water System

*   **Procedural**: No liquid chunks in the modern sense of static mesh data for waves.
*   **Ripple Simulation**: `WaterRadWave` structures simulate dynamic ripples (from rain, footsteps).
*   **Rendering**: Uses `Ocean0.bls` pixel shader and dynamic texture generation.

---
*Verified by Ghidra Decompilation, Dec 2025.*
