# WoW Alpha Client Architecture: The Definitive Comparison

**Date**: Jan 09 2026
**Subject**: Analysis of WoW Alpha versions 0.5.3, 0.5.5, and 0.6.0.

## 1. Executive Summary
The transition from Alpha 0.5.3 to 0.6.0 represents a massive architectural shift in the World of Warcraft engine.
*   **0.5.3 (Build 3368)**: The "Old World" anchor. Uses Miles Audio, Monolithic Maps, and WMO v14.
*   **0.5.5 (Build 3494)**: The "Hybrid" bridge. Introducing FMOD and the modern UI, it retains the Old Map formats but hides a fully functional prototype of the Modern ADT system.
*   **0.6.0 (Build 3592)**: The "New World" foundation. Launches the Split ADT format and WMO v17 basics, setting the standard for Vanilla WoW.

## 2. Detailed Comparison Matrix

| Feature | Alpha 0.5.3 | Alpha 0.5.5 | Alpha 0.6.0 |
| :--- | :--- | :--- | :--- |
| **Build Date** | Dec 11 2003 | ~Jan 2004 | ~Feb 2004 |
| **Map Architecture** | **Monolithic WDT** | **Monolithic WDT** (Latent ADT) | **Split ADT** (WDT + ADT) |
| **WMO Format** | **v14** | **v14** | **v16** (Hybrid) |
| **Audio Engine** | **Miles (MSS)** | **FMOD** | **FMOD** |
| **Texture Format** | **BLP2** (Type 1) | **BLP2** (Type 1) | **BLP2** |
| **UI Engine** | **XML / Lua 5.0** | **XML / Lua 5.0** | **XML / Lua 5.1** |
| **Database** | WDBC | WDBC | WDBC |
| **Development Tools** | **Active Sound Editor** / Active GM cmds / Latent MDL Exporter | Latent ADT Loader | Console CVars |

## 3. Key Discoveries

### 3.1 The "BLP1" Myth
Contrary to popular community knowledge, **Alpha 0.5.3 does NOT use BLP1 (Jpeg)**. It strictly checks for the `BLP2` magic (`0x32504C42`). This implies the switch to BLP2 happened much earlier in development (pre-0.5.3).

### 3.2 The UI Revolution
Both 0.5.3 and 0.5.5 use the modern **XML/Lua** UI system. The idea that these clients used a "Hardcoded" UI is false; they merely lacked the exposed `Interface\` folder structure of later builds, but the engine was active.

### 3.3 The 0.5.5 "Rosetta Stone"
Alpha 0.5.5 is the most fascinating build. It runs on the old Monolithic Map format (WDT v14) but contains a hard-gated, fully functional loader for **Prototype ADT** files. This loader expects a "Headerless" ADT structure with "Implicit" MCNK data chunks, a unique format found nowhere else.

### 3.4 WMO Evolution
*   **v14**: Used in 0.5.3 and 0.5.5. Features `MOLM`/`MOLD` lightmaps.
*   **v16**: Used in 0.6.0. A hybrid format using v17 Group headers but retaining v14-style material definitions (referencing `MOBA` batches).

## 4. Recommendations for Converter Tools
1.  **Likewolf Conversion**:
    *   Target **WMO v14** for 0.5.3 compatibility.
    *   Target **Monolithic WDT** for maps.
2.  **Coordinate Systems**:
    *   Always use **XZY** order for object placement and bounding boxes. This is consistent across all Alpha versions.
3.  **Textures**:
    *   Ensure all textures are converted to **BLP2 (Type 1)**. Do not generate BLP1 files.

## 5. References
*   `Alpha_0.5.3_Analysis.md`: Detailed breakdown of Title 0.5.3.
*   `Alpha_0.5.5_Comparisons.md`: Feature comparison and deep dive.
*   `Alpha_0.5.5_ADT_Spec.md`: Technical spec of the prototype ADT format.
*   `WMO_Format_Evolution.md`: Technical spec of WMO v14/v16.
