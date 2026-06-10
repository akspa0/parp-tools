# Alpha 0.5.5 Analysis & Comparisons

**Binary**: `WoWClient.exe` (Alpha 0.5.5)

This document details the features of the Alpha 0.5.5 client, positioning it as a **Transition Client** between the 0.5.3 (Monolithic) and 0.6.0 (Modular) eras.

## 1. WMO Format: v14 (Confirmed)
Alpha 0.5.5 uses the EXACT same WMO format as 0.5.3.

*   **Version**: v14 (Implied).
*   **Lighting**: **Embedded Lightmaps** are supported and verified.
    *   **Evidence**: The Group Parser (`FUN_006bb8c0`) explicitly asserts for `MOLM` (Lightmap Header) and `MOLD` (Lightmap Data) tokens.
    *   **Implication**: WMO files generated for 0.5.5 must include lightmaps if the flag is set. v16-style (0.6.0) WMOs will crash or fail to load.

## 2. Map Format: Monolithic WDT (Primary) with Latent ADT Support
Alpha 0.5.5 sits right on the fence of the WDT/ADT split.

*   **Primary Mode**: **Monolithic WDT** (like 0.5.3).
    *   **Evidence**: The binary contains the `MAIN` token string (`008abaa8`), which is the signature of the old WDT format containing embedded map chunks.
    *   **Evidence**: The string `"%s\%s.wdt"` exists, but `"%s_%d_%d.adt"` does NOT.
    *   **Evidence**: MCNK parser references `s_CMap__wdtFile` errors.
*   **Latent Mode**: **Split ADT** (Experimental / Developer Only).
    *   **Evidence**: A full ADT loader function (`FUN_006b6390`) exists, checking for `MHDR` and `MCIN`.
    *   **Format**: The loader expects a **Headerless ADT**. It checks for `MHDR` immediately at the start of the file data. It **DOES NOT** check for or skip an `MVER` chunk.
    *   **MCNK Structure**: The Map Chunk structure appears to use **Fixed Offsets** for primary data.
        *   `MCVT` (Heights) and `MCNR` (Normals) data are embedded immediately after the header (implicit).
        *   `MCLY` (Layers) is checked at a hardcoded offset (likely 1164 bytes), suggesting a strict layout.
        *   `MCRF` (References) is explicitly checked by token.
    *   **Mechanism**: Loading via this path is gated by a runtime flag (`DAT_008ab3e4`).
    *   **Investigation**: This flag is **hardcoded to reset to 0** after specific internal operations (`FUN_006901b0`), effectively disabling ADT loading for the player. The loader code is fully intact, however, meaning a client patch (nop-ing the reset) could theoretically enable native ADT support.

**Converter Recommendation**: Generate **Monolithic WDT** files (all data in `.wdt`) for 0.5.5 to ensure maximum compatibility without client patching.

## 3. Audio System: FMOD (New)
A major architectural shift occurred here.

*   **Library**: **FMOD** (`fmod.dll`).
*   **Functions**: Imports `_FSOUND_Init`, `_FSOUND_Stream_PlayEx`, etc.
*   **Formats**: Supports `.mp3`, `.ogg`, `.wav`, `.mid`, `.dls` via FMOD's internal codecs.

## 4. Legacy Model Support: MDX (Confirmed)
Alpha 0.5.5 retains a fully functional **Warcraft 3 MDX Loader**.

*   **Evidence**: Strings `Spells\*.mdx` and `Environments\Stars\stars.mdl` are present and used.
*   **Header**: The model parser (`FUN_0041fc20`) explicitly checks for the `MDLX` (0x584c444d) magic token.
*   **Version**: The parser logic (`FUN_0041ece0`) expects `TEXS` (Texture) chunk entries to be **268 bytes**. This aligns with **Version 800+** of the MDX format (Reign of Chaos/The Frozen Throne).
*   **Usage**: Primary usage appears to be for Spell Effects (`FlamestrikeSmall`, `CallLightning`) and Skyboxes (`stars.mdl`), indicating M2 had not yet fully replaced all assets.

## 5. Comparison Table

|Feature|Alpha 0.5.3|Alpha 0.5.5|Alpha 0.6.0|
|:---|:---|:---|:---|
|**WMO Version**|**v14**|**v14**|**v16** (Hybrid)|
|**WMO Lighting**|Lightmaps / VertColors|Lightmaps / VertColors|VertColors Only|
|**Map Format**|**Monolithic WDT**|**Monolithic WDT** (ADT latent)|**Split ADT** (.wdt + .adt)|
|**Chunk Format**|`MAIN` header|`MAIN` header|`MHDR` header|
|**Audio Engine**|Miles (MSS)|**FMOD**|**FMOD**|
|**MDX Support**|Likely|**Yes** (800)|**Yes** (800)|
|**UI System**|Hardcoded|**XML / Lua 5.0**|**XML / Lua 5.1**|
|**Textures**|BLP1?|**BLP2** (Type 1)|BLP2|
|**Database**|WDBC|**WDBC**|WDBC|

## 6. Major Discoveries (0.5.5)
*   **The First Modern Client**: Despite using v14 WMOs and Monolithic maps, the 0.5.5 client is architecturally closer to 1.0.0 than 0.5.3. It runs the **XML/Lua UI** system and the **FMOD** audio engine.
*   **BLP2 Support**: It explicitly checks for the `BLP2` magic, rejecting older formats if strict checking is enabled.
*   **WDBC**: Uses the standard `WDBC` database format.

## Conclusion
Alpha 0.5.5 is functionally identical to 0.5.3 regarding file formats (WMO v14, Monolithic WDT) but introduces the **FMOD** audio engine, paving the way for the 0.6.0 architecture.
