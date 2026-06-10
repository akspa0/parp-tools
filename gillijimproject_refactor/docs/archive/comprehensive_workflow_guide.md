# Universal World of Warcraft Map Conversion Guide

This document outlines the complete workflows for converting map data between World of Warcraft versions, specifically focusing on:
1.  **Alpha (0.5.3) → WotLK (3.3.5)**: Stabilized, automated pipeline.
2.  **Modern (11.0+) → Alpha (0.5.3)**: "Backporting" pipeline with asset preservation.

---

## Part 1: Alpha (0.5.3) to WotLK (3.3.5)

This workflow is fully automated by the **WoWRollback** toolchain. It converts terrain, textures, and objects, and patches AreaIDs using stabilized crosswalks.

### Prerequisites
- **WoWRollback** (built from source)
- **Alpha Data**: Extracted 0.5.3 client data (WDT/ADT/DBC)
- **LK Data**: 3.3.5 `DBFilesClient` (for crosswalks)

### Workflow

1.  **Prepare Data Layout**
    Ensure your input directory follows the standard tree structure:
    ```
    test_data/
    ├── 0.5.3/tree/
    │   ├── DBFilesClient/ (AreaTable.dbc, Map.dbc)
    │   └── World/Maps/<MapName>/
    └── 3.3.5/tree/DBFilesClient/
    ```

2.  **Run Conversion**
    Use the unified orchestrator to process a map (e.g., `Shadowfang`).
    
    ```powershell
    dotnet run --project WoWRollback.Orchestrator -- \
      --maps Shadowfang \
      --versions 0.5.3 \
      --alpha-root ../test_data \
      --lk-dbc-dir ../test_data/3.3.5/tree/DBFilesClient \
      --serve --port 8080
    ```

3.  **Verify Results**
    The tool will:
    - Generate crosswalks (handling prototype maps like Kalidar/Map 17 by forcing AreaID 0).
    - Convert ADTs to 3.3.5 format.
    - Fix object coordinates (swapping Y/Z axes).
    - Launch a web viewer to inspect the result.

    **Key Output Locations:**
    - ADTs: `parp_out/session_.../03_adts`
    - Crosswalks: `parp_out/session_.../02_crosswalks`

---

## Part 2: Modern (CASC / 11.0+) to Alpha (0.5.3)

This workflow "backports" modern terrain to the Alpha client. It involves handling CASC extraction, texture resizing, and coordinate transformation.

### Prerequisites
- **CASC Explorer** (or `wow.tools.local`) to extract modern assets.
- **BlpResize**: Tool to downscale/convert modern high-res BLP textures to Alpha-compatible formats.
- **AlphaWdtAnalyzer** (with write capabilities) or custom backport script.

### Workflow

1.  **Extract Modern Data**
    Extract the desired map (WDT and ADTs) and referenced assets (BLP, M2, WMO) from the CASC client.
    - *Note*: Modern ADTs use chunked structures (MCNK) similar to LK but with newer headers (MHDR/MH2O updates).

2.  **Texture Reprocessing (BlpResize)**
    Alpha clients have strict texture limitations (size and format).
    - Run `BlpResize` on all extracted `.blp` files.
    - **Goal**: Resize to max 256x256 (or 512x512 depending on client) and ensure non-DX11 formats.
    - *Command (Example)*: `BlpResize --input "extracted/World/Textures" --output "alpha/World/Textures" --max-size 256`

3.  **Coordinate Transformation (Critical)**
    Modern WoW and WotLK use a **Z-up** coordinate system (Height = Z).
    Alpha (0.5.3) uses a **Y-up** coordinate system (Height = Y).
    
    When writing the Alpha ADT/WDT files, you **MUST** swap the Y and Z coordinates for all placements (M2, WMO).
    
    *Logic:*
    ```csharp
    // Modern (X, Y, Z_Height) -> Alpha (X, Z_Height, Y)
    float alphaX = modernX;
    float alphaY = modernZ_Height; // Swap
    float alphaZ = modernY;        // Swap
    ```
    *Refer to `AdtAlpha.SwapYAndZ` in `GillijimProject` for the implementation reference.*

4.  **ADT Conversion**
    Convert the Modern ADT chunks to Alpha format.
    - **Header**: Strip modern chunks (`MFBO`, `MTXF`, etc.) not supported by Alpha.
    - **MCNK**: Rebuild `MCNK` headers. Alpha `MCNK` headers are smaller (without `Mclq` pointers in the same way).
    - **Liquid**: Downgrade liquid data (MH2O) back to MCNK-embedded liquid or simple MCLQ.

5.  **Import to Client**
    - Place the generated `.wdt` and `.adt` files into the Alpha client's `World/Maps/<MapName>/`.
    - Drop the processed textures into `World/Textures/...`.
    - Update `Map.dbc` and `AreaTable.dbc` in the Alpha MPQ if adding a new map ID.

### Troubleshooting Backports

- **Missing Objects**: Usually caused by failing to swap Y/Z coordinates. Check if objects are floating (Z issue) or shifted (Y issue).
- **Black Textures**: Ensure `BlpResize` generated valid BLP1/BLP2 formats compatible with the 0.5.3 engine.
- **Crashes**: Check `Map.dbc` consistency and ensure no modern chunk types (e.g., `MH2O`) remain in the Alpha ADT inputs if the reader doesn't support them.
