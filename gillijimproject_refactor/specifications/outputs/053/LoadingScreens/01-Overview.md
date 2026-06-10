# WoW Alpha 0.5.3 Loading Screen System Overview

## Executive Summary

The loading screen system in World of Warcraft Alpha 0.5.3.3368 is significantly simpler than in later versions. Unlike the Beta and Release clients, which use `LoadingScreens.dbc` to select map-specific artwork, the Alpha client appears to use a **single static loading screen** for all map transitions.

## Key Findings

1.  **No LoadingScreens.dbc**: The `LoadingScreens.dbc` file is not loaded by the client, and there are no string references to it.
2.  **Static Artwork**: The loading screen texture is hardcoded to `Interface\Glues\loading.blp`.
3.  **No Map-Specific Logic**: The `EnableLoadingScreen` function does not take any arguments (like Map ID) and uses a static array of texture definitions.
4.  **Progress Bar**: The progress bar is rendered using a textured quad (`Interface\Glues\LoadingBar\Loading-BarFill`) whose width is scaled by a global progress variable.
5.  **Render Pipeline**: The loading screen is implemented as a screen layer with a callback function (`LoadingScreenPaint`) that draws 3 quads (Background, Border, Fill) using `GxPrimDrawElements`.

## Architecture

The system consists of:
*   **EnableLoadingScreen**: Initializes the system, loads textures, and creates the screen layer.
*   **LoadingScreenPaint**: The render callback that draws the screen every frame.
*   **UpdateProgress**: Updates the global progress variable based on loading status.
*   **DisableLoadingScreen**: Cleans up resources and hides the screen.

## Comparison with Release Client

| Feature | Alpha 0.5.3 | Release 1.12 |
| :--- | :--- | :--- |
| **Selection Logic** | Static (Single Image) | Dynamic (Map.dbc -> LoadingScreens.dbc) |
| **DBC Support** | None | LoadingScreens.dbc |
| **Widescreen** | No | Yes (4:3 and 16:9 variants) |
| **Tips** | None | LoadingScreenTips.dbc |
| **Progress Bar** | Textured Quad | XML/Lua UI Element |
