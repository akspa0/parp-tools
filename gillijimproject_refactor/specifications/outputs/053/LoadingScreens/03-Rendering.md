# Loading Screen Rendering Pipeline

## Overview

The loading screen is rendered using a custom screen layer callback (`LoadingScreenPaint`) that draws 3 textured quads using `GxPrimDrawElements`. The rendering is performed in an orthographic projection.

## Render Function: LoadingScreenPaint (004087e5)

The `LoadingScreenPaint` function is called every frame by the screen layer system. It performs the following steps:

1.  **Setup**:
    *   Pushes the render state (`GxRsPush`).
    *   Disables lighting (`GxRsSet(GxRs_Lighting, 0)`).
    *   Disables fog (`GxRsSet(GxRs_Fog, 0)`).
    *   Selects the pass-through vertex shader (`GxVertexShaderSelect(GxVS_PassThru)`).

2.  **Layer Iteration**:
    *   Iterates through the 3 texture definitions (Background, Border, Fill).
    *   For each layer:
        *   Calculates vertex positions based on the definition (X, Y, Width, Height).
        *   Locks vertex pointers (`GxPrimLockVertexPtrs`).
        *   Sets the texture (`GxRsSet(GxRs_Texture0, textureHandle)`).
        *   Draws the quad (`GxPrimDrawElements(GxPrim_TriangleStrip, 4, ...)`).
        *   Unlocks vertex pointers (`GxPrimUnlockVertexPtrs`).

3.  **Cleanup**:
    *   Pops the render state (`GxRsPop`).

## Progress Bar Rendering

The progress bar fill layer (Layer 3) has special logic:
*   It checks a flag in the texture definition (offset 4).
*   If the flag is set (likely for the fill layer), it scales the **width** of the quad by the global progress variable (`_DAT_008c3d24`).
*   The progress variable ranges from 0.0 to 1.0.
*   This creates a horizontal fill effect.

## Coordinate System

The rendering uses an orthographic projection set up by `ScrnLayerCreate` (0.0 to 1.0 range).
*   **X**: 0.0 (Left) to 1.0 (Right)
*   **Y**: 0.0 (Bottom) to 1.0 (Top) - Note: Y-axis direction might be inverted depending on engine conventions, but usually 0,0 is bottom-left in OpenGL/D3D.
*   **Z**: 0.0 (Near) to 1.0 (Far)

## Text Rendering

No explicit text rendering calls were found in `LoadingScreenPaint`. The "Loading..." text is likely baked into the background texture (`Interface\Glues\loading.blp`) or rendered by a separate UI layer (though no evidence of a separate UI layer was found).

## Implementation Notes

*   Use `GxPrimDrawElements` with `GxPrim_TriangleStrip` for quads.
*   Ensure correct texture coordinates (0,0 to 1,1) are mapped to the quad vertices.
*   Handle alpha blending correctly for the progress bar layers.
