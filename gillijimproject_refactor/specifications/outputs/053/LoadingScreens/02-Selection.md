# Loading Screen Selection Logic

## Overview

In Alpha 0.5.3, the loading screen selection logic is **static**. The client does not use `LoadingScreens.dbc` or `Map.dbc` to determine the loading screen texture. Instead, it uses a hardcoded path for all map transitions.

## Implementation Details

### EnableLoadingScreen (00408172)

The `EnableLoadingScreen` function is responsible for initializing the loading screen. It performs the following steps:

1.  Calls `DisableLoadingScreen` to ensure any previous loading screen is cleaned up.
2.  Iterates through a static array of texture definitions at `00802c10`.
3.  Calls `LoadImage` for each definition (3 times).
4.  Creates a screen layer using `ScrnLayerCreate` with `LoadingScreenPaint` as the callback.
5.  Registers input handlers.
6.  Initializes the progress bar.

### Texture Definitions (00802c10)

The static array at `00802c10` contains 3 entries, each defining a layer of the loading screen. Based on string references and usage patterns, the layers are:

1.  **Background**: `Interface\Glues\loading.blp`
2.  **Progress Bar Border**: `Interface\Glues\LoadingBar\Loading-BarBorder.blp`
3.  **Progress Bar Fill**: `Interface\Glues\LoadingBar\Loading-BarFill.blp`

### Map ID Usage

Although the Map ID is available (stored in `DAT_008c3bd8` by `NewWorldHandler`), it is **not used** by `EnableLoadingScreen` to select the texture. The function takes no arguments and accesses only global state related to the screen layer handle.

## Conclusion

To replicate the Alpha 0.5.3 loading screen behavior:
*   Always display `Interface\Glues\loading.blp` as the background.
*   Do not implement map-specific loading screens unless you want to deviate from the original client behavior (e.g., for modern QoL).
