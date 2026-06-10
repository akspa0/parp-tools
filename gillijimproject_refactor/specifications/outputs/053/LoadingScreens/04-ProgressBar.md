# Progress Bar Logic

## Overview

The progress bar is a simple textured quad whose width is scaled by a global progress variable. The progress variable is updated by the engine during loading and ranges from 0.0 to 1.0.

## Implementation Details

### Progress Variable (008c3d24)

The global progress variable (`_DAT_008c3d24`) stores the current progress ratio (0.0 to 1.0). It is updated by `UpdateProgress` (0040842d).

### UpdateProgress (0040842d)

The `UpdateProgress` function calculates the progress ratio:

```c
void UpdateProgress(void) {
    _DAT_008c3d24 = 0.0f;
    if (DAT_008c3cdc != 0) {
        // Calculate ratio: current / max
        _DAT_008c3d24 = ((float)_DAT_008c3d5c / (float)DAT_008c3cdc) * 0.75f; // Scale by 0.75?
    }
    if (DAT_008c3cd0 != 0) {
        // If world loaded flag is set, add 0.25?
        _DAT_008c3d24 += 0.25f;
    }
    // Clamp to 0.0 - 1.0
    if (_DAT_008c3d24 > 1.0f) _DAT_008c3d24 = 1.0f;
    if (_DAT_008c3d24 < 0.0f) _DAT_008c3d24 = 0.0f;
}
```

*   `_DAT_008c3d5c`: Current progress value (int).
*   `DAT_008c3cdc`: Max progress value (int).
*   `DAT_008c3cd0`: World loaded flag (bool).

### FrameXMLProgressCallback (0040820c)

This callback is registered with the engine to receive progress updates. It updates the global variables:

```c
void FrameXMLProgressCallback(int current, int max) {
    _DAT_008c3d5c = current;
    DAT_008c3cdc = max;
    UpdateProgressBar();
}
```

### UpdateProgressBar (00408229)

This function triggers a render update:

1.  Calls `UpdateProgress` to recalculate the ratio.
2.  Sets up an orthographic projection (`GxXformSetProjection`).
3.  Calls `LoadingScreenPaint` to draw the screen.
4.  Calls `GxScenePresent` to swap buffers.

### LoadingScreenRegisterWorldLoaded (0040895a)

Called when the world map has finished loading (`CWorld::LoadMap` returns). It sets the `DAT_008c3cd0` flag to 1, which likely pushes the progress bar to 100% (0.75 + 0.25 = 1.0).

## Conclusion

To implement the progress bar:
1.  Maintain `current` and `max` progress counters.
2.  Calculate ratio: `ratio = (current / max) * 0.75`.
3.  When map loading finishes, set `ratio = 1.0`.
4.  Render the progress bar fill quad with width scaled by `ratio`.
