# Alpha 0.6.0 Console Variables (CVars)

**Based on String Analysis of WoW Alpha 0.6.0 Client**

This document lists confirmed CVars found in the binary, categorized by their subsystem.

## 1. Graphics & Display
| CVar | Default | Description |
| :--- | :--- | :--- |
| `gxColorBits` | 16/32? | Color depth. |
| `gxDepthBits` | 16/24? | Z-Buffer depth. |
| `gxResolution` | "640x480" | Screen resolution. |
| `gxRefresh` | 60? | Refresh rate. |
| `gxWindow` | 0 | Windowed mode (0=Fullscreen, 1=Windowed). |
| `pixelShaders` | 0/1 | Enable/Disable Pixel Shaders (Arb/NV). |
| `hwDetect` | 1 | Hardware detection flag (Debug). |

## 2. World & Rendering
| CVar | Description |
| :--- | :--- |
| `farclip` | View distance (Range 177.0 - 777.0). |
| `horizonfarclip` | Horizon distance (Range check confirmed). |
| `DistCull` | Distance culling for objects (Range 1.0 - 500.0). |

## 3. Sound (FMOD Based)
Audio is handled by **FMOD** (`_FSOUND_` imports detected).
*   `Sound_` prefix implies standard sound CVars (Enable, Volume, etc.) though precise names were constructed at runtime or in string tables.
*   Functions identify `FSOUND_SetMaxHardwareChannels`, `FSOUND_SetMixer`, etc.

## 4. Debug & Misc
| CVar | Description |
| :--- | :--- |
| `error` | Error reporting level? |
| `scriptErrors` | Lua script error reporting. |
