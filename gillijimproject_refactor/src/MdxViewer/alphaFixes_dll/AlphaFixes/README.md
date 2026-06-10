# AlphaFixes - Performance Patches for WoW Alpha 0.5.3

## Overview

AlphaFixes is a DLL injection module that fixes timing-related performance issues in WoW Alpha 0.5.3 on modern hardware. It's modeled after the VanillaFixes project for the 1.12.1 client.

## Features

1. **TSC Calibration Fix**
   - Replaces the inaccurate 30-second startup calibration
   - Uses QueryPerformanceCounter as a stable reference
   - Multiple samples for accuracy
   - Thread priority boost during calibration

2. **Priority Boost for Timing**
   - Boosts thread priority during time snapshot captures
   - Reduces timing jitter from context switches

3. **CPU Frequency Override**
   - Returns calibrated TSC frequency instead of runtime measurement
   - Prevents stuttering from repeated calibration attempts

## Requirements

- Visual Studio 2015 or later (with C++ support)
- CMake 3.10 or later
- WoW Alpha 0.5.3 client

## Building

### Using Visual Studio Developer Command Prompt

1. Open "Developer Command Prompt for VS"
2. Navigate to the AlphaFixes directory
3. Run:
   ```
   mkdir build
   cd build
   cmake .. -G "NMake Makefiles"
   nmake
   ```

### Using Visual Studio IDE

1. Open CMakeLists.txt in Visual Studio
2. Select build configuration (Release)
3. Build the solution

## Installation

### Method 1: DLL Injection Launcher

Copy `build/alphafixes.dll` to your WoW Alpha directory and use a DLL injector or launcher to load it before the game starts.

### Method 2: AppInit_DLLs (Legacy)

Add to registry:
```
[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows]
"AppInit_DLLs"="alphafixes.dll"
"LoadAppInit_DLLs"=dword:00000001
```

**Note:** This method may not work on Windows 10/11 due to security changes.

## Configuration

Create `alphafixes.ini` in the game directory:

```ini
[General]
EnableTSC=1              ; Enable TSC calibration fix (default: 1)
EnablePriorityBoost=1   ; Enable thread priority boost (default: 1)
EnableTimerResolution=1 ; Enable timer resolution increase (default: 1)
CalibrationTime=1000    ; Calibration time in ms (default: 1000)
DebugOutput=0           ; Enable debug output (default: 0)

[Advanced]
ManualTSC=0             ; Manual TSC frequency override (0 = auto)
PinToCpu=-1             ; Pin to specific CPU (-1 = auto, 0+ = CPU number)
UseQPCFallback=1       ; Use QPC as fallback if TSC fails (default: 1)
```

## Troubleshooting

### Black Screen on Launch
- Disable in `alphafixes.ini`:
  ```
  EnableTSC=0
  ```

### Still Stuttering
- Check debug output (set `DebugOutput=1`)
- Verify CPU supports constant TSC
- Try manual TSC override:
  ```
  ManualTSC=3000000000
  ```

### Game Crashes
- Update to latest build
- Disable priority boost:
  ```
  EnablePriorityBoost=0
  ```

## Architecture

```
AlphaFixes.dll
├── alpha_main.c        # DllMain, initialization
├── alpha_tsc.c         # TSC calibration hook
├── alpha_snapshot.c    # Priority boost patch
├── alpha_cpufreq.c    # CPU frequency override
└── alpha_helpers.c    # Utility functions
```

## Binary Targets

| Function | Address | Purpose |
|----------|---------|---------|
| TimeKeeper | 0x0045c100 | Timer calibration thread |
| Snapshot | 0x0045c0a0 | Time capture function |
| OsGetAsyncTimeClocks | 0x0045b960 | RDTSC wrapper |
| CGxDevice::CpuFrequency | 0x005951b0 | CPU frequency query |

## References

- [VanillaFixes Project](https://github.com/GregLukasz/vanillafixes)
- [Lua 5.0 Documentation](http://www.lua.org/manual/5.0/)
- [RDTSC Timing](https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquire-high-resolution-time-stamps)

## License

This project is provided for educational purposes. Use at your own risk.
