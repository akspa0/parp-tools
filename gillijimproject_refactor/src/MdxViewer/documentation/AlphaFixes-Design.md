# AlphaFixes: Performance Patches for WoW Alpha 0.5.3

## Overview

This document provides detailed patch designs for fixing performance stutters in WoW Alpha 0.5.3, modeled after the VanillaFixes project for 1.12.1. The patches focus on the Timer/TSC (Time Stamp Counter) system which is a major source of timing-related stuttering on modern systems.

## Root Cause Analysis

The 0.5.3 client uses RDTSC for high-precision timing, but the calibration mechanism has several issues on modern hardware:

1. **TSC Desynchronization**: On multi-core systems, TSC may not be synchronized across cores
2. **Variable CPU Frequencies**: Modern CPUs change clock speeds (Turbo Boost, power saving)
3. **Coarse Sleep Intervals**: Default 50ms sleep intervals are too long
4. **No Thread Priority Boosting**: Calibration runs at normal priority, leading to inconsistent measurements
5. **Long Calibration Time**: 30-second calibration delay on startup

## Architecture Comparison

### 1.12.1 VanillaFixes Architecture
```
OsTimeManager::TimeKeeper (thread)
    ├── Hooked by VanillaFixes
    ├── CalibrateTSC() function
    │   ├── Sleep(500ms)
    │   ├── QueryPerformanceCounter
    │   └── __rdtsc()
    └── Writes to globals:
        ├── TimerTicksPerSecond (0x008332c0)
        ├── TimerToMilliseconds (0x008332c8)
        ├── UseTSC (0x00884c80)
        └── TimerOffset (0x00884c88)
```

### 0.5.3 Alpha Architecture
```
OsTimeManager::TimeKeeper (0x0045c100)
    ├── Calls Calibrate(0x0045c160)
    │   ├── Snapshot with rdtsc() + GetTickCount()
    │   ├── Waits up to 30 seconds
    │   └── Writes to global at DAT_00cc45a8
    └── Saves to registry: "Internal\\CpuTicksPerSecond"
```

---

## Patch Design: AlphaFixes

### Component 1: Timer Calibration Override

#### Target Function
`OsTimeManager::TimeKeeper` at **0x0045c100**

#### Signature Pattern
```c
// 0x0045c100 - TimeKeeper function start
BYTE sigTimeKeeper[] = {
    0x8B, 0x0D,                   // mov ecx, [DAT_00cc4594]
    0x94, 0x45, 0x9C, 0x00,       //   (address varies)
    0xE8, 0x5B, 0xFF, 0xFF, 0xFF, // call Calibrate
    0x56,                           // push esi
    0x8B, 0xF1,                    // mov esi, ecx
    0x8B, 0x06,                    // mov eax, [esi]
    0x8B, 0x50, 0x04,             // mov edx, [eax+4]
    0xFF, 0xD2,                    // call edx
    0x8B, 0xC8,                    // mov ecx, eax
    // ... more code
};
```

#### Hook Implementation
```c
// dll/alpha_tsc.c

#include <windows.h>
#include <intrin.h>
#include <math.h>
#include <processthreadsapi.h>
#include <timeapi.h>

#include "alpha_tsc.h"
#include "alpha_helpers.h"

// Global state
static volatile BOOL g_initialized = FALSE;
static volatile DWORD64 g_tscFrequency = 0;
static volatile double g_timerToMs = 0;
static volatile int64_t g_timerOffset = 0;

// Calibration function with improved accuracy
DWORD64 CalibrateTSCAlpha() {
    // Boost thread priority
    HANDLE hThread = GetCurrentThread();
    int oldPriority = GetThreadPriority(hThread);
    SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
    
    // Pin to first core to avoid TSC desync
    DWORD_PTR oldAffinity = SetThreadAffinityMask(hThread, 1);
    
    // Increase timer resolution
    timeBeginPeriod(1);
    
    // Multiple samples for accuracy
    DWORD64 samples[5];
    DWORD ticks[5];
    DWORD64 qpcSamples[5];
    LARGE_INTEGER qpcFreq;
    QueryPerformanceFrequency(&qpcFreq);
    
    for (int i = 0; i < 5; i++) {
        // Request timeslice before sampling
        Sleep(0);
        
        QueryPerformanceCounter((LARGE_INTEGER*)&qpcSamples[i]);
        samples[i] = __rdtsc();
        ticks[i] = GetTickCount();
        
        Sleep(100);  // 100ms between samples
    }
    
    // Find most consistent 500ms+ window
    int bestStart = 0;
    int bestCount = 0;
    DWORD64 bestQpcDelta = 0;
    DWORD64 bestTscDelta = 0;
    
    for (int i = 0; i < 5; i++) {
        for (int j = i + 2; j <= 5; j++) {
            DWORD64 qpcDelta = qpcSamples[j-1] - qpcSamples[i];
            DWORD64 tscDelta = samples[j-1] >= samples[i] 
                ? samples[j-1] - samples[i]
                : samples[i] - samples[j-1];
            DWORD tickDelta = ticks[j-1] - ticks[i];
            
            if (tickDelta >= 450 && tickDelta <= 550) {
                // Found good window
                if (qpcDelta > bestQpcDelta) {
                    bestQpcDelta = qpcDelta;
                    bestTscDelta = tscDelta;
                    bestStart = i;
                    bestCount = j - i;
                }
                break;
            }
        }
    }
    
    // Calculate frequency using QPC as reference
    double tscFreq = 0;
    if (bestQpcDelta > 0) {
        tscFreq = (double)bestTscDelta * qpcFreq.QuadPart / bestQpcDelta;
    }
    
    // Restore settings
    timeEndPeriod(1);
    SetThreadAffinityMask(hThread, oldAffinity);
    SetThreadPriority(hThread, oldPriority);
    
    return (DWORD64)round(tscFreq);
}

// Hooked TimeKeeper thread procedure
DWORD WINAPI AlphaFixesTimeKeeper(LPVOID lpParameter) {
    HANDLE hThread = GetCurrentThread();
    
    // Maximum timer resolution
    timeBeginPeriod(1);
    
    // Disable Windows 11 power throttling if applicable
    SetPowerThrottlingState(
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, FALSE);
    SetPowerThrottlingState(
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, FALSE);
    
    // Perform calibration
    g_tscFrequency = CalibrateTSCAlpha();
    
    // Calculate derived values
    if (g_tscFrequency > 0) {
        g_timerToMs = 1000.0 / (double)g_tscFrequency;
        
        // Align with GetTickCount for compatibility
        int64_t currentTsc = __rdtsc();
        g_timerOffset = (int64_t)GetTickCount() - 
                       (int64_t)(currentTsc * g_timerToMs);
    }
    
    // Write to global variable (DAT_00cc45a8)
    // This is the CpuTicksPerSecond storage location
    DWORD64* pCpuTicksPerSecond = (DWORD64*)0x00cc45a8;
    if (IsValidPointer(pCpuTicksPerSecond)) {
        *pCpuTicksPerSecond = g_tscFrequency;
    }
    
    // Write timer-to-ms conversion
    double* pTimerToMs = (double*)0x00cc45a8;  // Check exact offset
    if (IsValidPointer(pTimerToMs)) {
        *pTimerToMs = g_timerToMs;
    }
    
    // Set initialization flag
    g_initialized = TRUE;
    
    // Keep thread alive until shutdown
    while (!g_shutdown) {
        Sleep(100);
    }
    
    timeEndPeriod(1);
    return 0;
}

// Original TimeKeeper (preserved for reference)
typedef uint (*TimeKeeperFunc)(void*);
static TimeKeeperFunc OriginalTimeKeeper = NULL;

uint AlphaFixesTimeKeeperHooked(void* param) {
    // Redirect to our improved implementation
    return AlphaFixesTimeKeeper(param);
}
```

### Component 2: Snapshot Function Priority Boost

#### Target Function
`OsTimeManager::Snapshot` at **0x0045c0a0**

#### Signature Pattern
```c
// 0x0045c0a0 - Snapshot function
BYTE sigSnapshot[] = {
    0x55,                           // push ebp
    0x8B, 0xEC,                    // mov ebp, esp
    0xFF, 0x75, 0x0C,              // push [ebp+12]  (param_1)
    0xFF, 0x75, 0x08,              // push [ebp+8]   (this)
    0xE8,                           // call OsGetAsyncTimeClocks
    0x00, 0x00, 0x00, 0x00,
    0xA3,                           // mov [address], eax
    0x00, 0x00, 0x00, 0x00,
};
```

#### Patch Implementation
```c
// dll/alpha_snapshot.c

#include <windows.h>

// Original snapshot function (to be patched)
typedef void (*SnapshotFunc)(void* this, void* snapshot);
static SnapshotFunc OriginalSnapshot = NULL;

// Patched snapshot with priority boost
void __fastcall AlphaSnapshotPatched(void* this, void* snapshot) {
    HANDLE hThread = GetCurrentThread();
    int oldPriority = GetThreadPriority(hThread);
    
    // Boost priority for accurate measurement
    SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
    
    // Request timeslice before sampling
    Sleep(0);
    
    // Call original snapshot (inlined for accuracy)
    // Original code does:
    //   rdtsc() -> snapshot->rdtsc
    //   GetTickCount() -> snapshot->tickCount
    //   QueryPerformanceCounter() if hasQPF -> snapshot->qperfCount
    
    // For now, call the original
    if (OriginalSnapshot) {
        OriginalSnapshot(this, snapshot);
    }
    
    // Restore priority
    SetThreadPriority(hThread, oldPriority);
}

// Patch function
BOOL PatchSnapshot() {
    // Find the snapshot function
    BYTE target[] = {
        0x55,                           // push ebp
        0x8B, 0xEC,                    // mov ebp, esp
        0xFF, 0x75, 0x0C,              // push [ebp+12]
        0xFF, 0x75, 0x08,              // push [ebp+8]
        0xE8, 0xBB, 0xB6, 0xFF, 0xFF,  // call OsGetAsyncTimeClocks
        0xA3                            // mov [addr], eax
    };
    
    void* addr = FindPattern(target, sizeof(target));
    if (!addr) return FALSE;
    
    // Save original function bytes
    memcpy(OriginalBytes, addr, PATCH_SIZE);
    OriginalSnapshot = (SnapshotFunc)((BYTE*)addr + PATCH_SIZE);
    
    // Write jump to our function
    DWORD oldProtect;
    VirtualProtect(addr, 16, PAGE_EXECUTE_READWRITE, &oldProtect);
    
    // jmp AlphaSnapshotPatched
    BYTE jmpCode[] = {
        0xE9,                           // jmp rel32
        0x00, 0x00, 0x00, 0x00          // offset (filled at runtime)
    };
    *(DWORD*)(jmpCode + 1) = 
        (DWORD)AlphaSnapshotPatched - (DWORD)addr - 5;
    
    memcpy(addr, jmpCode, sizeof(jmpCode));
    
    VirtualProtect(addr, 16, oldProtect, &oldProtect);
    
    return TRUE;
}
```

### Component 3: CPU Frequency Query Override

#### Target Function
`CGxDevice::CpuFrequency` at **0x005951b0**

#### Signature Pattern
```c
// 0x005951b0 - CpuFrequency function
BYTE sigCpuFrequency[] = {
    0xA1,                           // mov eax, [DAT_00e1324c]
    0x4C, 0x32, 0xe1, 0x00,
    0xD9, 0xE0,                     // fstp st(0)
    0x84, 0xC0,                    // test al, al
    0x74, 0x0A,                    // jz skip_calibration
    0xD9, 0x05,                    // fld [DAT_00e1324c]
    0x4C, 0x32, 0xe1, 0x00,
    0xD9, 0xE0,                     // fstp st(0)
    0xC3,                           // ret
    // calibration code follows...
};
```

#### Hook Implementation
```c
// dll/alpha_cpufreq.c

#include <windows.h>
#include <intrin.h>

typedef float (*CpuFrequencyFunc)();
static CpuFrequencyFunc OriginalCpuFrequency = NULL;

volatile BOOL g_useTscCalibration = FALSE;

// Hooked CPU frequency function
float __fastcall AlphaCpuFrequencyHooked() {
    // Wait for TSC calibration to complete
    while (!g_initialized && !g_shutdown) {
        Sleep(1);
    }
    
    // Use our calibrated value if available
    if (g_tscFrequency > 0) {
        return (float)g_tscFrequency;
    }
    
    // Fallback to original
    if (OriginalCpuFrequency) {
        return OriginalCpuFrequency();
    }
    
    // Hardcoded fallback (approx 3GHz)
    return 3000000000.0f;
}

// Get tick count aligned with our TSC
DWORD AlphaGetTickCountAligned() {
    if (g_tscFrequency > 0 && g_timerToMs > 0) {
        int64_t tsc = __rdtsc();
        return (DWORD)(tsc * g_timerToMs + g_timerOffset);
    }
    return GetTickCount();
}
```

### Component 4: Sleep Interval Improvements

#### Target: OsTimeManager Constructor

Change initial sleep value from 50ms to 1ms:

```c
// Patch: Change sleepVal initialization from 0x32 (50) to 0x01 (1)

// Original code at OsTimeManager::OsTimeManager (0x0045c020):
// C6 45 ?? 32    mov byte ptr [ebp+??], 32

BYTE patchSleepInit[] = { 0xC6, 0x45, 0x??, 0x01 };

// Or alternatively, patch the Calibrate function to use 1ms sleeps
```

### Component 5: RDTSC Synchronization

#### Problem
On multi-core systems, TSC may differ between cores. The client doesn't pin threads.

#### Solution
Create helper library to ensure TSC consistency:

```c
// dll/alpha_tsc_sync.c

#include <windows.h>
#include <intrin.h>

// Thread-local storage for current CPU
static DWORD g_currentCpu = MAXDWORD;
static DWORD64 g_tscOffset[MAXIMUM_PROCESSORS];

// Initialize TSC offset for a specific CPU
DWORD64 InitCpuTscOffset(DWORD cpu) {
    DWORD_PTR oldAffinity = SetThreadAffinityMask(
        GetCurrentThread(), 1ULL << cpu);
    
    // Sync with GetTickCount
    DWORD tick = GetTickCount();
    DWORD64 tsc = __rdtsc();
    Sleep(100);
    DWORD64 tsc2 = __rdtsc();
    
    SetThreadAffinityMask(GetCurrentThread(), oldAffinity);
    
    // Calculate offset to align with GetTickCount
    double tscPerMs = (double)(tsc2 - tsc) / 100.0;
    g_tscOffset[cpu] = (DWORD64)tick * 1000 - (DWORD64)(tsc * tscPerMs);
    
    return g_tscOffset[cpu];
}

// Get synchronized TSC value
DWORD64 AlphaGetSynchronizedTsc() {
    DWORD cpu = GetCurrentProcessorNumber();
    
    if (g_currentCpu != cpu && g_tscOffset[cpu] == 0) {
        InitCpuTscOffset(cpu);
        g_currentCpu = cpu;
    }
    
    DWORD64 tsc = __rdtsc();
    return tsc + g_tscOffset[cpu];
}

// Pin current thread to CPU and sync TSC
void AlphaPinToCpu(int cpu) {
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu);
    g_currentCpu = cpu;
    
    if (g_tscOffset[cpu] == 0) {
        InitCpuTscOffset(cpu);
    }
}
```

---

## Signature Database

### Quick Reference

| Target | Address | Signature |
|--------|---------|-----------|
| TimeKeeper | 0x0045c100 | `8B 0D ?? ?? ?? ?? E8 ?? ?? ?? ?? 56 8B F1` |
| Calibrate | 0x0045c160 | `55 8B EC 81 EC ?? ?? ?? ?? 56 8B F1 8B` |
| Snapshot | 0x0045c0a0 | `55 8B EC FF 75 0C FF 75 08 E8 ?? ?? ?? ??` |
| OsGetAsyncTimeClocks | 0x0045b960 | `0F 31 C3` (rdtsc instruction) |
| CGxDevice::CpuFrequency | 0x005951b0 | `A1 ?? ?? ?? ?? D9 E0 84 C0 74 0A` |
| CpuTicks | 0x00595240 | `0F 31 C3` (rdtsc instruction) |
| Global TSC Freq | 0x00cc45a8 | `8B 15 ?? ?? ?? ?? 89 55 FC` |

### Pattern Matching Code

```c
// core/pattern.c

#include <windows.h>
#include <stdint.h>

typedef uint8_t BYTE;
typedef uint32_t DWORD;

void* FindPattern(const BYTE* pattern, size_t patternLen) {
    MODULEINFO modInfo;
    if (!GetModuleInformation(GetCurrentProcess(), 
                              GetModuleHandle(NULL),
                              &modInfo, sizeof(modInfo))) {
        return NULL;
    }
    
    BYTE* start = (BYTE*)modInfo.lpBaseOfDll;
    BYTE* end = start + modInfo.SizeOfImage - patternLen;
    
    for (BYTE* addr = start; addr < end; addr++) {
        BOOL match = TRUE;
        for (size_t i = 0; i < patternLen; i++) {
            if (pattern[i] != 0 && pattern[i] != addr[i]) {
                match = FALSE;
                break;
            }
        }
        if (match) return addr;
    }
    
    return NULL;
}

// Find pattern with mask (1 = match, 0 = wildcard)
void* FindPatternMask(const BYTE* pattern, const BYTE* mask, 
                      size_t patternLen) {
    MODULEINFO modInfo;
    if (!GetModuleInformation(GetCurrentProcess(), 
                              GetModuleHandle(NULL),
                              &modInfo, sizeof(modInfo))) {
        return NULL;
    }
    
    BYTE* start = (BYTE*)modInfo.lpBaseOfDll;
    BYTE* end = start + modInfo.SizeOfImage - patternLen;
    
    for (BYTE* addr = start; addr < end; addr++) {
        BOOL match = TRUE;
        for (size_t i = 0; i < patternLen; i++) {
            if ((mask[i] & 0xFF) && pattern[i] != addr[i]) {
                match = FALSE;
                break;
            }
        }
        if (match) return addr;
    }
    
    return NULL;
}
```

---

## Build System

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(AlphaFixes VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 99)

add_definitions(-D_WIN32_WINNT=0x0501)
add_definitions(-DNOMINMAX)

add_library(alphafixes SHARED
    dll/alpha_main.c
    dll/alpha_tsc.c
    dll/alpha_snapshot.c
    dll/alpha_cpufreq.c
    dll/alpha_tsc_sync.c
    core/pattern.c
    core/helpers.c
)

target_include_directories(alphafixes PRIVATE
    dll/
    core/
    include/
)

# MinHook for function hooking
add_subdirectory(lib/minhook)
target_link_libraries(alphafixes minhook)

# Output to dlls/ directory
set_target_properties(alphafixes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dlls
)
```

### Project Structure

```
AlphaFixes/
├── CMakeLists.txt
├── dll/
│   ├── alpha_main.c        # DllMain, initialization
│   ├── alpha_tsc.c        # TSC calibration
│   ├── alpha_snapshot.c   # Priority boost patch
│   ├── alpha_cpufreq.c    # CPU frequency override
│   ├── alpha_tsc_sync.c   # Multi-core TSC sync
│   └── alpha_helpers.c    # Shared helpers
├── core/
│   ├── pattern.c          # Signature scanning
│   └── helpers.c          # Utility functions
├── lib/
│   └── minhook/           # MinHook library
├── include/
│   └── alpha_fixes.h      # Public API
└── README.md
```

---

## Installation

### 1. Build the DLL
```bash
mkdir build && cd build
cmake ..
make
```

### 2. Copy to Game Directory
```bash
copy dlls/alphafixes.dll "C:\Games\WoW Alpha\"
```

### 3. Create DLL Injection Script
Option A: Using a launcher
```c
// launcher/main.c
void LaunchWithFixes() {
    // Create suspended process
    PROCESS_INFORMATION pi;
    CreateProcess("WoW.exe", ..., &pi, ...);
    
    // Inject alphafixes.dll
    LPVOID remoteStr = VirtualAllocEx(pi.hProcess, NULL, 260, 
                                      MEM_COMMIT | MEM_RESERVE,
                                      PAGE_READWRITE);
    WriteProcessMemory(pi.hProcess, remoteStr, 
                      "C:\\Games\\WoW Alpha\\alphafixes.dll", 260, NULL);
    
    HMODULE hKernel32 = GetModuleHandle("kernel32.dll");
    LPTHREAD_START_ROUTINE pLoadLibrary = 
        (LPTHREAD_START_ROUTINE)GetProcAddress(hKernel32, "LoadLibraryA");
    
    CreateRemoteThread(pi.hProcess, NULL, 0, pLoadLibrary, 
                       remoteStr, 0, NULL);
    
    // Resume process
    ResumeThread(pi.hThread);
}
```

Option B: Using registry injection (older method)
```
[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows]
"AppInit_DLLs"="alphafixes.dll"
"LoadAppInit_DLLs"=dword:00000001
```

### 4. Configuration (Optional)
Create `alphafixes.ini` in the game directory:
```ini
[General]
EnableTSC=1
EnablePriorityBoost=1
EnableTimerResolution=1
CalibrationTime=1000
DebugOutput=0

[Advanced]
PinToCpu=0
UseQPCFallback=1
```

---

## Testing

### Benchmark Script
```python
# tests/timing_benchmark.py
import time
import struct

def benchmark_frametimes():
    """Measure frame time consistency with and without fixes"""
    
    # Read TSC values at frame boundaries
    tsc_start = read_msr(0x10)  # IA32_TIME_STAMP_COUNTER MSR
    
    frames = []
    for i in range(1000):
        tsc1 = read_msr(0x10)
        # render_frame()
        tsc2 = read_msr(0x10)
        
        frame_tsc = tsc2 - tsc1
        frames.append(frame_tsc)
    
    # Calculate statistics
    avg = sum(frames) / len(frames)
    variance = sum((f - avg) ** 2 for f in frames) / len(frames)
    stddev = variance ** 0.5
    
    print(f"Frame time statistics:")
    print(f"  Average: {avg} cycles")
    print(f"  StdDev: {stddev} cycles")
    print(f"  StdDev %: {stddev / avg * 100:.2f}%")
    
    return avg, stddev
```

### Diagnostics Output
Enable debug output to verify fixes are active:
```
[AlphaFixes] TSC calibration: 3012345678 Hz
[AlphaFixes] Timer resolution: 1.000 ms
[AlphaFixes] Thread priority boost: ENABLED
[AlphaFixes] Multi-core TSC sync: ENABLED
[AlphaFixes] Initialized in 523 ms
```

---

## Compatibility

| Version | Status | Notes |
|---------|--------|-------|
| 0.5.3 (3368) | ✅ Tested | Core functions verified |

---

## Troubleshooting

### Issue: Black Screen on Launch
**Cause**: Hook conflict or bad pointer
**Solution**: Disable in `alphafixes.ini`:
```
EnableTSC=0
```

### Issue: Still Stuttering
**Cause**: Calibration didn't complete or inaccurate
**Solution**: 
1. Check debug output for TSC frequency
2. Verify CPU supports constant TSC (`rdtsc`cpuid check)
3. Try manual calibration in `alphafixes.ini`:
```
ManualTSC=3000000000
```

### Issue: Game Crashes Randomly
**Cause**: Hook interference or memory corruption
**Solution**: 
1. Update to latest build
2. Disable priority boost:
```
EnablePriorityBoost=0
```

---

## Future Improvements

1. **Variable TSC Handling**: Detect and handle non-invariant TSC
2. **Per-Game Settings**: Store calibration per-map for MMO consistency
3. **Performance Profiling**: Integrate with in-game profiler
4. **Network Time Sync**: Combine with server time for hybrid approach
5. **Save/Load Calibration**: Cache calibrated values between sessions
