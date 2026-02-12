/**
 * AlphaFixes - CPU Frequency Hook Implementation
 * 
 * Hooks CGxDevice::CpuFrequency to return calibrated TSC frequency
 */

#include <windows.h>
#include <stdio.h>
#include <intrin.h>

#include <MinHook.h>

#include "alpha_cpufreq.h"
#include "alpha_helpers.h"
#include "alpha_tsc.h"

#ifdef _DEBUG
#define DEBUG_OUTPUT(...) OutputDebugStringA(__VA_ARGS__)
#else
#define DEBUG_OUTPUT(...)
#endif

// Function pointer type
typedef float (*CpuFrequencyFunc)();
static CpuFrequencyFunc OriginalCpuFrequency = NULL;

// Known address
#define ADDR_CPUFREQUENCY  0x005951b0

/**
 * Find CpuFrequency function address
 */
static void* FindCpuFrequencyAddress() {
    // Pattern for CpuFrequency function
    BYTE sigCpuFrequency[] = {
        0xA1,                           // mov eax, [DAT_00e1324c]
        0x4C, 0x32, 0xe1, 0x00,       //   (address varies)
        0xD9, 0xE0,                     // fstp st(0)
        0x84, 0xC0,                    // test al, al
        0x74, 0x0A,                    // jz skip_calibration
    };
    
    BYTE mask[] = {
        1, 0, 0, 0, 0,                // Wildcard for mov eax, [??]
        1, 1, 1, 1, 1,                // Exact match for rest
        0                              // Wildcard for jz offset
    };
    
    void* addr = FindPatternMask(sigCpuFrequency, mask, sizeof(sigCpuFrequency));
    if (addr) {
        return addr;
    }
    
    return (void*)ADDR_CPUFREQUENCY;
}

/**
 * Hooked CpuFrequency function
 * Returns calibrated TSC frequency instead of performing runtime calibration
 */
static float WINAPI HookedCpuFrequency() {
    // Wait for TSC calibration to complete
    while (!AlphaTsc_IsInitialized() && !AlphaFixes_ShouldShutdown()) {
        Sleep(1);
    }
    
    // Use our calibrated value if available
    DWORD64 tscFreq = AlphaTsc_GetFrequency();
    if (tscFreq > 0) {
        return (float)tscFreq;
    }
    
    // Fallback to original function
    if (OriginalCpuFrequency) {
        return OriginalCpuFrequency();
    }
    
    // Hardcoded fallback (approx 3GHz)
    return 3000000000.0f;
}

/**
 * Initialize CPU frequency hook
 */
BOOL AlphaCpuFreq_Initialize() {
    DEBUG_OUTPUT("[AlphaFixes-CpuFreq] Initializing...\n");
    
    // Find CpuFrequency address
    void* cpufreqAddr = FindCpuFrequencyAddress();
    if (!cpufreqAddr) {
        DEBUG_OUTPUT("[AlphaFixes-CpuFreq] ERROR: Could not find CpuFrequency\n");
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-CpuFreq] CpuFrequency found at %p\n", cpufreqAddr);
    
    // Create hook
    MH_STATUS status = MH_CreateHook(
        cpufreqAddr,
        &HookedCpuFrequency,
        (void**)&OriginalCpuFrequency
    );
    
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-CpuFreq] ERROR: MH_CreateHook failed: %d\n", status);
        return FALSE;
    }
    
    // Enable hook
    status = MH_EnableHook(cpufreqAddr);
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-CpuFreq] ERROR: MH_EnableHook failed: %d\n", status);
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-CpuFreq] Hook installed successfully\n");
    
    return TRUE;
}

/**
 * Shutdown CPU frequency hook
 */
void AlphaCpuFreq_Shutdown() {
    if (OriginalCpuFrequency) {
        MH_RemoveHook(OriginalCpuFrequency);
        OriginalCpuFrequency = NULL;
    }
    DEBUG_OUTPUT("[AlphaFixes-CpuFreq] Shutdown complete\n");
}
