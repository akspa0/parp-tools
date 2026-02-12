/**
 * AlphaFixes - TSC Calibration Module
 * 
 * Provides accurate TSC frequency calibration and timer fixes
 */

#include <windows.h>
#include <intrin.h>
#include <math.h>
#include <processthreadsapi.h>
#include <timeapi.h>

#include <MinHook.h>

#include "alpha_config.h"
#include "alpha_helpers.h"

#ifdef _DEBUG
#define DEBUG_OUTPUT(...) OutputDebugStringA(__VA_ARGS__)
#else
#define DEBUG_OUTPUT(...)
#endif

// Global state
static volatile BOOL g_tsc_initialized = FALSE;
static volatile DWORD64 g_tsc_frequency = 0;
static volatile double g_timer_to_ms = 0;
static volatile INT64 g_timer_offset = 0;

// Function pointers for original functions
typedef uint (*TimeKeeperOriginal)(void*);
static TimeKeeperOriginal OriginalTimeKeeper = NULL;

// Known addresses (from Ghidra analysis)
#define ADDR_TIMEEKEEPER     0x0045c100
#define ADDR_CPU_TICKS_SEC   0x00cc45a8

/**
 * Read RDTSC counter
 */
static inline DWORD64 ReadTSC() {
    DWORD64 tsc;
#if defined(_M_IX86)
    __asm {
        rdtsc
        mov DWORD PTR [tsc], eax
        mov DWORD PTR [tsc+4], edx
    }
#elif defined(_M_AMD64)
    tsc = __rdtsc();
#else
    // Fallback for other architectures
    LARGE_INTEGER perf;
    QueryPerformanceCounter(&perf);
    tsc = perf.QuadPart;
#endif
    return tsc;
}

/**
 * Perform accurate TSC calibration
 */
static DWORD64 CalibrateTSC() {
    HANDLE hThread = GetCurrentThread();
    DWORD_PTR oldAffinity;
    int oldPriority;
    
    // Get configuration
    ALPHA_CONFIG* config = AlphaFixes_GetConfig();
    
    // Save original thread settings
    oldAffinity = SetThreadAffinityMask(hThread, 1ULL << 0);
    oldPriority = GetThreadPriority(hThread);
    
    // Boost priority
    SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
    
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
        samples[i] = ReadTSC();
        ticks[i] = GetTickCount();
        
        Sleep(100);  // 100ms between samples
    }
    
    // Find most consistent window around calibration target
    int bestStart = 0;
    int bestCount = 0;
    DWORD64 bestQpcDelta = 0;
    DWORD64 bestTscDelta = 0;
    int calibrationTarget = config->calibration_time;  // Default: 1000ms
    
    for (int i = 0; i < 5; i++) {
        for (int j = i + 2; j <= 5; j++) {
            DWORD64 qpcDelta = qpcSamples[j-1] - qpcSamples[i];
            DWORD64 tscDelta;
            if (samples[j-1] >= samples[i]) {
                tscDelta = samples[j-1] - samples[i];
            } else {
                // Handle TSC wraparound
                tscDelta = samples[i] - samples[j-1];
            }
            DWORD tickDelta = ticks[j-1] - ticks[i];
            
            // Check if window is within acceptable range
            int windowSize = j - i;
            int expectedTicks = windowSize * 100;  // Each sleep is 100ms
            if (tickDelta >= expectedTicks - 50 && 
                tickDelta <= expectedTicks + 50) {
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
    if (bestQpcDelta > 0 && qpcFreq.QuadPart > 0) {
        tscFreq = (double)bestTscDelta * (double)qpcFreq.QuadPart / 
                  (double)bestQpcDelta;
    }
    
    // Restore settings
    timeEndPeriod(1);
    SetThreadAffinityMask(hThread, oldAffinity);
    SetThreadPriority(hThread, oldPriority);
    
    DEBUG_OUTPUT("[AlphaFixes-TSC] Calibration complete\n");
    
    return (DWORD64)round(tscFreq);
}

/**
 * Hooked TimeKeeper thread procedure
 */
static DWORD WINAPI HookedTimeKeeper(LPVOID lpParameter) {
    ALPHA_CONFIG* config = AlphaFixes_GetConfig();
    
    // Check for manual TSC override
    if (config->manual_tsc > 0) {
        g_tsc_frequency = config->manual_tsc;
        DEBUG_OUTPUT("[AlphaFixes-TSC] Using manual TSC frequency\n");
    } else {
        // Perform calibration
        g_tsc_frequency = CalibrateTSC();
        DEBUG_OUTPUT("[AlphaFixes-TSC] Calibrated TSC: %llu\n", g_tsc_frequency);
    }
    
    // Calculate derived values
    if (g_tsc_frequency > 0) {
        g_timer_to_ms = 1000.0 / (double)g_tsc_frequency;
        
        // Calculate offset to align with GetTickCount
        DWORD64 currentTsc = ReadTSC();
        DWORD currentTick = GetTickCount();
        g_timer_offset = (INT64)currentTick - 
                        (INT64)(currentTsc * g_timer_to_ms);
    }
    
    // Write to global variable (CpuTicksPerSecond)
    DWORD64* pCpuTicksPerSecond = (DWORD64*)ADDR_CPU_TICKS_SEC;
    if (IsValidPointer(pCpuTicksPerSecond)) {
        if (WriteMemory(pCpuTicksPerSecond, &g_tsc_frequency, 
                       sizeof(g_tsc_frequency))) {
            DEBUG_OUTPUT("[AlphaFixes-TSC] Wrote TSC frequency to memory\n");
        }
    }
    
    // Set initialization flag
    g_tsc_initialized = TRUE;
    
    // Wait for shutdown (or indefinitely if we want to keep the thread alive)
    while (!AlphaFixes_ShouldShutdown()) {
        Sleep(100);
    }
    
    return 0;
}

/**
 * Find TimeKeeper function address using pattern
 */
static void* FindTimeKeeperAddress() {
    // Try pattern matching first
    BYTE sigTimeKeeper[] = {
        0x8B, 0x0D,                   // mov ecx, [DAT_00cc4594]
        0x94, 0x45, 0x9C, 0x00,       //   (address varies)
        0xE8, 0x5B, 0xFF, 0xFF, 0xFF, // call Calibrate
        0x56,                           // push esi
        0x8B, 0xF1,                    // mov esi, ecx
    };
    
    BYTE mask[] = {
        1, 1, 0, 0, 0, 0,             // Match mov ecx, [??]
        1, 1, 1, 1, 1,                // Match call
        1, 1, 1, 1, 1                 // Match mov esi, ecx
    };
    
    void* addr = FindPatternMask(sigTimeKeeper, mask, sizeof(sigTimeKeeper));
    if (addr) {
        return addr;
    }
    
    // Fallback to known address
    return (void*)ADDR_TIMEEKEEPER;
}

/**
 * Initialize TSC calibration module
 */
BOOL AlphaTsc_Initialize() {
    ALPHA_CONFIG* config = AlphaFixes_GetConfig();
    
    if (!config->enable_tsc) {
        DEBUG_OUTPUT("[AlphaFixes-TSC] TSC fix disabled in config\n");
        return TRUE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-TSC] Initializing...\n");
    
    // Find TimeKeeper address
    void* timeKeeperAddr = FindTimeKeeperAddress();
    if (!timeKeeperAddr) {
        DEBUG_OUTPUT("[AlphaFixes-TSC] ERROR: Could not find TimeKeeper\n");
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-TSC] TimeKeeper found at %p\n", timeKeeperAddr);
    
    // Create hook
    MH_STATUS status = MH_CreateHook(
        timeKeeperAddr,
        &HookedTimeKeeper,
        (void**)&OriginalTimeKeeper
    );
    
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-TSC] ERROR: MH_CreateHook failed: %d\n", status);
        return FALSE;
    }
    
    // Enable hook
    status = MH_EnableHook(timeKeeperAddr);
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-TSC] ERROR: MH_EnableHook failed: %d\n", status);
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-TSC] Hook installed successfully\n");
    
    return TRUE;
}

/**
 * Shutdown TSC calibration module
 */
void AlphaTsc_Shutdown() {
    if (OriginalTimeKeeper) {
        MH_RemoveHook(OriginalTimeKeeper);
        OriginalTimeKeeper = NULL;
    }
    
    g_tsc_initialized = FALSE;
    DEBUG_OUTPUT("[AlphaFixes-TSC] Shutdown complete\n");
}

/**
 * Get calibrated TSC frequency
 */
DWORD64 AlphaTsc_GetFrequency() {
    return g_tsc_frequency;
}

/**
 * Get timer to milliseconds conversion
 */
double AlphaTsc_GetTimerToMs() {
    return g_timer_to_ms;
}

/**
 * Get timer offset
 */
INT64 AlphaTsc_GetTimerOffset() {
    return g_timer_offset;
}

/**
 * Check if TSC is initialized
 */
BOOL AlphaTsc_IsInitialized() {
    return g_tsc_initialized;
}

/**
 * Get tick count aligned with our TSC
 */
DWORD AlphaTsc_GetAlignedTickCount() {
    if (g_tsc_frequency > 0 && g_timer_to_ms > 0) {
        DWORD64 tsc = ReadTSC();
        return (DWORD)(tsc * g_timer_to_ms + g_timer_offset);
    }
    return GetTickCount();
}
