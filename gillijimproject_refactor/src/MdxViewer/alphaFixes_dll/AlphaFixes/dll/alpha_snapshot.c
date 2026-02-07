/**
 * AlphaFixes - Snapshot Priority Boost Implementation
 * 
 * Patches OsTimeManager::Snapshot to boost thread priority during timing capture
 */

#include <windows.h>
#include <stdio.h>
#include <intrin.h>

#include <MinHook.h>

#include "alpha_snapshot.h"
#include "alpha_helpers.h"
#include "alpha_tsc.h"

#ifdef _DEBUG
#define DEBUG_OUTPUT(...) OutputDebugStringA(__VA_ARGS__)
#else
#define DEBUG_OUTPUT(...)
#endif

// Function pointer types
typedef void (__thiscall *SnapshotFunc)(void* thisPtr, void* snapshot);
static SnapshotFunc OriginalSnapshot = NULL;

// Known addresses
#define ADDR_SNAPSHOT  0x0045c0a0
#define ADDR_RDTSC     0x0045b960

/**
 * Find snapshot function address
 */
static void* FindSnapshotAddress() {
    // Pattern for Snapshot function
    BYTE sigSnapshot[] = {
        0x55,                           // push ebp
        0x8B, 0xEC,                    // mov ebp, esp
        0xFF, 0x75, 0x0C,              // push [ebp+12]
        0xFF, 0x75, 0x08,              // push [ebp+8]
        0xE8,                           // call OsGetAsyncTimeClocks
        0x00, 0x00, 0x00, 0x00,
    };
    
    BYTE mask[] = {
        1, 1, 1, 1,                   // Exact match for prologue
        1, 1, 1,                       // push [ebp+12]
        1, 1, 1,                       // push [ebp+8]
        0, 0, 0, 0                     // Wildcard for call
    };
    
    void* addr = FindPatternMask(sigSnapshot, mask, sizeof(sigSnapshot));
    if (addr) {
        return addr;
    }
    
    return (void*)ADDR_SNAPSHOT;
}

/**
 * Patched snapshot function with priority boost
 */
static void __fastcall PatchedSnapshot(void* thisPtr, void* snapshot) {
    HANDLE hThread = GetCurrentThread();
    int oldPriority = GetThreadPriority(hThread);
    
    // Boost priority for accurate measurement
    SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
    
    // Request timeslice before sampling
    Sleep(0);
    
    // Get current TSC (inlined for accuracy)
    DWORD64 tsc = ReadTSCWrapper();
    
    // Get tick count
    DWORD tick = GetTickCount();
    
    // Write to snapshot structure (following game's structure)
    // Structure layout (from Ghidra analysis):
    // Offset 0: rdtsc (QWORD)
    // Offset 8: tickCount (DWORD)
    // Offset 12: qperfCount (QWORD, if hasQPF)
    
    DWORD* snapshotPtr = (DWORD*)snapshot;
    
    // Write TSC (little-endian: low DWORD first, then high DWORD)
    // Note: Assuming standard x86 little-endian, so we can write as DWORD
    // The actual implementation may vary based on the structure
    
    // Call original function if available
    if (OriginalSnapshot) {
        OriginalSnapshot(thisPtr, snapshot);
    }
    
    // Restore priority
    SetThreadPriority(hThread, oldPriority);
}

/**
 * Read RDTSC wrapper (to be called from patches)
 */
DWORD64 ReadTSCWrapper() {
#if defined(_M_IX86)
    DWORD low, high;
    __asm {
        rdtsc
        mov low, eax
        mov high, edx
    }
    return ((DWORD64)high << 32) | low;
#elif defined(_M_AMD64)
    return __rdtsc();
#else
    LARGE_INTEGER perf;
    QueryPerformanceCounter(&perf);
    return perf.QuadPart;
#endif
}

/**
 * Initialize snapshot patch
 */
BOOL AlphaSnapshot_Initialize() {
    DEBUG_OUTPUT("[AlphaFixes-Snapshot] Initializing...\n");
    
    // Find Snapshot address
    void* snapshotAddr = FindSnapshotAddress();
    if (!snapshotAddr) {
        DEBUG_OUTPUT("[AlphaFixes-Snapshot] ERROR: Could not find Snapshot\n");
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-Snapshot] Snapshot found at %p\n", snapshotAddr);
    
    // Create hook
    MH_STATUS status = MH_CreateHook(
        snapshotAddr,
        &PatchedSnapshot,
        (void**)&OriginalSnapshot
    );
    
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-Snapshot] ERROR: MH_CreateHook failed: %d\n", status);
        return FALSE;
    }
    
    // Enable hook
    status = MH_EnableHook(snapshotAddr);
    if (status != MH_OK) {
        DEBUG_OUTPUT("[AlphaFixes-Snapshot] ERROR: MH_EnableHook failed: %d\n", status);
        return FALSE;
    }
    
    DEBUG_OUTPUT("[AlphaFixes-Snapshot] Hook installed successfully\n");
    
    return TRUE;
}

/**
 * Shutdown snapshot patch
 */
void AlphaSnapshot_Shutdown() {
    if (OriginalSnapshot) {
        MH_RemoveHook(OriginalSnapshot);
        OriginalSnapshot = NULL;
    }
    DEBUG_OUTPUT("[AlphaFixes-Snapshot] Shutdown complete\n");
}
