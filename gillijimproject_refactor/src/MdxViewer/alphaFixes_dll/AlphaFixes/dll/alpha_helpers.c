/**
 * AlphaFixes - Helper functions implementation
 */

#include <windows.h>
#include <psapi.h>
#include <stdio.h>
#include <string.h>

#include "alpha_helpers.h"

/**
 * Find a byte pattern in memory
 */
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
            if (pattern[i] != addr[i]) {
                match = FALSE;
                break;
            }
        }
        if (match) return addr;
    }
    
    return NULL;
}

/**
 * Find a byte pattern with wildcard mask (1 = match, 0 = wildcard)
 */
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

/**
 * Read memory safely
 */
BOOL ReadMemory(void* addr, void* buffer, size_t size) {
    if (!addr || !buffer || !IsValidPointer(addr)) {
        return FALSE;
    }
    
    DWORD oldProtect;
    if (!VirtualProtect(addr, size, PAGE_READONLY, &oldProtect)) {
        return FALSE;
    }
    
    memcpy(buffer, addr, size);
    
    VirtualProtect(addr, size, oldProtect, &oldProtect);
    return TRUE;
}

/**
 * Write memory with protection handling
 */
BOOL WriteMemory(void* addr, const void* data, size_t size) {
    if (!addr || !data || !IsValidPointer(addr)) {
        return FALSE;
    }
    
    DWORD oldProtect;
    if (!VirtualProtect(addr, size, PAGE_EXECUTE_READWRITE, &oldProtect)) {
        return FALSE;
    }
    
    memcpy(addr, data, size);
    
    VirtualProtect(addr, size, oldProtect, &oldProtect);
    return TRUE;
}

/**
 * Get the base address of the main module
 */
void* GetModuleBase() {
    static void* base = NULL;
    if (!base) {
        base = GetModuleHandle(NULL);
    }
    return base;
}

/**
 * Check if a pointer is valid
 */
BOOL IsValidPointer(void* ptr) {
    if (!ptr) return FALSE;
    
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(ptr, &mbi, sizeof(mbi)) == 0) {
        return FALSE;
    }
    
    // Check if memory is committed and readable
    if (mbi.State != MEM_COMMIT) {
        return FALSE;
    }
    
    if (mbi.Protect == PAGE_NOACCESS || 
        mbi.Protect == PAGE_EXECUTE) {
        return FALSE;
    }
    
    return TRUE;
}

/**
 * Get the size of the main module
 */
SIZE_T GetModuleSize() {
    MODULEINFO modInfo;
    if (GetModuleInformation(GetCurrentProcess(), 
                           GetModuleHandle(NULL),
                           &modInfo, sizeof(modInfo))) {
        return modInfo.SizeOfImage;
    }
    return 0;
}
