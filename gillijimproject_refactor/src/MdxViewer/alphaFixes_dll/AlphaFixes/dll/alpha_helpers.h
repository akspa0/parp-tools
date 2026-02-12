/**
 * AlphaFixes - Helper functions header
 */

#ifndef ALPHA_HELPERS_H
#define ALPHA_HELPERS_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Find a byte pattern in memory
 */
void* FindPattern(const BYTE* pattern, size_t patternLen);

/**
 * Find a byte pattern with wildcard mask
 */
void* FindPatternMask(const BYTE* pattern, const BYTE* mask, 
                     size_t patternLen);

/**
 * Read memory safely
 */
BOOL ReadMemory(void* addr, void* buffer, size_t size);

/**
 * Write memory with protection handling
 */
BOOL WriteMemory(void* addr, const void* data, size_t size);

/**
 * Get the base address of the main module
 */
void* GetModuleBase();

/**
 * Check if a pointer is valid
 */
BOOL IsValidPointer(void* ptr);

/**
 * Get the size of the main module
 */
SIZE_T GetModuleSize();

#ifdef __cplusplus
}
#endif

#endif // ALPHA_HELPERS_H
