/**
 * MinHook - Minimal x86/x64 API Hooking Library
 * Header file
 */

#ifndef MINHOOK_H
#define MINHOOK_H

#include <windows.h>

#ifdef _MSC_VER
#pragma once
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Disable structure packing warning for MSVC
#if defined(_MSC_VER) && _MSC_VER >= 1200
#pragma warning(push)
#pragma warning(disable: 4324)
#endif

/**
 * MH_STATUS - Return codes for MinHook functions
 */
typedef enum MH_STATUS {
    MH_UNKNOWN = -1,
    MH_OK = 0,
    MH_ERROR_ALREADY_INITIALIZED,
    MH_ERROR_NOT_INITIALIZED,
    MH_ERROR_ALREADY_CREATED,
    MH_ERROR_NOT_CREATED,
    MH_ERROR_ENABLED,
    MH_ERROR_DISABLED,
    MH_ERROR_NOT_EXECUTABLE,
    MH_ERROR_UNSUPPORTED_FUNCTION,
    MH_ERROR_MEMORY_ALLOC,
    MH_ERROR_MEMORY_PROTECT,
    MH_ERROR_NO_MORE_HOOKS,
} MH_STATUS;

/**
 * Initialize MinHook
 */
MH_STATUS WINAPI MH_Initialize();

/**
 * Uninitialize MinHook
 */
MH_STATUS WINAPI MH_Uninitialize();

/**
 * Create a hook for a function
 * 
 * @param pTarget      Address of the target function
 * @param pDetour      Address of the detour function
 * @param ppOriginal   Address to store the original function pointer
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_CreateHook(
    LPVOID pTarget,
    LPVOID pDetour,
    LPVOID *ppOriginal
);

/**
 * Create a hook for a function (ex version with UTF-8 support)
 */
MH_STATUS WINAPI MH_CreateHookEx(
    LPCSTR pszTarget,
    LPCSTR pszDetour,
    LPVOID *ppOriginal
);

/**
 * Enable a hook
 * 
 * @param pTarget  Address of the target function (or NULL for all)
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_EnableHook(LPVOID pTarget);

/**
 * Disable a hook
 * 
 * @param pTarget  Address of the target function (or NULL for all)
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_DisableHook(LPVOID pTarget);

/**
 * Remove a hook
 * 
 * @param pTarget  Address of the target function
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_RemoveHook(LPVOID pTarget);

/**
 * Queue a hook to be enabled
 * 
 * @param pTarget  Address of the target function
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_QueueEnableHook(LPVOID pTarget);

/**
 * Queue a hook to be disabled
 * 
 * @param pTarget  Address of the target function
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_QueueDisableHook(LPVOID pTarget);

/**
 * Apply queued hooks
 * 
 * @return MH_STATUS
 */
MH_STATUS WINAPI MH_ApplyQueued();

/**
 * Get the library version
 * 
 * @return Version string
 */
LPCSTR WINAPI MH_GetVersion();

/**
 * Special hook enabling/disabling macros
 */
#define MH_ALL_HOOKS      NULL
#define MH_UNHOOK_CALL    0
#define MH_UNHOOK_JMP     1

#ifdef __cplusplus
}
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1200
#pragma warning(pop)
#endif

#endif // MINHOOK_H
