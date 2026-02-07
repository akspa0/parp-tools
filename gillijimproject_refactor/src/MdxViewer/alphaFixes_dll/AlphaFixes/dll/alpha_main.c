/**
 * AlphaFixes - Performance patches for WoW Alpha 0.5.3
 * 
 * Main DLL entry point and initialization
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <MinHook.h>

#include "alpha_config.h"
#include "alpha_helpers.h"
#include "alpha_tsc.h"
#include "alpha_snapshot.h"
#include "alpha_cpufreq.h"

// Global state
static HMODULE g_hSelf = NULL;
static volatile BOOL g_shutdown = FALSE;
static BOOL g_initialized = FALSE;

// Configuration
static ALPHA_CONFIG g_config;

#ifdef _DEBUG
#define DEBUG_OUTPUT(...) do { \
    char buf[512]; \
    snprintf(buf, sizeof(buf), __VA_ARGS__); \
    OutputDebugStringA(buf); \
} while(0)
#else
#define DEBUG_OUTPUT(...)
#endif

// Forward declarations
BOOL InitializeHooks();
void ShutdownHooks();
BOOL LoadConfig();

/**
 * DLL main entry point
 */
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    g_hSelf = hinstDLL;

    switch (fdwReason) {
        case DLL_PROCESS_ATTACH: {
            DEBUG_OUTPUT("[AlphaFixes] DLL_PROCESS_ATTACH\n");
            
            // Load configuration
            if (!LoadConfig()) {
                DEBUG_OUTPUT("[AlphaFixes] Warning: Could not load config\n");
            }
            
            // Initialize MinHook
            if (MH_Initialize() != MH_OK) {
                DEBUG_OUTPUT("[AlphaFixes] ERROR: MinHook initialization failed\n");
                return FALSE;
            }
            
            // Install hooks
            if (!InitializeHooks()) {
                DEBUG_OUTPUT("[AlphaFixes] ERROR: Hook installation failed\n");
                ShutdownHooks();
                return FALSE;
            }
            
            g_initialized = TRUE;
            DEBUG_OUTPUT("[AlphaFixes] Initialized successfully\n");
            break;
        }
        
        case DLL_PROCESS_DETACH: {
            DEBUG_OUTPUT("[AlphaFixes] DLL_PROCESS_DETACH\n");
            g_shutdown = TRUE;
            ShutdownHooks();
            break;
        }
        
        case DLL_THREAD_ATTACH: {
            // Thread initialization if needed
            break;
        }
        
        case DLL_THREAD_DETACH: {
            // Thread cleanup if needed
            break;
        }
    }

    return TRUE;
}

/**
 * Load configuration from INI file
 */
BOOL LoadConfig() {
    char path[MAX_PATH];
    GetModuleFileNameA(g_hSelf, path, MAX_PATH);
    PathRemoveFileSpecA(path);
    strcat(path, "\\alphafixes.ini");
    
    g_config.enable_tsc = GetPrivateProfileIntA("General", "EnableTSC", 1, path);
    g_config.enable_priority_boost = GetPrivateProfileIntA("General", "EnablePriorityBoost", 1, path);
    g_config.enable_timer_resolution = GetPrivateProfileIntA("General", "EnableTimerResolution", 1, path);
    g_config.calibration_time = GetPrivateProfileIntA("General", "CalibrationTime", 1000, path);
    g_config.debug_output = GetPrivateProfileIntA("General", "DebugOutput", 0, path);
    
    // Get manual TSC override if set
    g_config.manual_tsc = GetPrivateProfileIntA("Advanced", "ManualTSC", 0, path);
    g_config.pin_to_cpu = GetPrivateProfileIntA("Advanced", "PinToCpu", -1, path);
    g_config.use_qpc_fallback = GetPrivateProfileIntA("Advanced", "UseQPCFallback", 1, path);
    
    DEBUG_OUTPUT("[AlphaFixes] Config: TSC=%d, PriorityBoost=%d, TimerRes=%d\n",
                 g_config.enable_tsc, g_config.enable_priority_boost, 
                 g_config.enable_timer_resolution);
    
    return TRUE;
}

/**
 * Initialize all hooks
 */
BOOL InitializeHooks() {
    BOOL success = TRUE;
    
    // Hook TSC calibration functions
    if (g_config.enable_tsc) {
        if (!AlphaTsc_Initialize()) {
            DEBUG_OUTPUT("[AlphaFixes] WARNING: TSC hook failed\n");
            // Continue without TSC fix - game should still work
        }
    }
    
    // Hook snapshot for priority boost
    if (g_config.enable_priority_boost) {
        if (!AlphaSnapshot_Initialize()) {
            DEBUG_OUTPUT("[AlphaFixes] WARNING: Snapshot hook failed\n");
        }
    }
    
    // Hook CPU frequency query
    if (!AlphaCpuFreq_Initialize()) {
        DEBUG_OUTPUT("[AlphaFixes] WARNING: CPU frequency hook failed\n");
    }
    
    return success;
}

/**
 * Shutdown and remove all hooks
 */
void ShutdownHooks() {
    AlphaCpuFreq_Shutdown();
    AlphaSnapshot_Shutdown();
    AlphaTsc_Shutdown();
    
    MH_Uninitialize();
}

/**
 * Get configuration
 */
ALPHA_CONFIG* AlphaFixes_GetConfig() {
    return &g_config;
}

/**
 * Check if AlphaFixes is initialized
 */
BOOL AlphaFixes_IsInitialized() {
    return g_initialized;
}

/**
 * Get shutdown flag
 */
BOOL AlphaFixes_ShouldShutdown() {
    return g_shutdown;
}
