/**
 * AlphaFixes - Snapshot Priority Boost Module
 * 
 * Patches OsTimeManager::Snapshot to boost thread priority during timing capture
 */

#ifndef ALPHA_SNAPSHOT_H
#define ALPHA_SNAPSHOT_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize snapshot patch
 */
BOOL AlphaSnapshot_Initialize();

/**
 * Shutdown snapshot patch
 */
void AlphaSnapshot_Shutdown();

#ifdef __cplusplus
}
#endif

#endif // ALPHA_SNAPSHOT_H
