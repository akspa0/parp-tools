/**
 * AlphaFixes - CPU Frequency Hook Module
 * 
 * Hooks CGxDevice::CpuFrequency to return calibrated TSC frequency
 */

#ifndef ALPHA_CPUFREQ_H
#define ALPHA_CPUFREQ_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CPU frequency hook
 */
BOOL AlphaCpuFreq_Initialize();

/**
 * Shutdown CPU frequency hook
 */
void AlphaCpuFreq_Shutdown();

#ifdef __cplusplus
}
#endif

#endif // ALPHA_CPUFREQ_H
