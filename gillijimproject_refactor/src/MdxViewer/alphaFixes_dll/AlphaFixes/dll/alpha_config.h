/**
 * AlphaFixes - Configuration header
 */

#ifndef ALPHA_CONFIG_H
#define ALPHA_CONFIG_H

typedef struct {
    int enable_tsc;           // Enable TSC calibration fix
    int enable_priority_boost; // Enable priority boost for timing
    int enable_timer_resolution; // Enable timer resolution increase
    int calibration_time;      // Calibration time in ms (default: 1000)
    int debug_output;          // Enable debug output
    
    // Advanced options
    DWORD64 manual_tsc;       // Manual TSC frequency override (0 = auto)
    int pin_to_cpu;           // Pin to specific CPU (-1 = auto, 0+ = CPU number)
    int use_qpc_fallback;     // Use QPC as fallback if TSC fails
} ALPHA_CONFIG;

#endif // ALPHA_CONFIG_H
