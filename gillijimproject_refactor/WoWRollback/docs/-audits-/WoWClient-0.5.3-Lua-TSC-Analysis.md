# WoWClient.exe (0.5.3 Alpha) Lua Scripting & Timer System Analysis

## Overview

This document details the discoveries from Ghidra analysis of the WoWClient.exe binary (version 0.5.3 Alpha), focusing on:

1. The FrameScript Lua implementation
2. Timer/TSC (Time Stamp Counter) system
3. Performance-related issues and potential patches

---

## Lua Scripting System (FrameScript)

### Lua Version
The binary uses **Lua 5.0**, as evidenced by strings in the binary:
```
"Lua 5.0" @ 0x008a6a64
"Lua: Lua 5.0 Copyright (C) 1994-2003 Tecgraf, PUC-Rio"
```

### Core Lua Functions Found
The binary contains a complete implementation of Lua 5.0 with all standard functions:

| Function | Address | Purpose |
|----------|---------|---------|
| `lua_open` | 0x006db070 | Creates new Lua state |
| `lua_close` | 0x006db290 | Closes Lua state |
| `lua_pcall` | 0x006d8ea0 | Protected call |
| `lua_pushnumber` | 0x006d89c0 | Push number to stack |
| `lua_pushstring` | 0x006d8a40 | Push string to stack |
| `lua_pushcclosure` | 0x006d8ad0 | Push C function |
| `lua_settable` | 0x006d8d20 | Set table value |
| `lua_rawgeti` | 0x006d8c10 | Get table by integer index |
| `luaL_loadbuffer` | 0x006da130 | Load script from buffer |
| `luaL_ref` | 0x006d9e00 | Create reference |
| `luaL_unref` | 0x006d9ec0 | Release reference |

### FrameScript Architecture

#### FrameScript_Initialize (0x006e61d0)
Initializes the Lua state and registers built-in functions:

```c
int FrameScript_Initialize() {
    // Open Lua state with GC disabled
    DAT_010b3b68 = lua_open();
    lua_disablegc(DAT_010b3b68);
    
    // Register __framescript_meta table with __index
    lua_pushstring("__framescript_meta");
    lua_newtable();
    lua_pushstring("__index");
    lua_pushcclosure(FrameScript_Object::LookupScriptMethod, 0);
    lua_settable(-3);
    lua_settable(-10000);  // REGISTRY index
    
    // Register built-in functions
    lua_pushstring("getglobal");  // Custom getglobal
    lua_pushcclosure(getglobal_handler, 0);
    lua_settable(-10000);
    
    lua_pushstring("next");  // Iterator
    lua_pushcclosure(next_handler, 0);
    lua_settable(-10000);
    
    lua_pushstring("debuginfo");  // Debug info
    lua_pushcclosure(debug_handler, 0);
    lua_settable(-10000);
    
    // Open standard libraries
    luaopen_string();
    luaopen_table();
    luaopen_math();
    
    // Load compat.lua from Interface\FrameXML\compat.lua
    SFile::LoadFile("Interface_FrameXML_compat_lua", &local_8, &local_c, ...);
    luaL_loadbuffer(compat_lua_data, compat_lua_size, "compat.lua");
    lua_pcall(0, 0, 0);
    
    // Store _ERRORMESSAGE reference
    lua_pushstring("_ERRORMESSAGE");
    lua_gettable(-10000);
    DAT_010b3b6c = luaL_ref(-10000);
    
    return 1;
}
```

#### FrameScript_Execute (0x006e7360)
Core function execution with format string support:

```c
void FrameScript_ExecuteV(int scriptId, FrameScript_Object* obj, 
                          char* format, char* args) {
    lua_State* L = FrameScript_GetContext();
    
    // Parse format string and push arguments
    // Supported format specifiers:
    // %d - integer (lua_pushnumber)
    // %f - float (lua_pushnumber)
    // %s - string (lua_pushstring)
    // %u - unsigned integer (lua_pushnumber)
    
    if (obj != NULL) {
        FUN_006e7270(obj);  // Register object
    }
    
    // Get error function
    GetErrorFunction(L);
    
    // Get script by ID from registry
    lua_rawgeti(L, -10000, scriptId);
    
    // Execute script
    int result = lua_pcall(L, 0, 0, -2);
    
    if (result != 0) {
        // Error handling
        lua_settop(L, -2);
    }
    
    if (obj != NULL) {
        FUN_006e72f0();  // Cleanup
    }
}
```

#### FrameScript_RegisterFunction (0x006e6d00)
Registers a C function with Lua:

```c
void FrameScript_RegisterFunction(const char* name, 
                                   int (*func)(lua_State*)) {
    lua_State* L = FrameScript_GetContext();
    lua_pushcclosure(func, 0);
    lua_pushstring(name);
    lua_insert(-2);
    lua_settable(-10000);  // Store in registry
}
```

#### FrameScript_CompileFunction (0x006e70f0)
Compiles Lua code without executing:

```c
int FrameScript_CompileFunction(const char* code, const char* name) {
    lua_State* L = FrameScript_GetContext();
    size_t len = SStrLen(code);
    
    int result = luaL_loadbuffer(L, code, len, name);
    if (result == 0) {
        result = luaL_ref(L, -10000);  // Store in registry
        return result;
    }
    return -1;  // Compilation failed
}
```

### FrameScript_Object Class
Manages Lua object bindings:

```c
class FrameScript_Object {
    int lua_registered;      // Reference count
    int lua_objectRef;       // Lua reference (-2 = not registered)
    int m_onEvent;           // Event handlers
    void* vtable;            // Virtual function table
};
```

#### FrameScript_SignalEvent (0x006e6b30)
Dispatches events to registered handlers:

```c
void FrameScript_SignalEvent(uint eventId) {
    FrameScript_EventObject* event = 
        TSBaseArray<EventObject>::Get(eventId);
    
    // Iterate through listeners
    for (EVENTLISTENERNODE* node = event->listenerList.Head; 
         node != NULL; 
         node = node->Next) {
        
        FrameScript_Object* obj = node->callbackObject;
        char* eventName = event->name;
        
        obj->OnScriptEvent(eventName);
    }
}
```

### Lua Script Files Referenced
- `Interface\FrameXML\GlobalStrings.lua` @ 0x00845aa8
- `Interface\GlueXML\GlueStrings.lua` @ 0x0083bdcc
- `Interface\FrameXML\compat.lua` @ 0x008a778c

---

## Timer/TSC System Analysis

### OsTimeManager Class
The core time management system, similar to what VanillaFixes patches in later clients.

#### OsTimeManager Constructor (0x0045c020)
```c
OsTimeManager::OsTimeManager() {
    // Initialize synchronization objects
    SSyncObject::SSyncObject(&this->timeMgrThread);
    SEvent::SEvent(&this->shutdownEvt, 1, 0);
    
    // Store singleton reference
    DAT_00cc4594 = this;
    
    // Initial sleep value: 50ms
    this->sleepVal = 0x32;
    
    // Create TimeKeeper thread
    SThread::Create(TimeKeeper, NULL, &this->timeMgrThread, "OsTime");
}
```

#### OsTimeManager::TimeKeeper (0x0045c100)
Background thread that calibrates timer:

```c
uint OsTimeManager::TimeKeeper(void* param) {
    // Run calibration
    Calibrate(this);
    
    // Wait for shutdown, then cleanup
    do {
        this = InterlockedExchange(&DAT_00cc4594, 0);
    } while (this == NULL);
    
    ~OsTimeManager(this);
    MemFree(this);
    
    // Save calibrated value to registry
    SRegSaveData("Internal", "CpuTicksPerSecond", 
                 (HKEY)0x0, &DAT_00cc45a8, 8);
    
    return 0;
}
```

#### OsTimeManager::Calibrate (0x0045c160)
**Critical function** - Calibrates CPU frequency using QPF and RDTSC:

```c
void OsTimeManager::Calibrate(OsTimeManager* this) {
    // Check if QueryPerformanceCounter is available
    BOOL hasQPF = QueryPerformanceFrequency(&qPerfFreq);
    if (hasQPF && qPerfFreq != 0) {
        this->hasQPF = 1;
    }
    
    // Initial baseline snapshot
    TimeSnapshot baseTime;
    Snapshot(this, &baseTime);
    
    // Wait for shutdown event or timeout
    WaitMultiplePtr(1, &this->shutdownEvt, 1, this->sleepVal);
    
    while (true) {
        if (shutdown_signaled) return;
        
        // Get interval snapshot
        TimeSnapshot interval;
        Snapshot(this, &interval);
        
        // Calculate elapsed milliseconds
        uint elapsed = interval.tickCount - baseTime.tickCount;
        
        // Exit after ~30 seconds of calibration
        if (elapsed > 29999) break;
        
        // Calculate RDTSC delta (handling overflow)
        bool overflow = interval.rdtsc < baseTime.rdtsc;
        uint rdtscDelta = interval.rdtsc - baseTime.rdtsc;
        
        // If QPF available, use it for calibration
        if (this->hasQPF) {
            double qpfDelta = interval.qperfCount - baseTime.qperfCount;
            int64 freq = ftol();  // QPF frequency
            this->cpuTicksPerSecond_qp = freq;
        }
        
        // Calculate CPU ticks per millisecond
        if (elapsed != 0) {
            // TSC-based frequency calculation
            int64 result = __allmul(rdtscDelta, overflow ? -1 : 1, 1000, 0);
            result = __alldiv(result.high, result.low, elapsed, 0);
            this->cpuTicksPerSecond_ti = result;
        }
        
        // Store calibrated value globally
        InterlockedExchange64(&DAT_00cc45a8, 
                             &this->cpuTicksPerSecond_qp);
        
        // Increase sleep interval (up to 1000ms)
        this->sleepVal = this->sleepVal * 3 / 2;
        if (this->sleepVal > 1000) this->sleepVal = 1000;
        
        WaitMultiplePtr(1, &this->shutdownEvt, 1, this->sleepVal);
    }
}
```

#### OsTimeManager::Snapshot (0x0045c0a0)
Captures current time values:

```c
void OsTimeManager::Snapshot(TimeSnapshot* out) {
    // Preserve thread priority
    HANDLE thread = GetCurrentThread();
    int oldPriority = GetThreadPriority(thread);
    
    if (oldPriority != THREAD_PRIORITY_ERROR_RETURN) {
        SetThreadPriority(thread, oldPriority);
    }
    
    // Capture RDTSC
    out->rdtsc = OsGetAsyncTimeClocks();
    
    // Capture GetTickCount
    out->tickCount = GetTickCount();
    
    // Capture QPC if available
    if (this->hasQPF) {
        QueryPerformanceCounter(&out->qperfCount);
    }
}
```

#### OsGetAsyncTimeClocks (0x0045b960)
Wrapper for RDTSC instruction:

```c
__int64 OsGetAsyncTimeClocks() {
    return rdtsc();  // Returns Time Stamp Counter
}
```

### CGxDevice::CpuFrequency (0x005951b0)
**This is the GetCPUFrequency equivalent** - Returns calibrated CPU frequency:

```c
float CGxDevice::CpuFrequency() {
    // Check if already calibrated
    if (DAT_00e1324c != 0.0) {
        return DAT_00e1324c;
    }
    
    // Get current tick
    DWORD startTick = GetTickCount();
    int64 startTSC = CpuTicks();
    DWORD currentTick = startTick;
    
    // Wait for tick to change
    while (startTick == currentTick) {
        startTSC = CpuTicks();
        currentTick = GetTickCount();
    }
    
    // Wait ~250ms
    Sleep(250);
    
    int64 endTSC = CpuTicks();
    
    // Calculate TSC ticks per second
    int64 elapsed = endTSC - startTSC;
    elapsed = __allmul(elapsed, 4, 0);  // * 1000 / 250ms
    DAT_00e1324c = (float)elapsed;
    
    return DAT_00e1324c;
}
```

### CGxDevice::CpuTicks (0x00595240)
Direct RDTSC access:

```c
__int64 CGxDevice::CpuTicks() {
    return rdtsc();  // x86 RDTSC instruction
}
```

---

## Performance Issues & Patch Recommendations

### Issue 1: RDTSC Frequency Calibration

**Problem**: The calibration in `OsTimeManager::Calibrate` is designed to run for up to 30 seconds during startup. This calibration may be inaccurate on modern systems, especially:
- CPUs with variable clock speeds (Turbo Boost)
- CPUs with power-saving states
- Multi-core systems with desynchronized TSC

**VanillaFixes Approach** (1.12.1):
- Replaces `OsTimeManager::TimeKeeper` with custom thread
- Uses `Sleep(500)` for more accurate calibration
- Stores result in global variables for immediate use

**Recommended 0.5.3 Patches**:

1. **Hook TimeKeeper (0x0045c100)**
   ```c
   // Override TimeKeeper to use better calibration
   DWORD WINAPI VfTimeKeeperThread(LPVOID) {
       // Request high timer resolution (0.5ms)
       IncreaseTimerResolution(5000);
       
       // Disable Windows 11 power throttling
       SetPowerThrottlingState(
           PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, FALSE);
       SetPowerThrottlingState(
           PROCESS_POWER_THROTTLING_EXECUTION_SPEED, FALSE);
       
       // Better calibration with QPF
       LARGE_INTEGER qpf;
       QueryPerformanceFrequency(&qpf);
       
       int64 startTSC = CpuTicks();
       LARGE_INTEGER startQPC;
       QueryPerformanceCounter(&startQPC);
       
       Sleep(500);  // 500ms for accuracy
       
       int64 endTSC = CpuTicks();
       LARGE_INTEGER endQPC;
       QueryPerformanceCounter(&endQPC);
       
       // Calculate TSC frequency
       int64 elapsedQPC = endQPC.QuadPart - startQPC.QuadPart;
       int64 elapsedTSC = endTSC - startTSC;
       
       double tscFreq = (double)elapsedTSC * qpf.QuadPart / elapsedQPC;
       
       // Store globally
       *DAT_00cc45a8 = (int64)tscFreq;
       
       while (!shutdown) Sleep(1);
       return 0;
   }
   ```

2. **Signature for TimeKeeper (0x0045c100)**
   ```c
   // Pattern to find TimeKeeper function
   BYTE sigTimeKeeper[] = {
       0x8B, 0x0D, 0x00, 0x00, 0x00, 0x00,  // mov ecx, [global_ptr]
       0xE8, 0x00, 0x00, 0x00, 0x00,        // call Calibrate
       0x56,                                 // push esi
       0x8D, 0x64, 0x24                      // lea esp, [esp]
   };
   ```

### Issue 2: Timer Resolution

**Problem**: Default sleep intervals start at 50ms and grow to 1000ms, which is too coarse for smooth gameplay.

**Recommended Patch**:
```c
// In OsTimeManager::Calibrate, replace:
this->sleepVal = 0x32;  // 50ms

// With:
this->sleepVal = 1;     // 1ms (higher resolution)
```

### Issue 3: Thread Priority During Calibration

**Problem**: `OsTimeManager::Snapshot` doesn't boost thread priority, leading to inconsistent readings.

**Recommended Patch**:
```c
void OsTimeManager::Snapshot(TimeSnapshot* out) {
    HANDLE thread = GetCurrentThread();
    int oldPriority = GetThreadPriority(thread);
    
    // Boost to highest for accurate measurement
    SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL);
    
    out->rdtsc = OsGetAsyncTimeClocks();
    out->tickCount = GetTickCount();
    
    if (this->hasQPF) {
        QueryPerformanceCounter(&out->qperfCount);
    }
    
    // Restore priority
    SetThreadPriority(thread, oldPriority);
}
```

### Issue 4: CPU Frequency Caching

**Problem**: `CGxDevice::CpuFrequency` may return 0 on first call if called before calibration completes.

**Recommended Patch**:
```c
float CGxDevice::CpuFrequency() {
    // Wait for calibration if needed
    while (DAT_00e1324c == 0.0) {
        Sleep(1);
    }
    
    if (DAT_00e1324c != 0.0) {
        return DAT_00e1324c;
    }
    
    // Fallback calibration with better accuracy
    // (same as before)
}
```

---

## Binary Offsets Summary

| Component | Address | Notes |
|-----------|---------|-------|
| `lua_open` | 0x006db070 | Lua state creation |
| `lua_pcall` | 0x006d8ea0 | Script execution |
| `FrameScript_Initialize` | 0x006e61d0 | Script system init |
| `FrameScript_ExecuteV` | 0x006e7360 | Execute with args |
| `FrameScript_RegisterFunction` | 0x006e6d00 | Register C function |
| `FrameScript_CompileFunction` | 0x006e70f0 | Compile script |
| `FrameScript_GetContext` | 0x006e6c90 | Get Lua state |
| `FrameScript_SignalEvent` | 0x006e6b30 | Dispatch event |
| `OsTimeManager` | 0x0045c020 | Time manager class |
| `TimeKeeper` | 0x0045c100 | Calibration thread |
| `Calibrate` | 0x0045c160 | Calibration logic |
| `Snapshot` | 0x0045c0a0 | Time capture |
| `OsGetAsyncTimeClocks` | 0x0045b960 | RDTSC wrapper |
| `CGxDevice::CpuFrequency` | 0x005951b0 | CPU freq query |
| `CGxDevice::CpuTicks` | 0x00595240 | TSC counter |

---

## Lua Scripting API Reference

### Format Specifiers for FrameScript_Execute

| Specifier | C Type | Lua Function |
|-----------|--------|--------------|
| `%d` | `int` | `lua_pushnumber(L, (lua_Number)value)` |
| `%f` | `float/double` | `lua_pushnumber(L, (lua_Number)value)` |
| `%s` | `char*` | `lua_pushstring(L, value)` |
| `%u` | `unsigned int` | `lua_pushnumber(L, (lua_Number)value)` |

### Error Handling
The script system stores error handlers in the Lua registry at index `DAT_010b3b6c` (reference to `_ERRORMESSAGE`).

---

## References

- VanillaFixes Project: `lib/vanillafixes/src/dll/tsc.c`
- Lua 5.0 Documentation: http://www.lua.org/manual/5.0/
- RDTSC Timing: https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquire-high-resolution-time-stamps
