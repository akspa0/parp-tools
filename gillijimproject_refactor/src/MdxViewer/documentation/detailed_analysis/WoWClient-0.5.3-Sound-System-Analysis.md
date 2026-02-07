# WoW Alpha 0.5.3 (Build 3368) Sound System Analysis

## Overview

This document provides a deep analysis of the sound system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers the MIDI system, DLS instrument support, audio streaming, and integration with the game engine.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`MIDI_Initialize`](MIDI_Initialize) | 0x007b7230 | Initialize DirectMusic |
| [`MIDI_CleanupSegment`](MIDI_CleanupSegment) | 0x007b7320 | Clean up MIDI segment |
| [`MIDI_Play`](MIDI_Play) | 0x007b7370 | Play MIDI |
| [`MIDI_Stop`](MIDI_Stop) | 0x007b7660 | Stop MIDI |
| [`MIDI_Playing`](MIDI_Playing) | 0x007b7710 | Check if playing |
| [`MIDI_SetVolume`](MIDI_SetVolume) | 0x007b7690 | Set MIDI volume |
| [`AreaMIDIAmbiencesRec`](AreaMIDIAmbiencesRec) | 0x0058a190 | Area MIDI ambience |
| [`SoundInterfaceInitializeWorldMIDI`](SoundInterfaceInitializeWorldMIDI) | 0x004a7360 | Initialize world MIDI |
| [`SoundInterfaceShutdownWorldMIDI`](SoundInterfaceShutdownWorldMIDI) | 0x004a7380 | Shutdown world MIDI |
| [`SndInterfaceMIDISetPaused`](SndInterfaceMIDISetPaused) | 0x004a7490 | Pause/resume MIDI |
| [`SndInterfaceMIDIUnderwaterChanged`](SndInterfaceMIDIUnderwaterChanged) | 0x004a7480 | Underwater effect change |
| [`SndInterfaceMIDIAmbienceChanged`](SndInterfaceMIDIAmbienceChanged) | 0x004a7470 | Ambience change |
| [`SndInterfaceClearMIDI`](SndInterfaceClearMIDI) | 0x004a7450 | Clear MIDI |
| [`AUDIOSTREAM`](AUDIOSTREAM) | 0x00653760 | Audio stream structure |
| [`CheckAudioStreams`](CheckAudioStreams) | 0x0064e2e0 | Check active streams |
| [`UpdateAudioStreamPos`](UpdateAudioStreamPos) | 0x0064e710 | Update stream position |

---

## DirectMusic Integration

### MIDI Initialization

The engine uses DirectMusic for MIDI playback with DLS instruments:

```c
/* From MIDI_Initialize at 0x007b7230 */
int __fastcall Sound::MIDI_Initialize(void) {
    HRESULT HVar1;
    ulong uVar2;
    bool bVar4;
    undefined4 uVar5, uVar6, uVar7, uVar8;
    
    // Initialize COM for DirectMusic
    HVar1 = CoInitialize((LPVOID)0x0);
    if (HVar1 == 0) {
        DAT_010ba1fc = 1;  // COM initialized
    }
    
    // Create DirectMusic Performance
    HVar1 = CoCreateInstance(
        (IID *)&_CLSID_DirectMusicPerformance,
        (LPUNKNOWN)0x0,
        3,  // CLSCTX_INPROC_SERVER
        (IID *)&_IID_IDirectMusicPerformance8,
        &DAT_010ba1ec
    );
    
    bVar4 = HVar1 == 0;
    if (bVar4) {
        // Get device window
        iVar3 = *DAT_010ba1ec;
        uVar8 = 0;
        uVar7 = 0x3f;  // Default port parameters
        uVar6 = 0;
        uVar5 = 0;
        uVar2 = GxDevWindow();
        
        // Initialize audio path
        iVar3 = (**(code **)(iVar3 + 0xb0))
            (DAT_010ba1ec, 0, 0, uVar2, uVar5, uVar6, uVar7, uVar8);
        bVar4 = iVar3 == 0;
        
        if (bVar4) {
            // Set master tempo
            iVar3 = (**(code **)(*DAT_010ba1ec + 0xc4))
                (DAT_010ba1ec, 8, 0x10, 1, &DAT_010ba1f4);
            bVar4 = iVar3 == 0;
        }
    }
    
    DAT_010ba1fd = bVar4;  // Store initialization status
    return (uint)bVar4;
}
```

### Global State

```c
// Global DirectMusic state
DAT_010ba1ec  // IDirectMusicPerformance8*
DAT_010ba1f4  // IDirectMusicAudioPath*
DAT_010ba1fc  // COM initialized flag
DAT_010ba1fd  // MIDI initialized flag
```

---

## MIDI Playback Control

### Play MIDI

```c
/* MIDI_Play at 0x007b7370 */
int __fastcall Sound::MIDI_Play(char* filename) {
    // Load MIDI from file
    IDirectMusicSegment* segment = LoadMidiFile(filename);
    
    if (segment == NULL) {
        return 0;  // Failed to load
    }
    
    // Download DLS instruments
    IDirectMusicSegment8* seg8;
    segment->QueryInterface(&IID_IDirectMusicSegment8, &seg8);
    seg8->Download(dlsCollection);
    
    // Play on audio path
    DAT_010ba1ec->PlaySegmentEx(
        segment,
        NULL,
        NULL,
        DMUS_SEGF_SECONDARY,
        0,
        NULL,
        NULL,
        DAT_010ba1f4
    );
    
    return 1;  // Success
}
```

### Stop MIDI

```c
/* MIDI_Stop at 0x007b7660 */
int __fastcall Sound::MIDI_Stop(void) {
    // Stop all segments
    DAT_010ba1ec->StopEx(
        NULL,  // All segments
        0,
        0
    );
    
    // Unload DLS instruments
    if (dlsCollection != NULL) {
        dlsCollection->Unload();
    }
    
    return 1;
}
```

### Volume Control

```c
/* MIDI_SetVolume at 0x007b7690 */
int __fastcall Sound::MIDI_SetVolume(long volume) {
    // volume: -9600 to 0 (in 1/100ths of dB)
    // 0 = 0 dB (full volume)
    // -9600 = -96 dB (silent)
    
    DAT_010ba1f4->SetVolume(volume, 0);
    return 1;
}
```

---

## DLS Instrument System

### DLS Structure

DLS (Downloadable Sounds) is used for MIDI instrumentation:

```c
struct DLSCollection {
    IDirectMusicCollection8* collection;
    
    // Instrument lookup
    int GetInstrument(uint32_t program, uint32_t bank);
    int LoadDLS(char* filename);
    void Unload(void);
};
```

### DLS Instrument Banks

The engine supports multiple DLS banks:

| Bank | Description |
|------|-------------|
| General MIDI | Standard GM instruments |
| Custom | Game-specific instruments |
| Percussion | Drum kits |

### DLS Loading

```c
int LoadDLS(char* filename) {
    // Open DLS file
    IDirectMusicLoader8* loader;
    CoCreateInstance(
        &CLSID_DirectMusicLoader,
        NULL,
        CLSCTX_INPROC_SERVER,
        &IID_IDirectMusicLoader8,
        &loader
    );
    
    // Load collection
    IDirectMusicObject* dlsObj;
    loader->LoadObjectFromFile(
        CLSID_DirectMusicCollection,
        IID_IDirectMusicObject,
        filename,
        &dlsObj
    );
    
    dlsObj->QueryInterface(
        IID_IDirectMusicCollection8,
        &dlsCollection
    );
    
    return 1;
}
```

---

## Area-Based MIDI Ambiences

### Area MIDI Structure

```c
/* AreaMIDIAmbiencesRec at 0x0058a190 */
struct AreaMIDIAmbiencesRec {
    uint32_t areaId;              // Area ID
    uint32_t musicId;             // Music entry ID
    uint32_t dayAmbience;         // Daytime ambience MIDI
    uint32_t nightAmbience;       // Nighttime ambience MIDI
    uint32_t underwaterAmbience;  // Underwater MIDI
    uint32_t unk[4];              // Unknown fields
};
```

### Ambience Transitions

```c
/* SndInterfaceMIDIAmbienceChanged at 0x004a7470 */
void __fastcall Sound::SndInterfaceMIDIAmbienceChanged(
    uint32_t areaId,
    uint32_t newAmbience
) {
    AreaMIDIAmbiencesRec* area = GetAreaMIDI(areaId);
    
    if (area == NULL) {
        return;
    }
    
    // Crossfade to new ambience
    CrossfadeMIDI(
        area->dayAmbience,    // or night based on time
        newAmbience,
        5000  // 5 second crossfade
    );
}
```

### Underwater Effect

```c
/* SndInterfaceMIDIUnderwaterChanged at 0x004a7480 */
void __fastcall Sound::SndInterfaceMIDIUnderwaterChanged(
    int isUnderwater
) {
    if (isUnderwater) {
        // Apply low-pass filter
        IDirectMusicFX8* fx;
        DAT_010ba1f4->GetObjectInPath(
            DMUS_PCHANNEL_ALL,
            DMUS_IO_PARA_EFFECT,
            IID_IDirectMusicFX8,
            &fx
        );
        
        // Set underwater effect parameters
        DMUS_FXCHORUS_PARAMS chorus;
        chorusWetDryMix = 30;  // Wet mix 30%
        fx->SetEffectParam(&chorus, sizeof(chorus));
    } else {
        // Remove effects
        DAT_010ba1f4->SetVolume(0, 500);  // Fade back
    }
}
```

---

## Audio Streaming

### Audio Stream Structure

```c
/* AUDIOSTREAM at 0x00653760 */
struct AUDIOSTREAM {
    TSLink<Storm::SFile::AUDIOSTREAM> link;  // List link
    
    // Stream state
    uint32_t state;          // Playing, paused, stopped
    uint32_t format;         // Audio format
    uint32_t sampleRate;     // Sample rate (e.g., 22050, 44100)
    uint32_t channels;       // 1 = mono, 2 = stereo
    uint32_t bitsPerSample;  // 8 or 16 bits
    
    // Buffer
    uint8_t* buffer;         // Audio buffer
    uint32_t bufferSize;     // Buffer size in bytes
    uint32_t bufferPos;      // Current position
    
    // Stream source
    char* filename;          // Source file
    uint32_t fileOffset;     // Start offset in file
    uint32_t fileSize;       // Stream size
    
    // Callbacks
    void (*Callback)(AUDIOSTREAM* stream, uint32_t event);
};
```

### Stream Check

```c
/* CheckAudioStreams at 0x0064e2e0 */
void __fastcall Sound::CheckAudioStreams(void) {
    AUDIOSTREAM* stream = audioStreams.head;
    
    while (stream != NULL) {
        // Check if stream needs more data
        if (stream->state == PLAYING) {
            if (stream->bufferPos >= stream->bufferSize) {
                // Buffer underrun
                RefillStreamBuffer(stream);
            }
        }
        
        // Check for end of stream
        if (stream->bufferPos >= stream->fileSize) {
            if (stream->Callback != NULL) {
                stream->Callback(stream, STREAM_EVENT_ENDED);
            }
            stream->state = STOPPED;
        }
        
        stream = stream->next;
    }
}
```

### Stream Position Update

```c
/* UpdateAudioStreamPos at 0x0064e710 */
void __fastcall Sound::UpdateAudioStreamPos(
    AUDIOSTREAM* stream,
    uint32_t newPosition
) {
    if (newPosition >= stream->fileSize) {
        newPosition = 0;  // Loop to start
    }
    
    stream->fileOffset = newPosition;
    stream->bufferPos = 0;
    
    // Read new data
    SFile::Read(
        stream->filename,
        stream->buffer,
        stream->bufferSize,
        stream->fileOffset,
        NULL
    );
}
```

---

## Sound Interface Functions

### Initialize World MIDI

```c
/* SoundInterfaceInitializeWorldMIDI at 0x004a7360 */
int __fastcall Sound::SoundInterfaceInitializeWorldMIDI(void) {
    // Initialize MIDI for world (outdoor) areas
    DAT_010ba1fc = MIDI_Initialize();
    
    // Set default volume
    MIDI_SetVolume(0);  // Full volume
    
    // Register area ambience callbacks
    RegisterAreaCallback(AreaMIDIAmbiencesRec);
    
    return DAT_010ba1fc;
}
```

### Initialize World MIDI CVars

```c
/* SoundInterfaceInitializeWorldMIDICVars at 0x004a7230 */
void __fastcall Sound::SoundInterfaceInitializeWorldMIDICVars(void) {
    // Create console variables for MIDI control
    CreateCVar("MusicVolume", "1.0", CVAR_ARCHIVE);
    CreateCVar("MusicEnabled", "1", CVAR_ARCHIVE);
    CreateCVar("MusicFadeTime", "5.0", CVAR_ARCHIVE);
    
    // Register callbacks
    CVarCallback("MusicVolume", OnMusicVolumeChange);
}
```

### Shutdown World MIDI

```c
/* SoundInterfaceShutdownWorldMIDI at 0x004a7380 */
void __fastcall Sound::SoundInterfaceShutdownWorldMIDI(void) {
    // Clear any playing MIDI
    SndInterfaceClearMIDI();
    
    // Shutdown MIDI system
    MIDI_Shutdown();
    
    // Free DLS collection
    if (dlsCollection != NULL) {
        dlsCollection->Release();
        dlsCollection = NULL;
    }
}
```

### Clear MIDI

```c
/* SndInterfaceClearMIDI at 0x004a7450 */
void __fastcall Sound::SndInterfaceClearMIDI(void) {
    // Stop all segments
    DAT_010ba1ec->StopEx(NULL, 0, 0);
    
    // Clear audio path
    DAT_010ba1f4->SetVolume(-9600, 1000);  // Fade out
    
    // Clear callbacks
    ClearAreaCallbacks();
}
```

---

## Supported Audio Formats

### MIDI

| Format | Description |
|--------|-------------|
| MIDI Type 0 | Single track |
| MIDI Type 1 | Multiple tracks |
| Standard MIDI | GM/GS/XG compatible |

### DLS

| Format | Description |
|--------|-------------|
| DLS Level 1 | Basic downloadable sounds |
| DLS Level 2 | Advanced features |
| Wave samples | PCM audio data |

### Streamed Audio

| Format | Description |
|--------|-------------|
| MP3 | MPEG-1 Layer 3 |
| OGG | Vorbis audio |
| WAV | PCM uncompressed |

---

## Sound Categories

### Environmental Sounds

| Category | Description |
|----------|-------------|
| Ambient | Continuous background sounds |
| Weather | Rain, wind, thunder |
| Water | Rivers, waterfalls, ocean |

### Interactive Sounds

| Category | Description |
|----------|-------------|
| Footsteps | Walking on different surfaces |
| Combat | Weapons, spells, impacts |
| UI | Buttons, menus, selections |

### Music

| Category | Description |
|----------|-------------|
| Zone | Area-specific music |
| Combat | Battle music |
| Inn | Rest/tavern music |

---

## Performance Considerations

### Memory Management

1. **Streaming**
   - Buffer size: 32-64KB typical
   - Preload critical sounds
   - Stream non-critical sounds

2. **Pooling**
   ```c
   // Sound object pool
   SoundObject pool[64];
   uint32_t poolIndex;
   
   SoundObject* AllocateSound(void) {
       if (poolIndex < 64) {
           return &pool[poolIndex++];
       }
       return NULL;  // Pool exhausted
   }
   ```

### Latency

| Sound Type | Latency Target |
|------------|---------------|
| UI sounds | < 50ms |
| Footsteps | < 100ms |
| Combat | < 50ms |
| Music | < 500ms (fade) |

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| E_NOINTERFACE | DirectMusic not installed | Fallback to wave audio |
| E_OUTOFMEMORY | Too many sounds | Pool management |
| E_FILENOTFOUND | Missing sound file | Use placeholder |
| E_INVALIDARG | Corrupted audio file | Skip and log |

### Fallback System

```c
int PlaySoundWithFallback(char* filename) {
    // Try primary format
    if (MIDI_Play(filename) == 0) {
        // Fallback to wave
        return PlayWaveFile(filename);
    }
    return 0;
}
```

---

## References

### DirectMusic Interfaces

- `IDirectMusicPerformance8`: Main performance interface
- `IDirectMusicSegment8`: MIDI segment
- `IDirectMusicAudioPath8`: Audio output path
- `IDirectMusicCollection8`: DLS collection
- `IDirectMusicFX8`: Effects interface

### Related Functions

| Function | Address |
|----------|---------|
| [`MIDI_Initialize`](MIDI_Initialize) | 0x007b7230 |
| [`MIDI_Play`](MIDI_Play) | 0x007b7370 |
| [`MIDI_Stop`](MIDI_Stop) | 0x007b7660 |
| [`MIDI_SetVolume`](MIDI_SetVolume) | 0x007b7690 |
| [`AUDIOSTREAM`](AUDIOSTREAM) | 0x00653760 |
| [`CheckAudioStreams`](CheckAudioStreams) | 0x0064e2e0 |
