# Alpha 0.5.3 Sound System

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. MIDI Support (DirectMusic)

The Alpha client utilizes **Microsoft DirectMusic** (part of DirectX 8) for music playback, primarily for zone ambiences. This is distinct from the mp3/wav system used in later versions.

*   **Initialization**: Calls `CoCreateInstance(CLSID_DirectMusicPerformance)`.
*   **File Format**: `.dls` (Downloadable Sounds) banks and `.mid` (MIDI) sequences.
*   **Zone Integration**:
    *   The client uses `AreaMIDIAmbiences.dbc` (and `SndInterfaceSetMIDIArea`) to bind Zone IDs to MIDI tracks.
    *   Columns likely: `ID`, `DayAmbienceID`, `NightAmbienceID`. (Where AmbienceID maps to a DLS/MIDI resource).

---

## 2. Sound Emitters (MCSE)

Ambient sounds placed in the world (waterfalls, machinery, etc.) are handled via **MCSE** chunks within the MCNK terrain file.

*   **Parsing**: The `CMapChunk::Create` function parses `MCSE` (Sound Emitter) chunks after layers/refs.
*   **Object**: `CMapSoundEmitter` (Allocated via `AllocSoundEmitter`).
*   **Structure**: The MCSE chunk contains a list of emitters.
    *   `SoundID`: ID in `SoundEntries.dbc`.
    *   `Position`: XYZ coordinates.
    *   `Min/Max/Cutoff Distance`: Attenuation ranges.
    *   `Time Ranges`: Start/End time for the sound (e.g., night-only crickets).

---

## 3. MCNK Fixed Layout

The Alpha MCNK chunk has a **Strictly Fixed Layout** for the initial sub-chunks, unlike the relative-offset system in later versions:

1.  **Header**: 0x00 - 0x88 (136 bytes)
2.  **MCVT (Heights)**: 0x88 - 0x2CC (580 bytes)
3.  **MCNR (Normals)**: 0x2CC - 0x48C (448 bytes)
4.  **MCLY (Layers)**: Starts at 0x48C.
5.  **Refs/Alpha/Shadow/Sound**: Follow sequentially after MCLY.

This fixed structure is critical for parsing these chunks correctly.

---

## 4. API References

*   `Sound::MIDI_Initialize`: DirectMusic setup.
*   `SndInterfaceSetMIDIArea`: Triggers zone music changes.
*   `AllocSoundEmitter`: Creates `CMapSoundEmitter` instances from MCSE data.

---
*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
