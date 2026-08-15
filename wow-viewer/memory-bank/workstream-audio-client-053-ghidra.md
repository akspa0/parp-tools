# Audio Client Contract — WoW Alpha 0.5.3 Ghidra Evidence

Last verified: 2026-08-14 against the open Ghidra `WoWClient.exe` program at
`H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft\WoWClient.exe`.

This is a reverse-engineering workstream note, not a claim of audible or native-runtime proof.
The live Ghidra MCP session was read-only; no program symbols, bytes, or project state were edited.

## The missing ID 1 is two separate lookup problems

The staged 0.5.3 DBCs contain no `SoundEntries` record with ID 1. That is not evidence that ID 1
is a MIDI alias.

- `AreaMIDIAmbiences.dbc` row ID 1 contains `Sound\\Ambience\\MIDI\\TestMIDI1Day.mid`,
  `Sound\\Ambience\\MIDI\\TestMIDI1Night.mid`, and `Sound\\Ambience\\MIDI\\DLSBank001.dls`.
- `ZoneMusic.dbc` row ID 1 contains ordinary day/night `SoundEntries` references 2523 and 2533.
- The client resolves `AreaTable.MIDIAmbience` and `AreaTable.ZoneMusic` through separate tables.
  `ZoneMusic` must not be resolved as a direct `SoundEntries` ID.

### SoundEntries contract

`ReadFiles @ 0x004a09d0` walks `SoundEntriesRec` records with a 0x88-byte stride. The live
`BuildSoundFilesRec @ 0x004a4890` trace proves ten filename pointers at raw offsets `0x0c..0x30`
and ten frequency values at `0x34..0x58`. It emits only non-empty filename/frequency pairs and
constructs each path from the DBC directory base plus the declared filename.

`ReadFiles` then reads the remaining fields at `0x60..0x84`, including volume, pitch, priority,
channel, flags, min distance, cutoff distance, and reverb. `ISndInterfaceGetSndEntry @ 0x004a1580`
is an ID hash lookup; it does not redirect into MIDI data.

The client's ZoneMusic path is `SndInterfaceRegisterNewZone @ 0x004a5830` to `ZoneMusicRec`,
`ZoneMusicIdle @ 0x004a55a0`, and `PlayMusic @ 0x004a5770`. `PlayMusic` selects the current day/night
sound ID from the ZoneMusic row and calls ordinary `Sound::Play2D` after `SoundEntries` resolution.

## MIDI/DLS pairing and hand-off

`StartAmbience @ 0x004a73f0` stops the current segment, selects normal or underwater ambience,
selects the day/night sequence from the same `AreaMIDIAmbiencesRec`, and calls:

```text
MIDI_Play(sequencePath, dlsPath)
```

The client record layout used by that call is sequence at `+0x04` or `+0x08` and the shared DLS
path at `+0x0c`. `MIDI_Play @ 0x007b7370` asynchronously loads both paths through `SFile::Open`
via `InitLoader @ 0x007b73b0`; it does not assume loose filesystem paths.

`PostLoadCallback @ 0x007b74b0` creates a DirectMusic segment from the MIDI bytes, creates a
DirectMusic collection from the DLS bytes, connects the collection to the segment using
`GUID_ConnectToDLSCollection`, prepares/downloads the segment, sets repeat behavior, and calls
DirectMusic `PlaySegmentEx` on the standard audio path. Cleanup stops/unloads the segment and
releases both COM objects.

This establishes the DLS pairing rule: the bank belongs to the `AreaMIDIAmbiences` row, not to a
`SoundEntries` ID or to a filename-matching heuristic.

## MCSE emitter contract

`CMapChunk::Create @ 0x00698e10` reads the Alpha MCNK offset/count fields from the 128-byte MCNK
payload header, then copies each on-disk MCSE record with a **0x34-byte stride** into the client's
76-byte `CWSoundEmitter` structure:

| MCSE offset | field | width |
|---:|---|---:|
| `0x00` | soundPointID | uint32 |
| `0x04` | soundNameID | uint32 |
| `0x08`, `0x0c`, `0x10` | position X/Y/Z | float[3] |
| `0x14`, `0x18`, `0x1c` | min/max/cutoff distance | float[3] |
| `0x20`, `0x22`, `0x24` | start/end/mode | uint16[3] |
| `0x26`, `0x27` | loop count min/max | byte[2] |
| `0x28`, `0x2a` | group silence min/max | uint16[2] |
| `0x2c`, `0x2e` | play instances min/max | uint16[2] |
| `0x30`, `0x32` | inter-sound gap min/max | uint16[2] |

The client allocates/reuses a `CMapSoundEmitter` with `CMap::AllocSoundEmitter @ 0x00691a30`,
links it into the chunk emitter list, and calls `soundEmitterCreateHandler`. `CMapChunk::Purge
@ 0x00696c80` calls `soundEmitterDestroyHandler` before `CMap::FreeSoundEmitter @ 0x00691af0`.
`CWorld::SetSoundEmitterHandlers @ 0x00664e60` only assigns those callbacks.

The active 0.5.3 executable does not prove that a callback is installed: `CMapChunk::Initialize`
(`Initialize @ 0x00697920`) clears both callback globals, and the Ghidra xrefs show no in-process
caller that installs them. The callback API is therefore a valid ownership seam, not evidence that
this build automatically turns MCSE rows into audible sounds.

## Viewer consequence and next slice

- The Alpha reader was corrected from the previously assumed 76-byte MCSE entry to the proven
  0x34-byte on-disk layout and now preserves the additional scheduler fields.
- The area catalog still needs an explicit `ZoneMusic` reader/model so
  `AreaTable.ZoneMusic -> ZoneMusic row -> day/night SoundEntries IDs` is represented before any
  automatic area playback claim.
- MIDI/DLS remains a capability-gated diagnostic state until a platform-safe DirectMusic-equivalent
  or explicitly user-approved bridge is selected.
- The viewer must keep its own resident-emitter runtime separate from the client callback seam until
  callback installation and native listener/object ownership are proven.

## Viewer-side coordinate and legacy-input correction

The current viewer evidence identifies a separate hand-off bug: `AlphaTerrainAdapter.ConvertSoundPosition`
and the standard adapter equivalent apply the global `MapOrigin - raw` conversion directly to the MCSE
record position. In the observed session, a tile `(31,31)` emitter with raw `(3,-3,62.5)` was reported as
approximately `(17069.7,17063.7,62.5)` and rejected as roughly 23k units from a listener near the tile.
The raw values must remain available, but the audio path must compose them with the owning tile/chunk
origin before range checks, OpenAL placement, and `WorldPosition` diagnostics.

The viewer-side correction is now implemented in the shared `TerrainCoordinateTransform`: Alpha and
standard MCSE producers keep `RawPosition` and derive `Position` from the owning chunk corner using
the established axis convention. The legacy MCNK liquid producer also now places its candidate at
the chunk center using `corner - halfChunk`, matching the terrain/liquid renderer. Focused audio
contract tests pass; configured-client audible proof remains user-owned.

The terrain path decodes Alpha MCNK flag bits and produces `LiquidChunkData.Type`; those values now
feed both rendering and the bounded legacy environmental/water candidate producer on 0.5.3 maps where
MCSE is absent. This note does not claim the client-proven
SoundEntries mapping for each MCNK/liquid variant; unresolved mappings must remain visible rather than
being guessed. Later MCSE records are additive until a client-proven identity establishes a safe merge.
