# Contract: Spatial Audio Diagnostics

## Producer

The existing Alpha/Standard terrain adapters, SoundEntries/area catalog readers, MPQ data source,
decoder, and `WorldAudioRuntime` remain the producers. The UI is a consumer only.

## Required behavior

`WorldAudioRuntime` must be able to produce diagnostics without starting a source. For each current
residency-relevant trigger it must expose:

1. source record and tile/chunk ownership;
2. raw position and transformed world position with a named coordinate profile;
3. range/start/end/mode inputs and distance admission;
4. SoundEntries resolution and all candidate virtual paths;
5. archive/loose-file catalog and byte-read provenance;
6. format detection and decoder state;
7. OpenAL/backend state, mute state, and terminal reason.

No stage may be collapsed into a generic `NotPlaying` result.

## Failure states

The contract distinguishes `NotResident`, `UnresolvedSoundEntry`, `AmbiguousPath`, `MissingResource`,
`ReadFailed`, `UnsupportedFormat`, `DecodeFailed`, `BackendUnavailable`, `OutOfRange`, `Muted`,
`Active`, and `Stopped` (names may be represented by the project's existing enum conventions).

## UI rules

The audio panel shows a scrollable table for the actor's current tile/WMO context. It must remain
usable when OpenAL is missing and must not crash during finalization. A selected row may request a
preview, but preview failure must update the row rather than hide it.
