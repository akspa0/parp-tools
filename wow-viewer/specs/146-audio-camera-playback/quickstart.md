# Spec 146 Quickstart and Proof Gates

This document is for implementation and user validation. It does not claim that audio playback is
implemented yet.

## Phase 1 proof

The Alpha catalog is the existing metadata/source-resolution foundation for
area ambience. It joins `AreaTable.dbc` to `AreaMIDIAmbiences.dbc` and checks the
referenced `.mid`/`.dls` files; it does not provide playback. See the
[Alpha audio catalog guide](../../docs/architecture/alpha-audio-catalog.md)
before running the client proof below.

Run focused C# tests for:

- transport state transitions and generation invalidation;
- explicit camera/area/emitter binding provenance;
- capability reports for supported, missing, and unsupported sources;
- failure isolation when one source or DLS bank is unavailable;
- bounded emitter admission from resident tile/chunk candidates.

Then build the viewer and cross-platform project. Build success is not audible proof.

## User-run client proof

1. Load a configured client and record the exact build/root.
2. Load a zone with an `AreaMIDIAmbiences` binding and inspect day/night/underwater resolution.
3. Load a tile containing decoded MCSE entries and inspect source tile/chunk, position, and unresolved reasons.
4. Import the Undead FlyBy and verify its existing `CinematicCamera.dbc` placement still resolves tile `(28,28)`.
5. Exercise Play, pause, scrub, loop, and stop while observing the audio diagnostics surface.
6. Exercise Play + Video and record whether audio is muxed, separate, or unavailable.
7. Repeat with an archive-backed asset and a missing/unsupported asset to verify honest diagnostics.

## Capability matrix

Record one row per tested client/build/backend:

| Capability | Client/build | Source | Playback | Capture | Result/reason |
|---|---|---|---|---|---|
| WAV | pending | pending | pending | pending | pending |
| MP3 | pending | pending | pending | pending | pending |
| OGG | pending | pending | pending | pending | pending |
| MIDI sequence | pending | pending | pending | pending | pending |
| DLS/DirectSound bank | pending | pending | pending | pending | pending |
| MCSE positional emitter | pending | pending | pending | pending | pending |

Do not promote a capability to the README support table until this matrix has evidence.

## Future single-player boundary

Spec 146 does not implement the local server/session authority. A future spec must define how
Alpha-Core SQL data, terrain/object residency, NPC/game-object state, audio events, and client
presentation share ownership. The viewer audio runtime is designed to be a consumer of those world
events later, not their source of truth today.
