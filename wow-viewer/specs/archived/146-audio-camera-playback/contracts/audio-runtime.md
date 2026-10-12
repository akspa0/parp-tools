# Audio Runtime Contract

This is a viewer-independent contract sketch. It deliberately does not choose a decoder or mixer
library.

## Responsibilities

- `IAudioAssetResolver`: resolve a client virtual path and return source/provenance.
- `IAudioCapabilityProbe`: report format, platform, playback, and capture capabilities.
- `IAudioBindingResolver`: join proven client metadata or explicit project bindings to assets.
- `IAudioTransport`: own play/pause/stop/scrub/loop state and lifecycle generation.
- `IAudioRuntime`: load/unload bindings, update buses and bounded emitter candidates, and expose diagnostics.
- `IAudioBackend`: implement format playback and stream/mix handles behind the runtime contract.
- `IAudioCaptureBridge`: report whether the active mix can be muxed, written separately, or is unavailable.

## Invariants

1. One active transport generation owns a camera preview/capture session.
2. A stale generation cannot continue emitting audio after stop, map replacement, or client replacement.
3. Asset readers never depend on a concrete backend.
4. Unsupported formats and missing DLS banks are explicit capability states.
5. Emitter evaluation is bounded to resident/camera-relevant candidates.
6. Audio failures are isolated from rendering and video capture unless the user explicitly requires audio.
7. No source is treated as proven solely because a file with a matching name exists.

## Camera/capture lifecycle

```text
Stopped
  -> Preparing (resolve binding, assets, capabilities)
  -> Playing (camera preview)
  -> Recording (shared transport + video capture)
  -> Completed

Playing <-> Paused
Playing/Paused -> Stopped
Any state -> Failed (diagnostic retained; resources disposed)
```

Scrubbing changes `PlayheadMs` without starting a second backend stream. Looping creates a new
logical iteration under the same transport generation unless the backend requires a documented
reopen.

## Roadmap integration

Future world/session code may publish area, emitter, weather, time-of-day, and scripted sequence
events into the runtime. The runtime must not import SQL repositories, NPC AI, quest logic, or UI
types. Those authorities belong to a separate single-player client/server feature family.
