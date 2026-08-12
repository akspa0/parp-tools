# Implementation Plan: World Audio and Camera Playback

**Branch**: `146-audio-camera-playback` (manual artifacts on current branch unless branch creation is explicitly requested) | **Date**: 2026-08-12 | **Spec**: [spec.md](spec.md)

## Summary

Add a reusable, backend-neutral audio runtime to the viewer. The first implementation sequence joins
existing build-aware area ambience and MCSE reader contracts to audio asset resolution, introduces a
single logical transport for camera preview and capture, and adds capability diagnostics before any
format-specific library is promoted as authoritative. The eventual Alpha-Core-backed single-player
client/server remains a roadmap consumer of the runtime, not part of this implementation.

## Technical Context

**Language/Version**: C# / .NET 10; optional Python/CLI tooling only for offline inspection or conversion

**Primary Dependencies**: Existing `AlphaAreaAudioCatalog`, `AlphaAreaAudioAssetResolver`, `AdtMcseReader`, camera-path evaluator, capture queue, and ffmpeg route; a backend-neutral audio contract precedes third-party backend selection

**Storage**: Client virtual paths and archive/loose provenance; project-sidecar audio bindings; no proprietary audio assets committed to the repository

**Testing**: Focused core tests for transport, binding, capability, and emitter admission; build validation; user-run audible playback and synchronized capture proof

**Target Platform**: Windows viewer first, preserving cross-platform compilation and explicit unsupported capability states

**Project Type**: Desktop 3D viewer/runtime with reusable core audio contracts

**Performance Goals**: Audio updates evaluate only the active area, resident emitter candidates, and camera-relevant sources; no whole-map audio load or per-frame restart

**Constraints**: Reuse existing readers; DBC/DB2 definitions are authoritative; do not hardcode sound IDs or layouts; do not implement the single-player server in this feature

## Constitution Check

- New implementation remains under `wow-viewer`; legacy reference code remains read-only.
- Existing client-file readers and audio catalogs are reused, not rewritten.
- C# owns the interactive runtime; Python remains optional offline tooling.
- Format/backend claims require focused capability evidence and user-run audible proof.
- The audio service remains independent of ImGui and capture-panel types.
- The single-player client/server direction is recorded as a future integration boundary only.

## Phases

### Phase 1 — Audio contracts, capabilities, and transport

Define audio assets, bindings, buses, capabilities, diagnostics, and one logical playhead/lifecycle
contract. Add focused tests for start/pause/stop/loop/scrub and failure isolation. No decoder library
is treated as final until the capability matrix is recorded.

### Phase 2 — Area ambience and bounded MCSE emitter candidates

Build viewer-independent resolution from the existing area catalog and MCSE data. Preserve raw
provenance and unresolved reasons. Add resident tile/chunk admission and camera/player-head-based
attenuation inputs without forcing whole-map loading.

### Phase 3 — First interactive playback backend

Evaluate and add the smallest backend that satisfies the first proven capability slice, initially
expected to be WAV plus one compressed format. Keep MIDI/DLS and any unsupported format explicit in
the capability report rather than adding guessed conversion behavior.

### Phase 4 — Camera-path and capture integration

Bind explicit client/project audio to Camera Path, make preview share the transport with Play + Video,
and establish muxed/separate/silent capture reporting. Ensure stop, map replacement, and capture
cancellation dispose all audio resources.

### Phase 5 — Historical MIDI/DLS and wider client-era coverage

Only after representative samples and schema evidence exist, add MIDI sequence/DLS/DirectSound
support through a platform-aware backend or offline bridge. Extend capability fixtures and keep
unsupported platforms/builds fail-closed.

### Phase 6 — World/session seam and roadmap handoff

Expose audio events and source-state transitions in a runtime contract consumable by future world
session code. Record the Alpha-Core SQL-backed single-player client/server prerequisites and keep
server/session implementation in a separate future spec.

## Project Structure

```text
wow-viewer/
├── specs/146-audio-camera-playback/
│   ├── spec.md
│   ├── research.md
│   ├── plan.md
│   ├── data-model.md
│   ├── quickstart.md
│   ├── contracts/audio-runtime.md
│   └── tasks.md
├── src/core/WowViewer.Core/Audio/
│   ├── existing area catalog and audio asset reports
│   └── future backend-neutral audio contracts
├── src/core/WowViewer.Core.IO/Audio/
│   ├── existing area asset resolution and MCSE readers
│   └── future build-aware audio binding readers
├── src/viewer/WoWViewer/
│   ├── future audio runtime ownership and diagnostics
│   └── Camera Path / capture integration
└── tests/WowViewer.Core.Tests/
    └── focused transport, binding, capability, and emitter tests
```

## Validation Gates

1. Contract tests prove transport lifecycle, binding provenance, capability states, and failure isolation.
2. Viewer and cross-platform builds pass without moving parser ownership into the UI.
3. User-run client test proves one area ambience path and one MCSE emitter path audibly behave as expected.
4. User-run camera test proves built-in FlyBy preview and Play + Video share timing and stop cleanly.
5. A capability report records actual WAV/MP3/OGG/MIDI/DLS/playback/capture results for the tested client/build.
6. No single-player server/client claim is made until its separate design and session-authority gates exist.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Multiple playback backends may be required | Historical WoW clients use multiple audio families, including MIDI/DLS | One unproven library cannot honestly cover every requested format and platform |
| Separate world/session roadmap seam | The project goal includes a future single-player client/server | Coupling server work into audio would make the MVP untestable and obscure ownership |
