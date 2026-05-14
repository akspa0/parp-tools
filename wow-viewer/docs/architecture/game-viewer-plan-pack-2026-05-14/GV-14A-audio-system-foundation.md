# GV-14A Audio System Foundation

## Intent

Make audio a first-class engine subsystem beside rendering rather than a late app-side utility.

## Scope

- engine audio vocabulary
- subsystem ownership
- music plus multi-channel playback goals
- relation between profile audio data and engine-neutral audio runtime
- required asset-family coverage at the planning level

## Touched Surfaces

- `wow-viewer/docs/architecture/audio-engine-plan-2026-04-21.md`
- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`
- future `BASE/Audio` ownership docs
- future runtime/audio contracts

## Inputs And Assumptions

- the engine must support both music and world/object audio
- supported profiles will resolve audio differently
- the other media-focused project is expected to point generated metadata and derivative audio-facing artifacts at this engine later
- the minimum required audio families now explicitly include:
  - decoded audio: `wav`, `ogg`, `flac`
  - sequence-driven music: `midi`
  - instrument-bank families for MIDI: `SFP0`, DirectSound/DirectMusic `DLS`

## Outputs

- one audio subsystem story covering:
  - music playback
  - emitter/world audio playback
  - listener state
  - channel/layout policy
  - decoded-audio playback versus MIDI-plus-bank playback
  - diagnostics and proof surfaces
- one rule that audio runtime/backend interfaces are `BASE` concerns while profile-specific lookup rules stay in profile libraries

## Dependencies

- GV-00B
- GV-06A

## Proof

- future audio implementation slices can target subsystem ownership without reopening "is audio part of the engine?" debates

## Stop Conditions

- the audio subsystem can be explained in one paragraph using BASE/runtime/profile language

## Non-Goals

- no decoder implementation
- no backend choice finalization
