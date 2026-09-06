# GV-14D Audio Asset Family Support Matrix

## Intent

Define the minimum audio asset families the engine must support and keep simple decoded-audio playback distinct from sequence-driven instrument playback.

## Scope

- decoded audio asset families
- sequence-driven music families
- support-level vocabulary
- fallback and unresolved-reason rules

## Touched Surfaces

- audio planning docs
- future profile capability records
- future asset-resolution contracts
- future diagnostics/proof surfaces

## Inputs And Assumptions

- `wav`, `ogg`, and `flac` are decoded-audio families
- `midi` is a sequence/control family, not a directly playable sample format
- `midi` support in this engine requires instrument-bank support, initially through `SFP0` and DirectSound/DirectMusic-style `DLS`

## Outputs

- one support matrix covering at minimum:
  - `wav`
  - `ogg`
  - `flac`
  - `midi + SFP0`
  - `midi + DLS`
- one support-status vocabulary such as:
  - `native`
  - `adapter-backed`
  - `inspect-only`
  - `unsupported`
- one rule that unsupported audio must report exactly which family or dependency is missing

## Dependencies

- GV-14A
- GV-14B

## Proof

- future profile and backend work can cite one canonical family matrix instead of restating the required formats

## Stop Conditions

- a smaller model can answer whether a task is about decoded audio, MIDI sequencing, or instrument-bank support without guessing

## Non-Goals

- no decoder implementation
- no synth implementation
