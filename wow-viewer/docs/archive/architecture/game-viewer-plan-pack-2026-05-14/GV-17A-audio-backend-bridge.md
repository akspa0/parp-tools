# GV-17A Audio Backend Bridge

## Intent

Keep desktop audio playback behind one small backend bridge the same way rendering stays behind backend contracts.

## Scope

- audio backend interface
- device capability reporting
- decoded-stream/sample submission boundary
- null backend for diagnostics-first proof

## Touched Surfaces

- future engine audio backend interfaces
- future app diagnostics host
- future Windows-first playback backend

## Inputs And Assumptions

- the first proof should favor one desktop backend plus one null backend
- backend choice must not leak into profile lookup code
- music streams and short emitter samples may need different backend update paths
- decoded `wav`/`ogg`/`flac` playback and rendered output from MIDI synthesis should converge on the same backend-facing PCM/stream path where practical

## Outputs

- one `IAudioBackend`-style contract
- one backend capability record
- one null backend proof surface
- one rule for how decoded audio reaches the backend
- one rule that MIDI synthesis output must be rendered into the same backend-facing audio stream contract rather than inventing a second playback stack

## Dependencies

- GV-14A
- GV-14C
- GV-17

## Proof

- runtime can produce backend-neutral audio updates without knowing the concrete playback library

## Stop Conditions

- a first implementation story can target null backend first, then one real backend second

## Non-Goals

- no commitment to a specific playback package in this plan
