# GV-14B Profile Audio Resolution Contracts

## Intent

Split profile-specific audio lookup and asset resolution from the engine-neutral audio runtime.

## Scope

- profile audio resolvers
- music lookup contracts
- emitter/audio-event lookup contracts
- unresolved-reason reporting
- audio-family and bank-family resolution

## Touched Surfaces

- future WoW audio readers/resolvers
- future Warcraft 3 audio resolvers
- future Museums audio metadata resolvers
- inspect/report tooling for profile audio proof

## Inputs And Assumptions

- WoW Alpha MIDI/DLS lookup is not shaped like future Museums or later-era sound systems
- the engine still needs one consistent runtime-facing resolved-audio recipe
- unresolved audio must fail honestly with diagnostics
- profile audio resolution now needs to distinguish at least:
  - decoded assets such as `wav`, `ogg`, `flac`
  - sequence assets such as `midi`
  - required instrument-bank families such as `SFP0` or `DLS`

## Outputs

- a profile-side resolver contract such as:
  - `ResolveMusicCue`
  - `ResolveEmitterCue`
  - `ResolveAudioAsset`
  - `ResolveInstrumentBank`
  - `ExplainUnsupportedAudio`
- a runtime-facing result shape for resolved music/emitter recipes

## Dependencies

- GV-06A
- GV-14A

## Proof

- different supported profiles can produce one shared resolved-audio recipe shape without erasing their lookup differences

## Stop Conditions

- a smaller model can tell whether an audio task belongs in a profile resolver or in engine runtime/backend code

## Non-Goals

- no playback scheduling yet
