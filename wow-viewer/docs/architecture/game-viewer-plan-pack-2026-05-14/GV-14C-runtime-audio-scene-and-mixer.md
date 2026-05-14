# GV-14C Runtime Audio Scene And Mixer

## Intent

Define the engine-neutral runtime audio scene that consumes resolved profile audio and turns it into play/stop/update decisions.

## Scope

- listener state
- music channel ownership
- world emitter ownership
- multi-channel routing policy
- voice budgeting

## Touched Surfaces

- future runtime audio contracts
- future world-session listener updates
- future audio diagnostics surfaces

## Inputs And Assumptions

- the runtime must handle both background music and positioned/world audio
- multi-channel support should be explicit even if the first proof backend is simpler
- the mixer policy belongs to engine runtime, not to profile libraries

## Outputs

- candidate runtime contracts:
  - `AudioListenerState`
  - `AudioMusicState`
  - `AudioEmitterInstance`
  - `AudioSceneFrame`
  - `AudioPlaybackUpdate`
- a first policy split between:
  - music bus
  - ambient/world bus
  - UI or diagnostics bus
  - future voice/dialog bus

## Dependencies

- GV-14A
- GV-14B
- GV-18

## Proof

- a future null backend can report which music/emitter cues are active and how they are routed

## Stop Conditions

- the runtime audio scene is bounded enough that the first implementation can live in a small number of runtime files

## Non-Goals

- no DSP stack
- no final spatialization math
