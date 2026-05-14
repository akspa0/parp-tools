# GV-17B MIDI Synth And Instrument Bank Bridge

## Intent

Keep MIDI playback support explicit by separating sequence playback from sample decode and by naming the first required instrument-bank families.

## Scope

- MIDI sequencing bridge
- synth ownership boundary
- `SFP0` support seam
- DirectSound/DirectMusic `DLS` support seam
- rendered-audio handoff into the normal backend path

## Touched Surfaces

- future engine audio runtime/contracts
- future profile audio resolvers
- future backend-facing decoded-audio submission path
- diagnostics/proof tooling for MIDI bank resolution

## Inputs And Assumptions

- MIDI playback is not the same as playing a decoded `.wav`, `.ogg`, or `.flac`
- WoW Alpha already makes `DLS` a first-class proof target
- future profiles or Museums content may prefer `SFP0`

## Outputs

- one explicit sequencing/synth bridge contract such as:
  - `IMidiSequencePlayer`
  - `IInstrumentBankResolver`
  - `IRenderedAudioStreamSource`
- one rule that MIDI resolves into rendered PCM/stream output before the generic playback backend consumes it
- one rule that bank resolution failures must surface concrete missing-bank diagnostics

## Dependencies

- GV-14B
- GV-14C
- GV-17A

## Proof

- the engine can describe how `midi + SFP0` and `midi + DLS` become backend-neutral rendered audio without conflating them with simple sample decode

## Stop Conditions

- a future implementation slice can work on MIDI/soundfont support without reopening the rest of the backend design

## Non-Goals

- no commitment to one synth library
- no final timing/latency policy
