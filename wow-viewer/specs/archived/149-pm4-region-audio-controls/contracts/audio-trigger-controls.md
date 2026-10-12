# Contract: Audio Trigger Controls

## Producer and owner

Terrain loading supplies the decoded MCNK flag/liquid candidates and MCSE records for resident tiles.
The terrain adapters own source-coordinate normalization into `TerrainSoundEmitter.Position` while
`WorldAudioRuntime` owns source-to-SoundEntries diagnostics, range decisions, source lifecycle, and
effective start permission. `WorldScene` forwards the control/query methods. `ViewerApp_Audio.cs`
renders controls and sends explicit user intent.

## Required operations

```text
SetWorldTriggersEnabled(bool enabled)
SetTriggerEnabled(AudioTriggerInstanceKey key, bool enabled)
GetTriggerDiagnostics() -> IReadOnlyList<AudioTriggerDiagnostic>
StopWorldTriggers()                         # used by master/row disable and teardown
```

Names may vary during implementation, but the ownership and semantics are fixed.

## Invariants

1. The master world-trigger gate starts `false` on every new runtime/session.
2. No automatic MCSE or area/ZoneMusic source starts unless both the master gate and that instance are
   enabled.
3. No automatic MCNK environmental or liquid/water source starts unless both the master gate and that
   instance are enabled. MCNK rows remain enumerable when MCSE is absent, including on Alpha 0.5.3 maps.
4. MCSE raw/local coordinates are normalized with the owning tile/chunk origin before range checks,
   OpenAL placement, or an `InRange` diagnostic is reported. The raw value remains available for audit.
5. If MCNK and MCSE sources coexist, they remain separate rows unless a client-proven identity supports a
   safe merge; SoundEntries ID or position equality alone is not a merge rule.
6. Liquid family/type participates in MCNK water/environment selection. Missing or unproven mappings are
   visible as unresolved diagnostics rather than guessed SoundEntries IDs.
7. Disabling a row stops only that instance's owned source and blocks restart.
8. Disabling the master stops all world-trigger sources but does not stop deliberate SoundEntries
   preview unless the existing product policy explicitly says so.
9. Diagnostics enumerate disabled, unresolved, unsupported, and backend-unavailable rows just like
   enabled rows.
10. Trigger identity is not only SoundEntries ID; duplicate emitters and distinct MCNK/liquid instances
    remain independently controllable.
11. Map/client replacement clears enablement and sources.

## Diagnostic contract

The existing raw/transformed coordinates, area context, MCNK flags/liquid identity, candidate paths,
source provenance, decoder state, backend state, and terminal state remain available. The UI adds explicit
user-control state rather than overwriting `UnresolvedSoundEntry`, `DecodeFailed`, or
`BackendUnavailable` with a generic mute result. A coordinate diagnostic must distinguish raw/local,
normalized world, and listener distance so a tile-normalization failure cannot be mistaken for a missing
audio file.
