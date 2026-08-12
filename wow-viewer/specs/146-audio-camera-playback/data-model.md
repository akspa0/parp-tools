# Data Model: World Audio and Camera Playback

## AudioAsset

Represents one client-referenced sound, sequence, bank, or stream.

| Field | Type | Meaning |
|---|---|---|
| `VirtualPath` | string | Client path used for lookup |
| `ResolvedSource` | enum | None, loose file, archive, generated, or external bridge |
| `Format` | enum | WAV, MP3, OGG, MIDI, DLS, unknown |
| `Build` | string | Exact client build used for interpretation |
| `SchemaStatus` | enum | Proven, unavailable, mismatched, not applicable |
| `DecoderStatus` | enum | Available, unsupported, failed, not attempted |
| `Provenance` | string | Source/reader/binding evidence |

Validation: virtual paths are normalized; no asset is playable unless source and decoder status are
both available. Archive-backed assets remain virtual until the backend requests bytes.

## AudioCapability

Reports support for a format or runtime operation.

| Field | Type | Meaning |
|---|---|---|
| `CapabilityId` | string | Stable format/operation identifier |
| `Playback` | enum | Available, unsupported, failed, unavailable |
| `Capture` | enum | Muxed, separate, silent-only, unsupported, unknown |
| `Platform` | string | Runtime platform/backend identity |
| `BuildScope` | string | Client/build range tested |
| `Reason` | string | Human-readable evidence or failure reason |

## AudioBinding

Associates proven or explicitly authored assets with a source context.

| Field | Type | Meaning |
|---|---|---|
| `BindingId` | string | Stable authored/runtime identity |
| `SourceKind` | enum | Camera, sequence, area ambience, emitter, explicit user selection |
| `SourceIdentity` | string | Camera/model, area ID, emitter ID, or project identity |
| `Assets` | list | Ordered or layered asset references |
| `StartOffsetMs` | integer | Binding offset into the shared transport |
| `Loop` | bool | Whether the binding loops |
| `Provenance` | string | DBC/DB2/ADT/sidecar evidence |

Validation: filename similarity is not sufficient provenance. Unknown bindings remain unresolved.

## AudioTransportState

The one logical timebase shared by camera preview and capture.

| Field | Type | Meaning |
|---|---|---|
| `State` | enum | Stopped, Playing, Paused, Preparing, Completed, Failed |
| `PlayheadMs` | integer | Current logical time |
| `DurationMs` | integer | Active path or sequence duration |
| `Loop` | bool | Loop policy |
| `CaptureRelation` | enum | Preview-only, preparing-capture, recording, finalized |
| `Generation` | integer | Monotonic lifecycle token preventing stale streams |

State transitions must be serialized by the authoritative transport. A new generation invalidates
old backend handles.

## AudioBus

Independent volume/mute category: `Master`, `MusicAmbience`, `EmittersEffects`, `UI`, and optional
`Test`. Bus changes affect active sources without recreating their bindings.

## WorldAudioEmitter

Map-bound positional candidate derived from MCSE or a separately proven client record.

| Field | Type | Meaning |
|---|---|---|
| `SourceTile` | integer pair | ADT tile containing the source |
| `SourceChunk` | integer pair | ADT chunk containing the source |
| `Position` | Vector3? | Decoded world position when proven |
| `SoundIdentity` | string/int? | Resolved client identity, otherwise null |
| `Range` | float? | Proven attenuation range |
| `Volume` | float? | Proven source volume |
| `RawEntry` | bytes/reference | Preserved source evidence |
| `ResolutionState` | enum | Ready, unresolved identity, unsupported layout, unavailable |

Unknown range or identity must not be fabricated. Candidate admission is bounded by resident tile,
camera distance, and active audio budget.

## AreaAmbienceBinding

Existing `AlphaAreaAudioBinding` plus resolved day/night/underwater assets, DLS bank status, active
time-of-day choice, and transition state. Area transitions are edge-triggered, not frame-triggered.

## AudioDiagnostic

Structured result with source identity, requested/resolved path, build, schema state, decoder state,
bank state, playback state, capture state, and a stable failure reason. It is the proof record for
support claims and must distinguish missing data from unsupported runtime capability.
