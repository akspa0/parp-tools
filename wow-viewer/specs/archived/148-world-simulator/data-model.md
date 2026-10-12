# Data Model: Artifact World Simulator Runtime

## ArtifactProvenance

Represents how a virtual client resource was identified and obtained.

- `ClientBuild`: build/profile identity selected for the session.
- `VirtualPath`: normalized requested path.
- `SourceKind`: loose file, archive catalog, alpha MPQ wrapper, unknown, or missing.
- `SourceName`: archive or configured root label when known; never a machine-local assumption.
- `CatalogState`: not checked, visible, or absent.
- `ReadState`: not attempted, read, or failed with reason.
- `ByteLength`: readable byte count when available.

## AudioTriggerDiagnostic

One MCSE, area-music, or ambience decision as seen by the current world actor.

- `TriggerKind`: MCSE, zone music, area ambience, or other DBC-backed trigger.
- `Tile`, `Chunk`, and source record IDs.
- `RawPosition`: coordinates as stored by the era-specific reader.
- `WorldPosition`: transformed renderer coordinates.
- `CoordinateProfile`: named transform/profile and confidence/evidence state.
- `SoundPointId`, `SoundNameId`, and optional range/start/end/mode fields.
- `DistanceToActor`, effective range, and admission result.
- `SoundEntryId`, candidate virtual paths, selected path, and `ArtifactProvenance`.
- `Format`, decode state/reason, backend state/reason, mute state, and terminal status.

The model deliberately retains raw and transformed values. A coordinate conversion must not be
silently “fixed” from a screenshot without a format/evidence decision.

## CameraActorState

The authoritative exploration state at a timestamp.

- Position, forward/up orientation, and roll.
- Active map/tile/chunk and terrain/WMO context.
- Collision context and camera-path sample identity.
- Audio-listener position/orientation and mute/master state.
- Residency lease IDs and session/build identity.

## ResidencyLease

An explainable request keeping a tile, object, or resource resident.

- Owner: fog coverage, path warmup, explicit inspection, selected asset, or actor context.
- Target identity and coverage bounds.
- Created/last-refreshed/release timestamps.
- Hold interval and release reason.
- Conflict/union state when multiple owners retain the target.

## RenderPerformanceSample

A frame or capture sample used to compare renderer paths.

- Timestamp and actor snapshot ID.
- Resident tiles, WMO placements, unique models, and instances.
- Resource preparation, selection/culling, batch preparation, draw submission, terrain, WMO
  internal doodad, and audio durations.
- Draw calls/submissions by owner and total frame duration.
- Any missed residency or stale-resource reason.
