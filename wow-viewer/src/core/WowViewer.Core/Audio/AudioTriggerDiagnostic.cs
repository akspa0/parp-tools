using System.Numerics;

namespace WowViewer.Core.Audio;

/// <summary>Identifies the owner of a world audio trigger.</summary>
public enum AudioTriggerKind
{
    Mcse,
    McnkLiquid,
    ZoneMusic,
    AreaAmbience,
    Unknown
}

/// <summary>Terminal state of an audio trigger inspection.</summary>
public enum AudioTriggerTerminalState
{
    NotResident,
    Disabled,
    UnresolvedSoundEntry,
    MissingResource,
    ReadFailed,
    DecodePending,
    DecodeFailed,
    BackendUnavailable,
    OutOfRange,
    Muted,
    Ready,
    Active,
    Stopped
}

/// <summary>
/// A build-scoped, inspectable decision for one spatial audio trigger. Raw and transformed
/// positions are intentionally retained together so a coordinate mistake cannot be hidden by the
/// renderer-facing transform.
/// </summary>
public sealed record AudioTriggerDiagnostic(
    AudioTriggerKind TriggerKind,
    int TileX,
    int TileY,
    int ChunkX,
    int ChunkY,
    uint SoundPointId,
    uint SoundNameId,
    Vector3 RawPosition,
    Vector3 WorldPosition,
    string CoordinateProfile,
    float MinDistance,
    float MaxDistance,
    float CutoffDistance,
    float DistanceToListener,
    bool InRange,
    bool SoundEntryResolved,
    IReadOnlyList<string> CandidateVirtualPaths,
    string? SelectedVirtualPath,
    string ResourceSource,
    bool ResourceExists,
    bool BytesRead,
    string DecodeStatus,
    string BackendStatus,
    AudioTriggerTerminalState TerminalState,
    string Detail,
    uint McnkFlags = 0,
    int LiquidFamily = -1,
    int SoundWaterSubtype = 0);
