using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.Editor.Operations;

namespace WowViewer.Core.Editor.Staging;

/// <summary>
/// Metadata describing a mounted client source archive for cross-client staging.
/// </summary>
public sealed class ClientSourceDescriptor
{
    public string ClientId { get; init; } = string.Empty;
    public string DisplayName { get; init; } = string.Empty;
    public string BuildVersion { get; init; } = string.Empty;
    public string BasePath { get; init; } = string.Empty;
    public bool IsActive { get; set; } = true;
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.OrdinalIgnoreCase);
}

/// <summary>
/// Provenance tracking for a chunk or asset staged from a specific client archive.
/// </summary>
public sealed class StagedArtifactProvenance
{
    public string SourceClientId { get; init; } = string.Empty;
    public string SourceMapName { get; init; } = string.Empty;
    public int SourceTileX { get; init; }
    public int SourceTileY { get; init; }
    public int SourceChunkX { get; init; }
    public int SourceChunkY { get; init; }
    public string SourceBuild { get; init; } = string.Empty;
    public DateTime StagedAtUtc { get; init; } = DateTime.UtcNow;
    public string? ChecksumSha256 { get; set; }
}

/// <summary>
/// An active staging / restoration target map combining assets and terrain from multiple clients.
/// </summary>
public sealed class RestorationLibraryMap
{
    public string MapName { get; init; } = "RestorationMap";
    public string TargetEra { get; init; } = "3.3.5a";
    public DateTime CreatedAtUtc { get; init; } = DateTime.UtcNow;
    public Dictionary<GlobalChunkCoordinate, StagedChunkEntry> StagedChunks { get; } = new();
    public List<StagedPlacementEntry> StagedPlacements { get; } = new();
}

/// <summary>
/// Staged chunk record in the restoration library map.
/// </summary>
public sealed class StagedChunkEntry
{
    public GlobalChunkCoordinate TargetCoordinate { get; init; }
    public TransposedChunkRecord ChunkRecord { get; init; } = new();
    public StagedArtifactProvenance Provenance { get; init; } = new();
}

/// <summary>
/// Staged placement record in the restoration library map.
/// </summary>
public sealed class StagedPlacementEntry
{
    public TransposedObjectPlacement Placement { get; init; } = new();
    public StagedArtifactProvenance Provenance { get; init; } = new();
}

/// <summary>
/// Service managing multi-client archive ingestion, provenance tracking, and
/// cross-client staging into a unified restoration library map.
/// </summary>
public sealed class MultiClientMapStagingService
{
    private readonly Dictionary<string, ClientSourceDescriptor> _sources = new(StringComparer.OrdinalIgnoreCase);
    public RestorationLibraryMap? ActiveRestorationMap { get; private set; }

    public IReadOnlyCollection<ClientSourceDescriptor> MountedSources => _sources.Values;

    public bool RegisterClientSource(ClientSourceDescriptor descriptor)
    {
        if (string.IsNullOrWhiteSpace(descriptor.ClientId))
            throw new ArgumentException("ClientId cannot be empty.", nameof(descriptor));

        _sources[descriptor.ClientId] = descriptor;
        return true;
    }

    public bool UnregisterClientSource(string clientId)
    {
        return _sources.Remove(clientId);
    }

    public ClientSourceDescriptor? GetSource(string clientId)
    {
        _sources.TryGetValue(clientId, out var descriptor);
        return descriptor;
    }

    public RestorationLibraryMap CreateRestorationMap(string mapName, string targetEra = "3.3.5a")
    {
        ActiveRestorationMap = new RestorationLibraryMap
        {
            MapName = mapName,
            TargetEra = targetEra,
        };
        return ActiveRestorationMap;
    }

    public bool StageChunkPayload(
        string sourceClientId,
        string sourceMapName,
        int sourceTileX,
        int sourceTileY,
        int sourceChunkX,
        int sourceChunkY,
        GlobalChunkCoordinate targetCoord,
        TransposedChunkRecord chunkRecord,
        string? checksumSha256 = null)
    {
        if (ActiveRestorationMap == null)
            ActiveRestorationMap = CreateRestorationMap("DefaultRestorationMap");

        var source = GetSource(sourceClientId);
        var provenance = new StagedArtifactProvenance
        {
            SourceClientId = sourceClientId,
            SourceMapName = sourceMapName,
            SourceTileX = sourceTileX,
            SourceTileY = sourceTileY,
            SourceChunkX = sourceChunkX,
            SourceChunkY = sourceChunkY,
            SourceBuild = source?.BuildVersion ?? "Unknown",
            ChecksumSha256 = checksumSha256,
        };

        var entry = new StagedChunkEntry
        {
            TargetCoordinate = targetCoord,
            ChunkRecord = chunkRecord,
            Provenance = provenance,
        };

        ActiveRestorationMap.StagedChunks[targetCoord] = entry;
        return true;
    }

    public bool StagePlacement(
        string sourceClientId,
        string sourceMapName,
        int sourceTileX,
        int sourceTileY,
        TransposedObjectPlacement placement,
        string? checksumSha256 = null)
    {
        if (ActiveRestorationMap == null)
            ActiveRestorationMap = CreateRestorationMap("DefaultRestorationMap");

        var source = GetSource(sourceClientId);
        var provenance = new StagedArtifactProvenance
        {
            SourceClientId = sourceClientId,
            SourceMapName = sourceMapName,
            SourceTileX = sourceTileX,
            SourceTileY = sourceTileY,
            SourceBuild = source?.BuildVersion ?? "Unknown",
            ChecksumSha256 = checksumSha256,
        };

        ActiveRestorationMap.StagedPlacements.Add(new StagedPlacementEntry
        {
            Placement = placement,
            Provenance = provenance,
        });

        return true;
    }

    public int GetStagedChunkCount() => ActiveRestorationMap?.StagedChunks.Count ?? 0;
    public int GetStagedPlacementCount() => ActiveRestorationMap?.StagedPlacements.Count ?? 0;

    public void ClearStaging()
    {
        ActiveRestorationMap?.StagedChunks.Clear();
        ActiveRestorationMap?.StagedPlacements.Clear();
    }
}
