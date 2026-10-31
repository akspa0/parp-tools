using System;
using System.Collections.Generic;
using System.Linq;
using WoWViewer.Terrain;
using WowViewer.Core.Maps;

namespace WoWViewer.Workbench.Services.TerrainAnalysis;

/// <summary>
/// Service to detect solid burgundy-red underside fog colors in loaded tile MCCV data
/// and derive terrain chunk hole flags (HoleMask = 0xFFFF, HoleMask64 = ~0UL) for contiguous
/// clusters of 2 or more chunks, rebuilding GPU tile meshes dynamically (Spec 287 Phase 2).
/// </summary>
internal sealed class MccvHoleDetectionService
{
    private string? _lastStatusMessage;

    public string? LastStatusMessage => _lastStatusMessage;

    /// <summary>
    /// Analyzes all loaded tiles in the given <see cref="TerrainManager"/>, sets hole flags
    /// for clusters of >= 2 solid burgundy chunks, and rebuilds the GPU tile meshes.
    /// </summary>
    public (int tilesAffected, int candidatesFound, int chunksHoled) DeriveBurgundyHoles(TerrainManager terrainManager, int minClusterSize = 2)
    {
        ArgumentNullException.ThrowIfNull(terrainManager);

        int totalTiles = 0;
        int totalCandidates = 0;
        int totalHoled = 0;

        // Collect all currently loaded tile coordinates
        var loadedCoords = terrainManager.LoadedTiles.ToList();
        if (loadedCoords.Count == 0)
        {
            _lastStatusMessage = "No tiles are currently loaded in TerrainManager.";
            return (0, 0, 0);
        }

        foreach (var (tileX, tileY) in loadedCoords)
        {
            // Inspect cached chunks for this tile
            if (!terrainManager.TileCache.TryGetValue((tileX, tileY), out var cached) || cached == null || cached.Chunks.Count == 0)
                continue;

            var chunkMap = new Dictionary<(int, int), WoWViewer.Terrain.TerrainChunkData>();
            var candidates = new List<(int ChunkX, int ChunkY)>();

            foreach (var chunk in cached.Chunks)
            {
                chunkMap[(chunk.ChunkX, chunk.ChunkY)] = chunk;
                if (MccvHoleDetection.IsSolidBurgundyChunk(chunk.MccvColors))
                {
                    candidates.Add((chunk.ChunkX, chunk.ChunkY));
                }
            }

            if (candidates.Count == 0)
                continue;

            totalCandidates += candidates.Count;
            var admitted = MccvHoleDetection.FilterContiguousClusters(candidates, minClusterSize);
            if (admitted.Count == 0)
                continue;

            int holedInTile = 0;
            foreach (var coord in admitted)
            {
                if (chunkMap.TryGetValue(coord, out var chunk))
                {
                    chunk.HoleMask = 0xFFFF;
                    chunk.HoleMask64 = 0xFFFFFFFFFFFFFFFFUL;
                    holedInTile++;
                }
            }

            if (holedInTile > 0)
            {
                totalHoled += holedInTile;
                totalTiles++;
                // Rebuild GPU mesh for this tile
                terrainManager.ReplaceTileChunksAndRebuild(tileX, tileY, cached.Chunks);
            }
        }

        _lastStatusMessage = totalHoled > 0
            ? $"Set hole flags on {totalHoled} chunks across {totalTiles} tile(s) (detected {totalCandidates} candidate chunks; isolated single chunks excluded)."
            : totalCandidates > 0
                ? $"Detected {totalCandidates} candidate chunks, but none formed contiguous clusters of >= {minClusterSize} chunks."
                : "No solid burgundy-red MCCV chunks detected across loaded tiles.";

        return (totalTiles, totalCandidates, totalHoled);
    }
}
