using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;
using static WoWViewer.ViewerApp;
using static WoWViewer.InvestigationService;

namespace WoWViewer;

// TerrainQueryService: members moved from ViewerApp_Investigation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TerrainQueryService
{

    private readonly record struct McnkOverlayChunkInfo(
        int TileX,
        int TileY,
        int ChunkX,
        int ChunkY,
        Vector3 WorldPosition,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        McnkOverlayFlags Flags);

    private bool ShouldDrawMcnkFlagOverlay(TerrainRenderer? renderer)
    {
        return renderer != null && _showMcnkFlagOverlay && _mcnkOverlayFlags != McnkOverlayFlags.None;
    }

    private void BatchMcnkFlagOverlayGeometry(Terrain.BoundingBoxRenderer overlayRenderer)
    {
        _lastMcnkOverlayChunkCount = 0;
        _lastMcnkWeakCornerCount = 0;

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (!ShouldDrawMcnkFlagOverlay(renderer) || renderer == null)
            return;

        var loadedChunks = new Dictionary<(int globalChunkX, int globalChunkY), McnkOverlayChunkInfo>();

        if (_terrainManager != null)
        {
            foreach (var (tileX, tileY) in _terrainManager.LoadedTiles)
            {
                if (!_terrainManager.TryGetTileLoadResult(tileX, tileY, out TileLoadResult result))
                    continue;

                CollectMcnkFlagOverlayGeometry(renderer, overlayRenderer, result, loadedChunks);
            }
        }
        else if (_vlmTerrainManager != null)
        {
            foreach (var (tileX, tileY) in _vlmTerrainManager.LoadedTiles)
            {
                if (!_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out TileLoadResult result))
                    continue;

                CollectMcnkFlagOverlayGeometry(renderer, overlayRenderer, result, loadedChunks);
            }
        }

        if (_showMcnkWeakCorners && (_mcnkOverlayFlags & McnkOverlayFlags.Impassable) != 0)
            _lastMcnkWeakCornerCount = BatchMcnkWeakCornerMarkers(overlayRenderer, loadedChunks);
    }

    private void CollectMcnkFlagOverlayGeometry(
        TerrainRenderer renderer,
        Terrain.BoundingBoxRenderer overlayRenderer,
        TileLoadResult result,
        Dictionary<(int globalChunkX, int globalChunkY), McnkOverlayChunkInfo> loadedChunks)
    {
        for (int i = 0; i < result.Chunks.Count; i++)
        {
            TerrainChunkData chunk = result.Chunks[i];
            if (!renderer.TryGetChunkInfo(chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY, out TerrainRenderer.TerrainChunkInfo info))
                continue;

            var rawFlags = (McnkOverlayFlags)(uint)chunk.McnkFlags;
            loadedChunks[(chunk.TileX * 16 + chunk.ChunkX, chunk.TileY * 16 + chunk.ChunkY)] = new McnkOverlayChunkInfo(
                chunk.TileX,
                chunk.TileY,
                chunk.ChunkX,
                chunk.ChunkY,
                chunk.WorldPosition,
                info.BoundsMin,
                info.BoundsMax,
                rawFlags);

            McnkOverlayFlags matchedFlags = rawFlags & _mcnkOverlayFlags;
            if (matchedFlags == McnkOverlayFlags.None)
                continue;

            Vector3 color = ResolveMcnkOverlayColor(matchedFlags);
            float alpha = (matchedFlags & McnkOverlayFlags.Impassable) != 0 ? 0.26f : 0.18f;
            BatchChunkTopFaceOverlay(overlayRenderer, chunk, color, alpha);
            BatchChunkTopOutline(overlayRenderer, chunk, Vector3.Clamp(color * 1.18f, Vector3.Zero, Vector3.One));
            _lastMcnkOverlayChunkCount++;
        }
    }

    private int BatchMcnkWeakCornerMarkers(
        Terrain.BoundingBoxRenderer overlayRenderer,
        IReadOnlyDictionary<(int globalChunkX, int globalChunkY), McnkOverlayChunkInfo> loadedChunks)
    {
        int weakCornerCount = 0;
        float chunkStep = WoWConstants.ChunkSize / 16f;
        Vector3 weakCornerColor = new(1.0f, 0.96f, 0.18f);

        foreach (var pair in loadedChunks)
        {
            var anchorKey = pair.Key;
            McnkOverlayChunkInfo c00 = pair.Value;
            if (!loadedChunks.TryGetValue((anchorKey.globalChunkX + 1, anchorKey.globalChunkY), out McnkOverlayChunkInfo c10)
                || !loadedChunks.TryGetValue((anchorKey.globalChunkX, anchorKey.globalChunkY + 1), out McnkOverlayChunkInfo c01)
                || !loadedChunks.TryGetValue((anchorKey.globalChunkX + 1, anchorKey.globalChunkY + 1), out McnkOverlayChunkInfo c11))
            {
                continue;
            }

            bool c00Impassable = (c00.Flags & McnkOverlayFlags.Impassable) != 0;
            bool c10Impassable = (c10.Flags & McnkOverlayFlags.Impassable) != 0;
            bool c01Impassable = (c01.Flags & McnkOverlayFlags.Impassable) != 0;
            bool c11Impassable = (c11.Flags & McnkOverlayFlags.Impassable) != 0;

            bool weakDiagonalA = c00Impassable && c11Impassable && !c10Impassable && !c01Impassable;
            bool weakDiagonalB = c10Impassable && c01Impassable && !c00Impassable && !c11Impassable;
            if (!weakDiagonalA && !weakDiagonalB)
                continue;

            float markerZ = MathF.Max(MathF.Max(c00.BoundsMax.Z, c10.BoundsMax.Z), MathF.Max(c01.BoundsMax.Z, c11.BoundsMax.Z)) + 2.5f;
            Vector3 markerPosition = new(c00.WorldPosition.X - chunkStep, c00.WorldPosition.Y - chunkStep, markerZ);
            overlayRenderer.BatchPin(markerPosition, 6.0f, 0.85f, weakCornerColor);
            weakCornerCount++;
        }

        return weakCornerCount;
    }

    private static void BatchChunkTopFaceOverlay(Terrain.BoundingBoxRenderer overlayRenderer, TerrainChunkData chunk, Vector3 color, float alpha)
    {
        int[] indices = TerrainChunkMath.BuildChunkIndices(chunk.HoleMask);
        if (indices.Length < 3)
            return;

        const float surfaceLift = 0.18f;
        var positions = new Vector3[145];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = TerrainChunkMath.GetChunkVertexWorldPosition(chunk, chunk.Heights, i) + new Vector3(0f, 0f, surfaceLift);

        for (int t = 0; t + 2 < indices.Length; t += 3)
        {
            overlayRenderer.BatchTriangle(
                positions[indices[t + 0]],
                positions[indices[t + 1]],
                positions[indices[t + 2]],
                color,
                alpha);
        }
    }

    private static void BatchChunkTopOutline(Terrain.BoundingBoxRenderer overlayRenderer, TerrainChunkData chunk, Vector3 color)
    {
        const float outlineLift = 0.24f;
        var positions = new Vector3[145];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = TerrainChunkMath.GetChunkVertexWorldPosition(chunk, chunk.Heights, i) + new Vector3(0f, 0f, outlineLift);

        for (int outerCol = 0; outerCol < 8; outerCol++)
            overlayRenderer.BatchLine(positions[TerrainChunkMath.OuterIndex(0, outerCol)], positions[TerrainChunkMath.OuterIndex(0, outerCol + 1)], color);

        for (int outerRow = 0; outerRow < 8; outerRow++)
            overlayRenderer.BatchLine(positions[TerrainChunkMath.OuterIndex(outerRow, 8)], positions[TerrainChunkMath.OuterIndex(outerRow + 1, 8)], color);

        for (int outerCol = 8; outerCol > 0; outerCol--)
            overlayRenderer.BatchLine(positions[TerrainChunkMath.OuterIndex(8, outerCol)], positions[TerrainChunkMath.OuterIndex(8, outerCol - 1)], color);

        for (int outerRow = 8; outerRow > 0; outerRow--)
            overlayRenderer.BatchLine(positions[TerrainChunkMath.OuterIndex(outerRow, 0)], positions[TerrainChunkMath.OuterIndex(outerRow - 1, 0)], color);
    }

    private static Vector3 ResolveMcnkOverlayColor(McnkOverlayFlags flags)
    {
        Vector3 sum = Vector3.Zero;
        int count = 0;

        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.HasShadows, new Vector3(0.62f, 0.36f, 0.82f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.Impassable, new Vector3(0.98f, 0.22f, 0.18f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.River, new Vector3(0.18f, 0.84f, 0.96f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.Ocean, new Vector3(0.12f, 0.42f, 0.95f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.HasMagma, new Vector3(1.0f, 0.48f, 0.08f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.HasSlime, new Vector3(0.36f, 0.95f, 0.28f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.HasMccv, new Vector3(0.95f, 0.86f, 0.18f));
        AddColorIfPresent(ref sum, ref count, flags, McnkOverlayFlags.HasBakedShadows, new Vector3(0.78f, 0.78f, 0.78f));

        return count > 0
            ? Vector3.Clamp(sum / count, Vector3.Zero, Vector3.One)
            : new Vector3(1.0f, 1.0f, 1.0f);
    }

    private static void AddColorIfPresent(ref Vector3 sum, ref int count, McnkOverlayFlags activeFlags, McnkOverlayFlags flag, Vector3 color)
    {
        if ((activeFlags & flag) == 0)
            return;

        sum += color;
        count++;
    }
}
