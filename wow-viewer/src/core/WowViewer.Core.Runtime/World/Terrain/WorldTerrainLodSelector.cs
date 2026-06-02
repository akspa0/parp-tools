namespace WowViewer.Core.Runtime.World.Terrain;

public enum WorldTerrainLodLevel
{
    FullDetail,
    FadeToBaseLayer,
    BaseLayerOnly,
    LowDetail,
}

public readonly record struct WorldTerrainLodSelection(
    WorldTerrainLodLevel Level,
    int ActiveTextureLayerCount,
    float OverlayFadeFactor,
    int RenderableCellCount,
    bool UsesLowDetailMesh);

public static class WorldTerrainLodSelector
{
    public const float OverlayFadeDistance = 256.0f;

    public static WorldTerrainLodSelection Select(
        WorldTerrainChunkData chunk,
        float distance,
        float textureLodDistance,
        float fogEndDistance)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (float.IsNaN(distance) || float.IsInfinity(distance) || distance < 0f)
            throw new ArgumentOutOfRangeException(nameof(distance));

        int renderableCellCount = chunk.CellGrid.Cells.Count(static cell => !cell.IsHoled);
        int layerCount = Math.Max(0, chunk.LayerCount);
        int baseLayerCount = layerCount > 0 ? 1 : 0;

        if (fogEndDistance > 0f && distance >= fogEndDistance)
        {
            return new WorldTerrainLodSelection(
                WorldTerrainLodLevel.LowDetail,
                ActiveTextureLayerCount: 0,
                OverlayFadeFactor: 0f,
                RenderableCellCount: renderableCellCount,
                UsesLowDetailMesh: true);
        }

        if (layerCount <= 1)
        {
            return new WorldTerrainLodSelection(
                WorldTerrainLodLevel.FullDetail,
                ActiveTextureLayerCount: layerCount,
                OverlayFadeFactor: 1f,
                RenderableCellCount: renderableCellCount,
                UsesLowDetailMesh: false);
        }

        float lodStart = Math.Max(0f, textureLodDistance);
        float lodEnd = lodStart + OverlayFadeDistance;

        if (distance >= lodEnd)
        {
            return new WorldTerrainLodSelection(
                WorldTerrainLodLevel.BaseLayerOnly,
                ActiveTextureLayerCount: baseLayerCount,
                OverlayFadeFactor: 0f,
                RenderableCellCount: renderableCellCount,
                UsesLowDetailMesh: false);
        }

        if (distance >= lodStart)
        {
            float overlayFadeFactor = 1f - ((distance - lodStart) / OverlayFadeDistance);
            overlayFadeFactor = Math.Clamp(overlayFadeFactor, 0f, 1f);

            return new WorldTerrainLodSelection(
                WorldTerrainLodLevel.FadeToBaseLayer,
                ActiveTextureLayerCount: layerCount,
                OverlayFadeFactor: overlayFadeFactor,
                RenderableCellCount: renderableCellCount,
                UsesLowDetailMesh: false);
        }

        return new WorldTerrainLodSelection(
            WorldTerrainLodLevel.FullDetail,
            ActiveTextureLayerCount: layerCount,
            OverlayFadeFactor: 1f,
            RenderableCellCount: renderableCellCount,
            UsesLowDetailMesh: false);
    }
}
