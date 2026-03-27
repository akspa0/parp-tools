namespace WowViewer.Core.Maps;

public sealed class WdtSummary
{
    public WdtSummary(
        string sourcePath,
        bool isWmoBased,
        int tilesWithData,
        int totalTiles,
        int mainCellSizeBytes,
        int doodadNameCount,
        int worldModelNameCount,
        int doodadPlacementCount,
        int worldModelPlacementCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(tilesWithData);
        ArgumentOutOfRangeException.ThrowIfNegative(totalTiles);
        ArgumentOutOfRangeException.ThrowIfNegative(mainCellSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(worldModelNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadPlacementCount);
        ArgumentOutOfRangeException.ThrowIfNegative(worldModelPlacementCount);

        SourcePath = sourcePath;
        IsWmoBased = isWmoBased;
        TilesWithData = tilesWithData;
        TotalTiles = totalTiles;
        MainCellSizeBytes = mainCellSizeBytes;
        DoodadNameCount = doodadNameCount;
        WorldModelNameCount = worldModelNameCount;
        DoodadPlacementCount = doodadPlacementCount;
        WorldModelPlacementCount = worldModelPlacementCount;
    }

    public string SourcePath { get; }

    public bool IsWmoBased { get; }

    public int TilesWithData { get; }

    public int TotalTiles { get; }

    public int MainCellSizeBytes { get; }

    public int DoodadNameCount { get; }

    public int WorldModelNameCount { get; }

    public int DoodadPlacementCount { get; }

    public int WorldModelPlacementCount { get; }
}