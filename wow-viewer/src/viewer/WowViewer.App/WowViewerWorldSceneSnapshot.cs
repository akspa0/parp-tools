using WowViewer.Core.Maps;

namespace WowViewer.App;

internal sealed class WowViewerWorldSceneSnapshot
{
    public static readonly WowViewerWorldSceneSnapshot Empty = new(
        clientRoot: string.Empty,
        requestedMapInput: string.Empty,
        resolvedMapDirectory: string.Empty,
        loadDuration: TimeSpan.Zero,
        occupiedTiles: Array.Empty<WdtTileCoordinate>(),
        hasSelectedTile: false,
        selectedTileX: -1,
        selectedTileY: -1,
        activeTerrainTileCount: 0,
        placementSourcePath: string.Empty,
        terrainVisualWidth: 0,
        terrainVisualHeight: 0);

    public WowViewerWorldSceneSnapshot(
        string clientRoot,
        string requestedMapInput,
        string resolvedMapDirectory,
        TimeSpan loadDuration,
        IReadOnlyList<WdtTileCoordinate> occupiedTiles,
        bool hasSelectedTile,
        int selectedTileX,
        int selectedTileY,
        int activeTerrainTileCount,
        string placementSourcePath,
        int terrainVisualWidth,
        int terrainVisualHeight)
    {
        ClientRoot = clientRoot;
        RequestedMapInput = requestedMapInput;
        ResolvedMapDirectory = resolvedMapDirectory;
        LoadDuration = loadDuration;
        OccupiedTiles = occupiedTiles;
        HasSelectedTile = hasSelectedTile;
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        ActiveTerrainTileCount = activeTerrainTileCount;
        PlacementSourcePath = placementSourcePath;
        TerrainVisualWidth = terrainVisualWidth;
        TerrainVisualHeight = terrainVisualHeight;
    }

    public string ClientRoot { get; }

    public string RequestedMapInput { get; }

    public string ResolvedMapDirectory { get; }

    public TimeSpan LoadDuration { get; }

    public IReadOnlyList<WdtTileCoordinate> OccupiedTiles { get; }

    public bool HasSelectedTile { get; }

    public int SelectedTileX { get; }

    public int SelectedTileY { get; }

    public int ActiveTerrainTileCount { get; }

    public string PlacementSourcePath { get; }

    public int TerrainVisualWidth { get; }

    public int TerrainVisualHeight { get; }

    public static WowViewerWorldSceneSnapshot FromRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);
        WowViewerWorldSessionBootstrapResult session = runtimeFrame.Session;
        return new WowViewerWorldSceneSnapshot(
            session.ClientRoot,
            session.RequestedMapInput,
            session.ResolvedMapDirectory,
            session.LoadDuration,
            session.OccupiedTiles.ToArray(),
            hasSelectedTile: true,
            runtimeFrame.SelectedTileX,
            runtimeFrame.SelectedTileY,
            runtimeFrame.ActiveTerrainTiles.Count,
            runtimeFrame.PlacementSourcePath,
            runtimeFrame.TerrainVisualSnapshot.Width,
            runtimeFrame.TerrainVisualSnapshot.Height);
    }
}