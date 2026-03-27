namespace WowViewer.Core.IO.Files;

public static class MinimapService
{
    public const int TileSize = 256;

    public static string GetMinimapTilePath(string mapName, int x, int y)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        return $"textures/minimap/{mapName.ToLowerInvariant()}/map{x:D2}_{y:D2}.blp";
    }

    public static bool MinimapTileExists(string basePath, string mapName, int x, int y)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(basePath);
        return File.Exists(Path.Combine(basePath, GetMinimapTilePath(mapName, x, y)));
    }
}