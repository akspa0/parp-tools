using System;
using System.IO;

namespace WoWRollback.LkToAlphaModule.Utils;

public static class OutputPathResolver
{
    public static string GetDefaultRoot(string mapName, DateTime? now = null)
    {
        var t = (now ?? DateTime.Now).ToString("yyyyMMdd_HHmmss");
        return Path.Combine("project_output", $"{mapName}_{t}");
    }

    public static string GetTileOutPath(string root, string mapName, int tileX, int tileY)
    {
        var dir = Path.Combine(root, "World", "Maps", mapName);
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, $"{mapName}_{tileY:D2}_{tileX:D2}.adt");
    }
}
