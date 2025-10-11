using System;
using System.Collections.Generic;
using System.IO;

namespace WoWRollback.Orchestrator;

internal static class LooseAdtMaterializer
{
    // Build a merged on-disk view under session.Paths.AdtDir/<version>/tree/World/Maps by:
    // 1) Copying overlay files from overlayRoot (if present)
    // 2) Filling any missing files from baseAlphaRoot (no MPQ patching)
    // Supports overlays shaped as either <root>/World/Maps/... or <root>/tree/World/Maps/...
    public static void Materialize(SessionContext session, string overlayRoot, string baseAlphaRoot, IReadOnlyList<string> versions, IReadOnlyList<string> maps)
    {
        bool HasOverlay = !string.IsNullOrWhiteSpace(overlayRoot) && Directory.Exists(overlayRoot);

        foreach (var version in versions)
        {
            var destTree = Path.Combine(session.Paths.AdtDir, version, "tree");
            var destMapsRoot = Path.Combine(destTree, "World", "Maps");
            Directory.CreateDirectory(destMapsRoot);

            foreach (var map in maps)
            {
                var dstMap = Path.Combine(destMapsRoot, map);
                Directory.CreateDirectory(dstMap);

                // 1) Overlay copy first
                if (HasOverlay)
                {
                    var srcMap1 = Path.Combine(overlayRoot, "World", "Maps", map);
                    var srcMap2 = Path.Combine(overlayRoot, "tree", "World", "Maps", map);
                    var srcMap = Directory.Exists(srcMap1) ? srcMap1 : (Directory.Exists(srcMap2) ? srcMap2 : null);
                    if (srcMap is not null)
                    {
                        foreach (var file in Directory.EnumerateFiles(srcMap, "*.*", SearchOption.TopDirectoryOnly))
                        {
                            var ext = Path.GetExtension(file);
                            if (!ext.Equals(".adt", StringComparison.OrdinalIgnoreCase) &&
                                !ext.Equals(".wdt", StringComparison.OrdinalIgnoreCase)) continue;
                            var destPath = Path.Combine(dstMap, Path.GetFileName(file));
                            File.Copy(file, destPath, overwrite: true);
                        }
                    }
                }

                // 2) Fill remaining from base alpha root
                var baseMapDir = Path.Combine(baseAlphaRoot, version, "tree", "World", "Maps", map);
                if (Directory.Exists(baseMapDir))
                {
                    foreach (var file in Directory.EnumerateFiles(baseMapDir, "*.*", SearchOption.TopDirectoryOnly))
                    {
                        var name = Path.GetFileName(file);
                        var ext = Path.GetExtension(name);
                        if (!ext.Equals(".adt", StringComparison.OrdinalIgnoreCase) &&
                            !ext.Equals(".wdt", StringComparison.OrdinalIgnoreCase)) continue;
                        var destPath = Path.Combine(dstMap, name);
                        if (!File.Exists(destPath))
                        {
                            File.Copy(file, destPath, overwrite: false);
                        }
                    }
                }
            }
        }
    }
}
