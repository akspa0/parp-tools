using PM4NextExporter.Model;
using System.Linq;

namespace PM4NextExporter.Services
{
    internal sealed class SceneLoader
    {
        internal Scene LoadSingleTile(string inputPath, bool includeAdjacent, bool applyMscnRemap)
        {
            // Accept either a single PM4 file or a directory.
            // When includeAdjacent is true, aggregate all tiles in the directory of the input.
            if (string.IsNullOrWhiteSpace(inputPath))
                return Scene.Empty(inputPath ?? string.Empty);

            try
            {
                if (System.IO.Directory.Exists(inputPath))
                {
                    if (includeAdjacent)
                    {
                        // Load entire region directory
                        var global = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.LoadRegion(inputPath, "*.pm4", applyMscnRemap);
                        var unifiedScene = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.ToStandardScene(global);
                        return Scene.FromPm4Scene(unifiedScene, inputPath);
                    }

                    // If a directory is provided without includeAdjacent, pick the first .pm4 file
                    var first = System.IO.Directory.GetFiles(inputPath, "*.pm4").FirstOrDefault();
                    if (first is null)
                        return Scene.Empty(inputPath);

                    var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
                    var singleScene = adapter.Load(first);
                    return Scene.FromPm4Scene(singleScene, first);
                }
                else if (System.IO.File.Exists(inputPath))
                {
                    if (includeAdjacent)
                    {
                        var dir = System.IO.Path.GetDirectoryName(inputPath) ?? ".";
                        var global = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.LoadRegion(dir, "*.pm4", applyMscnRemap);
                        var unifiedScene = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.ToStandardScene(global);
                        return Scene.FromPm4Scene(unifiedScene, dir);
                    }

                    var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
                    var singleScene = adapter.Load(inputPath);
                    return Scene.FromPm4Scene(singleScene, inputPath);
                }
            }
            catch (System.Exception ex)
            {
                // Return empty scene but keep source for logging context
                System.Console.Error.WriteLine($"[SceneLoader] Failed to load '{inputPath}': {ex.Message}");
            }

            return Scene.Empty(inputPath);
        }
    }
}
