using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Service for exporting restored temporal stratigraphy and weak signal tiles to disk
/// as pre-computed loose LK ADT files or Alpha WDT map packages.
/// </summary>
public static class StratigraphyTileExporter
{
    private const int TileLatticeDim = 257;

    /// <summary>
    /// Exports a single restored LK ADT tile by applying the 257x257 height lattice to the source ADT bytes.
    /// </summary>
    public static void ExportRestoredLkAdt(
        string sourceAdtPath,
        string outputAdtPath,
        float[,] restoredHeights257,
        bool copyCompanionFiles = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceAdtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputAdtPath);
        ArgumentNullException.ThrowIfNull(restoredHeights257);

        float[] flatHeights = FlattenHeights257(restoredHeights257);
        AdtTerrainWriter.Write(sourceAdtPath, outputAdtPath, flatHeights);

        if (copyCompanionFiles)
        {
            string sourceDir = Path.GetDirectoryName(sourceAdtPath) ?? string.Empty;
            string outputDir = Path.GetDirectoryName(outputAdtPath) ?? string.Empty;
            string baseName = Path.GetFileNameWithoutExtension(sourceAdtPath);

            foreach (string suffix in new[] { "_obj0", "_tex0", "_lod" })
            {
                string companionSource = Path.Combine(sourceDir, $"{baseName}{suffix}.adt");
                string companionDest = Path.Combine(outputDir, $"{baseName}{suffix}.adt");
                if (File.Exists(companionSource) && !File.Exists(companionDest))
                {
                    File.Copy(companionSource, companionDest, overwrite: true);
                }
            }
        }
    }

    /// <summary>
    /// Flattens a 257x257 2D float array to a 66,049 1D float array.
    /// </summary>
    public static float[] FlattenHeights257(float[,] heights257)
    {
        ArgumentNullException.ThrowIfNull(heights257);
        var flat = new float[TileLatticeDim * TileLatticeDim];
        for (int y = 0; y < TileLatticeDim; y++)
            for (int x = 0; x < TileLatticeDim; x++)
                flat[y * TileLatticeDim + x] = heights257[y, x];
        return flat;
    }

    /// <summary>
    /// Expands a 66,049 1D float array to a 257x257 2D float array.
    /// </summary>
    public static float[,] ExpandHeights257(IReadOnlyList<float> flatHeights)
    {
        ArgumentNullException.ThrowIfNull(flatHeights);
        if (flatHeights.Count < TileLatticeDim * TileLatticeDim)
            throw new ArgumentException($"Flat heights array must have at least {TileLatticeDim * TileLatticeDim} elements.", nameof(flatHeights));

        var grid = new float[TileLatticeDim, TileLatticeDim];
        for (int y = 0; y < TileLatticeDim; y++)
            for (int x = 0; x < TileLatticeDim; x++)
                grid[y, x] = flatHeights[y * TileLatticeDim + x];
        return grid;
    }
}
