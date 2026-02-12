using System;
using System.IO;
using WoWRollback.LkToAlphaModule.Readers;
using WoWRollback.LkToAlphaModule.Writers;
using System.Linq;

namespace WoWRollback.LkToAlphaModule;

public sealed class LkToAlphaOrchestrator
{
    public LkToAlphaConversionResult ConvertLkToAlpha(
        string wdtPath,
        string lkInputDir,
        string outDir,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        if (string.IsNullOrWhiteSpace(wdtPath)) throw new ArgumentException("WDT path is required", nameof(wdtPath));
        if (string.IsNullOrWhiteSpace(outDir)) throw new ArgumentException("Output directory is required", nameof(outDir));
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("Map name is required", nameof(mapName));

        try
        {
            Directory.CreateDirectory(outDir);

            // Read LK WDT MAIN flags
            var reader = new LkWdtReader();
            var mainFlags = reader.ReadMainTileFlags(wdtPath);

            // Write minimal Alpha WDT (MVER + MAIN)
            var wdtOutPath = Path.Combine(outDir, mapName + ".wdt");
            var writer = new AlphaWdtWriter();
            writer.WriteAlphaWdt(wdtOutPath, mainFlags);

            return new LkToAlphaConversionResult(
                AlphaOutputDirectory: outDir,
                TilesProcessed: 0,
                Success: true);
        }
        catch (Exception ex)
        {
            return new LkToAlphaConversionResult(
                AlphaOutputDirectory: outDir,
                TilesProcessed: 0,
                Success: false,
                ErrorMessage: ex.Message);
        }
    }

    

    /// <summary>
    /// Packs a monolithic Alpha WDT (header + embedded ADTs) from LK inputs (terrain-only MCNKs).
    /// </summary>
    public LkToAlphaConversionResult PackMonolithicAlphaWdt(
        string wdtPath,
        string lkMapDirOrRoot,
        string outDir,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        if (string.IsNullOrWhiteSpace(wdtPath)) throw new ArgumentException("WDT path is required", nameof(wdtPath));
        if (string.IsNullOrWhiteSpace(outDir)) throw new ArgumentException("Output directory is required", nameof(outDir));
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("Map name is required", nameof(mapName));

        try
        {
            Directory.CreateDirectory(outDir);

            // Determine LK map directory
            string lkMapDir = Directory.Exists(lkMapDirOrRoot)
                ? lkMapDirOrRoot
                : Path.Combine(lkMapDirOrRoot, "World", "Maps", mapName);
            if (!Directory.Exists(lkMapDir))
            {
                var alt = Path.Combine(Path.GetDirectoryName(wdtPath) ?? string.Empty, "World", "Maps", mapName);
                if (Directory.Exists(alt)) lkMapDir = alt;
            }
            if (!Directory.Exists(lkMapDir)) throw new DirectoryNotFoundException($"LK map directory not found: {lkMapDirOrRoot}");

            // Count tiles we will pack (root ADTs only)
            var rootAdts = Directory.EnumerateFiles(lkMapDir, mapName + "_*.adt", SearchOption.TopDirectoryOnly)
                .Where(p => !p.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !p.Contains("_tex", StringComparison.OrdinalIgnoreCase))
                .ToList();

            var outWdtPath = Path.Combine(outDir, mapName + ".wdt");
            AlphaWdtMonolithicWriter.WriteMonolithic(wdtPath, lkMapDir, outWdtPath, opts);

            return new LkToAlphaConversionResult(
                AlphaOutputDirectory: outDir,
                TilesProcessed: rootAdts.Count,
                Success: true);
        }
        catch (Exception ex)
        {
            return new LkToAlphaConversionResult(
                AlphaOutputDirectory: outDir,
                TilesProcessed: 0,
                Success: false,
                ErrorMessage: ex.Message);
        }
    }
}
