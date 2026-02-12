using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using WoWRollback.LkToAlphaModule.Models;
using WoWRollback.LkToAlphaModule.Readers;

namespace WoWRollback.LkToAlphaModule.Writers;

public static class AlphaToLkMcseAppender
{
    // Appends MCSE chunks to existing LK root ADTs in lkDir for mapName based on Alpha WDT data.
    // For each tile with >=1 MCSE entry, append a named MCSE chunk at the end of <map>_YY_XX.adt.
    // If file is missing, skip with a warning.
    public static void AppendMcseFromAlpha(string alphaWdtPath, string lkDir, string mapName)
    {
        var alpha = AlphaWdtReader.Read(alphaWdtPath);
        int appended = 0, skipped = 0;
        foreach (var tile in alpha.Tiles)
        {
            if (tile.Mcse.Count == 0) continue;
            if (tile.FirstMcnk is null) continue; // no MCNK parsed

            // Compute YY_XX from tile.Index
            int yy = tile.Index / 64;
            int xx = tile.Index % 64;
            string adtName = $"{mapName}_{yy.ToString("D2", CultureInfo.InvariantCulture)}_{xx.ToString("D2", CultureInfo.InvariantCulture)}.adt";
            string adtPath = Path.Combine(lkDir, adtName);
            if (!File.Exists(adtPath))
            {
                // Try within World/Maps/<mapName>/
                string alt = Path.Combine(lkDir, "World", "Maps", mapName, adtName);
                if (File.Exists(alt)) adtPath = alt; else { skipped++; continue; }
            }

            // Build MCSE named chunk
            var chunk = McseWriterLk.BuildMcseChunk(tile.Mcse, prefer76Byte: true);
            var bytes = chunk.GetWholeChunk();

            // Append to end of file (LK readers scan chunks and will find MCSE)
            using (var fs = new FileStream(adtPath, FileMode.Append, FileAccess.Write, FileShare.None))
            {
                fs.Write(bytes, 0, bytes.Length);
            }

            appended++;
        }

        Console.WriteLine($"Appended MCSE to {appended} LK ADTs. Skipped {skipped} (file missing).\nSource: {alphaWdtPath}\nLK dir: {lkDir}");
    }
}
