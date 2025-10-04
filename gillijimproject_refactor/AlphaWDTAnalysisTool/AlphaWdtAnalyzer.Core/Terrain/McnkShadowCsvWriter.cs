using System;
using System.Collections.Generic;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Writes MCSH shadow map data to CSV format
/// </summary>
public sealed class McnkShadowCsvWriter
{
    public static void WriteCsv(List<McnkShadowEntry> entries, string outputPath)
    {
        if (entries.Count == 0)
        {
            Console.WriteLine($"[McnkShadowCsvWriter] No shadow entries to write");
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        using var writer = new StreamWriter(outputPath);

        // Write header
        writer.WriteLine("map,tile_row,tile_col,chunk_row,chunk_col,has_shadow,shadow_size,shadow_bitmap_base64");

        // Write data rows
        foreach (var entry in entries)
        {
            writer.WriteLine(string.Join(",",
                Csv(entry.Map),
                entry.TileRow,
                entry.TileCol,
                entry.ChunkRow,
                entry.ChunkCol,
                Csv(entry.HasShadow),
                entry.ShadowSize,
                Csv(entry.ShadowBitmapBase64)
            ));
        }

        int shadowCount = entries.Count(e => e.HasShadow);
        Console.WriteLine($"[McnkShadowCsvWriter] Wrote {shadowCount}/{entries.Count} shadow maps to {Path.GetFileName(outputPath)}");
    }

    private static string Csv(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        if (s.Contains('"') || s.Contains(','))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static string Csv(bool b) => b ? "true" : "false";
}
