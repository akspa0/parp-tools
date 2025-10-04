using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Writes MCNK terrain data to CSV format
/// </summary>
public sealed class McnkTerrainCsvWriter
{
    public static void WriteCsv(List<McnkTerrainEntry> entries, string outputPath)
    {
        if (entries.Count == 0)
        {
            Console.WriteLine($"[McnkTerrainCsvWriter] No terrain entries to write");
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        using var writer = new StreamWriter(outputPath);
        
        // Write header (23 columns)
        writer.WriteLine(
            "map,tile_row,tile_col,chunk_row,chunk_col," +
            "flags_raw,has_mcsh,impassible,lq_river,lq_ocean,lq_magma,lq_slime," +
            "has_mccv,high_res_holes,areaid,num_layers," +
            "has_holes,hole_type,hole_bitmap_hex,hole_count," +
            "position_x,position_y,position_z"
        );

        // Write data rows
        foreach (var entry in entries)
        {
            writer.WriteLine(string.Join(",",
                Csv(entry.Map),
                entry.TileRow,
                entry.TileCol,
                entry.ChunkRow,
                entry.ChunkCol,
                $"0x{entry.FlagsRaw:X8}",
                Csv(entry.HasMcsh),
                Csv(entry.Impassible),
                Csv(entry.LqRiver),
                Csv(entry.LqOcean),
                Csv(entry.LqMagma),
                Csv(entry.LqSlime),
                Csv(entry.HasMccv),
                Csv(entry.HighResHoles),
                entry.AreaId,
                entry.NumLayers,
                Csv(entry.HasHoles),
                entry.HoleType,
                entry.HoleBitmapHex,
                entry.HoleCount,
                entry.PositionX.ToString("F2", CultureInfo.InvariantCulture),
                entry.PositionY.ToString("F2", CultureInfo.InvariantCulture),
                entry.PositionZ.ToString("F2", CultureInfo.InvariantCulture)
            ));
        }

        Console.WriteLine($"[McnkTerrainCsvWriter] Wrote {entries.Count} entries to {Path.GetFileName(outputPath)}");
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
