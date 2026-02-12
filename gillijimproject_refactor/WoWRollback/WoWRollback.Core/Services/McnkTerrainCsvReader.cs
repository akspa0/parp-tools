using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Reads MCNK terrain CSV files from AlphaWDTAnalysisTool
/// </summary>
public static class McnkTerrainCsvReader
{
    public static List<McnkTerrainEntry> ReadCsv(string csvPath)
    {
        var entries = new List<McnkTerrainEntry>();

        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"Warning: MCNK terrain CSV not found: {csvPath}");
            return entries;
        }

        using var reader = new StreamReader(csvPath);
        reader.ReadLine(); // Skip header

        string? line;
        int lineNum = 1;
        while ((line = reader.ReadLine()) != null)
        {
            lineNum++;
            try
            {
                var parts = SplitCsvLine(line);
                if (parts.Length < 23)
                {
                    Console.WriteLine($"Warning: Line {lineNum} has {parts.Length} columns, expected 23");
                    continue;
                }

                entries.Add(new McnkTerrainEntry(
                    Map: parts[0],
                    TileRow: int.Parse(parts[1]),
                    TileCol: int.Parse(parts[2]),
                    ChunkRow: int.Parse(parts[3]),
                    ChunkCol: int.Parse(parts[4]),
                    FlagsRaw: ParseHex(parts[5]),
                    HasMcsh: ParseBool(parts[6]),
                    Impassible: ParseBool(parts[7]),
                    LqRiver: ParseBool(parts[8]),
                    LqOcean: ParseBool(parts[9]),
                    LqMagma: ParseBool(parts[10]),
                    LqSlime: ParseBool(parts[11]),
                    HasMccv: ParseBool(parts[12]),
                    HighResHoles: ParseBool(parts[13]),
                    AreaId: int.Parse(parts[14]),
                    NumLayers: int.Parse(parts[15]),
                    HasHoles: ParseBool(parts[16]),
                    HoleType: parts[17],
                    HoleBitmapHex: parts[18],
                    HoleCount: int.Parse(parts[19]),
                    PositionX: float.Parse(parts[20], CultureInfo.InvariantCulture),
                    PositionY: float.Parse(parts[21], CultureInfo.InvariantCulture),
                    PositionZ: float.Parse(parts[22], CultureInfo.InvariantCulture)
                ));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to parse line {lineNum}: {ex.Message}");
            }
        }

        Console.WriteLine($"Loaded {entries.Count} terrain chunks from {Path.GetFileName(csvPath)}");
        return entries;
    }

    private static string[] SplitCsvLine(string line)
    {
        var parts = new List<string>();
        var current = new System.Text.StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];

            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                parts.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }

        parts.Add(current.ToString());
        return parts.ToArray();
    }

    private static uint ParseHex(string hex)
    {
        if (hex.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            hex = hex.Substring(2);
        return Convert.ToUInt32(hex, 16);
    }

    private static bool ParseBool(string value)
    {
        return string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
    }
}
