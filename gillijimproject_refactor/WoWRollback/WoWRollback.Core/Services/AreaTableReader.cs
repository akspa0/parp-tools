using System;
using System.Collections.Generic;
using System.IO;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Reads AreaTable CSV files from AlphaWDTAnalysisTool
/// Files: AreaTable_Alpha.csv, AreaTable_335.csv
/// </summary>
public static class AreaTableReader
{
    public static Dictionary<int, string> ReadAreaTableCsv(string csvPath)
    {
        var areas = new Dictionary<int, string>();

        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"Warning: AreaTable CSV not found: {csvPath}");
            return areas;
        }

        using var reader = new StreamReader(csvPath);
        reader.ReadLine(); // Skip header

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            // Split on first comma only (area names may contain commas)
            var commaIndex = line.IndexOf(',');
            if (commaIndex < 0) continue;

            var idPart = line.Substring(0, commaIndex);
            var namePart = line.Substring(commaIndex + 1);

            if (int.TryParse(idPart, out int id))
            {
                // Remove CSV quotes if present
                string name = namePart.Trim('"');
                areas[id] = name;
            }
        }

        Console.WriteLine($"Loaded {areas.Count} area names from {Path.GetFileName(csvPath)}");
        return areas;
    }

    public static AreaTableLookup LoadForVersion(string versionRoot)
    {
        // Try to load both Alpha and LK AreaTables
        var alphaPath = Path.Combine(versionRoot, "AreaTable_Alpha.csv");
        var lkPath = Path.Combine(versionRoot, "AreaTable_335.csv");

        var alphaAreas = ReadAreaTableCsv(alphaPath);
        var lkAreas = ReadAreaTableCsv(lkPath);

        return new AreaTableLookup(alphaAreas, lkAreas);
    }
}
