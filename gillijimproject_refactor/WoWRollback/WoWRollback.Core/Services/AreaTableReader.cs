using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
        var header = reader.ReadLine(); // Read header: row_key,id,parent,continentId,name
        if (string.IsNullOrEmpty(header)) return areas;

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            // CSV format: row_key,id,parent,continentId,name
            // Example: 1,249,589824,0,Dreadmaul Rock
            var parts = line.Split(',');
            if (parts.Length < 5) continue;

            // Column 1 is the actual ID (column 0 is row_key)
            if (!int.TryParse(parts[1], out int id)) continue;

            // Column 4 is the name (may contain commas, so join the rest)
            string name = parts.Length > 5 
                ? string.Join(",", parts.Skip(4)) 
                : parts[4];
            
            name = name.Trim().Trim('"');
            
            if (!string.IsNullOrEmpty(name))
            {
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
