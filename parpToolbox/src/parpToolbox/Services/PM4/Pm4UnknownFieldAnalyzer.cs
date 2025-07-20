using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Performs correlation and distribution analysis for unknown fields in PM4 chunks
/// (currently MPRL and MSLK) to uncover hidden relationships such as index or reference
/// mappings.  Produces CSV reports into a dedicated output directory so that researchers
/// can inspect pivot tables in external tools.
/// </summary>
public static class Pm4UnknownFieldAnalyzer
{
    /// <summary>
    /// Runs the analysis and writes multiple CSV reports beneath
    /// <c>outputDir/unknown_field_analysis</c>.
    /// </summary>
    public static void AnalyzeUnknownFields(Pm4Scene scene, string outputDir)
    {
        var analysisDir = Path.Combine(outputDir, "unknown_field_analysis");
        Directory.CreateDirectory(analysisDir);

        ConsoleLogger.WriteLine($"[UnknownFieldAnalyzer] Writing results to {analysisDir}");

        WriteMprlDistributions(scene, analysisDir);
        WriteMslkDistributions(scene, analysisDir);
        WriteMprlToMslkCorrelation(scene, analysisDir);
        WriteSummary(scene, analysisDir);
        WriteOrphanReports(scene, analysisDir);
        WritePlacementIndexCorrelation(scene, analysisDir);
        WriteCombinedIndexCorrelation(scene, analysisDir);
        WriteMprlUnknown4ByteDistribution(scene, analysisDir);
        WriteMslkFlagDistributions(scene, analysisDir);
        WriteMsurDistributions(scene, analysisDir);
        WriteMsurRawDump(scene, analysisDir);

        ConsoleLogger.WriteLine("[UnknownFieldAnalyzer] Analysis complete!");
    }

    private static void WriteMprlDistributions(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "mprl_unknown_distribution.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("Field,Value,Count");

        WriteDistribution(writer, "Unknown0", scene.Placements.Select(p => p.Unknown0));
        WriteDistribution(writer, "Unknown2", scene.Placements.Select(p => (uint)p.Unknown2));
        WriteDistribution(writer, "Unknown4", scene.Placements.Select(p => p.Unknown4));
        WriteDistribution(writer, "Unknown6", scene.Placements.Select(p => p.Unknown6));
        WriteDistribution(writer, "Unknown14", scene.Placements.Select(p => (uint)p.Unknown14));
        WriteDistribution(writer, "Unknown16", scene.Placements.Select(p => p.Unknown16));

        ConsoleLogger.WriteLine("  Wrote MPRL distribution");
    }

    private static void WriteMslkDistributions(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "mslk_unknown_distribution.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("Field,Value,Count");

        WriteDistribution(writer, "Unknown_0x00", scene.Links.Select(l => l.Unknown_0x00));
        WriteDistribution(writer, "Unknown_0x01", scene.Links.Select(l => l.Unknown_0x01));
        WriteDistribution(writer, "Unknown_0x02", scene.Links.Select(l => l.Unknown_0x02));
        WriteDistribution(writer, "ParentIndex", scene.Links.Select(l => (uint)l.ParentIndex));
        WriteDistribution(writer, "ReferenceIndex", scene.Links.Select(l => (uint)l.ReferenceIndex));
        WriteDistribution(writer, "ReferenceIndexHigh", scene.Links.Select(l => (uint)l.ReferenceIndexHigh));
        WriteDistribution(writer, "ReferenceIndexLow", scene.Links.Select(l => (uint)l.ReferenceIndexLow));

        ConsoleLogger.WriteLine("  Wrote MSLK distribution");
    }

    /// <summary>
    /// Correlates MPRL.Unknown4 & Unknown14 to MSLK.ParentIndex / ReferenceIndex
    /// and outputs counts and matching row indices so we can visually confirm relationships.
    /// </summary>
    private static void WriteMprlToMslkCorrelation(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "mprl_to_mslk_correlation.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("MPRL_Index,MPRL.Unknown4,Matched_MSLK_Indices_Count,Matched_MSLK_Indices,Unknown14,MatchedRefIndex_Count");

        // Build lookup on ParentIndex for quick join
        var parentLookup = scene.Links
            .Select((l, idx) => new { l.ParentIndex, Index = idx })
            .GroupBy(x => x.ParentIndex)
            .ToDictionary(g => g.Key, g => g.Select(x => x.Index).ToList());

        var refLookup = scene.Links
            .Select((l, idx) => new { l.ReferenceIndex, Index = idx })
            .GroupBy(x => x.ReferenceIndex)
            .ToDictionary(g => g.Key, g => g.Select(x => x.Index).ToList());

        for (var i = 0; i < scene.Placements.Count; i++)
        {
            var pl = scene.Placements[i];
            parentLookup.TryGetValue((uint)pl.Unknown4, out var mslkMatches);
            refLookup.TryGetValue((ushort)pl.Unknown14, out var mslkRefMatches);

            writer.WriteLine($"{i},{pl.Unknown4},{mslkMatches?.Count ?? 0},\"{string.Join(';', mslkMatches ?? new List<int>())}\",{pl.Unknown14},{mslkRefMatches?.Count ?? 0}");
        }

        ConsoleLogger.WriteLine("  Wrote MPRL→MSLK correlation");
    }

    private static void WriteSummary(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "summary.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("Field,TotalValues,ValuesWithMatches,MatchRatePct");

        // Build lookups reused from correlation logic
        var parentSet = scene.Links.Select(l => l.ParentIndex).ToHashSet();
        var refSet = scene.Links.Select(l => (ushort)l.ReferenceIndex).ToHashSet();

        int totalUnknown4 = scene.Placements.Count;
        int matchedUnknown4 = scene.Placements.Count(p => parentSet.Contains((uint)p.Unknown4));
        writer.WriteLine($"Unknown4,{totalUnknown4},{matchedUnknown4},{matchedUnknown4 * 100.0 / totalUnknown4:0.##}");

        int totalUnknown14 = scene.Placements.Count;
        int matchedUnknown14 = scene.Placements.Count(p => refSet.Contains((ushort)p.Unknown14));
        writer.WriteLine($"Unknown14,{totalUnknown14},{matchedUnknown14},{matchedUnknown14 * 100.0 / totalUnknown14:0.##}");

        int totalParentIndex = scene.Links.Count;
        var placementUnknown4Set = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
        int matchedParent = scene.Links.Count(l => placementUnknown4Set.Contains(l.ParentIndex));
        writer.WriteLine($"ParentIndex,{totalParentIndex},{matchedParent},{matchedParent * 100.0 / totalParentIndex:0.##}");

        int totalReferenceIndex = scene.Links.Count;
        var placementUnknown14Set = scene.Placements.Select(p => (ushort)p.Unknown14).ToHashSet();
        int matchedRef = scene.Links.Count(l => placementUnknown14Set.Contains(l.ReferenceIndex));
        writer.WriteLine($"ReferenceIndex,{totalReferenceIndex},{matchedRef},{matchedRef * 100.0 / totalReferenceIndex:0.##}");

        ConsoleLogger.WriteLine("  Wrote summary.csv");
    }

    private static void WriteOrphanReports(Pm4Scene scene, string analysisDir)
    {
        // Placements whose Unknown4 has no matching ParentIndex
        var parentIndexSet = scene.Links.Select(l => l.ParentIndex).ToHashSet();
        var orphanPlacements = scene.Placements
            .Select((p, idx) => new { p.Unknown4, Index = idx })
            .Where(x => !parentIndexSet.Contains((uint)x.Unknown4))
            .ToList();
        var file1 = Path.Combine(analysisDir, "mprl_orphans.csv");
        using (var w = new StreamWriter(file1))
        {
            w.WriteLine("MPRL_Index,Unknown4");
            foreach (var o in orphanPlacements)
                w.WriteLine($"{o.Index},{o.Unknown4}");
        }
        ConsoleLogger.WriteLine($"  Wrote {orphanPlacements.Count} MPRL orphans");

        // ParentIndex values with no placement match
        var unknown4Set = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
        var orphanLinks = scene.Links
            .Select((l, idx) => new { l.ParentIndex, Index = idx })
            .Where(x => !unknown4Set.Contains(x.ParentIndex))
            .ToList();
        var file2 = Path.Combine(analysisDir, "mslk_orphans.csv");
        using (var w2 = new StreamWriter(file2))
        {
            w2.WriteLine("MSLK_Index,ParentIndex");
            foreach (var o in orphanLinks)
                w2.WriteLine($"{o.Index},{o.ParentIndex}");
        }
        ConsoleLogger.WriteLine($"  Wrote {orphanLinks.Count} MSLK orphans");
    }

    private static void WritePlacementIndexCorrelation(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "placementidx_to_mslk_parent.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("PlacementIdx,Matched_MSLK_Indices_Count,Matched_MSLK_Indices");

        var parentLookup = new Dictionary<uint, List<int>>();
        for (int i = 0; i < scene.Links.Count; i++)
        {
            uint key = scene.Links[i].ParentIndex;
            if (!parentLookup.TryGetValue(key, out var list))
            {
                list = new List<int>();
                parentLookup[key] = list;
            }
            list.Add(i);
        }

        for (int idx = 0; idx < scene.Placements.Count; idx++)
        {
            parentLookup.TryGetValue((uint)idx, out var matches);
            writer.WriteLine($"{idx},{matches?.Count ?? 0},\"{string.Join(';', matches ?? new List<int>())}\"");
        }
        ConsoleLogger.WriteLine("  Wrote placement index → MSLK.ParentIndex correlation");
    }

    private static void WriteCombinedIndexCorrelation(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "unknown14_16_to_referenceindex.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("PlacementIdx,CombinedUnknown14_16,Matched_MSLK_Count,Matched_MSLK_Indices");

        var refLookup = new Dictionary<uint, List<int>>();
        for (int i = 0; i < scene.Links.Count; i++)
        {
            uint combined = ((uint)scene.Links[i].ReferenceIndexHigh << 16) | scene.Links[i].ReferenceIndexLow;
            if (!refLookup.TryGetValue(combined, out var list))
            {
                list = new List<int>();
                refLookup[combined] = list;
            }
            list.Add(i);
        }

        for (int idx = 0; idx < scene.Placements.Count; idx++)
        {
            var pl = scene.Placements[idx];
            uint combinedPl = ((uint)pl.Unknown16 << 16) | (ushort)pl.Unknown14;
            refLookup.TryGetValue(combinedPl, out var matches);
            writer.WriteLine($"{idx},{combinedPl},{matches?.Count ?? 0},\"{string.Join(';', matches ?? new List<int>())}\"");
        }
        ConsoleLogger.WriteLine("  Wrote Unknown14/16 ↔ ReferenceIndex correlation");
    }

    private static void WriteMsurDistributions(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "msur_distribution.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("Field,Value,Count");

        void Dump<T>(string field, IEnumerable<T> vals)
        {
            var dist = vals.GroupBy(v => v)
                           .Select(g => new { Val = g.Key, Count = g.Count() })
                           .OrderByDescending(x => x.Count);
            foreach (var d in dist)
                writer.WriteLine($"{field},{d.Val},{d.Count}");
        }

        Dump("SurfaceGroupKey", scene.Surfaces.Select(s => s.SurfaceGroupKey));
        Dump("IndexCount", scene.Surfaces.Select(s => s.IndexCount));
        Dump("SurfaceAttributeMask", scene.Surfaces.Select(s => s.SurfaceAttributeMask));
        Dump("MsviFirstIndex", scene.Surfaces.Select(s => s.MsviFirstIndex));
        Dump("MdosIndex", scene.Surfaces.Select(s => s.MdosIndex));
        Dump("SurfaceKey_High16", scene.Surfaces.Select(s => s.SurfaceKey >> 16));
        Dump("SurfaceKey_Low16", scene.Surfaces.Select(s => s.SurfaceKey & 0xFFFF));

        ConsoleLogger.WriteLine("  Wrote MSUR distribution");
    }

    private static void WriteMsurRawDump(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "msur_raw_dump.csv");
        using var w = new StreamWriter(file);
        w.WriteLine("Row,SurfaceGroupKey,IndexCount,SurfaceAttributeMask,Nx,Ny,Nz,Height,MsviFirstIndex,MdosIndex,SurfaceKey");
        int row = 0;
        foreach (var s in scene.Surfaces)
        {
            w.WriteLine($"{row},{s.SurfaceGroupKey},{s.IndexCount},{s.SurfaceAttributeMask},{s.Nx},{s.Ny},{s.Nz},{s.Height},{s.MsviFirstIndex},{s.MdosIndex},{s.SurfaceKey}");
            row++;
        }
        ConsoleLogger.WriteLine($"  Wrote MSUR raw dump ({row} rows)");
    }

    private static void WriteMslkFlagDistributions(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "mslk_flags_distribution.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("Field,Value,Count");

        void Dump<T>(string field, IEnumerable<T> vals)
        {
            var dist = vals.GroupBy(v => v)
                           .Select(g => new { Val = g.Key, Count = g.Count() })
                           .OrderByDescending(x => x.Count);
            foreach (var d in dist)
                writer.WriteLine($"{field},{d.Val},{d.Count}");
        }

        Dump("Unknown_0x00", scene.Links.Select(l => l.Unknown_0x00));
        Dump("Unknown_0x01", scene.Links.Select(l => l.Unknown_0x01));
        Dump("Unknown_0x02", scene.Links.Select(l => l.Unknown_0x02));
        Dump("MspiIndexCount", scene.Links.Select(l => l.MspiIndexCount));

        ConsoleLogger.WriteLine("  Wrote MSLK flag distributions");
    }

    private static void WriteMprlUnknown4ByteDistribution(Pm4Scene scene, string analysisDir)
    {
        var file = Path.Combine(analysisDir, "mprl_unknown4_bytes_distribution.csv");
        using var writer = new StreamWriter(file);
        writer.WriteLine("BytePosition,ByteValue,Count");

        // High byte of Unknown4 (bits 15-8)
        var highDist = scene.Placements
            .GroupBy(p => (byte)(p.Unknown4 >> 8))
            .Select(g => new { Byte = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count);
        foreach (var item in highDist)
            writer.WriteLine($"High,{item.Byte},{item.Count}");

        // Low byte of Unknown4 (bits 7-0)
        var lowDist = scene.Placements
            .GroupBy(p => (byte)(p.Unknown4 & 0xFF))
            .Select(g => new { Byte = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count);
        foreach (var item in lowDist)
            writer.WriteLine($"Low,{item.Byte},{item.Count}");

        ConsoleLogger.WriteLine("  Wrote unknown4 byte distributions");
    }

    private static void WriteDistribution<T>(StreamWriter writer, string fieldName, IEnumerable<T> values)
    {
        var distribution = values
            .GroupBy(v => v)
            .Select(g => new { Value = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .ThenBy(x => x.Value);

        foreach (var entry in distribution)
        {
            writer.WriteLine($"{fieldName},{entry.Value},{entry.Count}");
        }
    }
}
