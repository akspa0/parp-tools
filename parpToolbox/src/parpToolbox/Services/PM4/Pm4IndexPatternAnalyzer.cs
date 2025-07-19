namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

/// <summary>
/// Analyzes index patterns and high/low pair encodings in PM4 chunks to discover
/// missing vertex references and proper index decoding methods.
/// </summary>
internal static class Pm4IndexPatternAnalyzer
{
    /// <summary>
    /// Analysis result for index patterns and encoding schemes.
    /// </summary>
    public record IndexAnalysis(
        int MaxVertexIndex,
        int AvailableVertices,
        int OutOfBoundsCount,
        Dictionary<string, List<uint>> HighLowPairs,
        List<uint> SuspiciousIndices,
        Dictionary<string, IndexStatistics> ChunkIndexStats,
        string Summary
    );

    /// <summary>
    /// Statistics for indices found in a chunk.
    /// </summary>
    public record IndexStatistics(
        uint MinIndex,
        uint MaxIndex,
        int TotalIndices,
        int ValidIndices,
        int OutOfBoundsIndices,
        List<uint> TopOutOfBoundsIndices
    );

    /// <summary>
    /// Comprehensive analysis of index patterns across all PM4 chunks.
    /// </summary>
    public static IndexAnalysis AnalyzeIndexPatterns(Pm4Scene scene, string outputDir)
    {
        ConsoleLogger.WriteLine("=== ANALYZING INDEX PATTERNS AND HIGH/LOW PAIRS ===");
        
        var maxVertexIndex = 0;
        var availableVertices = scene.Vertices.Count;
        var outOfBoundsCount = 0;
        var highLowPairs = new Dictionary<string, List<uint>>();
        var suspiciousIndices = new List<uint>();
        var chunkIndexStats = new Dictionary<string, IndexStatistics>();
        
        // Create analysis output directory
        var analysisDir = Path.Combine(outputDir, "index_analysis");
        Directory.CreateDirectory(analysisDir);
        
        ConsoleLogger.WriteLine($"Scene has {availableVertices} vertices (indices 0-{availableVertices - 1})");
        
        // Analyze MSUR surface indices
        AnalyzeMsurIndices(scene, ref maxVertexIndex, ref outOfBoundsCount, chunkIndexStats, highLowPairs, suspiciousIndices);
        
        // Analyze MSLK unknown fields for potential indices
        AnalyzeMslkUnknowns(scene, highLowPairs, suspiciousIndices, chunkIndexStats);
        
        // Analyze MPRL unknown fields for potential indices
        AnalyzeMprlUnknowns(scene, highLowPairs, suspiciousIndices, chunkIndexStats);
        
        // Look for high/low pair patterns that could encode larger indices
        AnalyzeHighLowPairPatterns(highLowPairs, availableVertices, suspiciousIndices);
        
        // Export detailed analysis
        ExportIndexAnalysis(maxVertexIndex, availableVertices, outOfBoundsCount, highLowPairs, 
            suspiciousIndices, chunkIndexStats, analysisDir);
        
        var summary = GenerateIndexAnalysisSummary(maxVertexIndex, availableVertices, outOfBoundsCount, 
            highLowPairs, chunkIndexStats);
        
        ConsoleLogger.WriteLine($"Max vertex index found: {maxVertexIndex}");
        ConsoleLogger.WriteLine($"Out-of-bounds accesses: {outOfBoundsCount}");
        ConsoleLogger.WriteLine($"Missing vertices: {maxVertexIndex + 1 - availableVertices}");
        ConsoleLogger.WriteLine(summary);
        
        return new IndexAnalysis(
            maxVertexIndex,
            availableVertices,
            outOfBoundsCount,
            highLowPairs,
            suspiciousIndices,
            chunkIndexStats,
            summary
        );
    }
    
    /// <summary>
    /// Analyzes MSUR surface vertex indices to find out-of-bounds patterns.
    /// </summary>
    private static void AnalyzeMsurIndices(Pm4Scene scene, ref int maxVertexIndex, ref int outOfBoundsCount,
        Dictionary<string, IndexStatistics> chunkIndexStats, Dictionary<string, List<uint>> highLowPairs,
        List<uint> suspiciousIndices)
    {
        ConsoleLogger.WriteLine($"Analyzing {scene.Surfaces.Count} MSUR surface indices...");
        
        var allIndices = new List<uint>();
        var validIndices = 0;
        var outOfBounds = 0;
        var topOutOfBounds = new List<uint>();
        
        foreach (var surface in scene.Surfaces)
        {
            for (int i = 0; i < surface.IndexCount; i++)
            {
                var index = surface.Indices[i];
                allIndices.Add(index);
                
                if (index >= scene.Vertices.Count)
                {
                    outOfBounds++;
                    if (topOutOfBounds.Count < 100) // Keep top 100 for analysis
                    {
                        topOutOfBounds.Add(index);
                    }
                    suspiciousIndices.Add(index);
                }
                else
                {
                    validIndices++;
                }
                
                if (index > maxVertexIndex)
                {
                    maxVertexIndex = (int)index;
                }
            }
        }
        
        outOfBoundsCount = outOfBounds;
        
        var minIndex = allIndices.Count > 0 ? allIndices.Min() : 0;
        var maxIndex = allIndices.Count > 0 ? allIndices.Max() : 0;
        
        chunkIndexStats["MSUR"] = new IndexStatistics(
            minIndex, maxIndex, allIndices.Count, validIndices, outOfBounds, topOutOfBounds);
        
        ConsoleLogger.WriteLine($"  MSUR indices: {allIndices.Count} total, {validIndices} valid, {outOfBounds} out-of-bounds");
        ConsoleLogger.WriteLine($"  Index range: {minIndex} - {maxIndex}");
    }
    
    /// <summary>
    /// Analyzes MSLK unknown fields for potential high/low index pairs.
    /// </summary>
    private static void AnalyzeMslkUnknowns(Pm4Scene scene, Dictionary<string, List<uint>> highLowPairs,
        List<uint> suspiciousIndices, Dictionary<string, IndexStatistics> chunkIndexStats)
    {
        ConsoleLogger.WriteLine($"Analyzing {scene.Links.Count} MSLK entries for unknown field patterns...");
        
        var unknownValues = new List<uint>();
        
        foreach (var link in scene.Links)
        {
            // Check each unknown field for potential index values
            unknownValues.Add(link.Unknown_0x00);
            unknownValues.Add(link.Unknown_0x01);
            unknownValues.Add(link.Unknown_0x02);
            unknownValues.Add(link.ParentIndex);
            unknownValues.Add((uint)link.MspiFirstIndex);
            unknownValues.Add(link.MspiIndexCount);
            unknownValues.Add(link.Unknown_0x0D);
            unknownValues.Add(link.Unknown_0x0E);
            unknownValues.Add(link.Unknown_0x10);
            unknownValues.Add(link.Unknown_0x12);
            unknownValues.Add(link.Unknown_0x14);
            unknownValues.Add(link.Unknown_0x16);
            
            // Look for potential high/low pairs
            CheckHighLowPair(link.Unknown_0x00, link.Unknown_0x01, "MSLK.Unknown_0x00+0x01", highLowPairs);
            CheckHighLowPair(link.Unknown_0x02, link.ParentIndex & 0xFFFF, "MSLK.Unknown_0x02+ParentIndex_Low", highLowPairs);
            CheckHighLowPair(link.Unknown_0x0D, link.Unknown_0x0E, "MSLK.Unknown_0x0D+0x0E", highLowPairs);
            CheckHighLowPair(link.Unknown_0x10, link.Unknown_0x12, "MSLK.Unknown_0x10+0x12", highLowPairs);
            CheckHighLowPair(link.Unknown_0x14, link.Unknown_0x16, "MSLK.Unknown_0x14+0x16", highLowPairs);
        }
        
        var validCount = unknownValues.Count(v => v < scene.Vertices.Count);
        var outOfBoundsCount = unknownValues.Count - validCount;
        var minVal = unknownValues.Count > 0 ? unknownValues.Min() : 0;
        var maxVal = unknownValues.Count > 0 ? unknownValues.Max() : 0;
        var topOutOfBounds = unknownValues.Where(v => v >= scene.Vertices.Count).Take(100).ToList();
        
        chunkIndexStats["MSLK"] = new IndexStatistics(
            minVal, maxVal, unknownValues.Count, validCount, outOfBoundsCount, topOutOfBounds);
        
        ConsoleLogger.WriteLine($"  MSLK unknowns: {unknownValues.Count} values, {validCount} in vertex range, {outOfBoundsCount} out-of-bounds");
    }
    
    /// <summary>
    /// Analyzes MPRL unknown fields for potential high/low index pairs.
    /// </summary>
    private static void AnalyzeMprlUnknowns(Pm4Scene scene, Dictionary<string, List<uint>> highLowPairs,
        List<uint> suspiciousIndices, Dictionary<string, IndexStatistics> chunkIndexStats)
    {
        ConsoleLogger.WriteLine($"Analyzing {scene.Placements.Count} MPRL entries for unknown field patterns...");
        
        var unknownValues = new List<uint>();
        
        foreach (var placement in scene.Placements)
        {
            unknownValues.Add(placement.Unknown0);
            unknownValues.Add((uint)placement.Unknown2);
            unknownValues.Add(placement.Unknown4);
            unknownValues.Add(placement.Unknown6);
            unknownValues.Add((uint)placement.Unknown14);
            unknownValues.Add(placement.Unknown16);
            
            // Look for potential high/low pairs
            CheckHighLowPair(placement.Unknown0, (uint)placement.Unknown2, "MPRL.Unknown0+Unknown2", highLowPairs);
            CheckHighLowPair(placement.Unknown4, placement.Unknown6, "MPRL.Unknown4+Unknown6", highLowPairs);
            CheckHighLowPair((uint)placement.Unknown14, placement.Unknown16, "MPRL.Unknown14+Unknown16", highLowPairs);
        }
        
        var validCount = unknownValues.Count(v => v < scene.Vertices.Count);
        var outOfBoundsCount = unknownValues.Count - validCount;
        var minVal = unknownValues.Count > 0 ? unknownValues.Min() : 0;
        var maxVal = unknownValues.Count > 0 ? unknownValues.Max() : 0;
        var topOutOfBounds = unknownValues.Where(v => v >= scene.Vertices.Count).Take(100).ToList();
        
        chunkIndexStats["MPRL"] = new IndexStatistics(
            minVal, maxVal, unknownValues.Count, validCount, outOfBoundsCount, topOutOfBounds);
        
        ConsoleLogger.WriteLine($"  MPRL unknowns: {unknownValues.Count} values, {validCount} in vertex range, {outOfBoundsCount} out-of-bounds");
    }
    
    /// <summary>
    /// Checks if two values could form a high/low pair encoding a larger index.
    /// </summary>
    private static void CheckHighLowPair(uint low, uint high, string pairName, Dictionary<string, List<uint>> highLowPairs)
    {
        // Combine as high<<16 | low to form 32-bit index
        uint combined = (high << 16) | low;
        
        if (!highLowPairs.ContainsKey(pairName))
        {
            highLowPairs[pairName] = new List<uint>();
        }
        
        highLowPairs[pairName].Add(combined);
    }
    
    /// <summary>
    /// Analyzes high/low pair patterns to find potential vertex indices.
    /// </summary>
    private static void AnalyzeHighLowPairPatterns(Dictionary<string, List<uint>> highLowPairs, 
        int availableVertices, List<uint> suspiciousIndices)
    {
        ConsoleLogger.WriteLine("Analyzing high/low pair patterns for potential vertex indices...");
        
        foreach (var kvp in highLowPairs)
        {
            var pairName = kvp.Key;
            var values = kvp.Value;
            
            var validIndices = values.Count(v => v < availableVertices);
            var potentialIndices = values.Count(v => v >= availableVertices && v < 200000); // Reasonable upper bound
            
            if (potentialIndices > 0)
            {
                ConsoleLogger.WriteLine($"  {pairName}: {values.Count} pairs, {validIndices} valid, {potentialIndices} potential indices");
                
                // Add the most suspicious ones for further analysis
                var suspicious = values.Where(v => v >= availableVertices && v < 200000).Take(10);
                suspiciousIndices.AddRange(suspicious);
            }
        }
    }
    
    /// <summary>
    /// Exports detailed index analysis to CSV files.
    /// </summary>
    private static void ExportIndexAnalysis(int maxVertexIndex, int availableVertices, int outOfBoundsCount,
        Dictionary<string, List<uint>> highLowPairs, List<uint> suspiciousIndices,
        Dictionary<string, IndexStatistics> chunkIndexStats, string analysisDir)
    {
        // Export high/low pair analysis
        var highLowFile = Path.Combine(analysisDir, "high_low_pairs.csv");
        using (var writer = new StreamWriter(highLowFile))
        {
            writer.WriteLine("PairName,CombinedValue,IsValidIndex,IsPotentialIndex");
            
            foreach (var kvp in highLowPairs)
            {
                foreach (var value in kvp.Value)
                {
                    var isValid = value < availableVertices;
                    var isPotential = value >= availableVertices && value < 200000;
                    writer.WriteLine($"{kvp.Key},{value},{isValid},{isPotential}");
                }
            }
        }
        
        // Export suspicious indices
        var suspiciousFile = Path.Combine(analysisDir, "suspicious_indices.csv");
        using (var writer = new StreamWriter(suspiciousFile))
        {
            writer.WriteLine("Index,OffsetFromMax,PotentialTileOffset");
            
            foreach (var index in suspiciousIndices.Distinct().OrderBy(x => x))
            {
                var offset = index - availableVertices;
                var tileOffset = offset / 65536; // Potential tile number
                writer.WriteLine($"{index},{offset},{tileOffset}");
            }
        }
        
        // Export chunk statistics
        var statsFile = Path.Combine(analysisDir, "chunk_index_stats.csv");
        using (var writer = new StreamWriter(statsFile))
        {
            writer.WriteLine("ChunkType,MinIndex,MaxIndex,TotalIndices,ValidIndices,OutOfBoundsIndices,OutOfBoundsPercentage");
            
            foreach (var kvp in chunkIndexStats)
            {
                var stats = kvp.Value;
                var percentage = stats.TotalIndices > 0 ? (stats.OutOfBoundsIndices * 100.0 / stats.TotalIndices) : 0;
                writer.WriteLine($"{kvp.Key},{stats.MinIndex},{stats.MaxIndex},{stats.TotalIndices},{stats.ValidIndices},{stats.OutOfBoundsIndices},{percentage:F2}%");
            }
        }
        
        ConsoleLogger.WriteLine($"Index analysis exported to {analysisDir}");
    }
    
    /// <summary>
    /// Generates a summary of the index analysis findings.
    /// </summary>
    private static string GenerateIndexAnalysisSummary(int maxVertexIndex, int availableVertices, int outOfBoundsCount,
        Dictionary<string, List<uint>> highLowPairs, Dictionary<string, IndexStatistics> chunkIndexStats)
    {
        var summary = $@"
INDEX PATTERN ANALYSIS SUMMARY:
- Available vertices: {availableVertices} (indices 0-{availableVertices - 1})
- Maximum index found: {maxVertexIndex}
- Missing vertices: {maxVertexIndex + 1 - availableVertices}
- Out-of-bounds accesses: {outOfBoundsCount}
- Data loss percentage: {(outOfBoundsCount * 100.0 / (outOfBoundsCount + availableVertices)):F2}%

HIGH/LOW PAIR ANALYSIS:
{string.Join("\n", highLowPairs.Select(kvp => $"- {kvp.Key}: {kvp.Value.Count} combinations"))}

CHUNK INDEX STATISTICS:
{string.Join("\n", chunkIndexStats.Select(kvp => $"- {kvp.Key}: {kvp.Value.OutOfBoundsIndices}/{kvp.Value.TotalIndices} out-of-bounds ({(kvp.Value.TotalIndices > 0 ? kvp.Value.OutOfBoundsIndices * 100.0 / kvp.Value.TotalIndices : 0):F1}%)"))}

RECOMMENDATIONS:
1. Implement global tile loading to access missing {maxVertexIndex + 1 - availableVertices} vertices
2. Investigate high/low pair encodings for extended index ranges
3. Consider vertex index remapping or offset calculations for tile boundaries
4. Analyze tile coordinate system and vertex pool organization
        ";
        
        return summary.Trim();
    }
}
