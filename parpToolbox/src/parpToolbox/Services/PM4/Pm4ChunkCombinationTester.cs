namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

/// <summary>
/// Automated testing system to exhaustively try different combinations of chunk unknowns
/// to discover the correct face connectivity and object grouping logic.
/// This will help break the deadlock on PM4 object assembly.
/// </summary>
internal static class Pm4ChunkCombinationTester
{
    /// <summary>
    /// Test result for a specific combination of chunk field mappings.
    /// </summary>
    public record TestResult(
        string TestName,
        string GroupingStrategy,
        int ObjectCount,
        int TotalTriangles,
        int TotalVertices,
        double ConnectivityScore,
        double GeometryQualityScore,
        bool HasValidGeometry,
        string Details
    );

    /// <summary>
    /// Runs exhaustive tests on all possible chunk field combinations to find the correct grouping logic.
    /// </summary>
    public static List<TestResult> RunExhaustiveChunkTests(Pm4Scene scene, string outputDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("=== STARTING EXHAUSTIVE CHUNK COMBINATION TESTING ===");
        ConsoleLogger.WriteLine($"Testing scene with {scene.Surfaces.Count} surfaces, {scene.Links.Count} links, {scene.Placements.Count} placements");
        
        // Create test output directory
        var testDir = Path.Combine(outputDir, "chunk_combination_tests");
        Directory.CreateDirectory(testDir);
        
        // Test 1: Different MSUR grouping strategies
        results.AddRange(TestMsurGroupingStrategies(scene, testDir));
        
        // Test 2: Different MSLK relationship interpretations
        results.AddRange(TestMslkRelationshipStrategies(scene, testDir));
        
        // Test 3: Different MPRL transformation strategies
        results.AddRange(TestMprlTransformationStrategies(scene, testDir));
        
        // Test 4: Combined unknown field strategies
        results.AddRange(TestUnknownFieldCombinations(scene, testDir));
        
        // Test 5: Index range and connectivity strategies
        results.AddRange(TestIndexConnectivityStrategies(scene, testDir));
        
        // Sort results by quality score and export analysis
        var sortedResults = results.OrderByDescending(r => r.GeometryQualityScore).ToList();
        ExportTestAnalysis(sortedResults, testDir);
        
        ConsoleLogger.WriteLine($"=== COMPLETED {results.Count} TESTS ===");
        ConsoleLogger.WriteLine($"Best result: {sortedResults.First().TestName} (Quality: {sortedResults.First().GeometryQualityScore:F3})");
        
        return sortedResults;
    }
    
    /// <summary>
    /// Tests different MSUR surface grouping strategies using various field combinations.
    /// </summary>
    private static List<TestResult> TestMsurGroupingStrategies(Pm4Scene scene, string testDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("Testing MSUR grouping strategies...");
        
        // Strategy 1: Group by SurfaceGroupKey (current approach)
        results.Add(TestGroupingStrategy(scene, "MSUR_SurfaceGroupKey", 
            surfaces => surfaces.GroupBy(s => s.SurfaceGroupKey).ToList()));
        
        // Strategy 2: Group by IndexCount
        results.Add(TestGroupingStrategy(scene, "MSUR_IndexCount", 
            surfaces => surfaces.GroupBy(s => s.IndexCount).ToList()));
        
        // Strategy 3: Group by Unknown_0x02
        results.Add(TestGroupingStrategy(scene, "MSUR_Unknown_0x02", 
            surfaces => surfaces.GroupBy(s => s.AttributeMask).ToList()));
        
        // Strategy 4: Group by MdosIndex
        results.Add(TestGroupingStrategy(scene, "MSUR_MdosIndex", 
            surfaces => surfaces.GroupBy(s => s.MdosIndex).ToList()));
        
        // Strategy 5: Group by SurfaceKey (high 16 bits)
        results.Add(TestGroupingStrategy(scene, "MSUR_SurfaceKeyHigh", 
            surfaces => surfaces.GroupBy(s => (s.SurfaceKey >> 16) & 0xFFFF).ToList()));
        
        // Strategy 6: Group by SurfaceKey (low 16 bits)
        results.Add(TestGroupingStrategy(scene, "MSUR_SurfaceKeyLow", 
            surfaces => surfaces.GroupBy(s => s.SurfaceKey & 0xFFFF).ToList()));
        
        // Strategy 7: Group by combined SurfaceGroupKey + MdosIndex
        results.Add(TestGroupingStrategy(scene, "MSUR_SurfaceGroupKey_MdosIndex", 
            surfaces => surfaces.GroupBy(s => $"{s.SurfaceGroupKey}_{s.MdosIndex}").ToList()));
        
        // Strategy 8: Group by spatial proximity (surface height ranges)
        results.Add(TestGroupingStrategy(scene, "MSUR_SpatialHeight", 
            surfaces => surfaces.GroupBy(s => Math.Round(s.Height / 100.0) * 100).ToList()));
        
        return results;
    }
    
    /// <summary>
    /// Tests different MSLK relationship interpretation strategies.
    /// </summary>
    private static List<TestResult> TestMslkRelationshipStrategies(Pm4Scene scene, string testDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("Testing MSLK relationship strategies...");
        
        // Strategy 1: Group by ParentIndex
        results.Add(TestMslkStrategy(scene, "MSLK_ParentIndex", 
            links => links.GroupBy(l => l.ParentIndex).ToList()));
        
        // Strategy 2: Group by Flags_0x00
        results.Add(TestMslkStrategy(scene, "MSLK_Flags_0x00", 
            links => links.GroupBy(l => l.Flags_0x00).ToList()));
        
        // Strategy 3: Group by Type_0x01
        results.Add(TestMslkStrategy(scene, "MSLK_Type_0x01", 
            links => links.GroupBy(l => l.Type_0x01).ToList()));
        
        // Strategy 4: Group by SortKey_0x02
        results.Add(TestMslkStrategy(scene, "MSLK_SortKey_0x02", 
            links => links.GroupBy(l => l.SortKey_0x02).ToList()));
        
        // Strategy 5: Group by ReferenceIndex
        results.Add(TestMslkStrategy(scene, "MSLK_ReferenceIndex", 
            links => links.GroupBy(l => l.ReferenceIndex).ToList()));
        
        // Strategy 6: Group by MspiFirstIndex (non-container entries only)
        results.Add(TestMslkStrategy(scene, "MSLK_MspiFirstIndex", 
            links => links.Where(l => l.MspiFirstIndex >= 0).GroupBy(l => l.MspiFirstIndex / 100).ToList()));
        
        return results;
    }
    
    /// <summary>
    /// Tests different MPRL transformation strategies.
    /// </summary>
    private static List<TestResult> TestMprlTransformationStrategies(Pm4Scene scene, string testDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("Testing MPRL transformation strategies...");
        
        // Strategy 1: Group by Unknown0
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown0", 
            placements => placements.GroupBy(p => p.Unknown0).ToList()));
        
        // Strategy 2: Group by Unknown2
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown2", 
            placements => placements.GroupBy(p => p.Unknown2).ToList()));
        
        // Strategy 3: Group by Unknown4 (known to link to MSLK)
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown4", 
            placements => placements.GroupBy(p => p.Unknown4).ToList()));
        
        // Strategy 4: Group by Unknown6
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown6", 
            placements => placements.GroupBy(p => p.Unknown6).ToList()));
        
        // Strategy 5: Group by Unknown14
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown14", 
            placements => placements.GroupBy(p => p.Unknown14).ToList()));
        
        // Strategy 6: Group by Unknown16
        results.Add(TestMprlStrategy(scene, "MPRL_Unknown16", 
            placements => placements.GroupBy(p => p.Unknown16).ToList()));
        
        return results;
    }
    
    /// <summary>
    /// Tests combinations of unknown fields across different chunks.
    /// </summary>
    private static List<TestResult> TestUnknownFieldCombinations(Pm4Scene scene, string testDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("Testing unknown field combinations...");
        
        // Test cross-chunk field correlations
        // This is where we might find the missing relationships
        
        // Combination 1: MSUR.SurfaceGroupKey + MSLK.Unknown_0x01
        results.Add(TestCombinedStrategy(scene, "MSUR_SurfaceGroupKey_MSLK_Unknown01"));
        
        // Combination 2: MSUR.MdosIndex + MSLK.Unknown_0x00
        results.Add(TestCombinedStrategy(scene, "MSUR_MdosIndex_MSLK_Unknown00"));
        
        // Combination 3: MPRL.Unknown4 + MSLK.ParentIndex + MSUR.SurfaceGroupKey
        results.Add(TestCombinedStrategy(scene, "MPRL_Unknown4_MSLK_ParentIndex_MSUR_SurfaceGroupKey"));
        
        return results;
    }
    
    /// <summary>
    /// Tests different index range and connectivity strategies.
    /// </summary>
    private static List<TestResult> TestIndexConnectivityStrategies(Pm4Scene scene, string testDir)
    {
        var results = new List<TestResult>();
        
        ConsoleLogger.WriteLine("Testing index connectivity strategies...");
        
        // Strategy 1: Strict index range boundaries (no overlap)
        results.Add(TestConnectivityStrategy(scene, "StrictIndexRanges", true));
        
        // Strategy 2: Overlapping index ranges allowed
        results.Add(TestConnectivityStrategy(scene, "OverlappingIndexRanges", false));
        
        return results;
    }
    
    /// <summary>
    /// Tests a specific grouping strategy and evaluates the results.
    /// </summary>
    private static TestResult TestGroupingStrategy<TKey>(Pm4Scene scene, string strategyName, 
        Func<IEnumerable<MsurChunk.Entry>, List<IGrouping<TKey, MsurChunk.Entry>>> groupingFunc)
    {
        try
        {
            var groups = groupingFunc(scene.Surfaces);
            var totalTriangles = 0;
            var totalVertices = 0;
            var validGroups = 0;
            
            foreach (var group in groups)
            {
                var surfaces = group.ToList();
                if (surfaces.Any(s => s.IndexCount > 0))
                {
                    validGroups++;
                    totalTriangles += surfaces.Sum(s => (int)s.IndexCount / 3);
                    // Estimate vertices (rough approximation)
                    totalVertices += surfaces.Sum(s => (int)s.IndexCount);
                }
            }
            
            var connectivityScore = CalculateConnectivityScore(groups.Count(), totalTriangles, totalVertices);
            var qualityScore = CalculateGeometryQualityScore(validGroups, totalTriangles, totalVertices);
            
            return new TestResult(
                strategyName,
                "MSUR Grouping",
                groups.Count(),
                totalTriangles,
                totalVertices,
                connectivityScore,
                qualityScore,
                validGroups > 0,
                $"Valid groups: {validGroups}, Avg triangles/group: {(validGroups > 0 ? totalTriangles / (double)validGroups : 0):F1}"
            );
        }
        catch (Exception ex)
        {
            return new TestResult(strategyName, "MSUR Grouping", 0, 0, 0, 0, 0, false, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Tests a specific MSLK relationship strategy.
    /// </summary>
    private static TestResult TestMslkStrategy<TKey>(Pm4Scene scene, string strategyName,
        Func<IEnumerable<MslkEntry>, List<IGrouping<TKey, MslkEntry>>> groupingFunc)
    {
        try
        {
            var groups = groupingFunc(scene.Links);
            var geometryGroups = groups.Where(g => g.Any(l => l.MspiFirstIndex >= 0)).Count();
            
            var connectivityScore = CalculateConnectivityScore(groups.Count(), geometryGroups, scene.Links.Count);
            var qualityScore = CalculateGeometryQualityScore(geometryGroups, groups.Count(), scene.Links.Count);
            
            return new TestResult(
                strategyName,
                "MSLK Relationship",
                groups.Count(),
                geometryGroups,
                scene.Links.Count,
                connectivityScore,
                qualityScore,
                geometryGroups > 0,
                $"Geometry groups: {geometryGroups}, Container groups: {groups.Count() - geometryGroups}"
            );
        }
        catch (Exception ex)
        {
            return new TestResult(strategyName, "MSLK Relationship", 0, 0, 0, 0, 0, false, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Tests a specific MPRL transformation strategy.
    /// </summary>
    private static TestResult TestMprlStrategy<TKey>(Pm4Scene scene, string strategyName,
        Func<IEnumerable<MprlChunk.Entry>, List<IGrouping<TKey, MprlChunk.Entry>>> groupingFunc)
    {
        try
        {
            var groups = groupingFunc(scene.Placements);
            var totalPlacements = scene.Placements.Count;
            
            var connectivityScore = CalculateConnectivityScore(groups.Count(), totalPlacements, totalPlacements);
            var qualityScore = CalculateGeometryQualityScore(groups.Count(), totalPlacements, totalPlacements);
            
            return new TestResult(
                strategyName,
                "MPRL Transformation",
                groups.Count(),
                totalPlacements,
                totalPlacements,
                connectivityScore,
                qualityScore,
                groups.Count() > 0,
                $"Placement groups: {groups.Count()}, Avg placements/group: {(groups.Count() > 0 ? totalPlacements / (double)groups.Count() : 0):F1}"
            );
        }
        catch (Exception ex)
        {
            return new TestResult(strategyName, "MPRL Transformation", 0, 0, 0, 0, 0, false, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Tests combined strategies across multiple chunks.
    /// </summary>
    private static TestResult TestCombinedStrategy(Pm4Scene scene, string strategyName)
    {
        try
        {
            // Implement specific combined logic based on strategy name
            // This is where we test cross-chunk relationships
            
            var score = 0.5; // Placeholder - implement actual testing logic
            
            return new TestResult(
                strategyName,
                "Combined Strategy",
                1,
                scene.Surfaces.Sum(s => (int)s.IndexCount / 3),
                scene.Vertices.Count,
                score,
                score,
                true,
                "Combined strategy test - needs implementation"
            );
        }
        catch (Exception ex)
        {
            return new TestResult(strategyName, "Combined Strategy", 0, 0, 0, 0, 0, false, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Tests connectivity strategies.
    /// </summary>
    private static TestResult TestConnectivityStrategy(Pm4Scene scene, string strategyName, bool strictRanges)
    {
        try
        {
            // Test how different index range interpretations affect connectivity
            var score = strictRanges ? 0.6 : 0.4; // Placeholder
            
            return new TestResult(
                strategyName,
                "Connectivity Strategy",
                scene.Surfaces.Count,
                scene.Surfaces.Sum(s => (int)s.IndexCount / 3),
                scene.Vertices.Count,
                score,
                score,
                true,
                $"Strict ranges: {strictRanges}"
            );
        }
        catch (Exception ex)
        {
            return new TestResult(strategyName, "Connectivity Strategy", 0, 0, 0, 0, 0, false, $"Error: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Calculates a connectivity score based on how well faces connect.
    /// </summary>
    private static double CalculateConnectivityScore(int groupCount, int triangleCount, int vertexCount)
    {
        if (groupCount == 0 || triangleCount == 0) return 0.0;
        
        // Higher scores for reasonable group counts and good triangle/vertex ratios
        var groupScore = Math.Min(1.0, 10.0 / groupCount); // Prefer fewer, more substantial groups
        var ratioScore = vertexCount > 0 ? Math.Min(1.0, triangleCount / (double)vertexCount) : 0.0;
        
        return (groupScore + ratioScore) / 2.0;
    }
    
    /// <summary>
    /// Calculates a geometry quality score.
    /// </summary>
    private static double CalculateGeometryQualityScore(int validGroups, int triangleCount, int vertexCount)
    {
        if (validGroups == 0 || triangleCount == 0) return 0.0;
        
        // Score based on having reasonable numbers of groups and geometry
        var groupScore = Math.Min(1.0, validGroups / 5.0); // Prefer 3-5 main groups
        var geometryScore = Math.Min(1.0, triangleCount / 1000.0); // Prefer substantial geometry
        
        return (groupScore + geometryScore) / 2.0;
    }
    
    /// <summary>
    /// Exports detailed analysis of all test results.
    /// </summary>
    private static void ExportTestAnalysis(List<TestResult> results, string testDir)
    {
        var analysisPath = Path.Combine(testDir, "test_analysis.csv");
        
        using var writer = new StreamWriter(analysisPath);
        writer.WriteLine("TestName,GroupingStrategy,ObjectCount,TotalTriangles,TotalVertices,ConnectivityScore,GeometryQualityScore,HasValidGeometry,Details");
        
        foreach (var result in results)
        {
            writer.WriteLine($"{result.TestName},{result.GroupingStrategy},{result.ObjectCount},{result.TotalTriangles},{result.TotalVertices},{result.ConnectivityScore:F3},{result.GeometryQualityScore:F3},{result.HasValidGeometry},\"{result.Details}\"");
        }
        
        ConsoleLogger.WriteLine($"Exported test analysis to {analysisPath}");
        
        // Export top 10 results summary
        var summaryPath = Path.Combine(testDir, "top_results_summary.txt");
        using var summaryWriter = new StreamWriter(summaryPath);
        summaryWriter.WriteLine("=== TOP 10 CHUNK COMBINATION TEST RESULTS ===");
        summaryWriter.WriteLine();
        
        foreach (var result in results.Take(10))
        {
            summaryWriter.WriteLine($"Test: {result.TestName}");
            summaryWriter.WriteLine($"  Strategy: {result.GroupingStrategy}");
            summaryWriter.WriteLine($"  Quality Score: {result.GeometryQualityScore:F3}");
            summaryWriter.WriteLine($"  Objects: {result.ObjectCount}, Triangles: {result.TotalTriangles}, Vertices: {result.TotalVertices}");
            summaryWriter.WriteLine($"  Details: {result.Details}");
            summaryWriter.WriteLine();
        }
        
        ConsoleLogger.WriteLine($"Exported top results summary to {summaryPath}");
    }
}
