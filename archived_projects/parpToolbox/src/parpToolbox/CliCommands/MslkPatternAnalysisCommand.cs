using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4.Database;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to analyze MSLK link patterns and understand object→geometry relationships.
    /// </summary>
    public class MslkPatternAnalysisCommand
    {
        /// <summary>
        /// Runs MSLK pattern analysis on the specified PM4 database.
        /// </summary>
        public async Task RunAsync(string databasePath)
        {
            ConsoleLogger.WriteLine("=== MSLK Link Pattern Analysis ===");
            
            if (!File.Exists(databasePath))
            {
                ConsoleLogger.WriteLine($"Database file not found: {databasePath}");
                return;
            }

            try
            {
                var analyzer = new Pm4MslkPatternAnalyzer(databasePath);
                var report = await analyzer.AnalyzeMslkPatternsAsync();
                
                // Print summary report
                PrintAnalysisReport(report);
                
                ConsoleLogger.WriteLine("\n=== MSLK Link Pattern Analysis Complete ===");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during MSLK pattern analysis: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private void PrintAnalysisReport(MslkAnalysisReport report)
        {
            ConsoleLogger.WriteLine("\n=== MSLK Link Analysis Summary ===");
            
            ConsoleLogger.WriteLine($"\nLink Field Diversity:");
            ConsoleLogger.WriteLine($"  ParentIndex: {report.ParentIndexUniqueCount} unique values");
            ConsoleLogger.WriteLine($"  MspiFirstIndex: {report.MspiFirstIndexDistribution.Count} unique values");
            ConsoleLogger.WriteLine($"  MspiIndexCount: {report.MspiIndexCountDistribution.Count} unique values");
            ConsoleLogger.WriteLine($"  ReferenceIndex: {report.ReferenceIndexUniqueCount} unique values");
            
            if (report.UnknownFieldDistributions.Any())
            {
                ConsoleLogger.WriteLine($"  Unknown fields: {report.UnknownFieldDistributions.Count}");
            }
            
            ConsoleLogger.WriteLine($"\nContainer vs Geometry Links:");
            ConsoleLogger.WriteLine($"  Container nodes (FirstIndex=-1): {report.ContainerNodeCount} ({report.ContainerNodePercentage:F1}%)");
            if (report.ContainerParentIndices.Any())
            {
                ConsoleLogger.WriteLine($"  Container parent indices: {report.ContainerParentIndices.Count} unique");
            }
            
            if (report.GeometryCoverage != null)
            {
                var gc = report.GeometryCoverage;
                ConsoleLogger.WriteLine($"\nGeometry Coverage:");
                ConsoleLogger.WriteLine($"  Total links: {gc.TotalLinks:N0}");
                ConsoleLogger.WriteLine($"  Geometry links: {gc.GeometryLinks:N0}");
                ConsoleLogger.WriteLine($"  Container links: {gc.ContainerLinks:N0}");
                ConsoleLogger.WriteLine($"  Estimated triangles: {gc.TotalTriangles:N0}");
                ConsoleLogger.WriteLine($"  Avg triangles/link: {gc.AverageTrianglesPerLink:F1}");
            }
            
            if (report.HierarchicalAnalysis != null)
            {
                var ha = report.HierarchicalAnalysis;
                ConsoleLogger.WriteLine($"\nHierarchical Structure:");
                ConsoleLogger.WriteLine($"  Object groups: {ha.ObjectGroups:N0}");
                ConsoleLogger.WriteLine($"  Links per object: [{ha.MinLinksPerObject}, {ha.MaxLinksPerObject}] avg: {ha.AverageLinksPerObject:F1}");
            }
            
            if (report.MprlMslkOverlap != null)
            {
                var mo = report.MprlMslkOverlap;
                ConsoleLogger.WriteLine($"\nMPRL↔MSLK Relationship Analysis:");
                ConsoleLogger.WriteLine($"  MPRL unique Object IDs: {mo.MprlUniqueObjectIds}");
                ConsoleLogger.WriteLine($"  MSLK unique ParentIndices: {mo.MslkUniqueParentIndices}");
                ConsoleLogger.WriteLine($"  Overlapping IDs: {mo.OverlappingIds} ({mo.OverlapPercentage:F1}% overlap)");
                
                if (mo.MprlOnlyIds > 0)
                {
                    ConsoleLogger.WriteLine($"  MPRL-only IDs: {mo.MprlOnlyIds} (objects without geometry links)");
                }
                if (mo.MslkOnlyIds > 0)
                {
                    ConsoleLogger.WriteLine($"  MSLK-only IDs: {mo.MslkOnlyIds} (geometry without object placements)");
                }
            }
            
            ConsoleLogger.WriteLine("\n=== Interpretation Recommendations ===");
            
            // Provide intelligent recommendations based on patterns
            if (report.ContainerNodePercentage > 30)
            {
                ConsoleLogger.WriteLine("• High container node percentage suggests hierarchical object assembly");
            }
            
            if (report.MprlMslkOverlap?.OverlapPercentage > 90)
            {
                ConsoleLogger.WriteLine("• Strong MPRL↔MSLK overlap confirms object instance→geometry link system");
            }
            else if (report.MprlMslkOverlap?.OverlapPercentage < 70)
            {
                ConsoleLogger.WriteLine("• Weak MPRL↔MSLK overlap suggests missing cross-tile references or complex linking");
            }
            
            if (report.HierarchicalAnalysis?.AverageLinksPerObject > 10)
            {
                ConsoleLogger.WriteLine("• High links per object suggests complex multi-component object assembly");
            }
            
            if (report.GeometryCoverage?.TotalTriangles > 500000)
            {
                ConsoleLogger.WriteLine("• Large triangle count confirms significant geometry content in this tile");
            }
            
            if (report.UnknownFieldDistributions.Any())
            {
                ConsoleLogger.WriteLine("• Unknown fields detected - may contain additional link metadata (rotation, scale, flags)");
            }
        }
    }
}
