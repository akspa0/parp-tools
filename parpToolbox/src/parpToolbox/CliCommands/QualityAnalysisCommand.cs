using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4.Database;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Analyzes data quality issues in PM4 database exports.
    /// </summary>
    public static class QualityAnalysisCommand
    {
        /// <summary>
        /// Runs data quality analysis on a PM4 database.
        /// </summary>
        /// <param name="args">CLI arguments</param>
        /// <param name="databasePath">Path to the PM4 database file</param>
        /// <returns>Exit code</returns>
        public static async Task<int> Run(string[] args, string databasePath)
        {
            try
            {
                if (!File.Exists(databasePath))
                {
                    ConsoleLogger.WriteLine($"Error: Database file not found: {databasePath}");
                    return 1;
                }

                ConsoleLogger.WriteLine($"Analyzing data quality for database: {databasePath}");
                
                var analyzer = new Pm4DatabaseQualityAnalyzer(databasePath);
                var report = await analyzer.AnalyzeDataQualityAsync();
                
                // Print comprehensive report
                PrintQualityReport(report);
                
                // Generate recommendations
                PrintRecommendations(report);
                
                ConsoleLogger.WriteLine("Data quality analysis complete.");
                return 0;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during quality analysis: {ex.Message}");
                return 1;
            }
        }

        private static void PrintQualityReport(QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("\n=== DATA QUALITY ANALYSIS REPORT ===");
            
            // Coordinate Quality
            ConsoleLogger.WriteLine("\n--- COORDINATE DATA QUALITY ---");
            if (report.SurfaceCoordinateStats != null)
            {
                var stats = report.SurfaceCoordinateStats;
                ConsoleLogger.WriteLine($"Surface Coordinates:");
                ConsoleLogger.WriteLine($"  Total: {stats.Count:N0}, Valid: {stats.ValidCount:N0}");
                ConsoleLogger.WriteLine($"  Range: [{stats.Min:F2}, {stats.Max:F2}]");
                ConsoleLogger.WriteLine($"  Average: {stats.Average:F2} ± {stats.StandardDeviation:F2}");
                ConsoleLogger.WriteLine($"  Outliers detected: {report.SurfaceOutliers.Count:N0}");
                
                if (report.SurfaceOutliers.Count > 0)
                {
                    ConsoleLogger.WriteLine("  Sample outliers:");
                    for (int i = 0; i < Math.Min(5, report.SurfaceOutliers.Count); i++)
                    {
                        ConsoleLogger.WriteLine($"    {report.SurfaceOutliers[i]}");
                    }
                }
            }
            
            if (report.VertexCoordinateStats != null)
            {
                var stats = report.VertexCoordinateStats;
                ConsoleLogger.WriteLine($"Vertex Coordinates (sample):");
                ConsoleLogger.WriteLine($"  Total: {stats.Count:N0}, Valid: {stats.ValidCount:N0}");
                ConsoleLogger.WriteLine($"  Range: [{stats.Min:F2}, {stats.Max:F2}]");
                ConsoleLogger.WriteLine($"  Average: {stats.Average:F2} ± {stats.StandardDeviation:F2}");
            }
            
            // Placement Quality
            ConsoleLogger.WriteLine("\n--- PLACEMENT DATA QUALITY ---");
            ConsoleLogger.WriteLine($"Total Placements: {report.TotalPlacements:N0}");
            ConsoleLogger.WriteLine($"Non-Zero Placements: {report.NonZeroPlacements:N0}");
            ConsoleLogger.WriteLine($"Zero Placement Rate: {report.ZeroPlacementPercentage:F1}%");
            
            // Links Quality
            ConsoleLogger.WriteLine("\n--- LINKS DATA QUALITY ---");
            ConsoleLogger.WriteLine($"Total Links: {report.TotalLinks:N0}");
            ConsoleLogger.WriteLine($"Links with Valid Data: {report.LinksWithValidData:N0}");
            
            // Field Completeness
            ConsoleLogger.WriteLine("\n--- FIELD COMPLETENESS ---");
            ConsoleLogger.WriteLine($"Empty Link Fields: {report.EmptyLinkFields:N0}");
            ConsoleLogger.WriteLine($"Empty Placement Fields: {report.EmptyPlacementFields:N0}");
            
            // Raw Chunk Coverage
            ConsoleLogger.WriteLine("\n--- RAW CHUNK COVERAGE ---");
            ConsoleLogger.WriteLine($"Total Raw Chunks: {report.TotalRawChunks:N0}");
            ConsoleLogger.WriteLine($"Total Raw Data Size: {report.TotalRawDataSize:N0} bytes");
            
            if (report.ChunkTypeCounts.Any())
            {
                ConsoleLogger.WriteLine("Chunk Type Distribution:");
                foreach (var (type, count) in report.ChunkTypeCounts.OrderByDescending(kvp => kvp.Value))
                {
                    ConsoleLogger.WriteLine($"  {type}: {count:N0} chunks");
                }
            }
        }

        private static void PrintRecommendations(QualityAnalysisReport report)
        {
            ConsoleLogger.WriteLine("\n=== RECOMMENDATIONS ===");
            
            // Coordinate issues
            if (report.SurfaceOutliers.Count > 10)
            {
                ConsoleLogger.WriteLine("⚠️  HIGH PRIORITY: Massive coordinate outliers detected");
                ConsoleLogger.WriteLine("   → Values like ±3.4e+38 suggest MAX_FLOAT/invalid data");
                ConsoleLogger.WriteLine("   → Review coordinate field extraction logic");
                ConsoleLogger.WriteLine("   → Implement coordinate validation and filtering");
            }
            
            // Placement issues
            if (report.ZeroPlacementPercentage > 50)
            {
                ConsoleLogger.WriteLine("⚠️  CRITICAL: Most placements are zero coordinates");
                ConsoleLogger.WriteLine("   → MPRL chunk parsing is likely broken");
                ConsoleLogger.WriteLine("   → Review MPRL field extraction methods");
                ConsoleLogger.WriteLine("   → Verify field name patterns and offsets");
            }
            
            // Field completeness issues
            var emptyFieldRate = (report.EmptyLinkFields + report.EmptyPlacementFields) * 100.0 / 
                                (Math.Max(1, report.TotalLinks + report.TotalPlacements));
            if (emptyFieldRate > 25)
            {
                ConsoleLogger.WriteLine("⚠️  MODERATE: High rate of empty field data");
                ConsoleLogger.WriteLine("   → Many chunks have missing field extractions");
                ConsoleLogger.WriteLine("   → Review dynamic field extraction logic");
                ConsoleLogger.WriteLine("   → Add more comprehensive field name patterns");
            }
            
            // Raw chunk coverage
            if (report.TotalRawChunks == 0)
            {
                ConsoleLogger.WriteLine("⚠️  INFO: No raw chunks captured");
                ConsoleLogger.WriteLine("   → Enable CaptureRawData option during PM4 loading");
            }
            else if (report.ChunkTypeCounts.Count < 5)
            {
                ConsoleLogger.WriteLine("⚠️  INFO: Limited chunk type coverage");
                ConsoleLogger.WriteLine("   → May be missing some PM4 chunk types");
                ConsoleLogger.WriteLine("   → Verify all chunk signatures are recognized");
            }

            ConsoleLogger.WriteLine("\n💡 NEXT STEPS:");
            ConsoleLogger.WriteLine("1. Fix MPRL placement parsing (zero coordinates)");
            ConsoleLogger.WriteLine("2. Filter coordinate outliers (±3.4e+38 values)");
            ConsoleLogger.WriteLine("3. Enhance field extraction completeness");
            ConsoleLogger.WriteLine("4. Verify multi-file processing works correctly");
            ConsoleLogger.WriteLine("5. Re-run export and validate improvements");
        }
    }
}
