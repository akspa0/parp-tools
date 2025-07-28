using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4.Database;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to analyze MPRL placement field patterns and understand data encoding.
    /// </summary>
    public class MprlPatternAnalysisCommand
    {
        /// <summary>
        /// Runs MPRL pattern analysis on the specified PM4 database.
        /// </summary>
        public async Task RunAsync(string databasePath)
        {
            ConsoleLogger.WriteLine("=== MPRL Pattern Analysis ===");
            
            if (!File.Exists(databasePath))
            {
                ConsoleLogger.WriteLine($"Database file not found: {databasePath}");
                return;
            }

            try
            {
                var analyzer = new Pm4MprlPatternAnalyzer(databasePath);
                var report = await analyzer.AnalyzeMprlPatternsAsync();
                
                // Print summary report
                PrintAnalysisReport(report);
                
                ConsoleLogger.WriteLine("\n=== MPRL Pattern Analysis Complete ===");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during MPRL pattern analysis: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private void PrintAnalysisReport(MprlAnalysisReport report)
        {
            ConsoleLogger.WriteLine("\n=== MPRL Field Analysis Summary ===");
            
            ConsoleLogger.WriteLine($"\nField Diversity:");
            ConsoleLogger.WriteLine($"  Unknown0: {report.Unknown0UniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Unknown2: {report.Unknown2UniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Unknown4: {report.Unknown4UniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Unknown6: {report.Unknown6UniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Unknown14: {report.Unknown14UniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Unknown16: {report.Unknown16UniqueCount} unique values");
            
            if (!string.IsNullOrEmpty(report.Unknown4Analysis))
            {
                ConsoleLogger.WriteLine($"\nUnknown4 Analysis: {report.Unknown4Analysis}");
            }
            
            if (!string.IsNullOrEmpty(report.Unknown6Analysis))
            {
                ConsoleLogger.WriteLine($"Unknown6 Analysis: {report.Unknown6Analysis}");
            }
            
            if (!string.IsNullOrEmpty(report.Unknown16Analysis))
            {
                ConsoleLogger.WriteLine($"Unknown16 Analysis: {report.Unknown16Analysis}");
            }
            
            if (report.CoordinateStats != null)
            {
                var stats = report.CoordinateStats;
                ConsoleLogger.WriteLine($"\nCoordinate Statistics:");
                ConsoleLogger.WriteLine($"  Valid Positions: {stats.ValidCount}/{stats.Count}");
                ConsoleLogger.WriteLine($"  X Range: [{stats.MinX:F2}, {stats.MaxX:F2}] (avg: {stats.AvgX:F2})");
                ConsoleLogger.WriteLine($"  Y Range: [{stats.MinY:F2}, {stats.MaxY:F2}] (avg: {stats.AvgY:F2})");
                ConsoleLogger.WriteLine($"  Z Range: [{stats.MinZ:F2}, {stats.MaxZ:F2}] (avg: {stats.AvgZ:F2})");
            }
            
            if (report.FieldCorrelations.Any())
            {
                ConsoleLogger.WriteLine($"\nField Correlations Found: {report.FieldCorrelations.Count}");
            }
            
            ConsoleLogger.WriteLine("\n=== Interpretation Recommendations ===");
            
            // Provide intelligent recommendations based on patterns
            if (report.Unknown0UniqueCount == 1)
            {
                ConsoleLogger.WriteLine("• Unknown0: Single value suggests object type/category ID");
            }
            else if (report.Unknown0UniqueCount < 10)
            {
                ConsoleLogger.WriteLine("• Unknown0: Few unique values suggest enumeration/type system");
            }
            
            if (report.Unknown2UniqueCount <= 2 && report.Unknown2Distribution.ContainsKey(-1))
            {
                ConsoleLogger.WriteLine("• Unknown2: Limited values including -1 suggest flag/state field");
            }
            
            if (report.Unknown4UniqueCount == 1)
            {
                ConsoleLogger.WriteLine("• Unknown4: Single value confirms all placements reference same MSLK ParentIndex");
            }
            
            if (report.Unknown6Distribution.ContainsKey(32768))
            {
                ConsoleLogger.WriteLine("• Unknown6: 32768 (0x8000) suggests bit flag with bit 15 set");
            }
            
            if (report.Unknown16Distribution.ContainsKey(0) && report.Unknown16Distribution.ContainsKey(16383))
            {
                ConsoleLogger.WriteLine("• Unknown16: 0/16383 pattern suggests binary state (off/on with 14-bit mask)");
            }
        }
    }
}
