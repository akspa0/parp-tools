using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Analyzes MPRL chunk unknown field patterns to hypothesize their semantic meanings.
    /// </summary>
    public class Pm4MprlFieldAnalyzer
    {
        private readonly string _databasePath;
        private readonly string _outputDirectory;
        
        public Pm4MprlFieldAnalyzer(string databasePath, string outputDirectory)
        {
            _databasePath = databasePath ?? throw new ArgumentNullException(nameof(databasePath));
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
            
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }
        
        /// <summary>
        /// Main analysis entry point.
        /// </summary>
        public async Task AnalyzeAsync()
        {
            ConsoleLogger.WriteLine("[MPRL ANALYZER] Starting comprehensive MPRL field pattern analysis...");

            using var context = new Pm4DatabaseContext(_databasePath);

            var placements = await context.Placements.ToListAsync();

            if (!placements.Any())
            {
                ConsoleLogger.WriteLine("No MPRL placements found in database.");
                return;
            }

            ConsoleLogger.WriteLine($"[MPRL ANALYZER] Analyzing {placements.Count} MPRL placements...");

            // Analyze available fields only
            await AnalyzeUnknown4Patterns(placements);
            await AnalyzeUnknown6Patterns(placements);
            await AnalyzePositionPatterns(placements);
            await GenerateFieldAnalysisReport(placements);

            ConsoleLogger.WriteLine($"[MPRL ANALYZER] Analysis complete. Check output: {_outputDirectory}");
        }

        private async Task AnalyzeUnknown4Patterns(List<Pm4Placement> placements)
        {
            var unknown4Stats = placements
                .GroupBy(p => p.Unknown4)
                .Select(g => new { Value = g.Key, Count = g.Count() })
                .OrderByDescending(x => x.Count)
                .ToList();

            ConsoleLogger.WriteLine($"[MPRL] Unknown4 Analysis - {unknown4Stats.Count} unique values:");
            foreach (var stat in unknown4Stats.Take(10))
            {
                ConsoleLogger.WriteLine($"  Unknown4={stat.Value}: {stat.Count:N0} placements ({stat.Count * 100.0 / placements.Count:F1}%)");
            }

            var csvPath = Path.Combine(_outputDirectory, "mprl_unknown4_analysis.csv");
            await File.WriteAllTextAsync(csvPath, 
                "Unknown4,Count,Percentage,Hypothesis\n" +
                string.Join("\n", unknown4Stats.Select(s => 
                    $"{s.Value},{s.Count},{s.Count * 100.0 / placements.Count:F2},Object/Group ID")));
        }

        private async Task AnalyzeUnknown6Patterns(List<Pm4Placement> placements)
        {
            var unknown6Stats = placements
                .GroupBy(p => p.Unknown6)
                .Select(g => new { Value = g.Key, Count = g.Count() })
                .OrderByDescending(x => x.Count)
                .ToList();

            ConsoleLogger.WriteLine($"[MPRL] Unknown6 Analysis - {unknown6Stats.Count} unique values:");
            foreach (var stat in unknown6Stats.Take(10))
            {
                ConsoleLogger.WriteLine($"  Unknown6={stat.Value}: {stat.Count:N0} placements ({stat.Count * 100.0 / placements.Count:F1}%)");
            }

            var csvPath = Path.Combine(_outputDirectory, "mprl_unknown6_analysis.csv");
            await File.WriteAllTextAsync(csvPath, 
                "Unknown6,Count,Percentage,Hypothesis\n" +
                string.Join("\n", unknown6Stats.Select(s => 
                    $"{s.Value},{s.Count},{s.Count * 100.0 / placements.Count:F2},Property/Type Flag")));
        }

        private async Task AnalyzePositionPatterns(List<Pm4Placement> placements)
        {
            var positionStats = new
            {
                MinX = placements.Min(p => p.PositionX),
                MaxX = placements.Max(p => p.PositionX),
                MinY = placements.Min(p => p.PositionY),
                MaxY = placements.Max(p => p.PositionY),
                MinZ = placements.Min(p => p.PositionZ),
                MaxZ = placements.Max(p => p.PositionZ)
            };

            ConsoleLogger.WriteLine("[MPRL] Position Analysis:");
            ConsoleLogger.WriteLine($"  X Range: {positionStats.MinX:F2} to {positionStats.MaxX:F2}");
            ConsoleLogger.WriteLine($"  Y Range: {positionStats.MinY:F2} to {positionStats.MaxY:F2}");
            ConsoleLogger.WriteLine($"  Z Range: {positionStats.MinZ:F2} to {positionStats.MaxZ:F2}");

            var csvPath = Path.Combine(_outputDirectory, "mprl_position_analysis.csv");
            await File.WriteAllTextAsync(csvPath, 
                "Axis,Min,Max,Range\n" +
                $"X,{positionStats.MinX},{positionStats.MaxX},{positionStats.MaxX - positionStats.MinX}\n" +
                $"Y,{positionStats.MinY},{positionStats.MaxY},{positionStats.MaxY - positionStats.MinY}\n" +
                $"Z,{positionStats.MinZ},{positionStats.MaxZ},{positionStats.MaxZ - positionStats.MinZ}\n");
        }

        private async Task GenerateFieldAnalysisReport(List<Pm4Placement> placements)
        {
            var csvPath = Path.Combine(_outputDirectory, "mprl_comprehensive_analysis.csv");
            
            var csvContent = "Id,Unknown4,Unknown6,PositionX,PositionY,PositionZ,RawFieldsJson\n" +
                string.Join("\n", placements.Select(p => 
                    $"{p.Id},{p.Unknown4},{p.Unknown6},{p.PositionX},{p.PositionY},{p.PositionZ},\"{p.RawFieldsJson.Replace("\"", "\\\"")}\""));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"[MPRL] Comprehensive analysis saved to: {csvPath}");
        }
    }
}
