using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Analyzes MPRL placement data patterns to understand field meanings and encoding.
    /// </summary>
    public class Pm4MprlPatternAnalyzer
    {
        private readonly string _databasePath;

        public Pm4MprlPatternAnalyzer(string databasePath)
        {
            _databasePath = databasePath;
        }

        /// <summary>
        /// Performs comprehensive MPRL pattern analysis to understand field meanings.
        /// </summary>
        public async Task<MprlAnalysisReport> AnalyzeMprlPatternsAsync()
        {
            using var context = new Pm4DatabaseContext(_databasePath);
            
            ConsoleLogger.WriteLine("[MPRL ANALYSIS] Starting comprehensive MPRL pattern analysis...");
            
            var placements = await context.Placements.ToListAsync();
            var report = new MprlAnalysisReport();
            
            if (!placements.Any())
            {
                ConsoleLogger.WriteLine("[MPRL ANALYSIS] No placements found in database.");
                return report;
            }

            // Parse raw field data to extract Unknown values
            var unknownFields = ExtractUnknownFields(placements);
            
            // Analyze field patterns
            AnalyzeUnknown0Patterns(unknownFields, report);
            AnalyzeUnknown2Patterns(unknownFields, report);
            AnalyzeUnknown4Patterns(unknownFields, report);
            AnalyzeUnknown6Patterns(unknownFields, report);
            AnalyzeUnknown14Patterns(unknownFields, report);
            AnalyzeUnknown16Patterns(unknownFields, report);
            
            // Analyze coordinate patterns
            AnalyzeCoordinatePatterns(placements, report);
            
            // Look for correlations between fields
            AnalyzeFieldCorrelations(unknownFields, report);
            
            ConsoleLogger.WriteLine("[MPRL ANALYSIS] MPRL pattern analysis complete.");
            return report;
        }

        private List<MprlUnknownFields> ExtractUnknownFields(List<Pm4Placement> placements)
        {
            var unknownFields = new List<MprlUnknownFields>();
            
            foreach (var placement in placements)
            {
                try
                {
                    var rawData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(placement.RawFieldsJson);
                    
                    var fields = new MprlUnknownFields
                    {
                        PlacementId = placement.Id,
                        Position = new System.Numerics.Vector3(placement.PositionX, placement.PositionY, placement.PositionZ),
                        Unknown4 = placement.Unknown4,
                        Unknown6 = placement.Unknown6
                    };
                    
                    // Extract Unknown fields from raw JSON
                    if (rawData?.ContainsKey("Unknown0") == true && rawData["Unknown0"] is JsonElement elem0)
                        fields.Unknown0 = elem0.GetUInt16();
                    if (rawData?.ContainsKey("Unknown2") == true && rawData["Unknown2"] is JsonElement elem2)
                        fields.Unknown2 = elem2.GetInt16();
                    if (rawData?.ContainsKey("Unknown14") == true && rawData["Unknown14"] is JsonElement elem14)
                        fields.Unknown14 = elem14.GetInt16();
                    if (rawData?.ContainsKey("Unknown16") == true && rawData["Unknown16"] is JsonElement elem16)
                        fields.Unknown16 = elem16.GetUInt16();
                        
                    unknownFields.Add(fields);
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Error parsing placement {placement.Id}: {ex.Message}");
                }
            }
            
            return unknownFields;
        }

        private void AnalyzeUnknown0Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown0Values = fields.Select(f => f.Unknown0).ToList();
            var distribution = unknown0Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown0Distribution = distribution;
            report.Unknown0UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown0: {report.Unknown0UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences ({count * 100.0 / fields.Count:F1}%)");
            }
        }

        private void AnalyzeUnknown2Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown2Values = fields.Select(f => f.Unknown2).ToList();
            var distribution = unknown2Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown2Distribution = distribution;
            report.Unknown2UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown2: {report.Unknown2UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences ({count * 100.0 / fields.Count:F1}%)");
            }
        }

        private void AnalyzeUnknown4Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown4Values = fields.Select(f => f.Unknown4).ToList();
            var distribution = unknown4Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown4Distribution = distribution;
            report.Unknown4UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown4: {report.Unknown4UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences ({count * 100.0 / fields.Count:F1}%)");
            }
            
            // Special analysis for Unknown4 (likely MSLK ParentIndex reference)
            if (report.Unknown4UniqueCount == 1)
            {
                report.Unknown4Analysis = "Single value - likely references one MSLK ParentIndex (building group)";
            }
            else
            {
                report.Unknown4Analysis = $"Multiple values - references {report.Unknown4UniqueCount} different MSLK ParentIndex groups";
            }
        }

        private void AnalyzeUnknown6Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown6Values = fields.Select(f => f.Unknown6).ToList();
            var distribution = unknown6Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown6Distribution = distribution;
            report.Unknown6UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown6: {report.Unknown6UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences | Binary: {Convert.ToString(value, 2).PadLeft(16, '0')}");
            }
            
            // Special analysis for Unknown6 (likely bit flags)
            if (unknown6Values.All(v => v == 32768))
            {
                report.Unknown6Analysis = "All values are 32768 (0x8000, bit 15 set) - likely consistent flag/type";
            }
            else
            {
                report.Unknown6Analysis = "Multiple bit patterns - analyze as flags or states";
            }
        }

        private void AnalyzeUnknown14Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown14Values = fields.Select(f => f.Unknown14).ToList();
            var distribution = unknown14Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown14Distribution = distribution;
            report.Unknown14UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown14: {report.Unknown14UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences ({count * 100.0 / fields.Count:F1}%)");
            }
        }

        private void AnalyzeUnknown16Patterns(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            var unknown16Values = fields.Select(f => f.Unknown16).ToList();
            var distribution = unknown16Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.Unknown16Distribution = distribution;
            report.Unknown16UniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Unknown16: {report.Unknown16UniqueCount} unique values");
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences | Binary: {Convert.ToString(value, 2).PadLeft(16, '0')}");
            }
            
            // Special analysis for Unknown16 (common values: 0, 16383)
            if (unknown16Values.All(v => v == 0 || v == 16383))
            {
                report.Unknown16Analysis = "Binary pattern: 0 or 16383 (0x3FFF, 14 bits set) - likely state/flag toggle";
            }
        }

        private void AnalyzeCoordinatePatterns(List<Pm4Placement> placements, MprlAnalysisReport report)
        {
            var positions = placements.Select(p => new System.Numerics.Vector3(p.PositionX, p.PositionY, p.PositionZ)).ToList();
            
            if (positions.Any())
            {
                report.CoordinateStats = new MprlCoordinateStats
                {
                    Count = positions.Count,
                    ValidCount = positions.Count,
                    MinX = positions.Min(p => p.X),
                    MaxX = positions.Max(p => p.X),
                    MinY = positions.Min(p => p.Y),
                    MaxY = positions.Max(p => p.Y),
                    MinZ = positions.Min(p => p.Z),
                    MaxZ = positions.Max(p => p.Z),
                    AvgX = positions.Average(p => p.X),
                    AvgY = positions.Average(p => p.Y),
                    AvgZ = positions.Average(p => p.Z)
                };
                
                ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Coordinate Ranges:");
                ConsoleLogger.WriteLine($"  X: [{report.CoordinateStats.MinX:F2}, {report.CoordinateStats.MaxX:F2}] avg: {report.CoordinateStats.AvgX:F2}");
                ConsoleLogger.WriteLine($"  Y: [{report.CoordinateStats.MinY:F2}, {report.CoordinateStats.MaxY:F2}] avg: {report.CoordinateStats.AvgY:F2}");
                ConsoleLogger.WriteLine($"  Z: [{report.CoordinateStats.MinZ:F2}, {report.CoordinateStats.MaxZ:F2}] avg: {report.CoordinateStats.AvgZ:F2}");
            }
        }

        private void AnalyzeFieldCorrelations(List<MprlUnknownFields> fields, MprlAnalysisReport report)
        {
            ConsoleLogger.WriteLine($"[MPRL ANALYSIS] Field Correlations:");
            
            // Look for correlations between Unknown14 and Unknown16
            var unknown14_16_pairs = fields.GroupBy(f => new { f.Unknown14, f.Unknown16 })
                .ToDictionary(g => g.Key, g => g.Count());
                
            ConsoleLogger.WriteLine($"  Unknown14-Unknown16 combinations:");
            foreach (var (pair, count) in unknown14_16_pairs.OrderByDescending(kvp => kvp.Value))
            {
                ConsoleLogger.WriteLine($"    ({pair.Unknown14}, {pair.Unknown16}): {count} occurrences");
            }
            
            report.FieldCorrelations = unknown14_16_pairs.ToDictionary(
                kvp => $"Unknown14_{kvp.Key.Unknown14}_Unknown16_{kvp.Key.Unknown16}", 
                kvp => kvp.Value);
        }
    }

    public class MprlAnalysisReport
    {
        public Dictionary<ushort, int> Unknown0Distribution { get; set; } = new();
        public Dictionary<short, int> Unknown2Distribution { get; set; } = new();
        public Dictionary<uint, int> Unknown4Distribution { get; set; } = new();
        public Dictionary<uint, int> Unknown6Distribution { get; set; } = new();
        public Dictionary<short, int> Unknown14Distribution { get; set; } = new();
        public Dictionary<ushort, int> Unknown16Distribution { get; set; } = new();
        
        public int Unknown0UniqueCount { get; set; }
        public int Unknown2UniqueCount { get; set; }
        public int Unknown4UniqueCount { get; set; }
        public int Unknown6UniqueCount { get; set; }
        public int Unknown14UniqueCount { get; set; }
        public int Unknown16UniqueCount { get; set; }
        
        public string Unknown4Analysis { get; set; } = string.Empty;
        public string Unknown6Analysis { get; set; } = string.Empty;
        public string Unknown16Analysis { get; set; } = string.Empty;
        
        public MprlCoordinateStats? CoordinateStats { get; set; }
        public Dictionary<string, int> FieldCorrelations { get; set; } = new();
    }

    public class MprlUnknownFields
    {
        public int PlacementId { get; set; }
        public System.Numerics.Vector3 Position { get; set; }
        public ushort Unknown0 { get; set; }
        public short Unknown2 { get; set; }
        public uint Unknown4 { get; set; }
        public uint Unknown6 { get; set; }
        public short Unknown14 { get; set; }
        public ushort Unknown16 { get; set; }
    }

    public class MprlCoordinateStats
    {
        public int Count { get; set; }
        public int ValidCount { get; set; }
        public float MinX { get; set; }
        public float MaxX { get; set; }
        public float MinY { get; set; }
        public float MaxY { get; set; }
        public float MinZ { get; set; }
        public float MaxZ { get; set; }
        public double AvgX { get; set; }
        public double AvgY { get; set; }
        public double AvgZ { get; set; }
    }
}
