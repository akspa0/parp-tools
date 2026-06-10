using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;
using System.Text.Json;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Analyzes Surface table data for encoding patterns in fields like BoundsMaxZ.
    /// Identifies which fields contain spatial data vs encoded linkage/reference data.
    /// </summary>
    public class Pm4SurfaceEncodingAnalyzer
    {
        /// <summary>
        /// Analyzes Surface field encoding patterns by GroupKey.
        /// </summary>
        public static async Task AnalyzeSurfaceEncodingPatterns(string databasePath)
        {
            var startTime = DateTime.Now;
            
            ConsoleLogger.WriteLine("=== PM4 SURFACE ENCODING PATTERN ANALYSIS ===");
            ConsoleLogger.WriteLine($"Database: {databasePath}");
            ConsoleLogger.WriteLine();

            using var context = new Pm4DatabaseContext(databasePath);

            // Get all surface data grouped by GroupKey
            var surfacesByGroup = await context.Surfaces
                .GroupBy(s => s.GroupKey)
                .Select(g => new {
                    GroupKey = g.Key,
                    Count = g.Count(),
                    Surfaces = g.Select(s => new {
                        s.BoundsMinX, s.BoundsMinY, s.BoundsMinZ,
                        s.BoundsMaxX, s.BoundsMaxY, s.BoundsMaxZ,
                        s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ,
                        s.RawFlags, s.MsviFirstIndex, s.IndexCount
                    }).ToList()
                })
                .OrderBy(g => g.GroupKey)
                .ToListAsync();

            ConsoleLogger.WriteLine($"=== SURFACE GROUP SUMMARY ===");
            ConsoleLogger.WriteLine($"Total surface groups: {surfacesByGroup.Count}");
            ConsoleLogger.WriteLine($"Total surfaces: {surfacesByGroup.Sum(g => g.Count)}");
            ConsoleLogger.WriteLine();

            // Analyze each group for encoding patterns
            var spatialGroups = new List<byte>();
            var encodedGroups = new List<byte>();
            var mixedGroups = new List<byte>();

            ConsoleLogger.WriteLine("=== GROUP-BY-GROUP ENCODING ANALYSIS ===");

            foreach (var group in surfacesByGroup)
            {
                ConsoleLogger.WriteLine($"GroupKey {group.GroupKey}: {group.Count} surfaces");
                
                var encodingPattern = AnalyzeGroupEncodingPattern(group.Surfaces.Cast<object>().ToList());
                
                ConsoleLogger.WriteLine($"  Pattern: {encodingPattern.Type}");
                ConsoleLogger.WriteLine($"  BoundsMaxZ values: {string.Join(", ", encodingPattern.UniqueMaxZValues.Take(5))}");
                if (encodingPattern.UniqueMaxZValues.Count > 5)
                    ConsoleLogger.WriteLine($"    ... and {encodingPattern.UniqueMaxZValues.Count - 5} more values");
                
                ConsoleLogger.WriteLine($"  Field consistency:");
                ConsoleLogger.WriteLine($"    BoundsMaxZ: {encodingPattern.MaxZConsistency:P1}");
                ConsoleLogger.WriteLine($"    BoundsMaxY: {encodingPattern.MaxYConsistency:P1}");
                ConsoleLogger.WriteLine($"    BoundsCenterZ: {encodingPattern.CenterZConsistency:P1}");

                // Decode potential meanings for large values
                if (encodingPattern.HasLargeValues)
                {
                    ConsoleLogger.WriteLine($"  Potential encodings:");
                    foreach (var value in encodingPattern.UniqueMaxZValues.Where(v => Math.Abs(v) > 1000000).Take(3))
                    {
                        var decodedInfo = DecodeValue(value);
                        ConsoleLogger.WriteLine($"    {value}: {decodedInfo}");
                    }
                }

                // Categorize group
                switch (encodingPattern.Type)
                {
                    case "Spatial":
                        spatialGroups.Add(group.GroupKey);
                        break;
                    case "Encoded":
                        encodedGroups.Add(group.GroupKey);
                        break;
                    case "Mixed":
                        mixedGroups.Add(group.GroupKey);
                        break;
                }

                ConsoleLogger.WriteLine();
            }

            // Summary of findings
            ConsoleLogger.WriteLine("=== ENCODING PATTERN SUMMARY ===");
            ConsoleLogger.WriteLine($"Spatial groups (normal coordinates): {spatialGroups.Count}");
            ConsoleLogger.WriteLine($"  Groups: {string.Join(", ", spatialGroups)}");
            ConsoleLogger.WriteLine();
            
            ConsoleLogger.WriteLine($"Encoded groups (linkage/reference data): {encodedGroups.Count}");
            ConsoleLogger.WriteLine($"  Groups: {string.Join(", ", encodedGroups)}");
            ConsoleLogger.WriteLine();
            
            ConsoleLogger.WriteLine($"Mixed groups (spatial + encoded): {mixedGroups.Count}");
            if (mixedGroups.Count > 0)
                ConsoleLogger.WriteLine($"  Groups: {string.Join(", ", mixedGroups)}");
            ConsoleLogger.WriteLine();

            // Generate architectural hypothesis
            ConsoleLogger.WriteLine("=== ARCHITECTURAL HYPOTHESIS ===");
            GenerateArchitecturalHypothesis(spatialGroups, encodedGroups, mixedGroups);

            var analysisTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Analysis completed in {analysisTime.TotalSeconds:F1}s");
        }

        private static GroupEncodingPattern AnalyzeGroupEncodingPattern(List<object> surfaces)
        {
            var pattern = new GroupEncodingPattern();
            
            var maxZValues = new List<float>();
            var maxYValues = new List<float>();
            var centerZValues = new List<float>();
            
            foreach (dynamic surface in surfaces)
            {
                maxZValues.Add((float)surface.BoundsMaxZ);
                maxYValues.Add((float)surface.BoundsMaxY);
                centerZValues.Add((float)surface.BoundsCenterZ);
            }

            pattern.UniqueMaxZValues = maxZValues.Distinct().OrderBy(v => v).ToList();
            pattern.UniqueMaxYValues = maxYValues.Distinct().OrderBy(v => v).ToList();
            pattern.UniqueCenterZValues = centerZValues.Distinct().OrderBy(v => v).ToList();

            // Calculate consistency (how many surfaces share the most common value)
            pattern.MaxZConsistency = maxZValues.Count > 0 ? (double)maxZValues.GroupBy(v => v).Max(g => g.Count()) / maxZValues.Count : 0;
            pattern.MaxYConsistency = maxYValues.Count > 0 ? (double)maxYValues.GroupBy(v => v).Max(g => g.Count()) / maxYValues.Count : 0;
            pattern.CenterZConsistency = centerZValues.Count > 0 ? (double)centerZValues.GroupBy(v => v).Max(g => g.Count()) / centerZValues.Count : 0;

            // Determine if values are large (likely encoded)
            pattern.HasLargeValues = pattern.UniqueMaxZValues.Any(v => Math.Abs(v) > 1000000);

            // Classify pattern type
            if (pattern.HasLargeValues && pattern.MaxZConsistency > 0.8)
            {
                pattern.Type = "Encoded";
            }
            else if (!pattern.HasLargeValues && pattern.UniqueMaxZValues.All(v => Math.Abs(v) < 10000))
            {
                pattern.Type = "Spatial";
            }
            else
            {
                pattern.Type = "Mixed";
            }

            return pattern;
        }

        private static string DecodeValue(float value)
        {
            var intValue = (int)value;
            var uintValue = (uint)value;
            
            var decodings = new List<string>();
            
            // Hex representation
            decodings.Add($"Hex: 0x{intValue:X8}");
            
            // Possible tile coordinates (if divided by large factors)
            if (intValue > 1000000)
            {
                decodings.Add($"÷1000: {intValue / 1000}");
                decodings.Add($"÷10000: {intValue / 10000}");
            }
            
            // Check if it's a float bit pattern
            var bytes = BitConverter.GetBytes(intValue);
            var floatValue = BitConverter.ToSingle(bytes, 0);
            if (!float.IsNaN(floatValue) && !float.IsInfinity(floatValue) && Math.Abs(floatValue) < 1000000)
            {
                decodings.Add($"Float bits: {floatValue:F6}");
            }
            
            // Check high/low 16-bit pairs
            var high16 = (intValue >> 16) & 0xFFFF;
            var low16 = intValue & 0xFFFF;
            decodings.Add($"16-bit pairs: {high16}, {low16}");

            return string.Join(" | ", decodings);
        }

        private static void GenerateArchitecturalHypothesis(
            List<byte> spatialGroups, List<byte> encodedGroups, List<byte> mixedGroups)
        {
            var hypotheses = new List<string>();

            if (encodedGroups.Count > 0)
            {
                hypotheses.Add($"Groups {string.Join(", ", encodedGroups)} use encoded linkage data in coordinate fields");
                hypotheses.Add("These groups likely represent cross-tile or inter-object references");
            }

            if (spatialGroups.Count > 0)
            {
                hypotheses.Add($"Groups {string.Join(", ", spatialGroups)} contain normal spatial coordinate data");
                hypotheses.Add("These groups represent local geometry within the current tile");
            }

            if (mixedGroups.Count > 0)
            {
                hypotheses.Add($"Groups {string.Join(", ", mixedGroups)} mix spatial and encoded data");
                hypotheses.Add("These may represent boundary objects spanning tile edges");
            }

            hypotheses.Add("Surface GroupKey determines data interpretation: spatial vs encoded");
            hypotheses.Add("Different GroupKeys use different architectural roles in the global mesh");
            hypotheses.Add("Complete building assembly requires decoding both spatial and linkage groups");

            foreach (var hypothesis in hypotheses)
            {
                ConsoleLogger.WriteLine($"• {hypothesis}");
            }
        }

        private class GroupEncodingPattern
        {
            public string Type { get; set; } = "";
            public List<float> UniqueMaxZValues { get; set; } = new();
            public List<float> UniqueMaxYValues { get; set; } = new();
            public List<float> UniqueCenterZValues { get; set; } = new();
            public double MaxZConsistency { get; set; }
            public double MaxYConsistency { get; set; }
            public double CenterZConsistency { get; set; }
            public bool HasLargeValues { get; set; }
        }
    }
}
