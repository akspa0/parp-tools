using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Validates PM4 chunk parsing against authoritative wowdev.wiki documentation
    /// to ensure we're not hallucinating or misinterpreting fields.
    /// </summary>
    public class Pm4ChunkValidationTool
    {
        public class MsurFieldValidation
        {
            public string FieldName { get; set; } = "";
            public string OfficialName { get; set; } = "";
            public string OfficialType { get; set; } = "";
            public string OfficialDescription { get; set; } = "";
            public object? CurrentValue { get; set; }
            public string CurrentType { get; set; } = "";
            public bool IsValidated { get; set; }
            public string ValidationNotes { get; set; } = "";
        }
        
        public class ChunkValidationReport
        {
            public string ChunkType { get; set; } = "";
            public int TotalFields { get; set; }
            public int ValidatedFields { get; set; }
            public int FabricatedFields { get; set; }
            public List<MsurFieldValidation> FieldValidations { get; set; } = new();
            public List<string> CriticalIssues { get; set; } = new();
            public Dictionary<string, int> ValueDistribution { get; set; } = new();
        }
        
        /// <summary>
        /// Validates MSUR chunk parsing against official wowdev.wiki documentation
        /// </summary>
        public ChunkValidationReport ValidateMsurChunk(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("[CHUNK VALIDATOR] Starting MSUR chunk validation against wowdev.wiki documentation...");
            
            var report = new ChunkValidationReport
            {
                ChunkType = "MSUR"
            };
            
            // Official MSUR structure from wowdev.wiki PD4.md
            var officialFields = new[]
            {
                new { Offset = "0x00", Name = "_0x00", Type = "uint8_t", Description = "flags (bitmask32)" },
                new { Offset = "0x01", Name = "_0x01", Type = "uint8_t", Description = "count of indices in MSVI" },
                new { Offset = "0x02", Name = "_0x02", Type = "uint8_t", Description = "unknown" },
                new { Offset = "0x03", Name = "_0x03", Type = "uint8_t", Description = "Always 0, padding" },
                new { Offset = "0x04", Name = "_0x04", Type = "float", Description = "Unknown float" },
                new { Offset = "0x08", Name = "_0x08", Type = "float", Description = "Unknown float" },
                new { Offset = "0x0C", Name = "_0x0c", Type = "float", Description = "Unknown float" },
                new { Offset = "0x10", Name = "_0x10", Type = "float", Description = "Unknown float" },
                new { Offset = "0x14", Name = "MSVI_first_index", Type = "uint32_t", Description = "Vertex index start" },
                new { Offset = "0x18", Name = "_0x18", Type = "uint32_t", Description = "Unknown uint32" },
                new { Offset = "0x1C", Name = "_0x1c", Type = "uint32_t", Description = "Unknown uint32" }
            };
            
            ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Official MSUR structure has {officialFields.Length} fields");
            
            if (!scene.Surfaces.Any())
            {
                report.CriticalIssues.Add("No MSUR surfaces found in scene");
                return report;
            }
            
            // Analyze first surface entry to understand current field mapping
            var sampleSurface = scene.Surfaces.First();
            var surfaceType = sampleSurface.GetType();
            var properties = surfaceType.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            
            ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Current implementation has {properties.Length} properties");
            
            report.TotalFields = properties.Length;
            
            // Map current properties against official structure
            foreach (var prop in properties)
            {
                var validation = new MsurFieldValidation
                {
                    FieldName = prop.Name,
                    CurrentType = prop.PropertyType.Name
                };
                
                try
                {
                    validation.CurrentValue = prop.GetValue(sampleSurface);
                }
                catch (Exception ex)
                {
                    validation.ValidationNotes = $"Error reading value: {ex.Message}";
                }
                
                // Map against official fields
                switch (prop.Name)
                {
                    case "FlagsOrUnknown_0x00":
                        validation.OfficialName = "_0x00";
                        validation.OfficialType = "uint8_t";
                        validation.OfficialDescription = "flags (bitmask32)";
                        validation.IsValidated = true;
                        break;
                        
                    case "IndexCount":
                        validation.OfficialName = "_0x01";
                        validation.OfficialType = "uint8_t";
                        validation.OfficialDescription = "count of indices in MSVI";
                        validation.IsValidated = true;
                        break;
                        
                    case "Unknown_0x02":
                        validation.OfficialName = "_0x02";
                        validation.OfficialType = "uint8_t";
                        validation.OfficialDescription = "unknown";
                        validation.IsValidated = true;
                        break;
                        
                    case "Padding_0x03":
                        validation.OfficialName = "_0x03";
                        validation.OfficialType = "uint8_t";
                        validation.OfficialDescription = "Always 0, padding";
                        validation.IsValidated = true;
                        break;
                        
                    case "Nx":
                        validation.OfficialName = "_0x04";
                        validation.OfficialType = "float";
                        validation.OfficialDescription = "Unknown float (NOT confirmed as normal X)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as normal X but not confirmed in official docs";
                        break;
                        
                    case "Ny":
                        validation.OfficialName = "_0x08";
                        validation.OfficialType = "float";
                        validation.OfficialDescription = "Unknown float (NOT confirmed as normal Y)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as normal Y but not confirmed in official docs";
                        break;
                        
                    case "Nz":
                        validation.OfficialName = "_0x0c";
                        validation.OfficialType = "float";
                        validation.OfficialDescription = "Unknown float (NOT confirmed as normal Z)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as normal Z but not confirmed in official docs";
                        break;
                        
                    case "Height":
                        validation.OfficialName = "_0x10";
                        validation.OfficialType = "float";
                        validation.OfficialDescription = "Unknown float (NOT confirmed as height)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as height but not confirmed in official docs";
                        break;
                        
                    case "MsviFirstIndex":
                        validation.OfficialName = "MSVI_first_index";
                        validation.OfficialType = "uint32_t";
                        validation.OfficialDescription = "Vertex index start";
                        validation.IsValidated = true;
                        break;
                        
                    case "MdosIndex":
                        validation.OfficialName = "_0x18";
                        validation.OfficialType = "uint32_t";
                        validation.OfficialDescription = "Unknown uint32 (NOT confirmed as MDOS index)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as MDOS index but not confirmed in official docs";
                        break;
                        
                    case "PackedParams":
                        validation.OfficialName = "_0x1c";
                        validation.OfficialType = "uint32_t";
                        validation.OfficialDescription = "Unknown uint32 (NOT confirmed as packed params)";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "ASSUMPTION: Interpreted as packed params but not confirmed in official docs";
                        break;
                        
                    default:
                        validation.OfficialName = "UNKNOWN";
                        validation.OfficialType = "UNKNOWN";
                        validation.OfficialDescription = "Field not found in official documentation";
                        validation.IsValidated = false;
                        validation.ValidationNotes = "FABRICATED: This field does not exist in official documentation";
                        break;
                }
                
                report.FieldValidations.Add(validation);
            }
            
            // Calculate statistics
            report.ValidatedFields = report.FieldValidations.Count(f => f.IsValidated);
            report.FabricatedFields = report.FieldValidations.Count(f => !f.IsValidated);
            
            // Analyze value distributions for key fields
            AnalyzeValueDistributions(scene, report);
            
            // Generate critical issues
            GenerateCriticalIssues(report);
            
            // Write validation report
            WriteValidationReport(report, outputDirectory);
            
            ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Validation complete:");
            ConsoleLogger.WriteLine($"  Validated fields: {report.ValidatedFields}/{report.TotalFields}");
            ConsoleLogger.WriteLine($"  Fabricated/unconfirmed fields: {report.FabricatedFields}");
            ConsoleLogger.WriteLine($"  Critical issues: {report.CriticalIssues.Count}");
            
            return report;
        }
        
        private void AnalyzeValueDistributions(Pm4Scene scene, ChunkValidationReport report)
        {
            ConsoleLogger.WriteLine("[CHUNK VALIDATOR] Analyzing value distributions...");
            
            // Analyze FlagsOrUnknown_0x00 (GroupKey) distribution
            var groupKeyValues = new Dictionary<byte, int>();
            var indexCountValues = new Dictionary<byte, int>();
            var unknown02Values = new Dictionary<byte, int>();
            
            foreach (var surface in scene.Surfaces.Take(10000)) // Sample to avoid memory issues
            {
                var surfaceType = surface.GetType();
                
                try
                {
                    var flagsProp = surfaceType.GetProperty("FlagsOrUnknown_0x00");
                    if (flagsProp != null)
                    {
                        var value = (byte)(flagsProp.GetValue(surface) ?? 0);
                        groupKeyValues[value] = groupKeyValues.GetValueOrDefault(value, 0) + 1;
                    }
                    
                    var indexProp = surfaceType.GetProperty("IndexCount");
                    if (indexProp != null)
                    {
                        var value = (byte)(indexProp.GetValue(surface) ?? 0);
                        indexCountValues[value] = indexCountValues.GetValueOrDefault(value, 0) + 1;
                    }
                    
                    var unknown02Prop = surfaceType.GetProperty("Unknown_0x02");
                    if (unknown02Prop != null)
                    {
                        var value = (byte)(unknown02Prop.GetValue(surface) ?? 0);
                        unknown02Values[value] = unknown02Values.GetValueOrDefault(value, 0) + 1;
                    }
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Error analyzing surface: {ex.Message}");
                }
            }
            
            report.ValueDistribution["GroupKey_UniqueValues"] = groupKeyValues.Count;
            report.ValueDistribution["IndexCount_UniqueValues"] = indexCountValues.Count;
            report.ValueDistribution["Unknown02_UniqueValues"] = unknown02Values.Count;
            
            ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Value distribution:");
            ConsoleLogger.WriteLine($"  GroupKey unique values: {groupKeyValues.Count}");
            ConsoleLogger.WriteLine($"  IndexCount unique values: {indexCountValues.Count}");
            ConsoleLogger.WriteLine($"  Unknown_0x02 unique values: {unknown02Values.Count}");
        }
        
        private void GenerateCriticalIssues(ChunkValidationReport report)
        {
            // Check for fabricated fields
            var fabricatedFields = report.FieldValidations.Where(f => !f.IsValidated).ToList();
            if (fabricatedFields.Any())
            {
                report.CriticalIssues.Add($"Found {fabricatedFields.Count} unvalidated/fabricated fields that may be causing grouping issues");
            }
            
            // Check for single-value distributions (indicating grouping problems)
            if (report.ValueDistribution.GetValueOrDefault("GroupKey_UniqueValues", 0) <= 1)
            {
                report.CriticalIssues.Add("GroupKey field has only 1 unique value - explains massive single group problem");
            }
            
            if (report.ValueDistribution.GetValueOrDefault("Unknown02_UniqueValues", 0) <= 1)
            {
                report.CriticalIssues.Add("Unknown_0x02 field has only 1 unique value - not suitable for grouping");
            }
        }
        
        private void WriteValidationReport(ChunkValidationReport report, string outputDirectory)
        {
            var reportPath = Path.Combine(outputDirectory, "msur_chunk_validation_report.txt");
            
            using var writer = new StreamWriter(reportPath);
            
            writer.WriteLine("=== MSUR CHUNK VALIDATION REPORT ===");
            writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine();
            
            writer.WriteLine("=== SUMMARY ===");
            writer.WriteLine($"Total fields: {report.TotalFields}");
            writer.WriteLine($"Validated fields: {report.ValidatedFields}");
            writer.WriteLine($"Fabricated/unconfirmed fields: {report.FabricatedFields}");
            writer.WriteLine();
            
            writer.WriteLine("=== CRITICAL ISSUES ===");
            foreach (var issue in report.CriticalIssues)
            {
                writer.WriteLine($"- {issue}");
            }
            writer.WriteLine();
            
            writer.WriteLine("=== FIELD VALIDATION DETAILS ===");
            foreach (var field in report.FieldValidations)
            {
                writer.WriteLine($"Field: {field.FieldName}");
                writer.WriteLine($"  Official: {field.OfficialName} ({field.OfficialType}) - {field.OfficialDescription}");
                writer.WriteLine($"  Current Type: {field.CurrentType}");
                writer.WriteLine($"  Sample Value: {field.CurrentValue}");
                writer.WriteLine($"  Validated: {field.IsValidated}");
                if (!string.IsNullOrEmpty(field.ValidationNotes))
                {
                    writer.WriteLine($"  Notes: {field.ValidationNotes}");
                }
                writer.WriteLine();
            }
            
            writer.WriteLine("=== VALUE DISTRIBUTIONS ===");
            foreach (var kvp in report.ValueDistribution)
            {
                writer.WriteLine($"{kvp.Key}: {kvp.Value}");
            }
            
            ConsoleLogger.WriteLine($"[CHUNK VALIDATOR] Report saved to: {reportPath}");
        }
    }
}
