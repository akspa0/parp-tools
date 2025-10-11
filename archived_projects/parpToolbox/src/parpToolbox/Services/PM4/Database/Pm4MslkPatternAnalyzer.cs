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
    /// Analyzes MSLK link patterns to understand object→geometry relationships and field meanings.
    /// </summary>
    public class Pm4MslkPatternAnalyzer
    {
        private readonly string _databasePath;

        public Pm4MslkPatternAnalyzer(string databasePath)
        {
            _databasePath = databasePath;
        }

        /// <summary>
        /// Performs comprehensive MSLK pattern analysis to understand link relationships.
        /// </summary>
        public async Task<MslkAnalysisReport> AnalyzeMslkPatternsAsync()
        {
            using var context = new Pm4DatabaseContext(_databasePath);
            
            ConsoleLogger.WriteLine("[MSLK ANALYSIS] Starting comprehensive MSLK link pattern analysis...");
            
            var links = await context.Links.ToListAsync();
            var placements = await context.Placements.ToListAsync();
            var report = new MslkAnalysisReport();
            
            if (!links.Any())
            {
                ConsoleLogger.WriteLine("[MSLK ANALYSIS] No links found in database.");
                return report;
            }

            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Analyzing {links.Count} MSLK links and {placements.Count} MPRL placements...");

            // Parse raw field data to extract link fields
            var linkFields = ExtractLinkFields(links);
            
            // Analyze link field patterns
            AnalyzeParentIndexPatterns(linkFields, report);
            AnalyzeMspiIndexPatterns(linkFields, report);
            AnalyzeReferenceIndexPatterns(linkFields, report);
            AnalyzeUnknownFieldPatterns(linkFields, report);
            
            // Analyze MPRL→MSLK relationships
            AnalyzeMprlToMslkRelationships(placements, linkFields, report);
            
            // Analyze geometry coverage patterns
            AnalyzeGeometryCoverage(linkFields, report);
            
            // Look for hierarchical patterns
            AnalyzeHierarchicalPatterns(linkFields, report);
            
            ConsoleLogger.WriteLine("[MSLK ANALYSIS] MSLK pattern analysis complete.");
            return report;
        }

        private List<MslkLinkFields> ExtractLinkFields(List<Pm4Link> links)
        {
            var linkFields = new List<MslkLinkFields>();
            
            foreach (var link in links)
            {
                try
                {
                    var rawData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(link.RawFieldsJson);
                    
                    var fields = new MslkLinkFields
                    {
                        LinkId = link.Id,
                        ParentIndex = link.ParentIndex,
                        MspiFirstIndex = link.MspiFirstIndex,
                        MspiIndexCount = link.MspiIndexCount,
                        ReferenceIndex = link.ReferenceIndex
                    };
                    
                    // Extract additional fields from raw JSON
                    if (rawData != null)
                    {
                        foreach (var kvp in rawData)
                        {
                            if (kvp.Key.StartsWith("Unknown") && kvp.Value is JsonElement elem)
                            {
                                fields.UnknownFields[kvp.Key] = elem.ToString();
                            }
                        }
                    }
                        
                    linkFields.Add(fields);
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Error parsing link {link.Id}: {ex.Message}");
                }
            }
            
            return linkFields;
        }

        private void AnalyzeParentIndexPatterns(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            var parentIndices = fields.Select(f => f.ParentIndex).ToList();
            var distribution = parentIndices.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.ParentIndexDistribution = distribution;
            report.ParentIndexUniqueCount = distribution.Count;
            report.ParentIndexRange = new ValueRange<uint>
            {
                Min = parentIndices.Min(),
                Max = parentIndices.Max(),
                Average = parentIndices.Select(x => (double)x).Average()
            };
            
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] ParentIndex: {report.ParentIndexUniqueCount} unique values");
            ConsoleLogger.WriteLine($"  Range: [{report.ParentIndexRange.Min}, {report.ParentIndexRange.Max}] avg: {report.ParentIndexRange.Average:F1}");
            
            // Show top parent indices by frequency
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} links ({count * 100.0 / fields.Count:F1}%)");
            }
        }

        private void AnalyzeMspiIndexPatterns(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            var mspiFirstIndices = fields.Select(f => f.MspiFirstIndex).ToList();
            var mspiIndexCounts = fields.Select(f => f.MspiIndexCount).ToList();
            
            // Analyze FirstIndex patterns
            var firstIndexDistribution = mspiFirstIndices.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            report.MspiFirstIndexDistribution = firstIndexDistribution;
            
            // Analyze IndexCount patterns  
            var indexCountDistribution = mspiIndexCounts.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            report.MspiIndexCountDistribution = indexCountDistribution;
            
            // Special analysis for -1 values (container nodes)
            var containerNodes = fields.Where(f => f.MspiFirstIndex == -1).ToList();
            report.ContainerNodeCount = containerNodes.Count;
            report.ContainerNodePercentage = containerNodes.Count * 100.0 / fields.Count;
            
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] MSPI Patterns:");
            ConsoleLogger.WriteLine($"  FirstIndex unique values: {firstIndexDistribution.Count}");
            ConsoleLogger.WriteLine($"  IndexCount unique values: {indexCountDistribution.Count}");
            ConsoleLogger.WriteLine($"  Container nodes (FirstIndex=-1): {report.ContainerNodeCount} ({report.ContainerNodePercentage:F1}%)");
            
            if (containerNodes.Any())
            {
                var containerParentIndices = containerNodes.Select(c => c.ParentIndex).Distinct().ToList();
                ConsoleLogger.WriteLine($"  Container parent indices: {containerParentIndices.Count} unique values");
                report.ContainerParentIndices = containerParentIndices;
            }
        }

        private void AnalyzeReferenceIndexPatterns(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            var referenceIndices = fields.Select(f => f.ReferenceIndex).ToList();
            var distribution = referenceIndices.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            report.ReferenceIndexDistribution = distribution;
            report.ReferenceIndexUniqueCount = distribution.Count;
            
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] ReferenceIndex: {report.ReferenceIndexUniqueCount} unique values");
            
            // Show patterns
            foreach (var (value, count) in distribution.OrderByDescending(kvp => kvp.Value).Take(5))
            {
                ConsoleLogger.WriteLine($"  {value}: {count} occurrences ({count * 100.0 / fields.Count:F1}%)");
            }
        }

        private void AnalyzeUnknownFieldPatterns(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            var allUnknownKeys = fields.SelectMany(f => f.UnknownFields.Keys).Distinct().ToList();
            
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Unknown fields found: {allUnknownKeys.Count}");
            foreach (var key in allUnknownKeys)
            {
                var values = fields.Where(f => f.UnknownFields.ContainsKey(key))
                                  .Select(f => f.UnknownFields[key])
                                  .GroupBy(v => v)
                                  .ToDictionary(g => g.Key, g => g.Count());
                                  
                ConsoleLogger.WriteLine($"  {key}: {values.Count} unique values");
                report.UnknownFieldDistributions[key] = values;
            }
        }

        private void AnalyzeMprlToMslkRelationships(List<Pm4Placement> placements, List<MslkLinkFields> linkFields, MslkAnalysisReport report)
        {
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Analyzing MPRL→MSLK relationships...");
            
            // Extract MPRL Unknown4 values for comparison
            var mprlObjectIds = new List<uint>();
            foreach (var placement in placements)
            {
                mprlObjectIds.Add(placement.Unknown4);
            }
            
            var uniqueMprlObjectIds = mprlObjectIds.Distinct().ToList();
            var uniqueMslkParentIndices = linkFields.Select(l => l.ParentIndex).Distinct().ToList();
            
            // Find overlaps
            var overlapping = uniqueMprlObjectIds.Intersect(uniqueMslkParentIndices).ToList();
            var mprlOnly = uniqueMprlObjectIds.Except(uniqueMslkParentIndices).ToList();
            var mslkOnly = uniqueMslkParentIndices.Except(uniqueMprlObjectIds).ToList();
            
            report.MprlMslkOverlap = new MprlMslkRelationshipAnalysis
            {
                MprlUniqueObjectIds = uniqueMprlObjectIds.Count,
                MslkUniqueParentIndices = uniqueMslkParentIndices.Count,
                OverlappingIds = overlapping.Count,
                MprlOnlyIds = mprlOnly.Count,
                MslkOnlyIds = mslkOnly.Count,
                OverlapPercentage = overlapping.Count * 100.0 / Math.Max(uniqueMprlObjectIds.Count, uniqueMslkParentIndices.Count)
            };
            
            ConsoleLogger.WriteLine($"  MPRL unique Object IDs: {report.MprlMslkOverlap.MprlUniqueObjectIds}");
            ConsoleLogger.WriteLine($"  MSLK unique ParentIndices: {report.MprlMslkOverlap.MslkUniqueParentIndices}");
            ConsoleLogger.WriteLine($"  Overlapping IDs: {report.MprlMslkOverlap.OverlappingIds} ({report.MprlMslkOverlap.OverlapPercentage:F1}%)");
            ConsoleLogger.WriteLine($"  MPRL-only IDs: {report.MprlMslkOverlap.MprlOnlyIds}");
            ConsoleLogger.WriteLine($"  MSLK-only IDs: {report.MprlMslkOverlap.MslkOnlyIds}");
        }

        private void AnalyzeGeometryCoverage(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            var geometryLinks = fields.Where(f => f.MspiFirstIndex != -1).ToList();
            var totalTriangles = geometryLinks.Sum(f => f.MspiIndexCount / 3); // Assuming triangles
            
            report.GeometryCoverage = new GeometryCoverageAnalysis
            {
                TotalLinks = fields.Count,
                GeometryLinks = geometryLinks.Count,
                ContainerLinks = fields.Count - geometryLinks.Count,
                TotalTriangles = totalTriangles,
                AverageTrianglesPerLink = geometryLinks.Any() ? geometryLinks.Average(f => f.MspiIndexCount / 3.0) : 0
            };
            
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Geometry Coverage:");
            ConsoleLogger.WriteLine($"  Total links: {report.GeometryCoverage.TotalLinks}");
            ConsoleLogger.WriteLine($"  Geometry links: {report.GeometryCoverage.GeometryLinks}");
            ConsoleLogger.WriteLine($"  Container links: {report.GeometryCoverage.ContainerLinks}");
            ConsoleLogger.WriteLine($"  Estimated triangles: {report.GeometryCoverage.TotalTriangles:N0}");
            ConsoleLogger.WriteLine($"  Avg triangles/link: {report.GeometryCoverage.AverageTrianglesPerLink:F1}");
        }

        private void AnalyzeHierarchicalPatterns(List<MslkLinkFields> fields, MslkAnalysisReport report)
        {
            ConsoleLogger.WriteLine($"[MSLK ANALYSIS] Hierarchical Patterns:");
            
            // Group by ParentIndex to see object structure
            var objectGroups = fields.GroupBy(f => f.ParentIndex).ToList();
            
            var groupSizes = objectGroups.Select(g => g.Count()).ToList();
            report.HierarchicalAnalysis = new HierarchicalAnalysis
            {
                ObjectGroups = objectGroups.Count,
                MinLinksPerObject = groupSizes.Min(),
                MaxLinksPerObject = groupSizes.Max(),
                AverageLinksPerObject = groupSizes.Average()
            };
            
            ConsoleLogger.WriteLine($"  Object groups (by ParentIndex): {report.HierarchicalAnalysis.ObjectGroups}");
            ConsoleLogger.WriteLine($"  Links per object: [{report.HierarchicalAnalysis.MinLinksPerObject}, {report.HierarchicalAnalysis.MaxLinksPerObject}] avg: {report.HierarchicalAnalysis.AverageLinksPerObject:F1}");
            
            // Show examples of complex objects
            var complexObjects = objectGroups.Where(g => g.Count() > 10).Take(5).ToList();
            ConsoleLogger.WriteLine($"  Complex objects (>10 links):");
            foreach (var obj in complexObjects)
            {
                var containerCount = obj.Count(l => l.MspiFirstIndex == -1);
                var geometryCount = obj.Count(l => l.MspiFirstIndex != -1);
                ConsoleLogger.WriteLine($"    ParentIndex {obj.Key}: {obj.Count()} links ({containerCount} containers, {geometryCount} geometry)");
            }
        }
    }

    public class MslkAnalysisReport
    {
        public Dictionary<uint, int> ParentIndexDistribution { get; set; } = new();
        public Dictionary<int, int> MspiFirstIndexDistribution { get; set; } = new();
        public Dictionary<int, int> MspiIndexCountDistribution { get; set; } = new();
        public Dictionary<uint, int> ReferenceIndexDistribution { get; set; } = new();
        public Dictionary<string, Dictionary<string, int>> UnknownFieldDistributions { get; set; } = new();
        
        public int ParentIndexUniqueCount { get; set; }
        public int ReferenceIndexUniqueCount { get; set; }
        public ValueRange<uint> ParentIndexRange { get; set; } = new();
        
        public int ContainerNodeCount { get; set; }
        public double ContainerNodePercentage { get; set; }
        public List<uint> ContainerParentIndices { get; set; } = new();
        
        public MprlMslkRelationshipAnalysis MprlMslkOverlap { get; set; } = new();
        public GeometryCoverageAnalysis GeometryCoverage { get; set; } = new();
        public HierarchicalAnalysis HierarchicalAnalysis { get; set; } = new();
    }

    public class MslkLinkFields
    {
        public int LinkId { get; set; }
        public uint ParentIndex { get; set; }
        public int MspiFirstIndex { get; set; }
        public int MspiIndexCount { get; set; }
        public uint ReferenceIndex { get; set; }
        public Dictionary<string, string> UnknownFields { get; set; } = new();
    }

    public class ValueRange<T> where T : struct
    {
        public T Min { get; set; }
        public T Max { get; set; }
        public double Average { get; set; }
    }

    public class MprlMslkRelationshipAnalysis
    {
        public int MprlUniqueObjectIds { get; set; }
        public int MslkUniqueParentIndices { get; set; }
        public int OverlappingIds { get; set; }
        public int MprlOnlyIds { get; set; }
        public int MslkOnlyIds { get; set; }
        public double OverlapPercentage { get; set; }
    }

    public class GeometryCoverageAnalysis
    {
        public int TotalLinks { get; set; }
        public int GeometryLinks { get; set; }
        public int ContainerLinks { get; set; }
        public int TotalTriangles { get; set; }
        public double AverageTrianglesPerLink { get; set; }
    }

    public class HierarchicalAnalysis
    {
        public int ObjectGroups { get; set; }
        public int MinLinksPerObject { get; set; }
        public int MaxLinksPerObject { get; set; }
        public double AverageLinksPerObject { get; set; }
    }
}
