using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Exports MSLK analysis results to structured JSON format.
    /// Provides per-PM4 file structured data export for further analysis.
    /// </summary>
    public class MslkJsonExporter
    {
        /// <summary>
        /// MSLK entry data structure for JSON export
        /// </summary>
        public class MslkEntryJson
        {
            public int EntryIndex { get; set; }
            public byte Flags_0x00 { get; set; }
            public byte Sequence_0x01 { get; set; }
            public ushort Padding_0x02 { get; set; }
            public uint ParentIndex_0x04 { get; set; }
            public int MspiFirstIndex { get; set; }
            public byte MspiIndexCount { get; set; }
            public uint Constant_0x0C { get; set; }
            public ushort MsurIndex_0x10 { get; set; }
            public ushort Constant_0x12 { get; set; }
            
            // Derived properties
            public bool HasGeometry => MspiFirstIndex >= 0;
            public string NodeType => HasGeometry ? "GEOMETRY" : "DOODAD";
        }

        /// <summary>
        /// Complete MSLK analysis result for JSON export
        /// </summary>
        public class MslkAnalysisJson
        {
            public string FileName { get; set; } = "";
            public DateTime AnalysisTimestamp { get; set; } = DateTime.UtcNow;
            public MslkStatisticsJson Statistics { get; set; } = new();
            public List<MslkEntryJson> Entries { get; set; } = new();
            public MslkHierarchyJson Hierarchy { get; set; } = new();
            public Dictionary<string, object> ValidationResults { get; set; } = new();
        }

        /// <summary>
        /// MSLK statistics summary
        /// </summary>
        public class MslkStatisticsJson
        {
            public int TotalEntries { get; set; }
            public int GeometryEntries { get; set; }
            public int DoodadEntries { get; set; }
            public Dictionary<byte, int> FlagDistribution { get; set; } = new();
            public int HierarchyLevels { get; set; }
            public int RootNodes { get; set; }
        }

        /// <summary>
        /// MSLK hierarchy structure
        /// </summary>
        public class MslkHierarchyJson
        {
            public List<int> RootNodeIndices { get; set; } = new();
            public Dictionary<int, List<int>> ParentChildMap { get; set; } = new();
            public Dictionary<int, int> NodeDepths { get; set; } = new();
            public int MaxDepth { get; set; }
        }

        private readonly JsonSerializerOptions _jsonOptions;

        public MslkJsonExporter()
        {
            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
        }

        /// <summary>
        /// Analyzes MSLK chunk and exports to JSON
        /// </summary>
        public MslkAnalysisJson AnalyzeAndExport(MSLK mslkChunk, string fileName)
        {
            var analysis = new MslkAnalysisJson
            {
                FileName = fileName
            };

            if (mslkChunk?.Entries == null)
            {
                analysis.ValidationResults["Error"] = "MSLK chunk is null or has no entries";
                return analysis;
            }

            // Convert entries
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var entry = mslkChunk.Entries[i];
                analysis.Entries.Add(new MslkEntryJson
                {
                    EntryIndex = i,
                    Flags_0x00 = entry.Unknown_0x00,
                    Sequence_0x01 = entry.Unknown_0x01,
                    Padding_0x02 = entry.Unknown_0x02,
                    ParentIndex_0x04 = entry.Unknown_0x04,
                    MspiFirstIndex = entry.MspiFirstIndex,
                    MspiIndexCount = entry.MspiIndexCount,
                    Constant_0x0C = entry.Unknown_0x0C,
                    MsurIndex_0x10 = entry.Unknown_0x10,
                    Constant_0x12 = entry.Unknown_0x12
                });
            }

            // Calculate statistics
            CalculateStatistics(analysis);
            
            // Build hierarchy
            BuildHierarchy(analysis);
            
            // Validate constants
            ValidateConstants(analysis);

            return analysis;
        }

        /// <summary>
        /// Exports analysis to JSON file
        /// </summary>
        public void ExportToFile(MslkAnalysisJson analysis, string outputPath)
        {
            var json = JsonSerializer.Serialize(analysis, _jsonOptions);
            File.WriteAllText(outputPath, json);
        }

        /// <summary>
        /// Exports analysis to JSON string
        /// </summary>
        public string ExportToString(MslkAnalysisJson analysis)
        {
            return JsonSerializer.Serialize(analysis, _jsonOptions);
        }

        private void CalculateStatistics(MslkAnalysisJson analysis)
        {
            analysis.Statistics.TotalEntries = analysis.Entries.Count;
            analysis.Statistics.GeometryEntries = 0;
            analysis.Statistics.DoodadEntries = 0;

            foreach (var entry in analysis.Entries)
            {
                // Count node types
                if (entry.HasGeometry)
                    analysis.Statistics.GeometryEntries++;
                else
                    analysis.Statistics.DoodadEntries++;

                // Track flag distribution
                if (!analysis.Statistics.FlagDistribution.ContainsKey(entry.Flags_0x00))
                    analysis.Statistics.FlagDistribution[entry.Flags_0x00] = 0;
                analysis.Statistics.FlagDistribution[entry.Flags_0x00]++;
            }
        }

        private void BuildHierarchy(MslkAnalysisJson analysis)
        {
            var parentChildMap = new Dictionary<uint, List<int>>();
            var nodeDepths = new Dictionary<int, int>();

            // Build parent-child relationships
            for (int i = 0; i < analysis.Entries.Count; i++)
            {
                var entry = analysis.Entries[i];
                var parentIndex = entry.ParentIndex_0x04;

                if (!parentChildMap.ContainsKey(parentIndex))
                    parentChildMap[parentIndex] = new List<int>();
                parentChildMap[parentIndex].Add(i);
            }

            // Find root nodes (entries that don't have their parent index as another entry's index)
            var allIndices = new HashSet<uint>();
            for (int i = 0; i < analysis.Entries.Count; i++)
            {
                allIndices.Add((uint)i);
            }

            foreach (var kvp in parentChildMap)
            {
                var parentIndex = kvp.Key;
                var children = kvp.Value;

                if (!allIndices.Contains(parentIndex))
                {
                    // This parent index doesn't exist as an entry index, so children are roots
                    analysis.Hierarchy.RootNodeIndices.AddRange(children);
                }

                analysis.Hierarchy.ParentChildMap[(int)parentIndex] = children;
            }

            // Calculate depths
            CalculateDepths(analysis.Hierarchy, nodeDepths);
            analysis.Hierarchy.NodeDepths = nodeDepths;
            analysis.Hierarchy.MaxDepth = nodeDepths.Values.Count > 0 ? nodeDepths.Values.Max() : 0;
            
            analysis.Statistics.HierarchyLevels = analysis.Hierarchy.MaxDepth + 1;
            analysis.Statistics.RootNodes = analysis.Hierarchy.RootNodeIndices.Count;
        }

        private void CalculateDepths(MslkHierarchyJson hierarchy, Dictionary<int, int> depths)
        {
            foreach (var rootIndex in hierarchy.RootNodeIndices)
            {
                CalculateDepthRecursive(rootIndex, 0, hierarchy.ParentChildMap, depths);
            }
        }

        private void CalculateDepthRecursive(int nodeIndex, int depth, Dictionary<int, List<int>> parentChildMap, Dictionary<int, int> depths)
        {
            depths[nodeIndex] = depth;

            if (parentChildMap.ContainsKey(nodeIndex))
            {
                foreach (var childIndex in parentChildMap[nodeIndex])
                {
                    CalculateDepthRecursive(childIndex, depth + 1, parentChildMap, depths);
                }
            }
        }

        private void ValidateConstants(MslkAnalysisJson analysis)
        {
            var constantValidation = new Dictionary<string, object>();
            
            bool padding0x02Valid = analysis.Entries.All(e => e.Padding_0x02 == 0x0000);
            bool constant0x0CValid = analysis.Entries.All(e => e.Constant_0x0C == 0xFFFFFFFF);
            bool constant0x12Valid = analysis.Entries.All(e => e.Constant_0x12 == 0x8000);

            constantValidation["Padding_0x02_AllZero"] = padding0x02Valid;
            constantValidation["Constant_0x0C_AllFFFFFFFF"] = constant0x0CValid;
            constantValidation["Constant_0x12_All8000"] = constant0x12Valid;

            // Flag-geometry correlation validation
            var flagGeometryCorrelation = new Dictionary<string, object>();
            var flagGroups = analysis.Entries.GroupBy(e => e.Flags_0x00);
            
            foreach (var group in flagGroups)
            {
                var flag = group.Key;
                var hasGeometryCount = group.Count(e => e.HasGeometry);
                var noGeometryCount = group.Count(e => !e.HasGeometry);
                
                flagGeometryCorrelation[$"Flag_0x{flag:X2}"] = new
                {
                    TotalEntries = group.Count(),
                    HasGeometry = hasGeometryCount,
                    NoGeometry = noGeometryCount,
                    IsConsistent = hasGeometryCount == 0 || noGeometryCount == 0
                };
            }

            constantValidation["FlagGeometryCorrelation"] = flagGeometryCorrelation;
            analysis.ValidationResults = constantValidation;
        }
    }
} 