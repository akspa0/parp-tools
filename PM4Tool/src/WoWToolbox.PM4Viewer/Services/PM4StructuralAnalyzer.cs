using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.PM4Viewer.Services
{
    /// <summary>
    /// Advanced structural analysis of PM4 files to discover hidden patterns,
    /// hierarchical relationships, and decode unknown data fields.
    /// </summary>
    public class PM4StructuralAnalyzer
    {
        public class ChunkPaddingAnalysis
        {
            public string ChunkType { get; set; } = string.Empty;
            public int ExpectedSize { get; set; }
            public int ActualSize { get; set; }
            public int PaddingBytes { get; set; }
            public byte[] PaddingData { get; set; } = Array.Empty<byte>();
            public bool HasNonZeroPadding => PaddingData.Any(b => b != 0);
            public Dictionary<byte, int> PaddingValueFrequency { get; set; } = new();
        }

        public class UnknownFieldAnalysis
        {
            public string ChunkType { get; set; } = string.Empty;
            public string FieldName { get; set; } = string.Empty;
            public Type DataType { get; set; } = typeof(object);
            public List<object> Values { get; set; } = new();
            public List<object> UniqueValues { get; set; } = new();
            public Dictionary<object, int> ValueFrequency { get; set; } = new();
            public bool LooksLikeIndex { get; set; } = false;
            public bool LooksLikeFlags { get; set; } = false;
            public bool LooksLikeCount { get; set; } = false;
            public Range ValueRange { get; set; }
        }

        public class HierarchicalRelationship
        {
            public string ParentChunk { get; set; } = string.Empty;
            public string ChildChunk { get; set; } = string.Empty;
            public string RelationshipType { get; set; } = string.Empty; // "Index", "Count", "Reference"
            public string Evidence { get; set; } = string.Empty;
            public float ConfidenceScore { get; set; } // 0.0 to 1.0
        }

        public class NodeStructureAnalysis
        {
            public string ChunkType { get; set; } = string.Empty;
            public List<NodeGroup> NodeGroups { get; set; } = new();
            public Dictionary<uint, List<int>> GroupToIndices { get; set; } = new();
            public bool HasHierarchicalStructure => NodeGroups.Any(g => g.HasChildren);
        }

        public class NodeGroup
        {
            public uint GroupId { get; set; }
            public int EntryCount { get; set; }
            public List<int> EntryIndices { get; set; } = new();
            public List<uint> ChildGroups { get; set; } = new();
            public bool HasChildren => ChildGroups.Count > 0;
            public string GroupType { get; set; } = "Unknown";
        }

        public class StructuralAnalysisResult
        {
            public string FileName { get; set; } = string.Empty;
            public List<ChunkPaddingAnalysis> PaddingAnalysis { get; set; } = new();
            public List<UnknownFieldAnalysis> UnknownFields { get; set; } = new();
            public List<HierarchicalRelationship> Hierarchies { get; set; } = new();
            public List<NodeStructureAnalysis> NodeStructures { get; set; } = new();
            public Dictionary<string, object> Metadata { get; set; } = new();
        }

        public StructuralAnalysisResult AnalyzeFile(string pm4FilePath)
        {
            var result = new StructuralAnalysisResult
            {
                FileName = Path.GetFileName(pm4FilePath)
            };

            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);
                
                // 1. Analyze chunk padding for hidden metadata
                result.PaddingAnalysis = AnalyzeChunkPadding(pm4FilePath);
                
                // 2. Decode unknown fields across all chunks
                result.UnknownFields = AnalyzeUnknownFields(pm4File);
                
                // 3. Discover hierarchical relationships between chunks
                result.Hierarchies = DiscoverHierarchicalRelationships(pm4File);
                
                // 4. Analyze node-based structures and groupings
                result.NodeStructures = AnalyzeNodeStructures(pm4File);
                
                // 5. Extract file-level metadata patterns
                result.Metadata = ExtractMetadataPatterns(pm4File);
            }
            catch (Exception ex)
            {
                result.Metadata["Error"] = ex.Message;
            }

            return result;
        }

        private List<ChunkPaddingAnalysis> AnalyzeChunkPadding(string filePath)
        {
            var paddingAnalysis = new List<ChunkPaddingAnalysis>();
            
            try
            {
                using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                using var br = new BinaryReader(fs);
                
                while (fs.Position < fs.Length - 8) // At least 8 bytes for chunk header
                {
                    var chunkStart = fs.Position;
                    var signature = new string(br.ReadChars(4));
                    var chunkSize = br.ReadUInt32();
                    
                    var analysis = new ChunkPaddingAnalysis
                    {
                        ChunkType = signature,
                        ActualSize = (int)chunkSize
                    };
                    
                    // Read chunk data and look for padding patterns
                    var chunkData = br.ReadBytes((int)chunkSize);
                    
                    // Check for padding at end of chunk (common WoW pattern)
                    var paddingStart = chunkData.Length;
                    for (int i = chunkData.Length - 1; i >= 0; i--)
                    {
                        if (chunkData[i] != 0) break;
                        paddingStart = i;
                    }
                    
                    if (paddingStart < chunkData.Length)
                    {
                        analysis.PaddingBytes = chunkData.Length - paddingStart;
                        analysis.PaddingData = chunkData[paddingStart..];
                        
                        // Analyze padding value distribution
                        analysis.PaddingValueFrequency = analysis.PaddingData
                            .GroupBy(b => b)
                            .ToDictionary(g => g.Key, g => g.Count());
                    }
                    
                    analysis.ExpectedSize = paddingStart;
                    paddingAnalysis.Add(analysis);
                }
            }
            catch (Exception ex)
            {
                // Add error analysis
                paddingAnalysis.Add(new ChunkPaddingAnalysis 
                { 
                    ChunkType = "ERROR", 
                    PaddingData = System.Text.Encoding.UTF8.GetBytes(ex.Message) 
                });
            }
            
            return paddingAnalysis;
        }

        private List<UnknownFieldAnalysis> AnalyzeUnknownFields(PM4File pm4File)
        {
            var unknownFields = new List<UnknownFieldAnalysis>();
            
            // Analyze MSLK unknown fields (most promising for hierarchical data)
            if (pm4File.MSLK?.Entries != null)
            {
                unknownFields.AddRange(AnalyzeMslkUnknownFields(pm4File.MSLK.Entries));
            }
            
            // Analyze MSUR unknown fields
            if (pm4File.MSUR?.Entries != null)
            {
                unknownFields.AddRange(AnalyzeMsurUnknownFields(pm4File.MSUR.Entries));
            }
            
            // Analyze MSHD header unknowns
            if (pm4File.MSHD != null)
            {
                unknownFields.AddRange(AnalyzeMshdUnknownFields(pm4File.MSHD));
            }
            
            return unknownFields;
        }

        private List<UnknownFieldAnalysis> AnalyzeMslkUnknownFields(List<MSLKEntry> entries)
        {
            var analyses = new List<UnknownFieldAnalysis>();
            
            // Analyze Unknown_0x00 (suspected node type/flags)
            var unk00Analysis = new UnknownFieldAnalysis
            {
                ChunkType = "MSLK",
                FieldName = "Unknown_0x00",
                DataType = typeof(byte),
                Values = entries.Select(e => (object)e.Unknown_0x00).ToList()
            };
            AnalyzeFieldPattern(unk00Analysis);
            analyses.Add(unk00Analysis);
            
            // Analyze Unknown_0x04 (suspected group/internal index)
            var unk04Analysis = new UnknownFieldAnalysis
            {
                ChunkType = "MSLK",
                FieldName = "Unknown_0x04",
                DataType = typeof(uint),
                Values = entries.Select(e => (object)e.Unknown_0x04).ToList()
            };
            AnalyzeFieldPattern(unk04Analysis);
            analyses.Add(unk04Analysis);
            
            // Analyze Unknown_0x0C (suspected material/metadata)
            var unk0CAnalysis = new UnknownFieldAnalysis
            {
                ChunkType = "MSLK",
                FieldName = "Unknown_0x0C",
                DataType = typeof(uint),
                Values = entries.Select(e => (object)e.Unknown_0x0C).ToList()
            };
            AnalyzeFieldPattern(unk0CAnalysis);
            analyses.Add(unk0CAnalysis);
            
            return analyses;
        }

        private List<UnknownFieldAnalysis> AnalyzeMsurUnknownFields(List<MsurEntry> entries)
        {
            var analyses = new List<UnknownFieldAnalysis>();
            
            // Analyze FlagsOrUnknown_0x00
            var flagsAnalysis = new UnknownFieldAnalysis
            {
                ChunkType = "MSUR",
                FieldName = "FlagsOrUnknown_0x00",
                DataType = typeof(byte),
                Values = entries.Select(e => (object)e.FlagsOrUnknown_0x00).ToList()
            };
            AnalyzeFieldPattern(flagsAnalysis);
            analyses.Add(flagsAnalysis);
            
            return analyses;
        }

        private List<UnknownFieldAnalysis> AnalyzeMshdUnknownFields(MSHDChunk mshd)
        {
            var analyses = new List<UnknownFieldAnalysis>();
            
            // Analyze all MSHD unknown fields
            var unknownFields = new Dictionary<string, uint>
            {
                ["Unknown_0x00"] = mshd.Unknown_0x00,
                ["Unknown_0x04"] = mshd.Unknown_0x04,
                ["Unknown_0x08"] = mshd.Unknown_0x08,
                ["Unknown_0x0C"] = mshd.Unknown_0x0C,
                ["Unknown_0x10"] = mshd.Unknown_0x10,
                ["Unknown_0x14"] = mshd.Unknown_0x14,
                ["Unknown_0x18"] = mshd.Unknown_0x18,
                ["Unknown_0x1C"] = mshd.Unknown_0x1C
            };
            
            foreach (var (fieldName, value) in unknownFields)
            {
                var analysis = new UnknownFieldAnalysis
                {
                    ChunkType = "MSHD",
                    FieldName = fieldName,
                    DataType = typeof(uint),
                    Values = new List<object> { value }
                };
                AnalyzeFieldPattern(analysis);
                analyses.Add(analysis);
            }
            
            return analyses;
        }

        private void AnalyzeFieldPattern(UnknownFieldAnalysis analysis)
        {
            analysis.UniqueValues = analysis.Values.Distinct().ToList();
            analysis.ValueFrequency = analysis.Values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            
            // Detect patterns
            if (analysis.DataType == typeof(byte) || analysis.DataType == typeof(uint))
            {
                var numericValues = analysis.Values.Cast<IConvertible>().Select(v => Convert.ToUInt32(v)).ToList();
                
                // Check if looks like index (sequential or bounded)
                var max = numericValues.Max();
                var min = numericValues.Min();
                analysis.ValueRange = new Range((int)min, (int)max);
                
                // Sequential check
                var sortedUnique = numericValues.Distinct().OrderBy(x => x).ToList();
                bool isSequential = sortedUnique.Count > 1 && 
                                  sortedUnique.Zip(sortedUnique.Skip(1), (a, b) => b - a).All(diff => diff == 1);
                
                // Bounded check (values within reasonable index range)
                bool isBounded = max < numericValues.Count * 10; // heuristic
                
                analysis.LooksLikeIndex = isSequential || isBounded;
                
                // Check if looks like flags (powers of 2, OR combinations)
                var powerOf2Values = numericValues.Where(v => v > 0 && (v & (v - 1)) == 0).ToList();
                analysis.LooksLikeFlags = powerOf2Values.Count > analysis.UniqueValues.Count * 0.5;
                
                // Check if looks like count (small values, often correlates with other counts)
                analysis.LooksLikeCount = max < 100 && analysis.UniqueValues.Count < 20;
            }
        }

        private List<HierarchicalRelationship> DiscoverHierarchicalRelationships(PM4File pm4File)
        {
            var relationships = new List<HierarchicalRelationship>();
            
            // MSLK -> MSPI -> MSPV relationship chain
            if (pm4File.MSLK?.Entries != null && pm4File.MSPI?.Indices != null && pm4File.MSPV?.Vertices != null)
            {
                relationships.Add(new HierarchicalRelationship
                {
                    ParentChunk = "MSLK",
                    ChildChunk = "MSPI",
                    RelationshipType = "Index",
                    Evidence = "MSLK.MspiFirstIndex + MspiIndexCount references MSPI indices",
                    ConfidenceScore = 0.95f
                });
                
                relationships.Add(new HierarchicalRelationship
                {
                    ParentChunk = "MSPI", 
                    ChildChunk = "MSPV",
                    RelationshipType = "Index",
                    Evidence = "MSPI indices reference MSPV vertices",
                    ConfidenceScore = 0.90f
                });
            }
            
            // MSUR -> MSVI -> MSVT relationship chain
            if (pm4File.MSUR?.Entries != null && pm4File.MSVI?.Indices != null && pm4File.MSVT?.Vertices != null)
            {
                relationships.Add(new HierarchicalRelationship
                {
                    ParentChunk = "MSUR",
                    ChildChunk = "MSVI", 
                    RelationshipType = "Index",
                    Evidence = "MSUR.MsviFirstIndex + IndexCount references MSVI indices",
                    ConfidenceScore = 0.95f
                });
                
                relationships.Add(new HierarchicalRelationship
                {
                    ParentChunk = "MSVI",
                    ChildChunk = "MSVT",
                    RelationshipType = "Index", 
                    Evidence = "MSVI indices reference MSVT vertices",
                    ConfidenceScore = 0.95f
                });
            }
            
            // Discover potential Unknown_0x04 grouping relationships
            if (pm4File.MSLK?.Entries != null)
            {
                var groupedByUnk04 = pm4File.MSLK.Entries.GroupBy(e => e.Unknown_0x04).ToList();
                if (groupedByUnk04.Count > 1 && groupedByUnk04.Count < pm4File.MSLK.Entries.Count)
                {
                    relationships.Add(new HierarchicalRelationship
                    {
                        ParentChunk = "MSLK",
                        ChildChunk = "MSLK_Internal",
                        RelationshipType = "Grouping",
                        Evidence = $"Unknown_0x04 creates {groupedByUnk04.Count} distinct groups",
                        ConfidenceScore = 0.75f
                    });
                }
            }
            
            return relationships;
        }

        private List<NodeStructureAnalysis> AnalyzeNodeStructures(PM4File pm4File)
        {
            var nodeStructures = new List<NodeStructureAnalysis>();
            
            // Analyze MSLK as node hierarchy
            if (pm4File.MSLK?.Entries != null)
            {
                var mslkAnalysis = new NodeStructureAnalysis
                {
                    ChunkType = "MSLK"
                };
                
                // Group by Unknown_0x04 (suspected group/node ID)
                var groupedEntries = pm4File.MSLK.Entries
                    .Select((entry, index) => new { Entry = entry, Index = index })
                    .GroupBy(x => x.Entry.Unknown_0x04)
                    .ToList();
                
                foreach (var group in groupedEntries)
                {
                    var nodeGroup = new NodeGroup
                    {
                        GroupId = group.Key,
                        EntryCount = group.Count(),
                        EntryIndices = group.Select(x => x.Index).ToList()
                    };
                    
                    // Classify group type based on entry patterns
                    var firstEntry = group.First().Entry;
                    if (firstEntry.MspiFirstIndex == -1)
                    {
                        nodeGroup.GroupType = "Doodad_Node";
                    }
                    else if (firstEntry.MspiIndexCount > 0)
                    {
                        nodeGroup.GroupType = "Geometry_Node";
                    }
                    else
                    {
                        nodeGroup.GroupType = "Reference_Node";
                    }
                    
                    mslkAnalysis.NodeGroups.Add(nodeGroup);
                }
                
                // Build group to indices mapping
                mslkAnalysis.GroupToIndices = groupedEntries.ToDictionary(
                    g => g.Key,
                    g => g.Select(x => x.Index).ToList()
                );
                
                nodeStructures.Add(mslkAnalysis);
            }
            
            return nodeStructures;
        }

        private Dictionary<string, object> ExtractMetadataPatterns(PM4File pm4File)
        {
            var metadata = new Dictionary<string, object>();
            
            // Chunk presence analysis
            var chunkPresence = new Dictionary<string, bool>
            {
                ["MSVT"] = pm4File.MSVT != null,
                ["MSVI"] = pm4File.MSVI != null,
                ["MSUR"] = pm4File.MSUR != null,
                ["MSCN"] = pm4File.MSCN != null,
                ["MSLK"] = pm4File.MSLK != null,
                ["MSPI"] = pm4File.MSPI != null,
                ["MSPV"] = pm4File.MSPV != null,
                ["MPRL"] = pm4File.MPRL != null,
                ["MPRR"] = pm4File.MPRR != null,
                ["MDOS"] = pm4File.MDOS != null,
                ["MDSF"] = pm4File.MDSF != null,
                ["MDBH"] = pm4File.MDBH != null,
                ["MSRN"] = pm4File.MSRN != null
            };
            metadata["ChunkPresence"] = chunkPresence;
            
            // Count statistics
            var chunkCounts = new Dictionary<string, int>
            {
                ["MSVT_Vertices"] = pm4File.MSVT?.Vertices?.Count ?? 0,
                ["MSVI_Indices"] = pm4File.MSVI?.Indices?.Count ?? 0,
                ["MSUR_Surfaces"] = pm4File.MSUR?.Entries?.Count ?? 0,
                ["MSCN_Points"] = pm4File.MSCN?.ExteriorVertices?.Count ?? 0,
                ["MSLK_Entries"] = pm4File.MSLK?.Entries?.Count ?? 0,
                ["MSPI_Indices"] = pm4File.MSPI?.Indices?.Count ?? 0,
                ["MSPV_Vertices"] = pm4File.MSPV?.Vertices?.Count ?? 0
            };
            metadata["ChunkCounts"] = chunkCounts;
            
            // Ratio analysis (might reveal hidden relationships)
            if (chunkCounts["MSLK_Entries"] > 0)
            {
                var ratios = new Dictionary<string, float>
                {
                    ["MSPI_to_MSLK"] = (float)chunkCounts["MSPI_Indices"] / chunkCounts["MSLK_Entries"],
                    ["MSPV_to_MSLK"] = (float)chunkCounts["MSPV_Vertices"] / chunkCounts["MSLK_Entries"],
                    ["MSCN_to_MSLK"] = (float)chunkCounts["MSCN_Points"] / chunkCounts["MSLK_Entries"]
                };
                metadata["ChunkRatios"] = ratios;
            }
            
            return metadata;
        }
    }
} 