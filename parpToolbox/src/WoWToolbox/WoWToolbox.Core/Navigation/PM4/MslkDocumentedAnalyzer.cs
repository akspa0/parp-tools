using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// MSLK analyzer that strictly follows wowdev.wiki documentation structure.
    /// Uses only documented field names without statistical interpretations.
    /// 
    /// Per wowdev.wiki documentation:
    /// struct {
    ///   uint8_t _0x00;                 // flags? seen: &1; &2; &4; &8; &16
    ///   uint8_t _0x01;                 // 0…11-ish; position in some sequence?
    ///   uint16_t _0x02;                // Always 0 in version_48, likely padding
    ///   uint32_t _0x04;                // An index somewhere
    ///   int24_t MSPI_first_index;      // -1 if _0x0b is 0
    ///   uint8_t MSPI_index_count;
    ///   uint32_t _0x0c;                // Always 0xffffffff in version_48
    ///   uint16_t msur_index;
    ///   uint16_t _0x12;                // Always 0x8000 in version_48
    /// } mslk[];
    /// </summary>
    public class MslkDocumentedAnalyzer
    {
        public class DocumentedMslkEntry
        {
            public int Index { get; set; }
            public byte Flags_0x00 { get; set; }           // flags? seen: &1; &2; &4; &8; &16
            public byte Unknown_0x01 { get; set; }         // 0…11-ish; position in some sequence?
            public ushort Padding_0x02 { get; set; }      // Always 0 in version_48, likely padding
            public uint Index_0x04 { get; set; }          // An index somewhere
            public int MspiFirstIndex { get; set; }       // -1 if _0x0b is 0
            public byte MspiIndexCount { get; set; }      // Number of MSPI indices
            public uint Unknown_0x0C { get; set; }        // Always 0xffffffff in version_48
            public ushort MsurIndex { get; set; }         // Index into MSUR surfaces
            public ushort Constant_0x12 { get; set; }     // Always 0x8000 in version_48

            /// <summary>
            /// Gets flag analysis based on documentation: &1; &2; &4; &8; &16
            /// </summary>
            public List<int> ActiveFlags
            {
                get
                {
                    var flags = new List<int>();
                    if ((Flags_0x00 & 1) != 0) flags.Add(1);
                    if ((Flags_0x00 & 2) != 0) flags.Add(2);
                    if ((Flags_0x00 & 4) != 0) flags.Add(4);
                    if ((Flags_0x00 & 8) != 0) flags.Add(8);
                    if ((Flags_0x00 & 16) != 0) flags.Add(16);
                    return flags;
                }
            }

            /// <summary>
            /// Determines if this is a doodad node (MSPI_first_index == -1)
            /// </summary>
            public bool IsDoodadNode => MspiFirstIndex == -1;

            /// <summary>
            /// Determines if this references geometry (MSPI_first_index >= 0)
            /// </summary>
            public bool HasGeometry => MspiFirstIndex >= 0;
        }

        public class MslkAnalysisResult
        {
            public List<DocumentedMslkEntry> Entries { get; set; } = new();
            public Dictionary<byte, int> FlagsCounts { get; set; } = new();
            public Dictionary<byte, int> Unknown01Counts { get; set; } = new();
            public Dictionary<uint, int> Index04Counts { get; set; } = new();
            public Dictionary<uint, int> Unknown0CCounts { get; set; } = new();
            public Dictionary<ushort, int> MsurIndexCounts { get; set; } = new();
            public Dictionary<ushort, int> Constant12Counts { get; set; } = new();
            
            public int DoodadNodeCount { get; set; }
            public int GeometryEntryCount { get; set; }
            
            public List<string> ValidationWarnings { get; set; } = new();
        }

        /// <summary>
        /// Analyzes MSLK chunk using documented structure only
        /// </summary>
        public MslkAnalysisResult AnalyzeMslk(MSLK mslkChunk)
        {
            var result = new MslkAnalysisResult();
            
            if (mslkChunk?.Entries == null)
            {
                result.ValidationWarnings.Add("MSLK chunk is null or has no entries");
                return result;
            }

            // Convert to documented entries
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var entry = mslkChunk.Entries[i];
                var documented = new DocumentedMslkEntry
                {
                    Index = i,
                    Flags_0x00 = entry.Unknown_0x00,
                    Unknown_0x01 = entry.Unknown_0x01,
                    Padding_0x02 = entry.Unknown_0x02,
                    Index_0x04 = entry.Unknown_0x04,
                    MspiFirstIndex = entry.MspiFirstIndex,
                    MspiIndexCount = entry.MspiIndexCount,
                    Unknown_0x0C = entry.Unknown_0x0C,
                    MsurIndex = entry.Unknown_0x10,  // Per documentation: msur_index
                    Constant_0x12 = entry.Unknown_0x12
                };
                
                result.Entries.Add(documented);
                
                // Count occurrences for pattern analysis
                if (!result.FlagsCounts.ContainsKey(documented.Flags_0x00))
                    result.FlagsCounts[documented.Flags_0x00] = 0;
                result.FlagsCounts[documented.Flags_0x00]++;
                
                if (!result.Unknown01Counts.ContainsKey(documented.Unknown_0x01))
                    result.Unknown01Counts[documented.Unknown_0x01] = 0;
                result.Unknown01Counts[documented.Unknown_0x01]++;
                
                if (!result.Index04Counts.ContainsKey(documented.Index_0x04))
                    result.Index04Counts[documented.Index_0x04] = 0;
                result.Index04Counts[documented.Index_0x04]++;
                
                if (!result.Unknown0CCounts.ContainsKey(documented.Unknown_0x0C))
                    result.Unknown0CCounts[documented.Unknown_0x0C] = 0;
                result.Unknown0CCounts[documented.Unknown_0x0C]++;
                
                if (!result.MsurIndexCounts.ContainsKey(documented.MsurIndex))
                    result.MsurIndexCounts[documented.MsurIndex] = 0;
                result.MsurIndexCounts[documented.MsurIndex]++;
                
                if (!result.Constant12Counts.ContainsKey(documented.Constant_0x12))
                    result.Constant12Counts[documented.Constant_0x12] = 0;
                result.Constant12Counts[documented.Constant_0x12]++;
                
                // Count types
                if (documented.IsDoodadNode)
                    result.DoodadNodeCount++;
                else
                    result.GeometryEntryCount++;
            }

            // Validate documented constants
            ValidateDocumentedConstants(result);
            
            return result;
        }

        private void ValidateDocumentedConstants(MslkAnalysisResult result)
        {
            // Check if _0x02 is always 0 (padding)
            var nonZeroPadding = result.Entries.Where(e => e.Padding_0x02 != 0).ToList();
            if (nonZeroPadding.Any())
            {
                result.ValidationWarnings.Add($"Documentation states _0x02 should be 0 (padding), but found {nonZeroPadding.Count} non-zero values");
            }

            // Check if _0x0C is always 0xffffffff
            var non0xFFFF = result.Entries.Where(e => e.Unknown_0x0C != 0xFFFFFFFF).ToList();
            if (non0xFFFF.Any())
            {
                result.ValidationWarnings.Add($"Documentation states _0x0c should be 0xffffffff, but found {non0xFFFF.Count} different values");
            }

            // Check if _0x12 is always 0x8000
            var non0x8000 = result.Entries.Where(e => e.Constant_0x12 != 0x8000).ToList();
            if (non0x8000.Any())
            {
                result.ValidationWarnings.Add($"Documentation states _0x12 should be 0x8000, but found {non0x8000.Count} different values");
            }
        }

        /// <summary>
        /// Generates a Mermaid diagram showing MSLK relationships based on documented structure
        /// </summary>
        public string GenerateMermaidDiagram(MslkAnalysisResult analysis, string pm4FileName)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine($"graph TD");
            sb.AppendLine($"    subgraph PM4[\"{pm4FileName}\"]");
            sb.AppendLine($"        MSLK[\"MSLK Chunk<br/>{analysis.Entries.Count} entries\"]");
            sb.AppendLine($"        MSPI[\"MSPI Chunk<br/>Geometry Indices\"]");
            sb.AppendLine($"        MSUR[\"MSUR Chunk<br/>Surface Data\"]");
            sb.AppendLine($"    end");
            sb.AppendLine();

            // Group entries by flags for visualization
            var flagGroups = analysis.Entries.GroupBy(e => e.Flags_0x00).Take(10);
            
            foreach (var group in flagGroups)
            {
                var flag = group.Key;
                var count = group.Count();
                var doodadCount = group.Count(e => e.IsDoodadNode);
                var geomCount = group.Count(e => e.HasGeometry);
                
                var flagBits = string.Join("+", new DocumentedMslkEntry { Flags_0x00 = flag }.ActiveFlags);
                
                sb.AppendLine($"    subgraph FLAG{flag}[\"Flag 0x{flag:X2} ({flagBits})<br/>{count} entries\"]");
                
                if (doodadCount > 0)
                {
                    sb.AppendLine($"        DOODAD{flag}[\"Doodad Nodes<br/>{doodadCount} entries<br/>MSPI_first_index = -1\"]");
                    sb.AppendLine($"        DOODAD{flag} -.-> MSUR");
                }
                
                if (geomCount > 0)
                {
                    sb.AppendLine($"        GEOM{flag}[\"Geometry Entries<br/>{geomCount} entries<br/>MSPI_first_index >= 0\"]");
                    sb.AppendLine($"        GEOM{flag} --> MSPI");
                    sb.AppendLine($"        GEOM{flag} -.-> MSUR");
                }
                
                sb.AppendLine($"    end");
                sb.AppendLine();
            }

            // Add relationships
            sb.AppendLine($"    MSLK --> FLAG{flagGroups.First().Key}");
            if (flagGroups.Count() > 1)
            {
                foreach (var group in flagGroups.Skip(1))
                {
                    sb.AppendLine($"    MSLK --> FLAG{group.Key}");
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates a detailed analysis report using documented field names only
        /// </summary>
        public string GenerateAnalysisReport(MslkAnalysisResult analysis, string pm4FileName)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("=== DOCUMENTED MSLK ANALYSIS ===");
            sb.AppendLine($"File: {pm4FileName}");
            sb.AppendLine($"Total MSLK Entries: {analysis.Entries.Count}");
            sb.AppendLine($"Doodad Nodes (MSPI_first_index = -1): {analysis.DoodadNodeCount}");
            sb.AppendLine($"Geometry Entries (MSPI_first_index >= 0): {analysis.GeometryEntryCount}");
            sb.AppendLine();

            // Validation warnings
            if (analysis.ValidationWarnings.Any())
            {
                sb.AppendLine("=== DOCUMENTATION VALIDATION WARNINGS ===");
                foreach (var warning in analysis.ValidationWarnings)
                {
                    sb.AppendLine($"⚠️  {warning}");
                }
                sb.AppendLine();
            }

            // Flag analysis (documented as "flags? seen: &1; &2; &4; &8; &16")
            sb.AppendLine("=== FLAGS ANALYSIS (0x00) ===");
            sb.AppendLine("Documentation: \"flags? seen: &1; &2; &4; &8; &16\"");
            foreach (var kvp in analysis.FlagsCounts.OrderBy(x => x.Key).Take(10))
            {
                var flags = new DocumentedMslkEntry { Flags_0x00 = kvp.Key }.ActiveFlags;
                var flagStr = flags.Any() ? string.Join("+", flags) : "none";
                sb.AppendLine($"  0x{kvp.Key:X2} ({flagStr}): {kvp.Value} entries");
            }
            sb.AppendLine();

            // Unknown_0x01 analysis (documented as "0…11-ish; position in some sequence?")
            sb.AppendLine("=== UNKNOWN 0x01 ANALYSIS ===");
            sb.AppendLine("Documentation: \"0…11-ish; position in some sequence? index into something?\"");
            foreach (var kvp in analysis.Unknown01Counts.OrderBy(x => x.Key).Take(15))
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value} entries");
            }
            sb.AppendLine();

            // Index_0x04 analysis (documented as "An index somewhere")
            sb.AppendLine("=== INDEX 0x04 ANALYSIS ===");
            sb.AppendLine("Documentation: \"An index somewhere\"");
            var topIndices = analysis.Index04Counts.OrderByDescending(x => x.Value).Take(10);
            foreach (var kvp in topIndices)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value} entries");
            }
            sb.AppendLine();

            // Sample entries
            sb.AppendLine("=== SAMPLE ENTRIES (DOCUMENTED FIELDS) ===");
            foreach (var entry in analysis.Entries.Take(5))
            {
                sb.AppendLine($"Entry {entry.Index}:");
                sb.AppendLine($"  _0x00 (flags): 0x{entry.Flags_0x00:X2} [{string.Join("+", entry.ActiveFlags)}]");
                sb.AppendLine($"  _0x01 (sequence?): {entry.Unknown_0x01}");
                sb.AppendLine($"  _0x02 (padding): 0x{entry.Padding_0x02:X4}");
                sb.AppendLine($"  _0x04 (index): {entry.Index_0x04}");
                sb.AppendLine($"  MSPI_first_index: {entry.MspiFirstIndex}");
                sb.AppendLine($"  MSPI_index_count: {entry.MspiIndexCount}");
                sb.AppendLine($"  _0x0c: 0x{entry.Unknown_0x0C:X8}");
                sb.AppendLine($"  msur_index: {entry.MsurIndex}");
                sb.AppendLine($"  _0x12 (constant): 0x{entry.Constant_0x12:X4}");
                sb.AppendLine($"  Type: {(entry.IsDoodadNode ? "Doodad Node" : "Geometry Entry")}");
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
} 