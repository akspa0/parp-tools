using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Analyzes MSLK chunk relationships based on the official wowdev.wiki documentation.
    /// Creates object + node relationship maps and generates Mermaid diagrams.
    /// </summary>
    public class MslkRelationshipAnalyzer
    {
        public class MslkObjectNode
        {
            public int Index { get; set; }
            public byte Flags { get; set; }                    // _0x00: flags (&1, &2, &4, &8, &16)
            public byte SequencePosition { get; set; }         // _0x01: 0-11ish position in sequence
            public uint IndexReference { get; set; }           // _0x04: an index somewhere
            public int MspiFirstIndex { get; set; }            // MSPI_first_index: -1 for doodads
            public byte MspiIndexCount { get; set; }           // MSPI_index_count
            public uint AlwaysFFFFFFFF { get; set; }           // _0x0c: always 0xffffffff
            public ushort MsurIndex { get; set; }              // msur_index: links to MSUR
            public ushort Always8000 { get; set; }             // _0x12: always 0x8000
            
            // Derived properties
            public bool IsDoodadNode => MspiFirstIndex == -1;
            public bool IsGeometryNode => MspiFirstIndex >= 0;
            public string NodeType => IsDoodadNode ? "Doodad" : "Geometry";
            public List<int> FlagBits => GetFlagBits(Flags);
            public string FlagString => string.Join("|", FlagBits.Select(b => $"&{b}"));
        }

        public class MslkRelationshipMap
        {
            public string FileName { get; set; } = string.Empty;
            public List<MslkObjectNode> Nodes { get; set; } = new();
            public Dictionary<uint, List<MslkObjectNode>> GroupsByIndexReference { get; set; } = new();
            public Dictionary<byte, List<MslkObjectNode>> NodesBySequencePosition { get; set; } = new();
            public Dictionary<byte, List<MslkObjectNode>> NodesByFlags { get; set; } = new();
            public Dictionary<ushort, List<MslkObjectNode>> NodesByMsurIndex { get; set; } = new();
            public List<string> ValidationIssues { get; set; } = new();
        }

        public MslkRelationshipMap AnalyzeMslkRelationships(PM4File pm4File, string fileName = "")
        {
            var map = new MslkRelationshipMap { FileName = fileName };

            if (pm4File.MSLK?.Entries == null || pm4File.MSLK.Entries.Count == 0)
            {
                map.ValidationIssues.Add("No MSLK entries found");
                return map;
            }

            // Convert MSLK entries to documented structure
            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                var entry = pm4File.MSLK.Entries[i];
                var node = new MslkObjectNode
                {
                    Index = i,
                    Flags = entry.Unknown_0x00,                    // _0x00: flags
                    SequencePosition = entry.Unknown_0x01,         // _0x01: sequence position
                    IndexReference = entry.Unknown_0x04,           // _0x04: index reference
                    MspiFirstIndex = entry.MspiFirstIndex,         // MSPI_first_index
                    MspiIndexCount = entry.MspiIndexCount,         // MSPI_index_count
                    AlwaysFFFFFFFF = entry.Unknown_0x0C,           // _0x0c: should be 0xffffffff
                    MsurIndex = entry.Unknown_0x10,                // msur_index (was misinterpreted as reference index)
                    Always8000 = entry.Unknown_0x12                // _0x12: should be 0x8000
                };

                map.Nodes.Add(node);

                // Validate documented constants
                if (node.AlwaysFFFFFFFF != 0xFFFFFFFF)
                {
                    map.ValidationIssues.Add($"Node {i}: _0x0c should be 0xFFFFFFFF, got 0x{node.AlwaysFFFFFFFF:X8}");
                }
                if (node.Always8000 != 0x8000)
                {
                    map.ValidationIssues.Add($"Node {i}: _0x12 should be 0x8000, got 0x{node.Always8000:X4}");
                }
            }

            // Group nodes by various criteria
            map.GroupsByIndexReference = map.Nodes.GroupBy(n => n.IndexReference)
                .ToDictionary(g => g.Key, g => g.ToList());

            map.NodesBySequencePosition = map.Nodes.GroupBy(n => n.SequencePosition)
                .ToDictionary(g => g.Key, g => g.ToList());

            map.NodesByFlags = map.Nodes.GroupBy(n => n.Flags)
                .ToDictionary(g => g.Key, g => g.ToList());

            map.NodesByMsurIndex = map.Nodes.GroupBy(n => n.MsurIndex)
                .ToDictionary(g => g.Key, g => g.ToList());

            return map;
        }

        public string GenerateMermaidDiagram(MslkRelationshipMap map)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"graph TD");
            sb.AppendLine($"    subgraph PM4[\"{map.FileName} - MSLK Object Relationships\"]");

            // Create nodes with proper identification
            foreach (var node in map.Nodes)
            {
                var nodeId = $"N{node.Index}";
                var label = $"{node.NodeType}_{node.Index}<br/>Flags: {node.FlagString}<br/>Seq: {node.SequencePosition}<br/>MSUR: {node.MsurIndex}";
                
                if (node.IsDoodadNode)
                {
                    sb.AppendLine($"    {nodeId}[\"{label}\"]");
                    sb.AppendLine($"    {nodeId} --> |\"Doodad Node\"| {nodeId}");
                }
                else
                {
                    sb.AppendLine($"    {nodeId}[[\"{label}\"]]");
                    if (node.MspiIndexCount > 0)
                    {
                        sb.AppendLine($"    {nodeId} --> |\"MSPI[{node.MspiFirstIndex}:{node.MspiIndexCount}]\"| GEOM{node.Index}[\"Geometry Data\"]");
                    }
                }
            }

            // Group by Index Reference (organizational grouping)
            sb.AppendLine($"");
            sb.AppendLine($"    %% Groups by Index Reference");
            foreach (var group in map.GroupsByIndexReference.Where(g => g.Value.Count > 1))
            {
                var groupId = $"GROUP_{group.Key}";
                sb.AppendLine($"    subgraph {groupId}[\"Group {group.Key}\"]");
                foreach (var node in group.Value)
                {
                    sb.AppendLine($"        N{node.Index}");
                }
                sb.AppendLine($"    end");
            }

            // Show MSUR connections
            sb.AppendLine($"");
            sb.AppendLine($"    %% MSUR Surface Connections");
            foreach (var surfaceGroup in map.NodesByMsurIndex.Where(g => g.Key != 0 && g.Value.Count > 0))
            {
                var surfaceId = $"SURF_{surfaceGroup.Key}";
                sb.AppendLine($"    {surfaceId}[\"MSUR Surface {surfaceGroup.Key}\"]");
                foreach (var node in surfaceGroup.Value)
                {
                    sb.AppendLine($"    N{node.Index} --> {surfaceId}");
                }
            }

            // Show sequence ordering
            sb.AppendLine($"");
            sb.AppendLine($"    %% Sequence Ordering");
            var sequenceNodes = map.NodesBySequencePosition.Keys.OrderBy(k => k);
            for (int i = 0; i < sequenceNodes.Count() - 1; i++)
            {
                var current = sequenceNodes.ElementAt(i);
                var next = sequenceNodes.ElementAt(i + 1);
                
                if (map.NodesBySequencePosition[current].Count == 1 && map.NodesBySequencePosition[next].Count == 1)
                {
                    var currentNode = map.NodesBySequencePosition[current][0];
                    var nextNode = map.NodesBySequencePosition[next][0];
                    sb.AppendLine($"    N{currentNode.Index} -.-> |\"Sequence {current}→{next}\"| N{nextNode.Index}");
                }
            }

            sb.AppendLine($"    end");

            // Add style classes
            sb.AppendLine($"");
            sb.AppendLine($"    classDef doodadNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px");
            sb.AppendLine($"    classDef geometryNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px");
            sb.AppendLine($"    classDef surfaceNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px");

            return sb.ToString();
        }

        public void GenerateAnalysisReport(MslkRelationshipMap map, string outputPath)
        {
            using var writer = new System.IO.StreamWriter(outputPath, false, Encoding.UTF8);
            
            writer.WriteLine($"# MSLK Relationship Analysis: {map.FileName}");
            writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine($"Based on wowdev.wiki documentation structure");
            writer.WriteLine();

            // Summary statistics
            writer.WriteLine("## Summary Statistics");
            writer.WriteLine($"- Total Nodes: {map.Nodes.Count}");
            writer.WriteLine($"- Doodad Nodes: {map.Nodes.Count(n => n.IsDoodadNode)}");
            writer.WriteLine($"- Geometry Nodes: {map.Nodes.Count(n => n.IsGeometryNode)}");
            writer.WriteLine($"- Unique Groups (Index Reference): {map.GroupsByIndexReference.Count}");
            writer.WriteLine($"- Unique Flag Values: {map.NodesByFlags.Count}");
            writer.WriteLine($"- Unique MSUR References: {map.NodesByMsurIndex.Count}");
            writer.WriteLine();

            // Flag analysis
            writer.WriteLine("## Flag Analysis");
            writer.WriteLine("Flags represent object types with bit patterns (&1, &2, &4, &8, &16):");
            foreach (var flagGroup in map.NodesByFlags.OrderBy(g => g.Key))
            {
                var sampleNode = flagGroup.Value[0];
                writer.WriteLine($"- Flag 0x{flagGroup.Key:X2} ({sampleNode.FlagString}): {flagGroup.Value.Count} nodes");
            }
            writer.WriteLine();

            // Sequence analysis
            writer.WriteLine("## Sequence Position Analysis");
            writer.WriteLine("Sequence positions (0-11ish) representing order in some sequence:");
            foreach (var seqGroup in map.NodesBySequencePosition.OrderBy(g => g.Key))
            {
                writer.WriteLine($"- Position {seqGroup.Key}: {seqGroup.Value.Count} nodes");
            }
            writer.WriteLine();

            // Group analysis
            writer.WriteLine("## Group Analysis (Index Reference)");
            foreach (var group in map.GroupsByIndexReference.OrderBy(g => g.Key))
            {
                writer.WriteLine($"- Group {group.Key}: {group.Value.Count} nodes");
                if (group.Value.Count > 1)
                {
                    var doodads = group.Value.Count(n => n.IsDoodadNode);
                    var geometry = group.Value.Count(n => n.IsGeometryNode);
                    writer.WriteLine($"  - Doodads: {doodads}, Geometry: {geometry}");
                }
            }
            writer.WriteLine();

            // MSUR connections
            writer.WriteLine("## MSUR Surface Connections");
            foreach (var surfGroup in map.NodesByMsurIndex.Where(g => g.Key != 0).OrderBy(g => g.Key))
            {
                writer.WriteLine($"- MSUR Index {surfGroup.Key}: {surfGroup.Value.Count} nodes");
            }
            writer.WriteLine();

            // Validation issues
            if (map.ValidationIssues.Any())
            {
                writer.WriteLine("## Validation Issues");
                foreach (var issue in map.ValidationIssues)
                {
                    writer.WriteLine($"- {issue}");
                }
                writer.WriteLine();
            }

            // Mermaid diagram
            writer.WriteLine("## Mermaid Relationship Diagram");
            writer.WriteLine("```mermaid");
            writer.WriteLine(GenerateMermaidDiagram(map));
            writer.WriteLine("```");
        }

        private static List<int> GetFlagBits(byte flags)
        {
            var bits = new List<int>();
            for (int i = 0; i < 8; i++)
            {
                if ((flags & (1 << i)) != 0)
                {
                    bits.Add(1 << i);
                }
            }
            return bits;
        }
    }
} 