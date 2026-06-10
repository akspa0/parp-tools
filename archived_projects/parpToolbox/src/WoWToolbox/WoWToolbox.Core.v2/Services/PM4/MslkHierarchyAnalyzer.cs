using System;
using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Analyzes the MSLK chunk treating it as a flat set of entries with metadata
    /// (self-references, range groupings, geometry vs. doodad nodes, etc.).
    /// This is a direct port of the legacy v1 analyser but trimmed of
    /// experimental hierarchy-building that introduced circular reference bugs.
    /// </summary>
    public class MslkHierarchyAnalyzer
    {
        #region Internal DTOs
        public class HierarchyNode
        {
            public int Index { get; set; }
            public byte Flags_0x00 { get; set; }
            public byte Sequence_0x01 { get; set; }
            public uint ParentIndex_0x04 { get; set; }
            public int MspiFirstIndex { get; set; }
            public byte MspiIndexCount { get; set; }
            public uint Unknown_0x0C { get; set; }
            public ushort CrossReference_0x10 { get; set; }

            // Derived helpers
            public bool HasGeometry => MspiFirstIndex >= 0;
            public bool IsDoodadNode => MspiFirstIndex < 0;

            // Relationships (not populated – kept for potential future use)
            public List<HierarchyNode> Children { get; } = new();
            public HierarchyNode? Parent { get; set; }
            public int Depth { get; set; }
        }

        public class HierarchyAnalysisResult
        {
            public List<HierarchyNode> AllNodes { get; } = new();
            public List<HierarchyNode> RootNodes { get; set; } = new();
            public Dictionary<uint, List<HierarchyNode>> NodesByParentIndex { get; set; } = new();
            public Dictionary<byte, List<HierarchyNode>> NodesByFlags { get; set; } = new();
            public Dictionary<byte, List<HierarchyNode>> NodesBySequence { get; set; } = new();
            public int MaxDepth { get; set; }
            public int GeometryNodeCount { get; set; }
            public int DoodadNodeCount { get; set; }
            public List<string> DiscoveredPatterns { get; } = new();
        }
        #endregion

        /// <summary>
        /// Entry-point: analyse an MSLK chunk and return statistics/patterns.
        /// </summary>
        public HierarchyAnalysisResult AnalyzeHierarchy(MSLK mslkChunk)
        {
            var result = new HierarchyAnalysisResult();
            if (mslkChunk?.Entries == null || !mslkChunk.Entries.Any())
            {
                result.DiscoveredPatterns.Add("No MSLK entries to analyse");
                return result;
            }

            ConvertToHierarchyNodes(mslkChunk, result);
            AnalyzeParentIndexPatterns(result);
            result.RootNodes = result.AllNodes; // no parent-child building
            CalculateHierarchyStats(result);
            return result;
        }

        private static void ConvertToHierarchyNodes(MSLK mslkChunk, HierarchyAnalysisResult result)
        {
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                var node = new HierarchyNode
                {
                    Index = i,
                    Flags_0x00 = e.Unknown_0x00,
                    Sequence_0x01 = e.Unknown_0x01,
                    ParentIndex_0x04 = e.Unknown_0x04,
                    MspiFirstIndex = e.MspiFirstIndex,
                    MspiIndexCount = e.MspiIndexCount,
                    Unknown_0x0C = e.Unknown_0x0C,
                    CrossReference_0x10 = e.Unknown_0x10
                };
                result.AllNodes.Add(node);
                if (node.HasGeometry) result.GeometryNodeCount++; else result.DoodadNodeCount++;
            }
        }

        private static void AnalyzeParentIndexPatterns(HierarchyAnalysisResult result)
        {
            var selfRefs = result.AllNodes.Count(n => n.ParentIndex_0x04 == n.Index);
            var validRefs = result.AllNodes.Count(n => n.ParentIndex_0x04 < result.AllNodes.Count && n.ParentIndex_0x04 != n.Index);
            var invalidRefs = result.AllNodes.Count - selfRefs - validRefs;
            result.DiscoveredPatterns.Add($"Self-refs: {selfRefs}, Valid refs: {validRefs}, Invalid refs: {invalidRefs}");
        }

        private static void CalculateHierarchyStats(HierarchyAnalysisResult result)
        {
            result.MaxDepth = 0; // no hierarchy depth when treated flat
        }

        /// <summary>
        /// Returns a mapping of GroupId (ParentIndex_0x04) to list of MSLK entry indices that contain geometry.
        /// Useful for exporters that want to treat each group as a logical object.
        /// </summary>
        public Dictionary<uint, List<int>> GroupGeometryNodeIndicesByGroupId(MSLK mslkChunk)
        {
            var map = new Dictionary<uint, List<int>>();
            if (mslkChunk?.Entries == null) return map;
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                if (e.MspiFirstIndex < 0) continue; // skip non-geometry nodes
                uint groupId = e.Unknown_0x04;
                if (!map.TryGetValue(groupId, out var list))
                {
                    list = new List<int>();
                    map[groupId] = list;
                }
                list.Add(i);
            }
            return map;
        }

        /// <summary>
        /// Returns a mapping of objectId (Unk10 / ReferenceIndex) to list of entry indices that contain geometry.
        /// This is based on statistical observation that Unknown_0x10 clusters multiple groups into one logical object.
        /// </summary>
        public Dictionary<ushort, List<int>> GroupGeometryNodeIndicesByObjectId(MSLK mslkChunk)
        {
            var map = new Dictionary<ushort, List<int>>();
            if (mslkChunk?.Entries == null) return map;
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                if (e.MspiFirstIndex < 0) continue;
                ushort objectId = e.Unknown_0x10;
                if (!map.TryGetValue(objectId, out var list))
                {
                    list = new List<int>();
                    map[objectId] = list;
                }
                list.Add(i);
            }
            return map;
        }

        /// <summary>
        /// Groups geometry nodes by Unknown_0x00 (flag byte).
        /// </summary>
        public Dictionary<byte, List<int>> GroupGeometryNodeIndicesByFlag(MSLK mslkChunk)
        {
            var map = new Dictionary<byte, List<int>>();
            if (mslkChunk?.Entries == null) return map;
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                if (e.MspiFirstIndex < 0) continue;
                byte flag = e.Unknown_0x00;
                if (!map.TryGetValue(flag, out var list))
                {
                    list = new List<int>();
                    map[flag] = list;
                }
                list.Add(i);
            }
            return map;
        }

        /// <summary>
        /// Groups geometry nodes by Unknown_0x01 (subtype byte).
        /// </summary>
        public Dictionary<byte, List<int>> GroupGeometryNodeIndicesBySubtype(MSLK mslkChunk)
        {
            var map = new Dictionary<byte, List<int>>();
            if (mslkChunk?.Entries == null) return map;
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                if (e.MspiFirstIndex < 0) continue;
                byte subtype = e.Unknown_0x01;
                if (!map.TryGetValue(subtype, out var list))
                {
                    list = new List<int>();
                    map[subtype] = list;
                }
                list.Add(i);
            }
            return map;
        }

        /// <summary>
        /// Groups by composite key (flag << 8 | highByte(ReferenceIndex)). Provides a finer-grained container grouping.
        /// </summary>
        public Dictionary<ushort, List<int>> GroupGeometryNodeIndicesByContainer(MSLK mslkChunk)
        {
            var map = new Dictionary<ushort, List<int>>();
            if (mslkChunk?.Entries == null) return map;
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var e = mslkChunk.Entries[i];
                if (e.MspiFirstIndex < 0) continue;
                byte flag = e.Unknown_0x00;
                byte highRef = (byte)(e.Unknown_0x10 >> 8);
                ushort key = (ushort)((flag << 8) | highRef);
                if (!map.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    map[key] = list;
                }
                list.Add(i);
            }
            return map;
        }
    }
}
