namespace ParpDataHarvester.Export.Builders
{
    using System;
    using System.Collections.Generic;
    using ParpDataHarvester.Export;
    using ParpToolbox.Formats.PM4;

    // Builds a minimal node grouping from MSLK.ParentIndex labels discovered on primitives
    internal sealed class MslkNodeGraphBuilder
    {
        internal sealed class NodeGraph
        {
            public Dictionary<uint, List<int>> PrimitiveIndicesByParent { get; } = new();
        }

        public NodeGraph Build(Pm4Scene scene, IReadOnlyList<GltfRawWriter.PrimitiveSpec> primitives)
        {
            if (scene is null) throw new ArgumentNullException(nameof(scene));
            var graph = new NodeGraph();

            for (int i = 0; i < primitives.Count; i++)
            {
                var prim = primitives[i];
                if (prim.Extras is null) continue;
                if (!prim.Extras.TryGetValue("parentIndex", out var parentObj)) continue;
                if (parentObj is null) continue;
                uint parentIdx = parentObj is uint u ? u : Convert.ToUInt32(parentObj);
                if (!graph.PrimitiveIndicesByParent.TryGetValue(parentIdx, out var list))
                {
                    list = new List<int>();
                    graph.PrimitiveIndicesByParent[parentIdx] = list;
                }
                list.Add(i);
            }

            return graph;
        }
    }
}
