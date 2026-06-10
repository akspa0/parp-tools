using System;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder.Pipeline
{
    /// <summary>
    /// Step B – Builds object graph based on ParentIndex linkage and logs metrics.
    /// Wraps the existing <see cref="Pm4ObjectAssembler"/> to avoid duplicating logic.
    /// </summary>
    internal static class ObjectGraphBuilder
    {
        public static List<Pm4ObjectAssembler.BuildingObject> Build(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine("[OBJECT GRAPH] Building linkage graph …");
            var objects = Pm4ObjectAssembler.AssembleObjects(scene);

            Console.WriteLine($"[OBJECT GRAPH] ParentIndex values: {objects.Count}");
            var connectedComponents = objects.Count; // Using one object = one component – detailed graph later.
            Console.WriteLine($"[OBJECT GRAPH] Connected components: {connectedComponents}");

            // Orphan detection placeholder – needs refined scene model.
            Console.WriteLine("[OBJECT GRAPH] Orphan surfaces: (analysis TBD)");

            return objects;
        }
    }
}
