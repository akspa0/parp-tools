using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder
{
    /// <summary>
    /// Analyzes how MSCN vertices are referenced/linked in MSLK and MSUR chunks.
    /// Now that MSCN geometry is unified with MSVT, we need to find the indexing mechanism.
    /// </summary>
    internal static class MscnLinkageAnalyzer
    {
        public static void AnalyzeLinkage(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine($"Analyzing MSCN linkage patterns...");
            Console.WriteLine($"MSVT vertices: {scene.Vertices.Count}, MSCN vertices: {scene.MscnVertices.Count}");
            Console.WriteLine($"Total combined vertex space: 0-{scene.Vertices.Count + scene.MscnVertices.Count - 1}");

            Directory.CreateDirectory(outputDir);

            // MSCN index space starts after MSVT vertices
            int mscnIndexStart = scene.Vertices.Count;
            int mscnIndexEnd = scene.Vertices.Count + scene.MscnVertices.Count - 1;

            Console.WriteLine($"MSCN index range: {mscnIndexStart}-{mscnIndexEnd}");

            // Analyze MSVI indices for MSCN references
            AnalyzeMsviIndices(scene, mscnIndexStart, mscnIndexEnd, outputDir);

            // Analyze MSLK entries for potential MSCN index fields
            AnalyzeMslkFields(scene, mscnIndexStart, mscnIndexEnd, outputDir);

            // Analyze MSUR entries for potential MSCN references
            AnalyzeMsurFields(scene, mscnIndexStart, mscnIndexEnd, outputDir);

            // Look for unknown/undocumented fields that might contain MSCN indices
            AnalyzeUnknownFields(scene, mscnIndexStart, mscnIndexEnd, outputDir);
        }

        private static void AnalyzeMsviIndices(Pm4Scene scene, int mscnStart, int mscnEnd, string outputDir)
        {
            string filePath = Path.Combine(outputDir, "msvi_mscn_analysis.txt");
            using var writer = new StreamWriter(filePath);

            writer.WriteLine("MSVI Index Analysis for MSCN References");
            writer.WriteLine($"MSVT range: 0-{scene.Vertices.Count - 1}");
            writer.WriteLine($"MSCN range: {mscnStart}-{mscnEnd}");
            writer.WriteLine();

            var mscnReferences = new List<int>();
            var outOfBoundsIndices = new List<int>();

            for (int i = 0; i < scene.Indices.Count; i++)
            {
                int idx = scene.Indices[i];
                
                if (idx >= mscnStart && idx <= mscnEnd)
                {
                    mscnReferences.Add(idx);
                }
                else if (idx >= scene.Vertices.Count + scene.MscnVertices.Count)
                {
                    outOfBoundsIndices.Add(idx);
                }
            }

            writer.WriteLine($"Total MSVI indices: {scene.Indices.Count}");
            writer.WriteLine($"MSCN references found: {mscnReferences.Count}");
            writer.WriteLine($"Out-of-bounds indices: {outOfBoundsIndices.Count}");
            writer.WriteLine();

            if (mscnReferences.Count > 0)
            {
                writer.WriteLine("MSCN Index References (first 50):");
                foreach (var idx in mscnReferences.Take(50))
                {
                    int mscnOffset = idx - mscnStart;
                    writer.WriteLine($"  MSVI[{Array.IndexOf(scene.Indices.ToArray(), idx)}] = {idx} (MSCN vertex {mscnOffset})");
                }
            }

            if (outOfBoundsIndices.Count > 0)
            {
                writer.WriteLine("\nOut-of-bounds indices (potential cross-tile references):");
                foreach (var idx in outOfBoundsIndices.Take(20))
                {
                    writer.WriteLine($"  Index: {idx} (exceeds vertex buffer by {idx - (scene.Vertices.Count + scene.MscnVertices.Count)})");
                }
            }

            Console.WriteLine($"MSVI analysis written to: {filePath}");
        }

        private static void AnalyzeMslkFields(Pm4Scene scene, int mscnStart, int mscnEnd, string outputDir)
        {
            string filePath = Path.Combine(outputDir, "mslk_mscn_analysis.txt");
            using var writer = new StreamWriter(filePath);

            writer.WriteLine("MSLK Field Analysis for MSCN References");
            writer.WriteLine($"Looking for fields in MSCN index range: {mscnStart}-{mscnEnd}");
            writer.WriteLine();

            var potentialMscnFields = new Dictionary<string, List<int>>();

            foreach (var link in scene.Links)
            {
                // Check all integer fields in MSLK for potential MSCN indices
                CheckFieldForMscnReference("ParentIndex", (int)link.ParentIndex, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("ReferenceIndex", (int)link.ReferenceIndex, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("MspiFirstIndex", link.MspiFirstIndex, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("MspiIndexCount", link.MspiIndexCount, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("SurfaceRefIndex", (int)link.SurfaceRefIndex, mscnStart, mscnEnd, potentialMscnFields);

                // Check for any other integer properties using reflection
                var properties = link.GetType().GetProperties()
                    .Where(p => p.PropertyType == typeof(int) || p.PropertyType == typeof(uint))
                    .Where(p => !new[] { "ParentIndex", "ReferenceIndex", "MspiFirstIndex", "MspiIndexCount", "SurfaceRefIndex" }.Contains(p.Name));

                foreach (var prop in properties)
                {
                    var value = prop.GetValue(link);
                    if (value != null)
                    {
                        // Safely convert to int, skipping values too large for reasonable index ranges
                        int intValue = 0;
                        bool canConvert = false;
                        
                        if (value is uint uintVal)
                        {
                            if (uintVal <= int.MaxValue)
                            {
                                intValue = (int)uintVal;
                                canConvert = true;
                            }
                        }
                        else if (value is int intVal)
                        {
                            intValue = intVal;
                            canConvert = true;
                        }
                        else
                        {
                            try
                            {
                                intValue = Convert.ToInt32(value);
                                canConvert = true;
                            }
                            catch (OverflowException)
                            {
                                // Skip values too large for reasonable index analysis
                                canConvert = false;
                            }
                        }
                        
                        if (canConvert)
                        {
                            CheckFieldForMscnReference(prop.Name, intValue, mscnStart, mscnEnd, potentialMscnFields);
                        }
                    }
                }
            }

            writer.WriteLine($"Total MSLK entries analyzed: {scene.Links.Count}");
            writer.WriteLine();

            foreach (var (fieldName, indices) in potentialMscnFields)
            {
                writer.WriteLine($"{fieldName}: {indices.Count} potential MSCN references");
                if (indices.Count > 0)
                {
                    writer.WriteLine($"  Range: {indices.Min()}-{indices.Max()}");
                    writer.WriteLine($"  Sample values: {string.Join(", ", indices.Take(10))}");
                }
                writer.WriteLine();
            }

            Console.WriteLine($"MSLK analysis written to: {filePath}");
        }

        private static void AnalyzeMsurFields(Pm4Scene scene, int mscnStart, int mscnEnd, string outputDir)
        {
            string filePath = Path.Combine(outputDir, "msur_mscn_analysis.txt");
            using var writer = new StreamWriter(filePath);

            writer.WriteLine("MSUR Field Analysis for MSCN References");
            writer.WriteLine($"Looking for fields in MSCN index range: {mscnStart}-{mscnEnd}");
            writer.WriteLine();

            var potentialMscnFields = new Dictionary<string, List<int>>();

            foreach (var surface in scene.Surfaces)
            {
                // Check all integer fields in MSUR for potential MSCN indices
                CheckFieldForMscnReference("SurfaceKey", (int)surface.SurfaceKey, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("IndexCount", (int)surface.IndexCount, mscnStart, mscnEnd, potentialMscnFields);
                CheckFieldForMscnReference("GroupKey", (int)surface.GroupKey, mscnStart, mscnEnd, potentialMscnFields);

                // Check for any other integer properties using reflection
                var properties = surface.GetType().GetProperties()
                    .Where(p => p.PropertyType == typeof(int) || p.PropertyType == typeof(uint))
                    .Where(p => !new[] { "SurfaceKey", "IndexCount", "GroupKey" }.Contains(p.Name));

                foreach (var prop in properties)
                {
                    var value = prop.GetValue(surface);
                    if (value != null)
                    {
                        CheckFieldForMscnReference(prop.Name, Convert.ToInt32(value), mscnStart, mscnEnd, potentialMscnFields);
                    }
                }
            }

            writer.WriteLine($"Total MSUR entries analyzed: {scene.Surfaces.Count}");
            writer.WriteLine();

            foreach (var (fieldName, indices) in potentialMscnFields)
            {
                writer.WriteLine($"{fieldName}: {indices.Count} potential MSCN references");
                if (indices.Count > 0)
                {
                    writer.WriteLine($"  Range: {indices.Min()}-{indices.Max()}");
                    writer.WriteLine($"  Sample values: {string.Join(", ", indices.Take(10))}");
                }
                writer.WriteLine();
            }

            Console.WriteLine($"MSUR analysis written to: {filePath}");
        }

        private static void AnalyzeUnknownFields(Pm4Scene scene, int mscnStart, int mscnEnd, string outputDir)
        {
            string filePath = Path.Combine(outputDir, "unknown_fields_mscn_analysis.txt");
            using var writer = new StreamWriter(filePath);

            writer.WriteLine("Unknown/Undocumented Field Analysis for MSCN References");
            writer.WriteLine($"Searching for any integer fields in MSCN range: {mscnStart}-{mscnEnd}");
            writer.WriteLine();

            // Analyze placement entries
            writer.WriteLine("MPRL Placement Fields:");
            var placementMscnFields = new Dictionary<string, List<int>>();
            foreach (var placement in scene.Placements)
            {
                var properties = placement.GetType().GetProperties()
                    .Where(p => p.PropertyType == typeof(int) || p.PropertyType == typeof(uint));

                foreach (var prop in properties)
                {
                    var value = prop.GetValue(placement);
                    if (value != null)
                    {
                        CheckFieldForMscnReference($"MPRL.{prop.Name}", Convert.ToInt32(value), mscnStart, mscnEnd, placementMscnFields);
                    }
                }
            }

            foreach (var (fieldName, indices) in placementMscnFields)
            {
                if (indices.Count > 0)
                {
                    writer.WriteLine($"  {fieldName}: {indices.Count} references in MSCN range");
                }
            }

            writer.WriteLine();
            writer.WriteLine("Summary: Fields containing values in MSCN index range may indicate");
            writer.WriteLine("direct references to MSCN vertices for object assembly.");

            Console.WriteLine($"Unknown fields analysis written to: {filePath}");
        }

        private static void CheckFieldForMscnReference(string fieldName, int value, int mscnStart, int mscnEnd, Dictionary<string, List<int>> results)
        {
            if (value >= mscnStart && value <= mscnEnd)
            {
                if (!results.ContainsKey(fieldName))
                    results[fieldName] = new List<int>();
                results[fieldName].Add(value);
            }
        }
    }
}
