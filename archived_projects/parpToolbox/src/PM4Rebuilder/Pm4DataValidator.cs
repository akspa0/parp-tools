using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder
{
    /// <summary>
    /// Comprehensive PM4 data validation and CSV export tool.
    /// Validates chunk reading completeness, surface/face accounting, and exports all unknown fields.
    /// </summary>
    internal static class Pm4DataValidator
    {
        public static void ValidateAndExport(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine("[DATA VALIDATOR] Starting comprehensive PM4 data validation...");
            
            Directory.CreateDirectory(outputDir);
            
            // 1. Raw data statistics
            ExportDataStatistics(scene, outputDir);
            
            // 2. All chunks to CSV (including unknowns)
            ExportAllChunksToCSV(scene, outputDir);
            
            // 3. Triangle vs Quad analysis
            ExportGeometryAnalysis(scene, outputDir);
            
            // 4. Surface completeness analysis
            ExportSurfaceCompleteness(scene, outputDir);
            
            // 5. Unknown field analysis
            ExportUnknownFieldsDetailed(scene, outputDir);
            
            // 6. Index buffer analysis
            ExportIndexBufferAnalysis(scene, outputDir);
            
            Console.WriteLine("[DATA VALIDATOR] Comprehensive validation complete. Check CSV files for detailed analysis.");
        }
        
        private static void ExportDataStatistics(Pm4Scene scene, string outputDir)
        {
            var sb = new StringBuilder();
            sb.AppendLine("PM4 RAW DATA STATISTICS");
            sb.AppendLine("======================");
            sb.AppendLine($"MSVT Vertices: {scene.Vertices.Count}");
            sb.AppendLine($"MSCN Vertices: {scene.MscnVertices.Count}");
            sb.AppendLine($"MSVI Indices: {scene.Indices.Count}");
            sb.AppendLine($"MSLK Links: {scene.Links.Count}");
            sb.AppendLine($"MSUR Surfaces: {scene.Surfaces.Count}");
            sb.AppendLine($"MPRL Placements: {scene.Placements.Count}");
            sb.AppendLine();
            sb.AppendLine("THEORETICAL TRIANGLE COUNT FROM INDICES:");
            sb.AppendLine($"Max triangles from MSVI: {scene.Indices.Count / 3}");
            sb.AppendLine();
            sb.AppendLine("MSLK INDEX REFERENCES:");
            var mslkIndexRefs = scene.Links.Where(l => l.MspiIndexCount > 0).Sum(l => l.MspiIndexCount);
            sb.AppendLine($"Total MSLK index references: {mslkIndexRefs}");
            sb.AppendLine($"MSLK triangles: {mslkIndexRefs / 3}");
            
            File.WriteAllText(Path.Combine(outputDir, "data_statistics.txt"), sb.ToString());
        }
        
        private static void ExportAllChunksToCSV(Pm4Scene scene, string outputDir)
        {
            // MSVT vertices
            ExportVerticesCSV(scene.Vertices, Path.Combine(outputDir, "msvt_vertices.csv"), "MSVT");
            
            // MSCN vertices  
            ExportVerticesCSV(scene.MscnVertices, Path.Combine(outputDir, "mscn_vertices.csv"), "MSCN");
            
            // MSVI indices
            ExportIndicesCSV(scene.Indices, Path.Combine(outputDir, "msvi_indices.csv"));
            
            // MSLK links
            ExportMslkCSV(scene.Links, Path.Combine(outputDir, "mslk_links.csv"));
            
            // MSUR surfaces
            ExportMsurCSV(scene.Surfaces, Path.Combine(outputDir, "msur_surfaces.csv"));
            
            // MPRL placements
            ExportMprlCSV(scene.Placements, Path.Combine(outputDir, "mprl_placements.csv"));
        }
        
        private static void ExportVerticesCSV(List<System.Numerics.Vector3> vertices, string path, string type)
        {
            var csv = new StringBuilder();
            csv.AppendLine($"Index,X,Y,Z,Type");
            
            for (int i = 0; i < vertices.Count; i++)
            {
                var v = vertices[i];
                csv.AppendLine($"{i},{v.X},{v.Y},{v.Z},{type}");
            }
            
            File.WriteAllText(path, csv.ToString());
        }
        
        private static void ExportIndicesCSV(List<int> indices, string path)
        {
            var csv = new StringBuilder();
            csv.AppendLine("Index,Value,TriangleGroup,Position");
            
            for (int i = 0; i < indices.Count; i++)
            {
                var triangleGroup = i / 3;
                var position = i % 3 == 0 ? "A" : (i % 3 == 1 ? "B" : "C");
                csv.AppendLine($"{i},{indices[i]},{triangleGroup},{position}");
            }
            
            File.WriteAllText(path, csv.ToString());
        }
        
        private static void ExportMslkCSV(List<MslkEntry> links, string path)
        {
            if (links.Count == 0) return;
            
            var csv = new StringBuilder();
            
            // Use reflection to get actual field names dynamically
            var firstLink = links[0];
            var properties = firstLink.GetType().GetProperties();
            var propertyNames = new string[properties.Length];
            for (int i = 0; i < properties.Length; i++)
            {
                propertyNames[i] = properties[i].Name;
            }
            
            // Write header with Index + all actual field names
            csv.AppendLine("Index," + string.Join(",", propertyNames));
            
            // Write data
            for (int i = 0; i < links.Count; i++)
            {
                var link = links[i];
                var values = new List<string> { i.ToString() };
                
                // Add all fields through reflection
                foreach (var prop in properties)
                {
                    var value = prop.GetValue(link)?.ToString() ?? "";
                    values.Add(value);
                }
                
                csv.AppendLine(string.Join(",", values));
            }
            
            File.WriteAllText(path, csv.ToString());
        }
        
        private static void ExportMsurCSV(IReadOnlyList<dynamic> surfaces, string path)
        {
            var csv = new StringBuilder();
            csv.AppendLine("Index,SurfaceKey,IndexCount,GroupKey,Unknown1,Unknown2,Unknown3,Unknown4,Unknown5");
            
            for (int i = 0; i < surfaces.Count; i++)
            {
                var surface = surfaces[i];
                var properties = surface.GetType().GetProperties();
                var values = new List<string> { i.ToString() };
                
                // Add all fields through reflection
                foreach (var prop in properties)
                {
                    var value = prop.GetValue(surface)?.ToString() ?? "";
                    values.Add(value);
                }
                
                csv.AppendLine(string.Join(",", values));
            }
            
            File.WriteAllText(path, csv.ToString());
        }
        
        private static void ExportMprlCSV(IReadOnlyList<dynamic> placements, string path)
        {
            if (placements.Count == 0) return;
            
            var csv = new StringBuilder();
            
            // Use reflection to get actual field names
            var firstPlacement = placements[0];
            var properties = firstPlacement.GetType().GetProperties();
            var propertyNames = new string[properties.Length];
            for (int i = 0; i < properties.Length; i++)
            {
                propertyNames[i] = properties[i].Name;
            }
            
            // Write header
            csv.AppendLine("Index," + string.Join(",", propertyNames));
            
            // Write data
            for (int i = 0; i < placements.Count; i++)
            {
                var placement = placements[i];
                var values = new List<string> { i.ToString() };
                
                foreach (var prop in properties)
                {
                    var value = prop.GetValue(placement)?.ToString() ?? "";
                    values.Add(value);
                }
                
                csv.AppendLine(string.Join(",", values));
            }
            
            File.WriteAllText(path, csv.ToString());
        }
        
        private static void ExportGeometryAnalysis(Pm4Scene scene, string outputDir)
        {
            var analysis = new StringBuilder();
            analysis.AppendLine("GEOMETRY ANALYSIS: TRIANGLES vs QUADS");
            analysis.AppendLine("====================================");
            
            // Check if indices form triangles or quads
            analysis.AppendLine($"Total MSVI indices: {scene.Indices.Count}");
            analysis.AppendLine($"If triangles: {scene.Indices.Count / 3} triangles");
            analysis.AppendLine($"If quads: {scene.Indices.Count / 4} quads");
            analysis.AppendLine($"Remainder for triangles: {scene.Indices.Count % 3}");
            analysis.AppendLine($"Remainder for quads: {scene.Indices.Count % 4}");
            analysis.AppendLine();
            
            // Index range analysis
            var minIndex = scene.Indices.Count > 0 ? scene.Indices.Min() : 0;
            var maxIndex = scene.Indices.Count > 0 ? scene.Indices.Max() : 0;
            var totalVertexSpace = scene.Vertices.Count + scene.MscnVertices.Count;
            
            analysis.AppendLine("INDEX RANGE ANALYSIS:");
            analysis.AppendLine($"Min index: {minIndex}");
            analysis.AppendLine($"Max index: {maxIndex}");
            analysis.AppendLine($"Total vertex space: {totalVertexSpace}");
            analysis.AppendLine($"Out-of-range indices: {scene.Indices.Count(idx => idx >= totalVertexSpace)}");
            
            File.WriteAllText(Path.Combine(outputDir, "geometry_analysis.txt"), analysis.ToString());
        }
        
        private static void ExportSurfaceCompleteness(Pm4Scene scene, string outputDir)
        {
            var completeness = new StringBuilder();
            completeness.AppendLine("SURFACE COMPLETENESS ANALYSIS");
            completeness.AppendLine("=============================");
            
            // Analyze MSLK references
            var mslkWithIndices = scene.Links.Where(l => l.MspiIndexCount > 0).ToList();
            var totalMslkIndices = mslkWithIndices.Sum(l => l.MspiIndexCount);
            
            completeness.AppendLine($"MSLK entries with indices: {mslkWithIndices.Count}/{scene.Links.Count}");
            completeness.AppendLine($"Total indices referenced by MSLK: {totalMslkIndices}");
            completeness.AppendLine($"Available MSVI indices: {scene.Indices.Count}");
            completeness.AppendLine($"Coverage: {(double)totalMslkIndices / scene.Indices.Count * 100:F1}%");
            completeness.AppendLine();
            
            // MSUR analysis
            completeness.AppendLine($"MSUR surfaces: {scene.Surfaces.Count}");
            completeness.AppendLine($"MSLK links: {scene.Links.Count}");
            completeness.AppendLine();
            
            // Cross-reference analysis
            var mslkSurfaceRefs = scene.Links.Where(l => l.SurfaceRefIndex >= 0 && l.SurfaceRefIndex < scene.Surfaces.Count).Count();
            completeness.AppendLine($"MSLK entries referencing valid MSUR: {mslkSurfaceRefs}/{scene.Links.Count}");
            
            File.WriteAllText(Path.Combine(outputDir, "surface_completeness.txt"), completeness.ToString());
        }
        
        private static void ExportUnknownFieldsDetailed(Pm4Scene scene, string outputDir)
        {
            var unknowns = new StringBuilder();
            unknowns.AppendLine("DETAILED UNKNOWN FIELDS ANALYSIS");
            unknowns.AppendLine("================================");
            unknowns.AppendLine();
            
            // Analyze all MSLK unknown fields
            unknowns.AppendLine("MSLK UNKNOWN FIELD PATTERNS:");
            AnalyzeUnknownFieldPatterns(scene.Links, unknowns);
            unknowns.AppendLine();
            
            // Analyze all MSUR unknown fields
            unknowns.AppendLine("MSUR UNKNOWN FIELD PATTERNS:");
            AnalyzeUnknownFieldPatterns(scene.Surfaces, unknowns);
            unknowns.AppendLine();
            
            // Analyze all MPRL unknown fields
            unknowns.AppendLine("MPRL UNKNOWN FIELD PATTERNS:");
            AnalyzeUnknownFieldPatterns(scene.Placements, unknowns);
            
            File.WriteAllText(Path.Combine(outputDir, "unknown_fields_detailed.txt"), unknowns.ToString());
        }
        
        private static void AnalyzeUnknownFieldPatterns<T>(List<T> entries, StringBuilder output)
        {
            if (entries.Count == 0) return;
            
            var type = typeof(T);
            var properties = type.GetProperties();
            
            foreach (var prop in properties)
            {
                if (prop.PropertyType == typeof(int) || prop.PropertyType == typeof(uint))
                {
                    var values = entries.Select(e => prop.GetValue(e)).Cast<object>().ToList();
                    var uniqueValues = values.Distinct().Count();
                    var minVal = values.Min();
                    var maxVal = values.Max();
                    
                    output.AppendLine($"{prop.Name}: Unique={uniqueValues}, Min={minVal}, Max={maxVal}");
                }
            }
        }
        
        private static void ExportIndexBufferAnalysis(Pm4Scene scene, string outputDir)
        {
            var analysis = new StringBuilder();
            analysis.AppendLine("INDEX BUFFER DETAILED ANALYSIS");
            analysis.AppendLine("==============================");
            
            // Triangle-by-triangle analysis
            for (int i = 0; i + 2 < scene.Indices.Count; i += 3)
            {
                var a = scene.Indices[i];
                var b = scene.Indices[i + 1];
                var c = scene.Indices[i + 2];
                
                var triangleIdx = i / 3;
                var aMsvt = a < scene.Vertices.Count;
                var bMsvt = b < scene.Vertices.Count;
                var cMsvt = c < scene.Vertices.Count;
                
                analysis.AppendLine($"Triangle {triangleIdx}: {a}({(aMsvt ? "MSVT" : "MSCN")}), {b}({(bMsvt ? "MSVT" : "MSCN")}), {c}({(cMsvt ? "MSVT" : "MSCN")})");
                
                if (triangleIdx > 20) // Limit output
                {
                    analysis.AppendLine("... (truncated, see CSV for full data)");
                    break;
                }
            }
            
            File.WriteAllText(Path.Combine(outputDir, "index_buffer_analysis.txt"), analysis.ToString());
        }
    }
}
