using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Utility for analyzing spatial distribution of PM4 chunks to identify coordinate alignment issues.
    /// </summary>
    public static class SpatialAnalysisUtility
    {
        /// <summary>
        /// Analyzes the spatial distribution of all chunk types in a PM4 file.
        /// </summary>
        /// <param name="pm4FilePath">Path to the PM4 file</param>
        /// <param name="outputDir">Directory to write analysis results</param>
        public static void AnalyzeChunkSpatialDistribution(string pm4FilePath, string outputDir)
        {
            Console.WriteLine("=== Analyzing Chunk Spatial Distribution ===");
            
            Directory.CreateDirectory(outputDir);
            
            if (!File.Exists(pm4FilePath))
            {
                Console.WriteLine($"⚠️  PM4 file not found: {pm4FilePath}");
                return;
            }

            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);
                var fileName = Path.GetFileNameWithoutExtension(pm4FilePath);
                
                Console.WriteLine($"📊 Analyzing spatial distribution for: {Path.GetFileName(pm4FilePath)}");
                
                // Analyze each chunk type's coordinate ranges
                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var msvtCoords = pm4File.MSVT.Vertices.Select(v => Pm4CoordinateTransforms.FromMsvtVertexSimple(v)).ToList();
                    var msvtBounds = CalculateBounds(msvtCoords);
                    Console.WriteLine($"\n🎯 MSVT Bounds: {msvtBounds} (Count: {msvtCoords.Count})");
                    
                    var msvtPath = Path.Combine(outputDir, $"{fileName}_msvt_only.obj");
                    ExportChunkToObj(msvtCoords, msvtPath, "MSVT");
                    Console.WriteLine($"📁 Exported MSVT → {msvtPath}");
                }
                
                if (pm4File.MSCN?.ExteriorVertices != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                {
                    var mscnCoords = pm4File.MSCN.ExteriorVertices.Select(v => Pm4CoordinateTransforms.FromMscnVertex(v)).ToList();
                    var mscnBounds = CalculateBounds(mscnCoords);
                    Console.WriteLine($"🛡️  MSCN Bounds: {mscnBounds} (Count: {mscnCoords.Count})");
                    
                    var mscnPath = Path.Combine(outputDir, $"{fileName}_mscn_only.obj");
                    ExportChunkToObj(mscnCoords, mscnPath, "MSCN");
                    Console.WriteLine($"📁 Exported MSCN → {mscnPath}");
                }
                
                if (pm4File.MSPV?.Vertices != null && pm4File.MSPV.Vertices.Count > 0)
                {
                    var mspvCoords = pm4File.MSPV.Vertices.Select(v => Pm4CoordinateTransforms.FromMspvVertex(v)).ToList();
                    var mspvBounds = CalculateBounds(mspvCoords);
                    Console.WriteLine($"📐 MSPV Bounds: {mspvBounds} (Count: {mspvCoords.Count})");
                    
                    var mspvPath = Path.Combine(outputDir, $"{fileName}_mspv_only.obj");
                    ExportChunkToObj(mspvCoords, mspvPath, "MSPV");
                    Console.WriteLine($"📁 Exported MSPV → {mspvPath}");
                }
                
                if (pm4File.MPRL?.Entries != null && pm4File.MPRL.Entries.Count > 0)
                {
                    var mprlCoords = pm4File.MPRL.Entries.Select(e => Pm4CoordinateTransforms.FromMprlEntry(e)).ToList();
                    var mprlBounds = CalculateBounds(mprlCoords);
                    Console.WriteLine($"📍 MPRL Bounds: {mprlBounds} (Count: {mprlCoords.Count})");
                    
                    var mprlPath = Path.Combine(outputDir, $"{fileName}_mprl_only.obj");
                    ExportChunkToObj(mprlCoords, mprlPath, "MPRL");
                    Console.WriteLine($"📁 Exported MPRL → {mprlPath}");
                }

                // Create a comparison combined file
                var combinedPath = Path.Combine(outputDir, $"{fileName}_chunks_analysis.obj");
                ExportCombinedChunks(pm4File, combinedPath);
                Console.WriteLine($"📁 Exported Combined Analysis → {combinedPath}");

                Console.WriteLine($"\n✅ Analysis complete. Check individual OBJ files in MeshLab to identify spatial issues.");
                Console.WriteLine($"📂 Output directory: {outputDir}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during spatial analysis: {ex.Message}");
                throw;
            }
        }

        private static string CalculateBounds(List<Vector3> coords)
        {
            if (!coords.Any()) return "No coordinates";
            
            var minX = coords.Min(c => c.X);
            var maxX = coords.Max(c => c.X);
            var minY = coords.Min(c => c.Y);
            var maxY = coords.Max(c => c.Y);
            var minZ = coords.Min(c => c.Z);
            var maxZ = coords.Max(c => c.Z);
            
            return $"X:[{minX:F2}, {maxX:F2}] Y:[{minY:F2}, {maxY:F2}] Z:[{minZ:F2}, {maxZ:F2}]";
        }

        private static void ExportChunkToObj(IEnumerable<Vector3> coords, string outputPath, string chunkType)
        {
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# {chunkType} Chunk Only - Spatial Analysis");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine($"o {chunkType}_Points");
            
            foreach (var coord in coords)
            {
                writer.WriteLine($"v {coord.X:F6} {coord.Y:F6} {coord.Z:F6}");
            }
        }

        private static void ExportCombinedChunks(PM4File pm4File, string outputPath)
        {
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# PM4 Chunks Combined - Spatial Analysis");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine();

            // MSVT - Render mesh vertices
            if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
            {
                writer.WriteLine("# MSVT Render Vertices");
                writer.WriteLine("o MSVT_Render");
                foreach (var vertex in pm4File.MSVT.Vertices)
                {
                    var coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                    writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
                }
                writer.WriteLine();
            }

            // MSCN - Collision boundaries
            if (pm4File.MSCN?.ExteriorVertices != null && pm4File.MSCN.ExteriorVertices.Count > 0)
            {
                writer.WriteLine("# MSCN Collision Boundaries");
                writer.WriteLine("o MSCN_Collision");
                foreach (var vertex in pm4File.MSCN.ExteriorVertices)
                {
                    var coords = Pm4CoordinateTransforms.FromMscnVertex(vertex);
                    writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
                }
                writer.WriteLine();
            }

            // MSPV - Geometric structure
            if (pm4File.MSPV?.Vertices != null && pm4File.MSPV.Vertices.Count > 0)
            {
                writer.WriteLine("# MSPV Geometry");
                writer.WriteLine("o MSPV_Geometry");
                foreach (var vertex in pm4File.MSPV.Vertices)
                {
                    var coords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                    writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
                }
                writer.WriteLine();
            }

            // MPRL - Reference points
            if (pm4File.MPRL?.Entries != null && pm4File.MPRL.Entries.Count > 0)
            {
                writer.WriteLine("# MPRL Reference Points");
                writer.WriteLine("o MPRL_Reference");
                foreach (var entry in pm4File.MPRL.Entries)
                {
                    var coords = Pm4CoordinateTransforms.FromMprlEntry(entry);
                    writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
                }
                writer.WriteLine();
            }
        }
    }
} 