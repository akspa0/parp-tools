using System;
using System.IO;
using System.Numerics;
using System.Text;
using WmoBspConverter.Bsp;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Generates Quake 3 .map file format from WMO and BSP data.
    /// .map files are the human-readable source format used in Quake 3 mapping.
    /// </summary>
    public class WmoMapGenerator
    {
        // Applies the single WMO -> Quake 3 coordinate transform used for map emission
        private static Vector3 TransformToQ3(Vector3 v)
        {
            const float scale = 100.0f;
            // WMO(X,Y,Z) -> Q3(X,Z,-Y) with uniform 100x scale
            return new Vector3(v.X * scale, v.Z * scale, -v.Y * scale);
        }

        public void GenerateMapFile(string outputPath, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            var mapContent = new StringBuilder();
            
            // Add header info
            mapContent.AppendLine("// Auto-generated from WMO v14 file");
            mapContent.AppendLine($"// Original: WMO v{wmoData.Version}");
            mapContent.AppendLine($"// Groups: {wmoData.Groups.Count}");
            mapContent.AppendLine($"// Textures: {wmoData.Textures.Count}");
            mapContent.AppendLine();
            
            // Open worldspawn entity and keep it open while emitting brushes
            StartWorldspawnEntity(mapContent, wmoData, bspFile);
            
            // Generate brushes from geometry inside worldspawn
            var defaultTex = bspFile.Textures.Count > 0 ? bspFile.Textures[0].Name : "textures/common/caulk";
            GenerateBrushesFromGeometry(mapContent, bspFile, defaultTex);
            
            // Close worldspawn entity
            EndWorldspawnEntity(mapContent);
            
            // Add a default player spawn entity so editors/games can load
            AddSpawnEntity(mapContent, bspFile);
            
            // Generate texture info
            GenerateTextureInfo(mapContent, wmoData);
            
            // Write to file
            File.WriteAllText(outputPath, mapContent.ToString());
            
            Console.WriteLine($"[INFO] Generated .map file: {outputPath}");
        }

        private void StartWorldspawnEntity(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            mapContent.AppendLine("// Worldspawn entity");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine($"\"message\" \"WMO v{wmoData.Version} converted to Quake 3\"");
            mapContent.AppendLine($"\"wmo_groups\" \"{wmoData.Groups.Count}\"");
            mapContent.AppendLine($"\"wmo_textures\" \"{wmoData.Textures.Count}\"");
            mapContent.AppendLine($"\"numpolygons\" \"{bspFile.Faces.Count}\"");
            mapContent.AppendLine($"\"numvertices\" \"{bspFile.Vertices.Count}\"");
            // worldspawn stays open; brushes will follow
        }

        private void EndWorldspawnEntity(StringBuilder mapContent)
        {
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private void GenerateBrushesFromGeometry(StringBuilder mapContent, BspFile bspFile, string defaultTexture)
        {
            mapContent.AppendLine("// Complete geometry brushes generated from WMO data");
            mapContent.AppendLine($"// Total faces: {bspFile.Faces.Count}, Total vertices: {bspFile.Vertices.Count}");
            
            // DEBUG: Output raw vertex data to console for analysis
            Console.WriteLine("\n[DEBUG] Raw WMO vertices (first 12):");
            for (int i = 0; i < Math.Min(12, bspFile.Vertices.Count); i++)
            {
                var v = bspFile.Vertices[i].Position;
                Console.WriteLine($"  v{i}: X={v.X:F4}, Y={v.Y:F4}, Z={v.Z:F4}");
            }
            
            // Generate brushes from all faces in the BSP data
            int brushCount = 0;
            int skippedCount = 0;
            foreach (var face in bspFile.Faces)
            {
                if (face.FirstVertex + 2 >= bspFile.Vertices.Count)
                    continue;
                
                var v0 = bspFile.Vertices[face.FirstVertex].Position;
                var v1 = bspFile.Vertices[face.FirstVertex + 1].Position;
                var v2 = bspFile.Vertices[face.FirstVertex + 2].Position;
                
                // Cull degenerate triangles
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                var n = Vector3.Cross(e1, e2);
                if (n.Length() < 1e-6f)
                {
                    skippedCount++;
                    continue;
                }

                // Create brush from actual triangle geometry
                var faceTex = (face.Texture >= 0 && face.Texture < bspFile.Textures.Count)
                    ? bspFile.Textures[face.Texture].Name
                    : defaultTexture;
                var brush = GenerateTriangleBrushFromGeometry(v0, v1, v2, faceTex);
                mapContent.Append(brush);
                
                brushCount++;
                
                // Add spacing every 10 brushes to keep file readable
                if (brushCount % 10 == 0)
                {
                    mapContent.AppendLine();
                }
            }
            
            mapContent.AppendLine();
            Console.WriteLine($"[DEBUG] Generated {brushCount} geometry brushes from WMO data");
            if (skippedCount > 0)
                Console.WriteLine($"[DEBUG] Skipped {skippedCount} degenerate triangles");
        }

        private static readonly string BR = ""; // inline appending only

        private string GenerateTriangleBrushFromGeometry(Vector3 v0, Vector3 v1, Vector3 v2, string textureName)
        {
            var brush = new StringBuilder();
            
            var tex = string.IsNullOrWhiteSpace(textureName) ? "textures/common/caulk" : textureName;
            brush.AppendLine($"// Triangle: v0={v0}, v1={v1}, v2={v2}");
            brush.AppendLine("{");
            
            v0 = TransformToQ3(v0);
            v1 = TransformToQ3(v1);
            v2 = TransformToQ3(v2);
            
            float minX = MathF.Min(v0.X, MathF.Min(v1.X, v2.X));
            float minY = MathF.Min(v0.Y, MathF.Min(v1.Y, v2.Y));
            float minZ = MathF.Min(v0.Z, MathF.Min(v1.Z, v2.Z));
            float maxX = MathF.Max(v0.X, MathF.Max(v1.X, v2.X));
            float maxY = MathF.Max(v0.Y, MathF.Max(v1.Y, v2.Y));
            float maxZ = MathF.Max(v0.Z, MathF.Max(v1.Z, v2.Z));

            const float pad = 1.0f;
            minX -= pad; maxX += pad;
            minY -= pad; maxY += pad;
            minZ -= pad; maxZ += pad;

            // -X
            brush.AppendLine($"  ( {minX} {minY} {minZ} ) ( {minX} {minY} {maxZ} ) ( {minX} {maxY} {maxZ} ) {tex} 0 0 0 1 1");
            // +X
            brush.AppendLine($"  ( {maxX} {minY} {minZ} ) ( {maxX} {maxY} {minZ} ) ( {maxX} {maxY} {maxZ} ) {tex} 0 0 0 1 1");
            // -Y
            brush.AppendLine($"  ( {minX} {minY} {minZ} ) ( {maxX} {minY} {minZ} ) ( {maxX} {minY} {maxZ} ) {tex} 0 0 0 1 1");
            // +Y
            brush.AppendLine($"  ( {minX} {maxY} {minZ} ) ( {minX} {maxY} {maxZ} ) ( {maxX} {maxY} {maxZ} ) {tex} 0 0 0 1 1");
            // -Z
            brush.AppendLine($"  ( {minX} {minY} {minZ} ) ( {minX} {maxY} {minZ} ) ( {maxX} {maxY} {minZ} ) {tex} 0 0 0 1 1");
            // +Z
            brush.AppendLine($"  ( {minX} {minY} {maxZ} ) ( {maxX} {minY} {maxZ} ) ( {maxX} {maxY} {maxZ} ) {tex} 0 0 0 1 1");
            
            brush.AppendLine("}");
            
            return brush.ToString();
        }

        private void AddSpawnEntity(StringBuilder mapContent, BspFile bspFile)
        {
            if (bspFile.Vertices.Count == 0) return;
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in bspFile.Vertices)
            {
                // Compute bounds in Q3 coordinates to match geometry emission
                var p = TransformToQ3(v.Position);
                if (p.X < min.X) min.X = p.X;
                if (p.Y < min.Y) min.Y = p.Y;
                if (p.Z < min.Z) min.Z = p.Z;
                if (p.X > max.X) max.X = p.X;
                if (p.Y > max.Y) max.Y = p.Y;
                if (p.Z > max.Z) max.Z = p.Z;
            }
            var center = (min + max) * 0.5f;
            var spawnZ = min.Z + 64f;
            mapContent.AppendLine("// Default spawn");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"info_player_deathmatch\"");
            mapContent.AppendLine($"\"origin\" \"{center.X:F1} {center.Y:F1} {spawnZ:F1}\"");
            mapContent.AppendLine("\"angle\" \"0\"");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private void GenerateTextureInfo(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData)
        {
            mapContent.AppendLine("// Texture information");
            mapContent.AppendLine("// Available textures from WMO file:");
            
            for (int i = 0; i < wmoData.Textures.Count && i < 10; i++)
            {
                mapContent.AppendLine($"// {i}: {wmoData.Textures[i]}");
            }
            
            if (wmoData.Textures.Count > 10)
            {
                mapContent.AppendLine($"// ... and {wmoData.Textures.Count - 10} more textures");
            }
            
            mapContent.AppendLine();
        }

        /// <summary>
        /// Alternative: Generate a simple cube as a test map
        /// </summary>
        public void GenerateSimpleTestMap(string outputPath)
        {
            var mapContent = new StringBuilder();
            
            mapContent.AppendLine("// Simple test cube map");
            mapContent.AppendLine("// Generated by WMO v14 to Quake 3 converter");
            mapContent.AppendLine();
            
            // Worldspawn
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine("\"message\" \"Test map from WMO v14 converter\"");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
            
            // Simple cube brush
            mapContent.AppendLine("// Test cube brush");
            mapContent.AppendLine("{");
            mapContent.AppendLine("  ( -64 0 0 ) ( 0 -64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 -64 0 ) ( 0 0 128 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 0 128 ) ( 64 0 0 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 64 0 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 0 128 ) NULL 0 0 0");
            mapContent.AppendLine("}");
            
            // Add a light entity
            mapContent.AppendLine();
            mapContent.AppendLine("// Test light");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"light\"");
            mapContent.AppendLine("\"origin\" \"0 0 32\"");
            mapContent.AppendLine("\"light\" \"300\"");
            mapContent.AppendLine("}");
            
            File.WriteAllText(outputPath, mapContent.ToString());
            
            Console.WriteLine($"[INFO] Generated simple test .map file: {outputPath}");
        }
    }
}