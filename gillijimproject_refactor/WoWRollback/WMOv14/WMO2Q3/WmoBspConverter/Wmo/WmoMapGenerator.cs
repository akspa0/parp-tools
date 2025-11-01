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
        public void GenerateMapFile(string outputPath, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            var mapContent = new StringBuilder();
            
            // Add header info
            mapContent.AppendLine("// Auto-generated from WMO v14 file");
            mapContent.AppendLine($"// Original: WMO v{wmoData.Version}");
            mapContent.AppendLine($"// Groups: {wmoData.Groups.Count}");
            mapContent.AppendLine($"// Textures: {wmoData.Textures.Count}");
            mapContent.AppendLine();
            
            // Generate worldspawn entity
            GenerateWorldspawnEntity(mapContent, wmoData, bspFile);
            
            // Generate brushes from geometry
            GenerateBrushesFromGeometry(mapContent, bspFile);
            
            // Generate texture info
            GenerateTextureInfo(mapContent, wmoData);
            
            // Write to file
            File.WriteAllText(outputPath, mapContent.ToString());
            
            Console.WriteLine($"[INFO] Generated .map file: {outputPath}");
        }

        private void GenerateWorldspawnEntity(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            mapContent.AppendLine("// Worldspawn entity");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine($"\"message\" \"WMO v{wmoData.Version} converted to Quake 3\"");
            mapContent.AppendLine($"\"wmo_groups\" \"{wmoData.Groups.Count}\"");
            mapContent.AppendLine($"\"wmo_textures\" \"{wmoData.Textures.Count}\"");
            mapContent.AppendLine($"\"numpolygons\" \"{bspFile.Faces.Count}\"");
            mapContent.AppendLine($"\"numvertices\" \"{bspFile.Vertices.Count}\"");
            
            // Add first texture as skybox if available
            if (wmoData.Textures.Count > 0)
            {
                mapContent.AppendLine($"\"texture\" \"{wmoData.Textures[0]}\"");
            }
            
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private void GenerateBrushesFromGeometry(StringBuilder mapContent, BspFile bspFile)
        {
            mapContent.AppendLine("// Complete geometry brushes generated from WMO data");
            mapContent.AppendLine($"// Total faces: {bspFile.Faces.Count}, Total vertices: {bspFile.Vertices.Count}");
            
            // Generate brushes from all faces in the BSP data
            int brushCount = 0;
            foreach (var face in bspFile.Faces)
            {
                if (face.FirstVertex + 2 >= bspFile.Vertices.Count)
                    continue;
                
                var v0 = bspFile.Vertices[face.FirstVertex].Position;
                var v1 = bspFile.Vertices[face.FirstVertex + 1].Position;
                var v2 = bspFile.Vertices[face.FirstVertex + 2].Position;
                
                // Create brush from actual triangle geometry
                var brush = GenerateTriangleBrushFromGeometry(v0, v1, v2, face.Texture);
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
        }

        private string GenerateTriangleBrushFromGeometry(Vector3 v0, Vector3 v1, Vector3 v2, int textureIndex)
        {
            var brush = new StringBuilder();
            
            // Create a triangular prism brush from the actual triangle geometry
            // This creates a 3D brush from the 2D triangle by extruding it along a normal
            var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
            var height = 64f; // Extrude height for 3D volume
            
            brush.AppendLine($"// Triangle brush {textureIndex} from geometry: v0={v0}, v1={v1}, v2={v2}");
            brush.AppendLine("{");
            
            // Create 6 faces of the triangular prism
            brush.AppendLine($"  ( {v0.X} {v0.Y} {v0.Z} ) ( {v1.X} {v1.Y} {v1.Z} ) ( {v2.X} {v2.Y} {v2.Z} ) NULL 0 0 0");
            brush.AppendLine($"  ( {v0.X + normal.X * height} {v0.Y + normal.Y * height} {v0.Z + normal.Z * height} ) ( {v1.X + normal.X * height} {v1.Y + normal.Y * height} {v1.Z + normal.Z * height} ) ( {v2.X + normal.X * height} {v2.Y + normal.Y * height} {v2.Z + normal.Z * height} ) NULL 0 0 0");
            
            // Side faces
            brush.AppendLine($"  ( {v0.X} {v0.Y} {v0.Z} ) ( {v1.X} {v1.Y} {v1.Z} ) ( {v1.X + normal.X * height} {v1.Y + normal.Y * height} {v1.Z + normal.Z * height} ) NULL 0 0 0");
            brush.AppendLine($"  ( {v1.X} {v1.Y} {v1.Z} ) ( {v2.X} {v2.Y} {v2.Z} ) ( {v2.X + normal.X * height} {v2.Y + normal.Y * height} {v2.Z + normal.Z * height} ) NULL 0 0 0");
            brush.AppendLine($"  ( {v2.X} {v2.Y} {v2.Z} ) ( {v0.X} {v0.Y} {v0.Z} ) ( {v0.X + normal.X * height} {v0.Y + normal.Y * height} {v0.Z + normal.Z * height} ) NULL 0 0 0");
            
            // Top and bottom (extruded)
            brush.AppendLine($"  ( {v0.X} {v0.Y} {v0.Z} ) ( {v2.X} {v2.Y} {v2.Z} ) ( {v1.X} {v1.Y} {v1.Z} ) NULL 0 0 0");
            
            brush.AppendLine("}");
            
            return brush.ToString();
        }

        private string GenerateSimpleCubeBrush(int brushIndex)
        {
            // Fallback method for testing
            var brush = new StringBuilder();
            
            brush.AppendLine($"// Simple cube brush {brushIndex}");
            brush.AppendLine("{");
            brush.AppendLine("  ( -64 0 0 ) ( 0 -64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            brush.AppendLine("  ( 0 0 0 ) ( 0 -64 0 ) ( 0 0 128 ) NULL 0 0 0");
            brush.AppendLine("  ( 0 0 0 ) ( 0 0 128 ) ( 64 0 0 ) NULL 0 0 0");
            brush.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 64 0 ) NULL 0 0 0");
            brush.AppendLine("  ( 0 0 0 ) ( 0 64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            brush.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 0 128 ) NULL 0 0 0");
            brush.AppendLine("}");
            
            return brush.ToString();
        }

        private string GenerateTriangleBrush(Vector3 v0, Vector3 v1, Vector3 v2, int textureIndex)
        {
            // This method is no longer used - replaced by GenerateTriangleBrushFromGeometry
            return GenerateTriangleBrushFromGeometry(v0, v1, v2, textureIndex);
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