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
            // WMO vertices are already stored as (X, Z, -Y) in the file format
            // Q3 uses (X, Y, Z) where Y is forward and Z is up
            // So we just need to pass through: WMO's (X, Z, -Y) -> Q3's (X, Y, Z)
            return new Vector3(v.X, v.Y, v.Z);
        }

        public void GenerateMapFilePerGroup(string baseOutputPath, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            // Export each WMO group as a separate .map file
            Console.WriteLine($"[INFO] Exporting {wmoData.Groups.Count} groups as separate .map files...");
            
            int groupIndex = 0;
            foreach (var group in wmoData.Groups)
            {
                if (group.Vertices.Count == 0)
                {
                    Console.WriteLine($"[SKIP] Group {groupIndex} has no vertices");
                    groupIndex++;
                    continue;
                }
                
                var groupPath = Path.Combine(
                    Path.GetDirectoryName(baseOutputPath) ?? "",
                    $"{Path.GetFileNameWithoutExtension(baseOutputPath)}_group{groupIndex:D3}.map"
                );
                
                // Create a mini BSP with just this group's data
                var groupBsp = new BspFile();
                
                // Add vertices
                foreach (var v in group.Vertices)
                {
                    groupBsp.Vertices.Add(new BspVertex { Position = v });
                }
                
                // Add faces (triangles from indices)
                for (int i = 0; i < group.Indices.Count; i += 3)
                {
                    if (i + 2 < group.Indices.Count)
                    {
                        groupBsp.Faces.Add(new BspFace 
                        { 
                            FirstVertex = group.Indices[i],
                            NumVertices = 3,
                            Texture = i / 3 < group.FaceMaterials.Count ? group.FaceMaterials[i / 3] : 0
                        });
                    }
                }
                
                // Copy textures
                groupBsp.Textures.AddRange(bspFile.Textures);
                
                GenerateMapFile(groupPath, wmoData, groupBsp);
                Console.WriteLine($"[INFO] Exported group {groupIndex}: {groupPath} ({group.Vertices.Count} verts, {group.Indices.Count / 3} faces)");
                
                groupIndex++;
            }
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
            
            // Create sealed worldspawn box to contain the WMO
            CreateSealedWorldspawn(mapContent, wmoData, bspFile);
            Console.WriteLine($"[DEBUG] After worldspawn, length: {mapContent.Length:N0}");
            
            // Add player spawn entity
            AddSpawnEntity(mapContent, bspFile);
            Console.WriteLine($"[DEBUG] After AddSpawn, length: {mapContent.Length:N0}");
            
            // Add WMO geometry as a func_group entity (separate from worldspawn!)
            StartFuncGroupEntity(mapContent, wmoData);
            var defaultTex = bspFile.Textures.Count > 0 ? bspFile.Textures[0].Name : "textures/common/caulk";
            GenerateBrushesFromGeometry(mapContent, bspFile, defaultTex);
            EndFuncGroupEntity(mapContent);
            Console.WriteLine($"[DEBUG] After func_group, length: {mapContent.Length:N0}");
            
            // Generate texture info
            GenerateTextureInfo(mapContent, wmoData);
            Console.WriteLine($"[DEBUG] After GenerateTextureInfo, length: {mapContent.Length:N0}");
            
            // Write to file
            Console.WriteLine($"[DEBUG] StringBuilder length: {mapContent.Length:N0} characters");
            var finalContent = mapContent.ToString();
            Console.WriteLine($"[DEBUG] Final string length: {finalContent.Length:N0} characters");
            File.WriteAllText(outputPath, finalContent);
            
            Console.WriteLine($"[INFO] Generated .map file: {outputPath}");
        }

        private void CreateSealedWorldspawn(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            // Calculate WMO bounds
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (var v in bspFile.Vertices)
            {
                var p = TransformToQ3(v.Position);
                if (p.X < min.X) min.X = p.X;
                if (p.Y < min.Y) min.Y = p.Y;
                if (p.Z < min.Z) min.Z = p.Z;
                if (p.X > max.X) max.X = p.X;
                if (p.Y > max.Y) max.Y = p.Y;
                if (p.Z > max.Z) max.Z = p.Z;
            }
            
            // Expand bounds to create room around WMO
            float padding = 128.0f;
            min.X -= padding;
            min.Y -= padding;
            min.Z -= padding;
            max.X += padding;
            max.Y += padding;
            max.Z += padding;
            
            mapContent.AppendLine("// Sealed worldspawn room");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine($"\"message\" \"WMO v{wmoData.Version} in sealed room\"");
            
            // Create 6-sided sealed box (hollow room)
            // Each face is a brush with caulk texture
            
            // Floor
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X+1} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z-16} ) ( {min.X} {min.Y+1} {min.Z-16} ) ( {min.X+1} {min.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {min.Z-16} ) ( {min.X+1} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z-16} ) ( {max.X} {max.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) ( {min.X} {min.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X+1} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // Ceiling
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z+16} ) ( {min.X} {min.Y+1} {max.Z+16} ) ( {min.X+1} {min.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X+1} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y} {max.Z+16} ) ( {min.X+1} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {max.Z} ) ( {max.X} {max.Y} {max.Z+16} ) ( {max.X} {max.Y+1} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) ( {min.X} {min.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {max.Z} ) ( {max.X+1} {max.Y} {max.Z} ) ( {max.X} {max.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // -X wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X-16} {min.Y} {min.Z} ) ( {min.X-16} {min.Y} {max.Z} ) ( {min.X-16} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) ( {min.X-16} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {min.X-16} {max.Y} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X-16} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) ( {min.X-16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // +X wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {max.X+16} {min.Y} {min.Z} ) ( {max.X+16} {min.Y+1} {min.Z} ) ( {max.X+16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X+16} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X+16} {max.Y} {min.Z} ) ( {max.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X+16} {min.Y} {min.Z} ) ( {max.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y+1} {max.Z} ) ( {max.X+16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // -Y wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) ( {max.X} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y-16} {min.Z} ) ( {max.X} {min.Y-16} {min.Z} ) ( {min.X} {min.Y-16} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y-16} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y-16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {min.Z} ) ( {min.X} {min.Y-16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y-16} {max.Z} ) ( {max.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // +Y wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {max.Y+16} {min.Z} ) ( {min.X} {max.Y+16} {max.Z} ) ( {max.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {min.X} {max.Y+16} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {max.Z} ) ( {max.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z} ) ( {min.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {max.Z} ) ( {min.X} {max.Y+16} {max.Z} ) ( {max.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            mapContent.AppendLine("}");
            mapContent.AppendLine();
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
            Console.WriteLine("[DEBUG] Closing worldspawn entity");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }
        
        private void StartFuncGroupEntity(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData)
        {
            mapContent.AppendLine("// WMO geometry as func_group");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"func_group\"");
            mapContent.AppendLine($"\"_wmo_name\" \"{Path.GetFileNameWithoutExtension(wmoData.FileBytes.Length > 0 ? "wmo" : "unknown")}\"");
            mapContent.AppendLine($"\"_wmo_groups\" \"{wmoData.Groups.Count}\"");
            // func_group stays open; brushes will follow
        }
        
        private void EndFuncGroupEntity(StringBuilder mapContent)
        {
            Console.WriteLine("[DEBUG] Closing func_group entity");
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
            
            var tex = string.IsNullOrWhiteSpace(textureName) ? "common/caulk" : textureName;
            
            // Transform to Q3 coordinates
            v0 = TransformToQ3(v0);
            v1 = TransformToQ3(v1);
            v2 = TransformToQ3(v2);
            
            // Reverse winding order to fix mirrored planes
            // Swap v1 and v2 to flip the triangle normal
            var temp = v1;
            v1 = v2;
            v2 = temp;
            
            // Skip degenerate triangles
            var edge1 = v1 - v0;
            var edge2 = v2 - v0;
            var normal = Vector3.Cross(edge1, edge2);
            if (normal.Length() < 0.001f)
                return string.Empty;
            
            // Create axis-aligned bounding box (6 planes) - Q3 standard format
            // This is simpler and guaranteed to work
            float minX = MathF.Min(v0.X, MathF.Min(v1.X, v2.X)) - 0.1f;
            float minY = MathF.Min(v0.Y, MathF.Min(v1.Y, v2.Y)) - 0.1f;
            float minZ = MathF.Min(v0.Z, MathF.Min(v1.Z, v2.Z)) - 0.1f;
            float maxX = MathF.Max(v0.X, MathF.Max(v1.X, v2.X)) + 0.1f;
            float maxY = MathF.Max(v0.Y, MathF.Max(v1.Y, v2.Y)) + 0.1f;
            float maxZ = MathF.Max(v0.Z, MathF.Max(v1.Z, v2.Z)) + 0.1f;
            
            brush.AppendLine("{");
            
            // 6 planes in standard Q3 format (matching test_cube.map)
            // -X plane (left face)
            WritePlaneLine(brush, new Vector3(minX, minY, minZ), new Vector3(minX, minY+1, minZ), new Vector3(minX, minY, minZ+1), tex);
            // -Y plane (back face)  
            WritePlaneLine(brush, new Vector3(minX, minY, minZ), new Vector3(minX, minY, minZ+1), new Vector3(minX+1, minY, minZ), tex);
            // -Z plane (bottom face)
            WritePlaneLine(brush, new Vector3(minX, minY, minZ), new Vector3(minX+1, minY, minZ), new Vector3(minX, minY+1, minZ), tex);
            // +Z plane (top face)
            WritePlaneLine(brush, new Vector3(maxX, maxY, maxZ), new Vector3(maxX, maxY+1, maxZ), new Vector3(maxX+1, maxY, maxZ), tex);
            // +Y plane (front face)
            WritePlaneLine(brush, new Vector3(maxX, maxY, maxZ), new Vector3(maxX+1, maxY, maxZ), new Vector3(maxX, maxY, maxZ+1), tex);
            // +X plane (right face)
            WritePlaneLine(brush, new Vector3(maxX, maxY, maxZ), new Vector3(maxX, maxY, maxZ+1), new Vector3(maxX, maxY+1, maxZ), tex);
            
            brush.AppendLine("}");
            
            return brush.ToString();
        }
        
        private void WritePlaneLine(StringBuilder brush, Vector3 p0, Vector3 p1, Vector3 p2, string texture)
        {
            // Quake 3 plane format: ( x y z ) ( x y z ) ( x y z ) TEXTURE offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value
            // Must have 8 parameters after texture name (not 5!)
            brush.AppendLine($"  ( {p0.X:F6} {p0.Y:F6} {p0.Z:F6} ) ( {p1.X:F6} {p1.Y:F6} {p1.Z:F6} ) ( {p2.X:F6} {p2.Y:F6} {p2.Z:F6} ) {texture} 0 0 0 0.5 0.5 0 0 0");
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
            
            // Place spawn ABOVE the WMO geometry (so player doesn't spawn inside geometry)
            var center = (min + max) * 0.5f;
            center.Z = max.Z + 64.0f; // 64 units above the WMO
            
            mapContent.AppendLine("// Default spawn");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"info_player_deathmatch\"");
            mapContent.AppendLine($"\"origin\" \"{center.X:F1} {center.Y:F1} {center.Z:F1}\"");
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