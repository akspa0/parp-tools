using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using WmoBspConverter.Bsp;
using WmoBspConverter.Export;

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
            // WMO stores coordinates as (X, Y, Z) with Z up. Quake 3 also uses Z up.
            // Align axes directly so forward (Y) stays forward and vertical (Z) stays vertical.
            return new Vector3(v.X, v.Y, v.Z);
        }

        private sealed record GeometryBounds(Vector3 Min, Vector3 Max)
        {
            public Vector3 Center => (Min + Max) * 0.5f;
            public Vector3 Size => Max - Min;
        }

        private sealed class MapContext
        {
            public GeometryBounds Bounds { get; init; } = new GeometryBounds(Vector3.Zero, Vector3.Zero);
            public Vector3 GeometryOffset { get; init; } = Vector3.Zero;
        }

        private static GeometryBounds ComputeGeometryBounds(BspFile bspFile)
        {
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            foreach (var vertex in bspFile.Vertices)
            {
                var p = TransformToQ3(vertex.Position);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }

            if (bspFile.Vertices.Count == 0)
            {
                min = Vector3.Zero;
                max = Vector3.Zero;
            }

            return new GeometryBounds(min, max);
        }

        private static Vector3 ComputeGeometryOffset(GeometryBounds bounds)
        {
            var center = bounds.Center;
            return new Vector3(center.X, center.Y, bounds.Min.Z);
        }

        private (MapContext context, GeometryBounds paddedBounds) PrepareContext(BspFile bspFile)
        {
            var bounds = ComputeGeometryBounds(bspFile);
            var offset = ComputeGeometryOffset(bounds);
            var padding = new Vector3(128f, 128f, 128f);
            var paddedBounds = new GeometryBounds(bounds.Min - padding - offset, bounds.Max + padding - offset);

            Console.WriteLine($"[DEBUG] Geometry bounds min={bounds.Min}, max={bounds.Max}, offset={offset}");

            return (new MapContext { Bounds = bounds, GeometryOffset = offset }, paddedBounds);
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
            
            var (context, paddedBounds) = PrepareContext(bspFile);
            var combinedMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var combinedMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            // Create sealed worldspawn box to contain the WMO
            CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
            Console.WriteLine($"[DEBUG] After worldspawn, length: {mapContent.Length:N0}");
            
            // Add player spawn entity
            AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);
            Console.WriteLine($"[DEBUG] After AddSpawn, length: {mapContent.Length:N0}");
            
            // Add WMO geometry as a func_group entity (separate from worldspawn!)
            StartFuncGroupEntity(mapContent, wmoData);
            var defaultTex = bspFile.Textures.Count > 0 ? bspFile.Textures[0].Name : "textures/common/caulk";
            GenerateBrushesFromGeometry(mapContent, bspFile, defaultTex, context);
            EndFuncGroupEntity(mapContent);
            Console.WriteLine($"[DEBUG] After func_group, length: {mapContent.Length:N0}");
            
            // Generate texture info
            GenerateTextureInfo(mapContent, wmoData);
            Console.WriteLine($"[DEBUG] After GenerateTextureInfo, length: {mapContent.Length:N0}");
            
            // Generate Portals (Hint brushes)
            GeneratePortalBrushes(mapContent, wmoData, context);
            Console.WriteLine($"[DEBUG] After GeneratePortalBrushes, length: {mapContent.Length:N0}");
            
            // Write to file
            Console.WriteLine($"[DEBUG] StringBuilder length: {mapContent.Length:N0} characters");
            var finalContent = mapContent.ToString();
            Console.WriteLine($"[DEBUG] Final string length: {finalContent.Length:N0} characters");
            File.WriteAllText(outputPath, finalContent);
            
            Console.WriteLine($"[INFO] Generated .map file: {outputPath}");
        }

        public void GenerateMapFileWithModels(
            string outputPath,
            string outputRootDir,
            string wmoName,
            WmoV14Parser.WmoV14Data wmoData,
            BspFile bspFile,
            AseWriter aseWriter)
        {
            var (context, paddedBounds) = PrepareContext(bspFile);
            var materialShaderNames = BuildMaterialShaderNames(wmoData);
            var placements = new List<(int GroupIndex, AseWriter.ExportResult Result)>();
            var modelMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var modelMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            int g = 0;
            foreach (var group in wmoData.Groups)
            {
                var intIndices = group.Indices.ConvertAll(i => (int)i);
                var faceMats = group.FaceMaterials.ConvertAll(f => (int)f);
                var result = aseWriter.ExportGroup(
                    outputRootDir,
                    wmoName,
                    g,
                    group.Vertices,
                    intIndices,
                    faceMats,
                    materialShaderNames,
                    context.GeometryOffset);

                placements.Add((g, result));
                modelMin = Vector3.Min(modelMin, result.RoomMin);
                modelMax = Vector3.Max(modelMax, result.RoomMax);
                g++;
            }

            var zShift = placements.Count == 0 ? 0f : paddedBounds.Min.Z - modelMin.Z;
            const float modelLift = 8.0f; // raise models slightly above floor to avoid z-fighting
            zShift += modelLift;

            var mapContent = new StringBuilder();
            mapContent.AppendLine("// Auto-generated from WMO v14 file (ASE model placement)");
            mapContent.AppendLine($"// Original: WMO v{wmoData.Version}");
            mapContent.AppendLine($"// Groups: {wmoData.Groups.Count}");
            mapContent.AppendLine($"// Textures: {wmoData.Textures.Count}");
            mapContent.AppendLine();

            // Sealed room and spawn
            CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
            AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);

            foreach (var placement in placements)
            {
                var result = placement.Result;
                var adjustedOrigin = new Vector3(
                    result.RoomCenter.X,
                    result.RoomCenter.Y,
                    result.RoomCenter.Z + zShift);

                mapContent.AppendLine("// WMO group model");
                mapContent.AppendLine("{");
                mapContent.AppendLine("\"classname\" \"misc_model\"");
                mapContent.AppendLine($"\"model\" \"{result.RelativeModelPath.Replace("\\", "/")}\"");
                mapContent.AppendLine($"\"origin\" \"{adjustedOrigin.X:F3} {adjustedOrigin.Y:F3} {adjustedOrigin.Z:F3}\"");
                mapContent.AppendLine($"\"_wmo_group\" \"{placement.GroupIndex}\"");
                mapContent.AppendLine("}");
                mapContent.AppendLine();
            }

            Console.WriteLine($"[DEBUG] Combined model bounds (room): min={modelMin}, max={modelMax}, zShift={zShift}");

            File.WriteAllText(outputPath, mapContent.ToString());
            Console.WriteLine($"[INFO] Generated .map (models): {outputPath}");
        }

        private void CreateSealedWorldspawn(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, GeometryBounds bounds)
        {
            var min = bounds.Min;
            var max = bounds.Max;

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

        private void GenerateBrushesFromGeometry(StringBuilder mapContent, BspFile bspFile, string defaultTexture, MapContext context)
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
                var brush = GenerateTriangleBrushFromGeometry(v0, v1, v2, faceTex, context.GeometryOffset);
                if (string.IsNullOrEmpty(brush))
                    continue;

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

        private const string CaulkTexture = "common/caulk";
        private const float TriangleThickness = 2.0f;
        private static readonly string BR = ""; // inline appending only

        private string GenerateTriangleBrushFromGeometry(Vector3 v0, Vector3 v1, Vector3 v2, string textureName, Vector3 geometryOffset)
        {
            var tex = string.IsNullOrWhiteSpace(textureName) ? CaulkTexture : textureName;

            // Transform to Q3 coordinates and translate into sealed room space
            var q0 = TransformToQ3(v0) - geometryOffset;
            var q1 = TransformToQ3(v1) - geometryOffset;
            var q2 = TransformToQ3(v2) - geometryOffset;

            // Skip degenerate triangles
            var edge1 = q1 - q0;
            var edge2 = q2 - q0;
            var normal = Vector3.Cross(edge1, edge2);
            var normalLength = normal.Length();
            if (normalLength < 1e-4f)
            {
                return string.Empty;
            }
            normal /= normalLength;

            var halfThickness = TriangleThickness * 0.5f;
            var offset = normal * halfThickness;

            var top0 = q0 + offset;
            var top1 = q1 + offset;
            var top2 = q2 + offset;
            var bottom0 = q0 - offset;
            var bottom1 = q1 - offset;
            var bottom2 = q2 - offset;

            var interiorPoint = (top0 + top1 + top2 + bottom0 + bottom1 + bottom2) / 6f;
            var caulk = CaulkTexture;

            // Pre-validate all 5 planes before writing anything
            // If any plane is degenerate, skip the entire brush
            var planes = new (Vector3 p0, Vector3 p1, Vector3 p2, string texture)[]
            {
                (top0, top1, top2, tex),                    // Top face (textured)
                (bottom0, bottom2, bottom1, caulk),         // Bottom face
                (top1, top0, bottom0, caulk),               // Edge 0-1
                (top2, top1, bottom1, caulk),               // Edge 1-2
                (top0, top2, bottom2, caulk),               // Edge 2-0
            };

            // Validate and compute corrected winding for each plane
            var validatedPlanes = new List<(Vector3 p0, Vector3 p1, Vector3 p2, string texture)>();
            foreach (var (p0, p1, p2, texture) in planes)
            {
                var planeNormal = Vector3.Cross(p1 - p0, p2 - p0);
                if (planeNormal.LengthSquared() < 1e-6f)
                {
                    // Degenerate plane - skip entire brush
                    return string.Empty;
                }

                // Correct winding so normal points away from interior
                var finalP1 = p1;
                var finalP2 = p2;
                if (Vector3.Dot(planeNormal, interiorPoint - p0) > 0f)
                {
                    finalP1 = p2;
                    finalP2 = p1;
                }
                validatedPlanes.Add((p0, finalP1, finalP2, texture));
            }

            // All planes valid - write the brush
            var brush = new StringBuilder();
            brush.AppendLine("{");

            foreach (var (p0, p1, p2, texture) in validatedPlanes)
            {
                WritePlaneLine(brush, p0, p1, p2, texture);
            }

            brush.AppendLine("}");

            return brush.ToString();
        }

        private void WritePlaneLine(StringBuilder brush, Vector3 p0, Vector3 p1, Vector3 p2, string texture)
        {
            // Quake 3 plane format: ( x y z ) ( x y z ) ( x y z ) TEXTURE offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value
            // Must have 8 parameters after texture name (not 5!)
            brush.AppendLine($"  ( {p0.X:F6} {p0.Y:F6} {p0.Z:F6} ) ( {p1.X:F6} {p1.Y:F6} {p1.Z:F6} ) ( {p2.X:F6} {p2.Y:F6} {p2.Z:F6} ) {texture} 0 0 0 0.5 0.5 0 0 0");
        }

        private void WriteBrushPlane(StringBuilder brush, Vector3 p0, Vector3 p1, Vector3 p2, string texture, Vector3 interiorPoint)
        {
            var normal = Vector3.Cross(p1 - p0, p2 - p0);
            if (normal.LengthSquared() < 1e-6f)
            {
                return;
            }

            if (Vector3.Dot(normal, interiorPoint - p0) > 0f)
            {
                (p1, p2) = (p2, p1);
            }

            WritePlaneLine(brush, p0, p1, p2, texture);
        }

        private void AddSpawnEntity(StringBuilder mapContent, MapContext context, float floorZ)
        {
            var bounds = context.Bounds;
            var center = bounds.Center - context.GeometryOffset;
            center.Z = floorZ + 16.0f; // place spawn slightly above sealed floor

            mapContent.AppendLine("// Default spawn");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"info_player_deathmatch\"");
            mapContent.AppendLine($"\"origin\" \"{center.X:F1} {center.Y:F1} {center.Z:F1}\"");
            mapContent.AppendLine("\"angle\" \"0\"");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private static IReadOnlyList<string> BuildMaterialShaderNames(WmoV14Parser.WmoV14Data wmoData)
        {
            var names = new List<string>();
            if (wmoData.Materials.Count > 0 && wmoData.MaterialTextureIndices.Count > 0)
            {
                for (int i = 0; i < wmoData.Materials.Count; i++)
                {
                    var texIdx = i < wmoData.MaterialTextureIndices.Count ? (int)wmoData.MaterialTextureIndices[i] : -1;
                    names.Add(BuildShaderNameForTexture(wmoData, texIdx));
                }
            }
            else if (wmoData.Textures.Count > 0)
            {
                for (int i = 0; i < wmoData.Textures.Count; i++)
                {
                    names.Add(BuildShaderNameForTexture(wmoData, i));
                }
            }

            if (names.Count == 0)
            {
                names.Add("textures/wmo/wmo_default");
            }

            return names;
        }

        private static string BuildShaderNameForTexture(WmoV14Parser.WmoV14Data wmoData, int textureIndex)
        {
            if (textureIndex >= 0 && textureIndex < wmoData.Textures.Count)
            {
                var tex = wmoData.Textures[textureIndex];
                var baseName = Path.GetFileNameWithoutExtension(tex).ToLowerInvariant();
                return $"textures/wmo/{baseName}.tga";
            }

            return "textures/wmo/wmo_default.tga";
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

        private void GeneratePortalBrushes(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, MapContext context)
        {
            if (wmoData.Portals.Count == 0 || wmoData.PortalVertices.Count == 0)
                return;

            mapContent.AppendLine("// Portal Hint Brushes");
            mapContent.AppendLine($"// Generated {wmoData.Portals.Count} portals from {wmoData.PortalVertices.Count} vertices");

            int pIdx = 0;
            foreach (var portal in wmoData.Portals)
            {
                if (portal.VertexCount < 3) continue;

                var poly = new List<Vector3>();
                for (int i = 0; i < portal.VertexCount; i++)
                {
                    int vIdx = portal.FirstVertex + i;
                    if (vIdx < wmoData.PortalVertices.Count)
                    {
                         poly.Add(wmoData.PortalVertices[vIdx]);
                    }
                }

                if (poly.Count < 3) continue;

                // Transform to Q3 and offset
                for(int i=0; i<poly.Count; i++) 
                {
                    poly[i] = TransformToQ3(poly[i]) - context.GeometryOffset;
                }

                // Generate Brush
                // Use common/hint for the portal face and common/skip for the rest
                string brush = GenerateBrushFromPolygon(poly, "common/hint", "common/skip");
                mapContent.Append(brush);
                pIdx++;
            }
        }

        private string GenerateBrushFromPolygon(List<Vector3> points, string frontTex, string otherTex)
        {
            // Compute normal from first 3 points (assuming planar)
            var v0 = points[0];
            var v1 = points[1];
            var v2 = points[2];
            var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
            
            // Thickness
            float thickness = 4.0f; // Thin brush
            var offset = normal * (thickness * 0.5f);
            
            var brush = new StringBuilder();
            brush.AppendLine("{");
            brush.AppendLine($"// Hint Brush (Poly: {points.Count} verts)");
            
            // Front Face (Original polygon + offset)
            // Winding: v0, v1, v2 is CCW looking from front.
            // Pushed forward by offset -> Still v0, v1, v2
            var f0 = v0 + offset;
            var f1 = v1 + offset;
            var f2 = v2 + offset;
            WritePlaneLine(brush, f0, f1, f2, frontTex);

            // Back Face (Original polygon - offset)
            // Pushed back. To face away, we reverse winding: v0, v2, v1
            var b0 = v0 - offset;
            var b1 = v1 - offset;
            var b2 = v2 - offset;
            WritePlaneLine(brush, b0, b2, b1, otherTex);

            // Side Faces
            for (int i = 0; i < points.Count; i++)
            {
                var pA = points[i];
                var pB = points[(i + 1) % points.Count];
                
                var topA = pA + offset;
                var topB = pB + offset;
                var botA = pA - offset;
                var botB = pB - offset;
                
                // Side plane: topB, topA, botA (CCW looking from outside)
                WritePlaneLine(brush, topB, topA, botA, otherTex);
            }

            brush.AppendLine("}");
            return brush.ToString();
        }
    }
}