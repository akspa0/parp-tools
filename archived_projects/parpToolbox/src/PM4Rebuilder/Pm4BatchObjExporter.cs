using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using PM4Scene = ParpToolbox.Formats.PM4.Pm4Scene;

namespace PM4Rebuilder
{
    /// <summary>
    /// OBJ exporter that applies the decoded field algorithm discovered through batch analysis.
    /// Uses HasGeometry flag, ParentIndex grouping, and decoded field meanings for accurate object assembly.
    /// </summary>
    internal static class Pm4BatchObjExporter
    {
        public static async Task<int> ExportAllWithDecodedFields(string inputDirectory, string outputDirectory)
        {
            Console.WriteLine($"[BATCH OBJ EXPORTER] Starting UNIFIED MAP OBJ export with decoded field algorithm...");
            Console.WriteLine($"[BATCH OBJ EXPORTER] CRITICAL: Loading ALL PM4s as single unified map object to preserve cross-tile MSLK linkages");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Input: {inputDirectory}");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Output: {outputDirectory}");
            
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var batchOutputDir = Path.Combine(outputDirectory, $"unified_map_export_{timestamp}");
            Directory.CreateDirectory(batchOutputDir);
            
            var perTileDir = Path.Combine(batchOutputDir, "per_tile_objects");
            var globalDir = Path.Combine(batchOutputDir, "global_unified");
            Directory.CreateDirectory(perTileDir);
            Directory.CreateDirectory(globalDir);
            
            // Find ALL PM4 files
            var pm4Files = Directory.GetFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly);
            Console.WriteLine($"[BATCH OBJ EXPORTER] Found {pm4Files.Length} PM4 files to process as unified map");
            
            // STEP 1: Load ALL PM4s into a single unified scene (preserving cross-tile linkages)
            Console.WriteLine($"[BATCH OBJ EXPORTER] STEP 1: Loading all PM4s into unified map object...");
            var unifiedScene = await LoadUnifiedMapScene(pm4Files);
            if (unifiedScene == null)
            {
                Console.WriteLine($"[BATCH OBJ EXPORTER] ERROR: Failed to load unified map scene");
                return 1;
            }
            
            Console.WriteLine($"[BATCH OBJ EXPORTER] Unified scene loaded: {unifiedScene.TileCount} tiles, {unifiedScene.GlobalVertices.Count} vertices, {unifiedScene.GlobalLinks.Count} links");
            
            // STEP 2: (Deprecated) Skip object extraction – exporting raw tile scenes only
            
            
            var exportResults = new List<TileExportResult>();
            var globalVertices = new List<string>();
            var globalFaces = new List<string>();
            
            
            
            // STEP 3: Export per-tile OBJ files (whole tile geometry)
            Console.WriteLine($"[BATCH OBJ EXPORTER] STEP 3: Exporting per-tile OBJ files...");

            foreach (var kvp in unifiedScene.TileScenes)
            {
                var tileName = kvp.Key;
                var tileScene = kvp.Value;
                
                try
                {
                    Console.WriteLine($"[BATCH OBJ EXPORTER] Exporting tile {tileName}: {tileScene.Vertices.Count} verts / {tileScene.Triangles.Count} tris");
                    
                    var tileOutputDir = Path.Combine(perTileDir, tileName);
                    Directory.CreateDirectory(tileOutputDir);
                    
                    // Dump the full tile geometry into a single OBJ file
                    var tileVertices = new List<string>();
                    var tileFaces = new List<string>();
                    var tileObjPath = Path.Combine(tileOutputDir, $"{tileName}.obj");
                    await ExportTileScene(tileScene, tileObjPath, tileVertices, tileFaces, globalVertices.Count);
                    
                    // Add to global collections
                    lock (globalVertices)
                    {
                        globalVertices.AddRange(tileVertices);
                        globalFaces.AddRange(tileFaces);
                    }
                    
                    exportResults.Add(new TileExportResult 
                    { 
                        FileName = tileName, 
                        Success = true, 
                        ObjectCount = 1,
                        VertexCount = tileVertices.Count,
                        FaceCount = tileFaces.Count
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[BATCH OBJ EXPORTER] ERROR processing {tileName}: {ex.Message}");
                    exportResults.Add(new TileExportResult { FileName = tileName, Success = false, ErrorMessage = ex.Message, ObjectCount = 0 });
                }
            }
            
            // Export global merged OBJ
            await ExportGlobalMergedOBJ(globalVertices, globalFaces, globalDir);
            
            // Generate export summary
            await GenerateExportSummary(exportResults, batchOutputDir);
            
            var totalSuccess = exportResults.Count(r => r.Success);
            var totalExportedObjects = exportResults.Sum(r => r.ObjectCount);
            Console.WriteLine($"[BATCH OBJ EXPORTER] Export complete!");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Successfully processed: {totalSuccess}/{pm4Files.Length} files");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Total objects exported: {totalExportedObjects}");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Global vertices: {globalVertices.Count:N0}");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Global faces: {globalFaces.Count:N0}");
            Console.WriteLine($"[BATCH OBJ EXPORTER] Output directory: {batchOutputDir}");
            
            return 0;
        }
        

        

        

        
        private static void ExtractSurfaceGeometry(dynamic surface, List<int> indices, List<System.Numerics.Vector3> vertices,
            List<(float X, float Y, float Z)> objectVertices, List<(int A, int B, int C)> objectFaces)
        {
            try
            {
                // Use reflection to get surface properties
                var surfaceType = surface.GetType();
                var startIndexProp = surfaceType.GetProperty("StartIndex");
                var indexCountProp = surfaceType.GetProperty("IndexCount");
                
                if (startIndexProp == null || indexCountProp == null) return;
                
                var startIndex = (int)startIndexProp?.GetValue(surface)!;
                var indexCount = (int)indexCountProp?.GetValue(surface)!;
                
                if (indexCount < 3) return; // Need at least a triangle
                
                // Extract vertex indices
                var surfaceIndices = new List<int>();
                for (int i = 0; i < indexCount && (startIndex + i) < indices.Count; i++)
                {
                    surfaceIndices.Add(indices[startIndex + i]);
                }
                
                if (surfaceIndices.Count < 3) return;
                
                // Add vertices and create faces
                var vertexMap = new Dictionary<int, int>();
                
                foreach (var vertexIndex in surfaceIndices)
                {
                    if (!vertexMap.ContainsKey(vertexIndex) && vertexIndex < vertices.Count)
                    {
                        var vertex = vertices[vertexIndex];
                        objectVertices.Add((vertex.X, vertex.Y, vertex.Z));
                        vertexMap[vertexIndex] = objectVertices.Count - 1;
                    }
                }
                
                // Create triangular faces (fan triangulation for N-gons)
                if (surfaceIndices.Count >= 3)
                {
                    for (int i = 1; i < surfaceIndices.Count - 1; i++)
                    {
                        var idx0 = surfaceIndices[0];
                        var idx1 = surfaceIndices[i];
                        var idx2 = surfaceIndices[i + 1];
                        
                        if (vertexMap.ContainsKey(idx0) && vertexMap.ContainsKey(idx1) && vertexMap.ContainsKey(idx2))
                        {
                            objectFaces.Add((vertexMap[idx0], vertexMap[idx1], vertexMap[idx2]));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BATCH OBJ EXPORTER] Warning: Failed to extract surface geometry: {ex.Message}");
            }
        }
        
        private static async Task ExportSingleObject(DecodedPm4Object obj, string objPath, 
            List<string> tileVertices, List<string> tileFaces, int globalVertexOffset)
        {
            var objContent = new StringBuilder();
            objContent.AppendLine($"# {obj.Name}");
            objContent.AppendLine($"# Source: {obj.SourceTile}");
            objContent.AppendLine($"# ParentIndex: {obj.ParentIndex}");
            objContent.AppendLine($"# Links: {obj.LinkCount}");
            objContent.AppendLine($"# Placement: {obj.PlacementPosition}");
            objContent.AppendLine($"# Vertices: {obj.Vertices.Count}, Faces: {obj.Faces.Count}");
            objContent.AppendLine();
            
            // Export vertices
            foreach (var vertex in obj.Vertices)
            {
                var vertexLine = $"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}";
                objContent.AppendLine(vertexLine);
                tileVertices.Add(vertexLine);
            }
            
            // Export faces (1-indexed)
            foreach (var face in obj.Faces)
            {
                var faceLine = $"f {face.A + 1} {face.B + 1} {face.C + 1}";
                objContent.AppendLine(faceLine);
                
                // For global export, adjust indices
                var globalFaceLine = $"f {face.A + globalVertexOffset + 1} {face.B + globalVertexOffset + 1} {face.C + globalVertexOffset + 1}";
                tileFaces.Add(globalFaceLine);
            }
            
            await File.WriteAllTextAsync(objPath, objContent.ToString());
        }
        
        private static async Task ExportTileAggregate(List<DecodedPm4Object> objects, string objPath,
            List<string> tileVertices, List<string> tileFaces, int globalVertexOffset)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# PM4 Tile Aggregate OBJ");
            sb.AppendLine($"# Tile: {Path.GetFileNameWithoutExtension(objPath)}");
            sb.AppendLine($"# Objects merged: {objects.Count}");
            sb.AppendLine();

            int localVertexOffset = 0;
            int globalOffsetBase = globalVertexOffset;

            foreach (var obj in objects)
            {
                sb.AppendLine($"o {obj.Name}");
                sb.AppendLine($"# ParentIndex: {obj.ParentIndex}, Links: {obj.LinkCount}");

                // Vertices
                foreach (var vertex in obj.Vertices)
                {
                    var vLine = $"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}";
                    sb.AppendLine(vLine);
                    tileVertices.Add(vLine);
                }

                // Faces
                foreach (var face in obj.Faces)
                {
                    var faceLocal = $"f {face.A + localVertexOffset + 1} {face.B + localVertexOffset + 1} {face.C + localVertexOffset + 1}";
                    sb.AppendLine(faceLocal);

                    var faceGlobal = $"f {face.A + globalOffsetBase + localVertexOffset + 1} {face.B + globalOffsetBase + localVertexOffset + 1} {face.C + globalOffsetBase + localVertexOffset + 1}";
                    tileFaces.Add(faceGlobal);
                }

                localVertexOffset += obj.Vertices.Count;
            }

            await File.WriteAllTextAsync(objPath, sb.ToString());
        }

        // Exports geometry for a single tile scene directly to OBJ
        private static async Task ExportTileScene(ParpToolbox.Formats.PM4.Pm4Scene scene, string objPath,
            List<string> tileVertices, List<string> tileFaces, int globalVertexOffset)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# PM4 Tile OBJ");
            sb.AppendLine($"# Tile: {Path.GetFileNameWithoutExtension(objPath)}");
            sb.AppendLine($"# Vertices: {scene.Vertices.Count}, Faces: {scene.Triangles.Count}");
            sb.AppendLine();

            // Vertices
            foreach (var v in scene.Vertices)
            {
                var vLine = $"v {v.X:F6} {v.Y:F6} {v.Z:F6}";
                sb.AppendLine(vLine);
                tileVertices.Add(vLine);
            }

            // Faces
            foreach (var tri in scene.Triangles)
            {
                var faceLocal = $"f {tri.A + 1} {tri.B + 1} {tri.C + 1}";
                sb.AppendLine(faceLocal);
                var faceGlobal = $"f {tri.A + globalVertexOffset + 1} {tri.B + globalVertexOffset + 1} {tri.C + globalVertexOffset + 1}";
                tileFaces.Add(faceGlobal);
            }

            await File.WriteAllTextAsync(objPath, sb.ToString());
        }

        private static async Task ExportGlobalMergedOBJ(List<string> globalVertices, List<string> globalFaces, string globalDir)
        {
            var globalObjPath = Path.Combine(globalDir, "all_pm4_objects_merged.obj");
            var globalContent = new StringBuilder();
            
            globalContent.AppendLine("# Global PM4 Objects - Merged from All Tiles");
            globalContent.AppendLine($"# Generated: {DateTime.Now}");
            globalContent.AppendLine($"# Total vertices: {globalVertices.Count:N0}");
            globalContent.AppendLine($"# Total faces: {globalFaces.Count:N0}");
            globalContent.AppendLine();
            
            // Write all vertices
            foreach (var vertex in globalVertices)
            {
                globalContent.AppendLine(vertex);
            }
            
            globalContent.AppendLine();
            
            // Write all faces
            foreach (var face in globalFaces)
            {
                globalContent.AppendLine(face);
            }
            
            await File.WriteAllTextAsync(globalObjPath, globalContent.ToString());
            Console.WriteLine($"[BATCH OBJ EXPORTER] Global merged OBJ exported: {globalObjPath}");
        }
        
        private static async Task GenerateExportSummary(List<TileExportResult> results, string outputDir)
        {
            var summary = new StringBuilder();
            summary.AppendLine("PM4 BATCH OBJ EXPORT SUMMARY");
            summary.AppendLine("============================");
            summary.AppendLine($"Generated: {DateTime.Now}");
            summary.AppendLine();
            
            var successful = results.Where(r => r.Success).ToList();
            var failed = results.Where(r => !r.Success).ToList();
            
            summary.AppendLine($"Total files processed: {results.Count}");
            summary.AppendLine($"Successful: {successful.Count} ({(double)successful.Count / results.Count * 100:F1}%)");
            summary.AppendLine($"Failed: {failed.Count} ({(double)failed.Count / results.Count * 100:F1}%)");
            summary.AppendLine();
            
            if (successful.Any())
            {
                summary.AppendLine("EXPORT STATISTICS:");
                summary.AppendLine($"Total objects exported: {successful.Sum(r => r.ObjectCount):N0}");
                summary.AppendLine($"Total vertices: {successful.Sum(r => r.VertexCount):N0}");
                summary.AppendLine($"Total faces: {successful.Sum(r => r.FaceCount):N0}");
                summary.AppendLine($"Average objects per tile: {successful.Average(r => r.ObjectCount):F1}");
                summary.AppendLine($"Max objects in single tile: {successful.Max(r => r.ObjectCount)}");
                summary.AppendLine();
            }
            
            if (failed.Any())
            {
                summary.AppendLine("FAILED FILES:");
                foreach (var failure in failed.Take(20)) // Show first 20 failures
                {
                    summary.AppendLine($"  {failure.FileName}: {failure.ErrorMessage}");
                }
                if (failed.Count > 20)
                {
                    summary.AppendLine($"  ... and {failed.Count - 20} more");
                }
            }
            
            await File.WriteAllTextAsync(Path.Combine(outputDir, "export_summary.txt"), summary.ToString());
        }
    
    internal class TileExportResult
    {
        public string FileName { get; set; } = "";
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        public int ObjectCount { get; set; }
        public int VertexCount { get; set; }
        public int FaceCount { get; set; }
    }
    
    internal class DecodedPm4Object
    {
        public string Name { get; set; } = "";
        public uint ParentIndex { get; set; }
        public List<(float X, float Y, float Z)> Vertices { get; set; } = new();
        public List<(int A, int B, int C)> Faces { get; set; } = new();
        public System.Numerics.Vector3 PlacementPosition { get; set; }
        public string SourceTile { get; set; } = "";
        public int LinkCount { get; set; }
    }
    
    /// <summary>
    /// Unified map scene containing all PM4 tiles with preserved cross-tile linkages
    /// </summary>
    internal class UnifiedMapScene
    {
        public int TileCount { get; set; }
        public List<System.Numerics.Vector3> GlobalVertices { get; set; } = new();
        public List<System.Numerics.Vector3> GlobalMscnVertices { get; set; } = new();
        public List<int> GlobalIndices { get; set; } = new();
        public List<dynamic> GlobalLinks { get; set; } = new();
        public List<dynamic> GlobalSurfaces { get; set; } = new();
        public List<dynamic> GlobalPlacements { get; set; } = new();
        public Dictionary<string, (int VertexOffset, int MscnOffset, int IndexOffset, int LinkOffset, int SurfaceOffset, int PlacementOffset)> TileOffsets { get; set; } = new();
        public Dictionary<string, PM4Scene> TileScenes { get; set; } = new();
    }
    
    /// <summary>
    /// Load all PM4 files into a single unified scene preserving cross-tile linkages
    /// </summary>
    private static async Task<UnifiedMapScene?> LoadUnifiedMapScene(string[] pm4Files)
    {
        var unifiedScene = new UnifiedMapScene();
        int successfulTiles = 0;
        
        Console.WriteLine($"[UNIFIED LOADER] Processing {pm4Files.Length} PM4 files...");
        
        foreach (var pm4File in pm4Files)
        {
            try
            {
                var tileName = Path.GetFileNameWithoutExtension(pm4File);
                Console.WriteLine($"[UNIFIED LOADER] Loading tile: {tileName}");
                
                var scene = await SceneLoaderHelper.LoadSceneAsync(pm4File, false, true, false);
                if (scene == null)
                {
                    Console.WriteLine($"[UNIFIED LOADER] WARNING: Failed to load {tileName}");
                    continue;
                }
                
                // Record offsets for this tile
                var vertexOffset = unifiedScene.GlobalVertices.Count;
                var mscnOffset = unifiedScene.GlobalMscnVertices.Count;
                var indexOffset = unifiedScene.GlobalIndices.Count;
                var linkOffset = unifiedScene.GlobalLinks.Count;
                var surfaceOffset = unifiedScene.GlobalSurfaces.Count;
                var placementOffset = unifiedScene.GlobalPlacements.Count;
                
                unifiedScene.TileOffsets[tileName] = (vertexOffset, mscnOffset, indexOffset, linkOffset, surfaceOffset, placementOffset);
                unifiedScene.TileScenes[tileName] = scene;
                
                // Append data to global pools
                if (scene.Vertices?.Any() == true)
                    unifiedScene.GlobalVertices.AddRange(scene.Vertices);
                    
                if (scene.MscnVertices?.Any() == true)
                    unifiedScene.GlobalMscnVertices.AddRange(scene.MscnVertices);
                    
                if (scene.Indices?.Any() == true)
                    unifiedScene.GlobalIndices.AddRange(scene.Indices);
                    
                if (scene.Links?.Any() == true)
                    unifiedScene.GlobalLinks.AddRange(scene.Links);
                    
                if (scene.Surfaces?.Any() == true)
                    unifiedScene.GlobalSurfaces.AddRange(scene.Surfaces);
                    
                if (scene.Placements?.Any() == true)
                    unifiedScene.GlobalPlacements.AddRange(scene.Placements);
                
                successfulTiles++;
                if (successfulTiles % 50 == 0)
                    Console.WriteLine($"[UNIFIED LOADER] Progress: {successfulTiles}/{pm4Files.Length} tiles loaded");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[UNIFIED LOADER] ERROR loading {Path.GetFileNameWithoutExtension(pm4File)}: {ex.Message}");
            }
        }
        
        unifiedScene.TileCount = successfulTiles;
        Console.WriteLine($"[UNIFIED LOADER] Unified scene complete: {successfulTiles} tiles, {unifiedScene.GlobalVertices.Count} vertices, {unifiedScene.GlobalLinks.Count} links");
        
        return successfulTiles > 0 ? unifiedScene : null;
    }
    
    /// <summary>
    /// Extract objects from unified scene with preserved cross-tile linkages
    /// </summary>
    private static List<DecodedPm4Object> ExtractObjectsFromUnifiedScene(UnifiedMapScene unifiedScene)
    {
        var objects = new List<DecodedPm4Object>();
        
        Console.WriteLine($"[UNIFIED EXTRACTOR] Processing unified scene: {unifiedScene.GlobalLinks.Count} links, {unifiedScene.GlobalPlacements.Count} placements");
        
        // Step 1: Filter MSLK entries by HasGeometry flag (decoded algorithm)
        var geometryLinks = unifiedScene.GlobalLinks.Where(link =>
        {
            var hasGeometryProp = link.GetType().GetProperty("HasGeometry");
            if (hasGeometryProp == null) return false;
            
            var hasGeometry = hasGeometryProp.GetValue(link);
            return hasGeometry is bool b && b;
        }).ToList();
        
        Console.WriteLine($"[UNIFIED EXTRACTOR] Global MSLK entries with HasGeometry=true: {geometryLinks.Count}");
        
        // Step 2: Group by ParentIndex (global cross-tile grouping)
        var objectGroups = geometryLinks.GroupBy(link =>
        {
            var parentIndexProp = link.GetType().GetProperty("ParentIndex");
            var value = parentIndexProp?.GetValue(link);
            if (value == null) return -1;
            
            // Handle both uint and int types safely
            if (value is uint uintVal)
                return (int)uintVal;
            if (value is int intVal)
                return intVal;
                
            return -1;
        }).Where(g => g.Key != -1).ToList();
        
        Console.WriteLine($"[UNIFIED EXTRACTOR] Global ParentIndex groups: {objectGroups.Count}");
    
    // Debug: Log some sample Unknown4 values from placements
    var samplePlacements = unifiedScene.GlobalPlacements.Take(10).ToList();
    Console.WriteLine($"[UNIFIED EXTRACTOR] Sample placement Unknown4 values:");
    foreach (var p in samplePlacements)
    {
        var unknown4Prop = p.GetType().GetProperty("Unknown4");
        var unknown4Value = unknown4Prop?.GetValue(p);
        Console.WriteLine($"[UNIFIED EXTRACTOR]   Placement Unknown4: {unknown4Value} (type: {unknown4Value?.GetType().Name})");
    }
    
    // Debug: Log some sample ParentIndex values from groups
    var sampleParentIndices = objectGroups.Take(10).Select(g => g.Key).ToList();
    Console.WriteLine($"[UNIFIED EXTRACTOR] Sample ParentIndex values: {string.Join(", ", sampleParentIndices)}");
        
        foreach (var group in objectGroups)
        {
            var parentIndex = group.Key;
            var links = group.ToList();
            
            // Step 3: Find matching MPRL placement (now with global search)
            var placement = unifiedScene.GlobalPlacements.FirstOrDefault(p =>
            {
                var unknown4Prop = p.GetType().GetProperty("Unknown4");
                if (unknown4Prop == null) return false;
                
                var unknown4Value = unknown4Prop.GetValue(p);
                if (unknown4Value == null) return false;
                
                // Handle both uint and int types safely
                if (unknown4Value is uint uintVal)
                    return (int)uintVal == parentIndex;
                if (unknown4Value is int intVal)
                    return intVal == parentIndex;
                    
                return false;
            });
            
            var objectVertices = new List<(float X, float Y, float Z)>();
            var objectFaces = new List<(int A, int B, int C)>();
            
            foreach (var link in links)
            {
                if (link.SurfaceRefIndex < unifiedScene.GlobalSurfaces.Count)
                {
                    var surface = unifiedScene.GlobalSurfaces[(int)link.SurfaceRefIndex];
                    ExtractSurfaceGeometry(surface, unifiedScene.GlobalIndices, unifiedScene.GlobalVertices, objectVertices, objectFaces);
                }
            }
            
            // Include MSCN vertices if available
            if (unifiedScene.GlobalMscnVertices?.Any() == true)
            {
                foreach (var mscnVertex in unifiedScene.GlobalMscnVertices)
                {
                    objectVertices.Add((mscnVertex.X, mscnVertex.Y, mscnVertex.Z));
                }
            }
            
            if (objectVertices.Any())
            {
                // Determine source tile for this object
                var sourceTile = DetermineSourceTile(links, unifiedScene);
                
                objects.Add(new DecodedPm4Object
                {
                    Name = $"global_parent_{parentIndex}",
                    ParentIndex = (uint)parentIndex,
                    Vertices = objectVertices,
                    Faces = objectFaces,
                    PlacementPosition = new System.Numerics.Vector3(0, 0, 0),
                    SourceTile = sourceTile,
                    LinkCount = links.Count
                });
            }
        }
        
        return objects;
    }
    
    /// <summary>
    /// Determine source tile for an object based on its links
    /// </summary>
    private static string DetermineSourceTile(List<dynamic> links, UnifiedMapScene unifiedScene)
    {
        // Use the tile with the most links for this object
        var tileCounts = new Dictionary<string, int>();
        
        foreach (var link in links)
        {
            foreach (var (tileName, (_, _, _, linkOffset, _, _)) in unifiedScene.TileOffsets)
            {
                var tileScene = unifiedScene.TileScenes[tileName];
                if (tileScene.Links?.Contains(link) == true)
                {
                    tileCounts[tileName] = tileCounts.GetValueOrDefault(tileName, 0) + 1;
                }
            }
        }
        
        return tileCounts.OrderByDescending(kvp => kvp.Value).FirstOrDefault().Key ?? "unknown";
    }
}
}
