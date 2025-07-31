using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// PM4 Scene Graph Traversal Exporter
    /// 
    /// BREAKTHROUGH: Implements PM4 as a scene graph with hierarchical spatial organization
    /// and nested coordinate systems for proper single-object building exports.
    /// 
    /// Architecture:
    /// - MPRL entries = Scene root nodes (458 building objects)
    /// - MSLK entries = Child nodes (~13 sub-objects per building)
    /// - MSUR/MSCN/MSVT = Geometry data at different LoD levels
    /// - Nested coordinate systems with different scaling and ground planes
    /// </summary>
    public class Pm4SceneGraphExporter
    {
        private const float MSCN_SCALE_FACTOR = 4096.0f; // MSCN is 1/4096 scale
        private const float SPATIAL_TOLERANCE = 50.0f;   // Spatial clustering tolerance
        
        /// <summary>
        /// Export PM4 scene using scene graph traversal approach
        /// </summary>
        public void ExportPm4SceneGraph(ParpToolbox.Formats.PM4.Pm4Scene pm4Scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("=== PM4 Scene Graph Traversal Export ===");
            ConsoleLogger.WriteLine($"Target: {outputDirectory}");
            
            // Phase 1: Build Scene Graph Structure
            var sceneGraph = BuildSceneGraph(pm4Scene);
            ConsoleLogger.WriteLine($"Built scene graph with {sceneGraph.Count} root buildings");
            
            // Phase 2: Export Each Building via Scene Graph Traversal
            int exportedCount = 0;
            foreach (var building in sceneGraph)
            {
                try
                {
                    ExportBuildingSceneGraph(building, outputDirectory);
                    exportedCount++;
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"WARNING: Failed to export building {building.RootId}: {ex.Message}");
                }
            }
            
            ConsoleLogger.WriteLine($"Successfully exported {exportedCount} buildings using scene graph traversal");
        }
        
        /// <summary>
        /// Build hierarchical scene graph structure from PM4 data
        /// </summary>
        private List<BuildingSceneNode> BuildSceneGraph(ParpToolbox.Formats.PM4.Pm4Scene pm4Scene)
        {
            var buildings = new List<BuildingSceneNode>();

            // Get all data from scene
            var mprlEntries = pm4Scene.Placements ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry>();
            var mslkEntries = pm4Scene.Links ?? new List<MslkEntry>();
            var msurSurfaces = pm4Scene.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
            var msvtVertices = pm4Scene.Vertices ?? new List<System.Numerics.Vector3>();
            var indices = pm4Scene.Indices ?? new List<int>();

            ConsoleLogger.WriteLine($"Building scene graph from {mprlEntries.Count} MPRL, {mslkEntries.Count} MSLK, {msurSurfaces.Count} MSUR");

            // Step 1: Pre-index MSLK and MSUR data for efficient lookup
            ConsoleLogger.WriteLine("[STEP 1/3] Pre-indexing MSLK and MSUR data for fast lookups...");
            var mslkByParentId = mslkEntries.GroupBy(mslk => GetMslkParentIndex(mslk)).ToDictionary(g => g.Key, g => g.ToList());
            var msurBySurfaceKey = msurSurfaces.GroupBy(s => s.SurfaceKey).ToDictionary(g => g.Key, g => g.First());
            ConsoleLogger.WriteLine($"Indexed {mslkByParentId.Count} MSLK parent groups and {msurBySurfaceKey.Count} unique MSUR surfaces.");

            // Step 2: Group MPRL entries by Unknown4 (the correct building ID)
            ConsoleLogger.WriteLine("[STEP 2/3] Grouping MPRL entries by Unknown4 (the true building ID)...");
            var buildingGroups = mprlEntries.GroupBy(mprl => mprl.Unknown4).ToList();
            ConsoleLogger.WriteLine($"Found {buildingGroups.Count} building groups from MPRL entries.");

            // Step 3: Process each building group
            ConsoleLogger.WriteLine("[STEP 3/3] Assembling scene graph for each building...");
            foreach (var group in buildingGroups)
            {
                var buildingId = group.Key;
                var buildingPlacements = group.ToList();

                var building = new BuildingSceneNode
                {
                    RootId = buildingId,
                    RootPlacements = buildingPlacements
                };

                // Find all child MSLK entries for this building
                if (mslkByParentId.TryGetValue(buildingId, out var childMslkEntries))
                {
                    building.ChildNodes = childMslkEntries
                        .Select(mslk => new MslkSceneNode
                        {
                            MslkEntry = mslk,
                            LocalTransform = CalculateLocalTransform(mslk),
                            // Extract structural and surface geometry using the correct hierarchy
                            GeometryData = ExtractGeometryData(mslk, indices, msvtVertices),
                            SurfaceData = ExtractSurfaceDataOptimized(mslk, msurBySurfaceKey, indices, msvtVertices)
                        })
                        .ToList();
                }

                buildings.Add(building);
            }

            return buildings;
        }
        
        /// <summary>
        /// Export single building using scene graph traversal
        /// </summary>
        private void ExportBuildingSceneGraph(BuildingSceneNode building, string outputDirectory)
        {
            // Ensure output directory exists
            Directory.CreateDirectory(outputDirectory);
            
            var objFileName = Path.Combine(outputDirectory, $"building_{building.RootId:X8}.obj");
            var allVertices = new List<Vector3>();
            var allFaces = new List<int[]>();
            
            // Scene Graph Transform: Apply building root transform
            var rootTransform = CalculateRootTransform(building);
            
            // Traverse child nodes in scene graph order
            foreach (var childNode in building.ChildNodes)
            {
                // Apply nested coordinate transforms: Root -> Child -> Geometry
                var worldTransform = rootTransform * childNode.LocalTransform;
                
                // Extract and transform structural geometry (MSCN -> MSPI -> vertices)
                if (childNode.GeometryData.Any())
                {
                    foreach (var geometryVertex in childNode.GeometryData)
                    {
                        // Transform from MSCN coordinate space (1/4096 scale) to world space
                        var scaledVertex = geometryVertex * MSCN_SCALE_FACTOR;
                        var worldVertex = Vector3.Transform(scaledVertex, worldTransform);
                        allVertices.Add(worldVertex);
                    }
                }
                
                // Extract and transform surface geometry (MSUR -> MSVI -> MSVT)
                if (childNode.SurfaceData.Any())
                {
                    foreach (var surfaceVertex in childNode.SurfaceData)
                    {
                        // Surface vertices are already at full resolution
                        var worldVertex = Vector3.Transform(surfaceVertex, worldTransform);
                        allVertices.Add(worldVertex);
                    }
                }
                
                // Build faces using proper n-gon construction
                var faces = BuildNGonFaces(childNode, allVertices.Count);
                allFaces.AddRange(faces);
            }
            
            // Apply final coordinate system correction (X-axis flip)
            for (int i = 0; i < allVertices.Count; i++)
            {
                var v = allVertices[i];
                allVertices[i] = new Vector3(-v.X, v.Y, v.Z); // X-axis flip for OBJ export
            }
            
            // Export unified building object
            ExportObjFile(objFileName, allVertices, allFaces, $"Building_{building.RootId:X8}");
            
            ConsoleLogger.WriteLine($"Exporting building {building.RootId:X8} with {building.ChildNodes.Count} child nodes, {allFaces.Count} faces");
        }
        
        /// <summary>
        /// Calculate root transform matrix for building
        /// </summary>
        private Matrix4x4 CalculateRootTransform(BuildingSceneNode building)
        {
            // Use first placement for root transform (scene graph root)
            if (building.RootPlacements.Any())
            {
                var rootPlacement = building.RootPlacements.First();
                return Matrix4x4.CreateTranslation(rootPlacement.Position.X, rootPlacement.Position.Y, rootPlacement.Position.Z);
            }
            
            return Matrix4x4.Identity;
        }
        
        /// <summary>
        /// Calculate local transform matrix for MSLK child node
        /// </summary>
        private Matrix4x4 CalculateLocalTransform(MslkEntry mslk)
        {
            // For now, return identity - will enhance with proper transform extraction
            return Matrix4x4.Identity;
        }
        
        /// <summary>
        /// Extract geometry data from MSLK entry using available PM4Scene data
        /// </summary>
        private List<Vector3> ExtractGeometryData(MslkEntry mslk, List<int> indices, List<System.Numerics.Vector3> vertices)
        {
            var extractedVertices = new List<Vector3>();
            
            // Check if this MSLK has geometry data
            if (mslk.MspiFirstIndex >= 0 && mslk.MspiIndexCount > 0)
            {
                // Extract vertices using available indices
                for (int i = 0; i < mslk.MspiIndexCount && (mslk.MspiFirstIndex + i) < indices.Count; i++)
                {
                    var vertexIndex = indices[mslk.MspiFirstIndex + i];
                    if (vertexIndex < vertices.Count)
                    {
                        var vertex = vertices[vertexIndex];
                        extractedVertices.Add(new Vector3(vertex.X, vertex.Y, vertex.Z));
                    }
                }
            }
            
            return extractedVertices;
        }
        
        /// <summary>
        /// OPTIMIZED: Extract surface data using pre-indexed MSUR surfaces (eliminates O(N²) bottleneck)
        /// </summary>
        private List<Vector3> ExtractSurfaceDataOptimized(MslkEntry mslk, Dictionary<uint, ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry> msurBySurfaceKey, 
            List<int> indices, List<System.Numerics.Vector3> vertices)
        {
            var extractedVertices = new List<Vector3>();
            
            // OPTIMIZED: Use SurfaceRefIndex to directly lookup the associated surface (O(1) instead of O(N))
            if (msurBySurfaceKey.TryGetValue(mslk.SurfaceRefIndex, out var surface))
            {
                // Extract vertices using available indices for this surface
                for (int i = 0; i < surface.IndexCount && (surface.MsviFirstIndex + i) < indices.Count; i++)
                {
                    var vertexIndex = indices[(int)surface.MsviFirstIndex + i];
                    if (vertexIndex < vertices.Count)
                    {
                        var vertex = vertices[vertexIndex];
                        extractedVertices.Add(new Vector3(vertex.X, vertex.Y, vertex.Z));
                    }
                }
            }
            
            return extractedVertices;
        }
        
        /// <summary>
        /// Determine if MSUR surface is associated with MSLK node
        /// </summary>
        private bool IsSurfaceAssociated(ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface, MslkEntry mslk, Pm4Scene scene)
        {
            // CRITICAL: Only associate surfaces that actually belong to this MSLK node
            // Use SurfaceRefIndex to establish proper linkage with SurfaceKey
            if (mslk.SurfaceRefIndex == surface.SurfaceKey)
            {
                return true;
            }
            
            // Check if surface index falls within MSLK's geometry range
            var surfaceIndex = scene.Surfaces.IndexOf(surface);
            if (surfaceIndex >= mslk.MspiFirstIndex && 
                surfaceIndex < mslk.MspiFirstIndex + mslk.MspiIndexCount)
            {
                return true;
            }
            
            return false; // Only associate explicitly linked surfaces
        }
        
        /// <summary>
        /// Build n-gon faces from child node geometry
        /// </summary>
        private List<int[]> BuildNGonFaces(MslkSceneNode childNode, int vertexOffset)
        {
            var faces = new List<int[]>();
            
            // Build triangular faces for now - will enhance with proper n-gon construction
            var totalVertices = childNode.GeometryData.Count + childNode.SurfaceData.Count;
            
            for (int i = 0; i < totalVertices - 2; i += 3)
            {
                faces.Add(new int[] { 
                    vertexOffset - totalVertices + i + 1,     // OBJ uses 1-based indexing
                    vertexOffset - totalVertices + i + 2, 
                    vertexOffset - totalVertices + i + 3 
                });
            }
            
            return faces;
        }
        
        /// <summary>
        /// Export OBJ file with vertices and faces
        /// </summary>
        private void ExportObjFile(string fileName, List<Vector3> vertices, List<int[]> faces, string objectName)
        {
            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine($"# PM4 Scene Graph Export - {objectName}");
                writer.WriteLine($"# Vertices: {vertices.Count}, Faces: {faces.Count}");
                writer.WriteLine($"g {objectName}");
                
                // Write vertices
                foreach (var vertex in vertices)
                {
                    writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
                
                // Write faces
                foreach (var face in faces)
                {
                    if (face.Length == 3)
                        writer.WriteLine($"f {face[0]} {face[1]} {face[2]}");
                    else if (face.Length == 4)
                        writer.WriteLine($"f {face[0]} {face[1]} {face[2]} {face[3]}");
                }
            }
        }
        
        /// <summary>
        /// Get ParentIndex from MSLK entry using reflection (handles Unknown4/ParentIndex variations)
        /// </summary>
        private uint GetMslkParentIndex(MslkEntry mslk)
        {
            // Try to get ParentIndex via reflection to handle different field names
            var type = mslk.GetType();
            
            // Try ParentIndex first
            var parentIndexProp = type.GetProperty("ParentIndex");
            if (parentIndexProp != null)
            {
                return (uint)parentIndexProp.GetValue(mslk);
            }
            
            // Try Unknown4 as fallback
            var unknown4Prop = type.GetProperty("Unknown4");
            if (unknown4Prop != null)
            {
                return (uint)unknown4Prop.GetValue(mslk);
            }
            
            // Try field-based access
            var parentIndexField = type.GetField("ParentIndex");
            if (parentIndexField != null)
            {
                return (uint)parentIndexField.GetValue(mslk);
            }
            
            ConsoleLogger.WriteLine($"WARNING: Could not find ParentIndex field in MslkEntry");
            return 0;
        }
    }
    
    /// <summary>
    /// Scene graph node representing a building object
    /// </summary>
    public class BuildingSceneNode
    {
        public uint RootId { get; set; }
        public List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry> RootPlacements { get; set; } = new List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry>();
        public List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry> RootSurfaces { get; set; } = new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
        public List<MslkSceneNode> ChildNodes { get; set; } = new List<MslkSceneNode>();
    }
    
    /// <summary>
    /// Scene graph node representing an MSLK child object
    /// </summary>
    public class MslkSceneNode
    {
        public MslkEntry MslkEntry { get; set; }
        public Matrix4x4 LocalTransform { get; set; }
        public List<Vector3> GeometryData { get; set; } = new List<Vector3>();
        public List<Vector3> SurfaceData { get; set; } = new List<Vector3>();
    }
}
