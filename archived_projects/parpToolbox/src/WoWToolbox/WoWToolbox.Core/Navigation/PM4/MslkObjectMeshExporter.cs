using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Exports individual MSLK objects as separate OBJ files with complete mesh data.
    /// Each object represents a logical scene graph node with its associated geometry.
    /// </summary>
    public class MslkObjectMeshExporter
    {
        /// <summary>
        /// Mesh data for a single MSLK object
        /// </summary>
        public class ObjectMeshData
        {
            public List<Vector3> Vertices { get; set; } = new();
            public List<int> TriangleIndices { get; set; } = new();
            public List<Vector3> Normals { get; set; } = new();
            public List<string> VertexSources { get; set; } = new(); // Track where each vertex came from
            public List<string> Comments { get; set; } = new(); // Metadata for OBJ file
            
            // ✨ NEW: Enhanced WMO-style object data
            public List<MprRenderBatch> RenderBatches { get; set; } = new(); // MPRR sequences as render batches
            public List<MsurSurfaceData> AssociatedSurfaces { get; set; } = new(); // MSUR surfaces
            public ObjectBoundingBox BoundingBox { get; set; } = new();
            public Dictionary<string, object> WmoGroupMetadata { get; set; } = new(); // WMO-style metadata
            public List<Vector3> RenderMeshVertices { get; set; } = new(); // MSVT-based render mesh
            public List<int> RenderMeshIndices { get; set; } = new(); // MSVI-based render mesh
            
            public int VertexCount => Vertices.Count;
            public int TriangleCount => TriangleIndices.Count / 3;
            public bool HasGeometry => Vertices.Count > 0;
            public bool HasRenderMesh => RenderMeshVertices.Count > 0;
        }

        /// <summary>
        /// Represents an MPRR sequence as a WMO-style render batch
        /// </summary>
        public class MprRenderBatch
        {
            public int SequenceIndex { get; set; }
            public List<ushort> Values { get; set; } = new();
            public int Length => Values.Count;
            public bool IsTerminated => Values.LastOrDefault() == 0xFFFF;
            public string BatchType { get; set; } = "Unknown"; // Triangle, Strip, Fan, etc.
            public List<int> ResolvedIndices { get; set; } = new(); // If we can resolve what the values reference
        }

        /// <summary>
        /// MSUR surface data associated with an object
        /// </summary>
        public class MsurSurfaceData
        {
            public int SurfaceIndex { get; set; }
            public required MsurEntry Surface { get; set; }
            public List<Vector3> SurfaceVertices { get; set; } = new();
            public List<int> SurfaceIndices { get; set; } = new();
            public int TriangleCount => SurfaceIndices.Count / 3;
        }

        /// <summary>
        /// Bounding box data for an object
        /// </summary>
        public class ObjectBoundingBox
        {
            public Vector3 Min { get; set; } = new Vector3(float.MaxValue);
            public Vector3 Max { get; set; } = new Vector3(float.MinValue);
            public Vector3 Center => (Min + Max) * 0.5f;
            public Vector3 Size => Max - Min;
            public bool IsValid => Min.X != float.MaxValue;
        }

        /// <summary>
        /// Exports mesh data for a single MSLK object to an OBJ file with comprehensive PM4 analysis
        /// ✨ ENHANCED: Now supports render-mesh-only mode for clean visual geometry
        /// </summary>
        public void ExportObjectMesh(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment, 
                                   PM4File pm4File, string outputPath, string sourceFileName = "", bool renderMeshOnly = false)
        {
            var meshData = ExtractObjectMeshData(objectSegment, pm4File, renderMeshOnly);
            
            // Generate metadata including WMO-style analysis
            GenerateObjectMetadata(meshData, objectSegment, pm4File, sourceFileName, renderMeshOnly);
            
            // Generate normals for better rendering
            if (meshData.Vertices.Count > 0 && meshData.TriangleIndices.Count > 0)
            {
                meshData.Normals = Pm4CoordinateTransforms.ComputeVertexNormals(meshData.Vertices, meshData.TriangleIndices);
            }
            
            // Export to OBJ file
            ExportToObjFile(meshData, outputPath, objectSegment.RootIndex, renderMeshOnly);
            
            var modeDesc = renderMeshOnly ? "RENDER-ONLY" : "COMPREHENSIVE";
            Console.WriteLine($"   📄 Object {objectSegment.RootIndex} ({modeDesc}): {meshData.VertexCount} vertices, {meshData.TriangleCount} triangles → {Path.GetFileName(outputPath)}");
            
            if (meshData.HasRenderMesh)
            {
                Console.WriteLine($"      🎨 + {meshData.RenderMeshVertices.Count} render vertices, {meshData.RenderBatches.Count} MPRR batches");
            }
        }

        /// <summary>
        /// ✨ ENHANCED: Extracts complete mesh data for a single MSLK object from all its geometry nodes
        /// NOW INCLUDES: All PM4 chunks with proper object correlation + render-mesh-only mode
        /// </summary>
        private ObjectMeshData ExtractObjectMeshData(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment, 
                                                   PM4File pm4File, bool renderMeshOnly = false)
        {
            var meshData = new ObjectMeshData();
            
            if (pm4File.MSLK?.Entries == null)
            {
                meshData.Comments.Add("No MSLK data available");
                return meshData;
            }

            // ✨ FIXED: Check if this is an individual geometry export for strict mode
            bool isIndividualGeometry = objectSegment.SegmentationType == "individual_geometry";
            bool isStrictMode = isIndividualGeometry || renderMeshOnly;

            if (renderMeshOnly)
            {
                // 🎯 RENDER-MESH-ONLY MODE: Always use strict extraction for per-object geometry
                meshData.Comments.Add("=== RENDER MESH ONLY MODE (Strict Per-Object Geometry) ===");
                ExtractStrictIndividualGeometry(objectSegment, pm4File, meshData, renderMeshOnly: true);
            }
            else
            {
                // 🎯 COMPREHENSIVE MODE: Extract all PM4 data (original behavior)
                meshData.Comments.Add("=== COMPREHENSIVE MODE (All PM4 Data) ===");
                
                if (isIndividualGeometry)
                {
                    // For individual geometry, be strict even in comprehensive mode
                    ExtractStrictIndividualGeometry(objectSegment, pm4File, meshData, renderMeshOnly: false);
                }
                else
                {
                    // STEP 1: Extract structure geometry from MSLK → MSPI → MSPV chain
                    foreach (var nodeIndex in objectSegment.GeometryNodeIndices)
                    {
                        if (nodeIndex < 0 || nodeIndex >= pm4File.MSLK.Entries.Count)
                            continue;
                            
                        var mslkEntry = pm4File.MSLK.Entries[nodeIndex];
                        ExtractGeometryNodeMesh(mslkEntry, nodeIndex, pm4File, meshData);
                    }

                    // STEP 2: Find and extract ALL associated MSUR surfaces for render mesh data
                    var associatedSurfaces = FindAllAssociatedMsurSurfaces(objectSegment, pm4File);
                    if (associatedSurfaces.Any())
                    {
                        ExtractCompleteRenderMeshFromSurfaces(associatedSurfaces, pm4File, meshData);
                    }

                    // STEP 3: Generate actual geometry from MPRR sequences
                    GenerateGeometryFromMprrSequences(objectSegment, pm4File, meshData);

                    // STEP 4: Extract ALL relevant MSVT render mesh data
                    ExtractCompleteObjectRenderMesh(objectSegment, pm4File, meshData);

                    // STEP 5: Extract MSCN exterior points if relevant to this object
                    ExtractMscnExteriorPoints(objectSegment, pm4File, meshData);
                }
            }

            // Always compute bounding box and WMO-style metadata
            ComputeObjectBoundingBox(meshData);
            GenerateWmoStyleMetadata(objectSegment, pm4File, meshData);

            return meshData;
        }

        /// <summary>
        /// Extracts mesh data from a single geometry node (MSLK entry with MspiFirstIndex >= 0)
        /// </summary>
        private void ExtractGeometryNodeMesh(MSLKEntry mslkEntry, int nodeIndex, PM4File pm4File, ObjectMeshData meshData)
        {
            if (mslkEntry.MspiFirstIndex < 0 || pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null)
                return;

            try
            {
                // Extract MSPV vertices via MSPI indices
                var baseVertexCount = meshData.VertexCount;
                var nodeVertices = new List<Vector3>();
                
                for (int i = 0; i < mslkEntry.MspiIndexCount && (mslkEntry.MspiFirstIndex + i) < pm4File.MSPI.Indices.Count; i++)
                {
                    var mspiIndex = mslkEntry.MspiFirstIndex + i;
                    var mspvIndex = pm4File.MSPI.Indices[mspiIndex];
                    
                    if (mspvIndex < pm4File.MSPV.Vertices.Count)
                    {
                        var vertex = pm4File.MSPV.Vertices[(int)mspvIndex];
                        var transformedVertex = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                        
                        meshData.Vertices.Add(transformedVertex);
                        meshData.VertexSources.Add($"Node{nodeIndex}_MSPV{mspvIndex}");
                        nodeVertices.Add(transformedVertex);
                    }
                }
                
                // ✨ IMPROVED: Generate clean triangular faces for this geometry node
                if (nodeVertices.Count >= 3)
                {
                    // Strategy: Create triangular faces from vertex sequences
                    // Use triangle fan approach for polygonal surfaces
                    var localBaseIndex = baseVertexCount;
                    var generatedFaces = 0;
                    
                    if (nodeVertices.Count == 3)
                    {
                        // Simple triangle
                        if (IsValidTriangleByVertices(nodeVertices[0], nodeVertices[1], nodeVertices[2]))
                        {
                            meshData.TriangleIndices.AddRange(new[] { 
                                localBaseIndex, localBaseIndex + 1, localBaseIndex + 2 
                            });
                            generatedFaces++;
                        }
                    }
                    else if (nodeVertices.Count == 4)
                    {
                        // Quad -> two triangles
                        if (IsValidTriangleByVertices(nodeVertices[0], nodeVertices[1], nodeVertices[2]))
                        {
                            meshData.TriangleIndices.AddRange(new[] { 
                                localBaseIndex, localBaseIndex + 1, localBaseIndex + 2 
                            });
                            generatedFaces++;
                        }
                        if (IsValidTriangleByVertices(nodeVertices[0], nodeVertices[2], nodeVertices[3]))
                        {
                            meshData.TriangleIndices.AddRange(new[] { 
                                localBaseIndex, localBaseIndex + 2, localBaseIndex + 3 
                            });
                            generatedFaces++;
                        }
                    }
                    else if (nodeVertices.Count > 4)
                    {
                        // Triangle fan from first vertex
                        for (int j = 1; j < nodeVertices.Count - 1; j++)
                        {
                            if (IsValidTriangleByVertices(nodeVertices[0], nodeVertices[j], nodeVertices[j + 1]))
                            {
                                meshData.TriangleIndices.AddRange(new[] { 
                                    localBaseIndex, localBaseIndex + j, localBaseIndex + j + 1 
                                });
                                generatedFaces++;
                            }
                        }
                    }
                    
                    meshData.Comments.Add($"Node {nodeIndex}: {nodeVertices.Count} vertices → {generatedFaces} triangular faces");
                }
                else
                {
                    meshData.Comments.Add($"Node {nodeIndex}: {nodeVertices.Count} vertices (insufficient for faces)");
                }
            }
            catch (Exception ex)
            {
                meshData.Comments.Add($"Node {nodeIndex}: Error extracting geometry - {ex.Message}");
            }
        }

        /// <summary>
        /// ✨ NEW: Validates if three vertices form a valid triangle (non-degenerate, non-zero area)
        /// </summary>
        private bool IsValidTriangleByVertices(Vector3 v1, Vector3 v2, Vector3 v3)
        {
            // Check for degenerate triangles (same vertices)
            const float epsilon = 0.001f;
            if ((v1 - v2).Length() < epsilon || (v2 - v3).Length() < epsilon || (v3 - v1).Length() < epsilon)
                return false;
                
            // Check for zero-area triangles (colinear vertices)
            var edge1 = v2 - v1;
            var edge2 = v3 - v1;
            var cross = Vector3.Cross(edge1, edge2);
            return cross.Length() > epsilon;
        }

        /// <summary>
        /// Finds MSUR surfaces that might be associated with the geometry nodes in this object
        /// </summary>
        private List<int> FindAssociatedMsurSurfaces(List<int> geometryNodeIndices, PM4File pm4File)
        {
            var associatedSurfaces = new List<int>();
            
            if (pm4File.MSUR?.Entries == null || pm4File.MSLK?.Entries == null)
                return associatedSurfaces;

            // Strategy 1: Look for MSUR surfaces that reference MSVI indices used by our geometry nodes
            var usedMsviIndices = new HashSet<uint>();
            
            foreach (var nodeIndex in geometryNodeIndices)
            {
                if (nodeIndex < 0 || nodeIndex >= pm4File.MSLK.Entries.Count)
                    continue;
                    
                var mslkEntry = pm4File.MSLK.Entries[nodeIndex];
                
                // Collect MSVI indices used by this node via MSPI
                if (mslkEntry.MspiFirstIndex >= 0 && pm4File.MSPI?.Indices != null)
                {
                    for (int i = 0; i < mslkEntry.MspiIndexCount && (mslkEntry.MspiFirstIndex + i) < pm4File.MSPI.Indices.Count; i++)
                    {
                        usedMsviIndices.Add(pm4File.MSPI.Indices[mslkEntry.MspiFirstIndex + i]);
                    }
                }
            }

            // Find MSUR surfaces that use overlapping MSVI ranges
            for (int surfaceIndex = 0; surfaceIndex < pm4File.MSUR.Entries.Count; surfaceIndex++)
            {
                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                if (pm4File.MSVI?.Indices != null && msur.MsviFirstIndex < pm4File.MSVI.Indices.Count)
                {
                    // Check if this surface uses any of our MSVI indices
                    for (int i = 0; i < msur.IndexCount && (msur.MsviFirstIndex + i) < pm4File.MSVI.Indices.Count; i++)
                    {
                        var msviIndex = msur.MsviFirstIndex + (uint)i;
                        if (usedMsviIndices.Contains(msviIndex))
                        {
                            associatedSurfaces.Add(surfaceIndex);
                            break; // Found association, move to next surface
                        }
                    }
                }
            }

            return associatedSurfaces;
        }

        /// <summary>
        /// Extracts render mesh data from MSUR surfaces using production-ready face generation
        /// </summary>
        private void ExtractRenderMeshFromSurfaces(List<int> surfaceIndices, PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MSUR?.Entries == null || pm4File.MSVI?.Indices == null || pm4File.MSVT?.Vertices == null)
                return;

            var baseVertexCount = meshData.VertexCount;
            var surfaceVertices = new List<Vector3>();
            var addedVertexIndices = new Dictionary<uint, int>(); // MSVT index -> local mesh index
            
            // Use signature-based duplicate surface elimination (production-ready approach)
            var processedSurfaceSignatures = new HashSet<string>();

            foreach (var surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex < 0 || surfaceIndex >= pm4File.MSUR.Entries.Count)
                    continue;

                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                if (msur.IndexCount < 3 || msur.MsviFirstIndex >= pm4File.MSVI.Indices.Count)
                    continue;

                // Get surface vertex indices
                var surfaceIndices_local = new List<uint>();
                for (int j = 0; j < msur.IndexCount; j++)
                {
                    int msviIdx = (int)(msur.MsviFirstIndex + j);
                    if (msviIdx >= 0 && msviIdx < pm4File.MSVI.Indices.Count)
                    {
                        uint msvtIdx = pm4File.MSVI.Indices[msviIdx];
                        if (msvtIdx < pm4File.MSVT.Vertices.Count)
                        {
                            surfaceIndices_local.Add(msvtIdx);
                        }
                    }
                }

                if (surfaceIndices_local.Count < 3)
                    continue;

                // Create signature for duplicate detection
                var signature = string.Join(",", surfaceIndices_local.OrderBy(x => x));
                if (processedSurfaceSignatures.Contains(signature))
                    continue; // Skip duplicate surface

                processedSurfaceSignatures.Add(signature);

                // Add vertices to mesh
                var localSurfaceIndices = new List<int>();
                foreach (var msvtIdx in surfaceIndices_local)
                {
                    var vertex = pm4File.MSVT.Vertices[(int)msvtIdx];
                    // ✨ FIXED: Use unified spatial alignment transform for MSVT render vertices
                    var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                    
                    meshData.Vertices.Add(transformedVertex);
                    meshData.VertexSources.Add($"MSVT_Surface{surfaceIndex}");
                    localSurfaceIndices.Add(baseVertexCount + surfaceVertices.Count);
                    surfaceVertices.Add(transformedVertex);
                }

                // Generate triangle fan faces
                if (localSurfaceIndices.Count >= 3)
                {
                    for (int k = 1; k < localSurfaceIndices.Count - 1; k++)
                    {
                        int idx1 = localSurfaceIndices[0];     // Fan center
                        int idx2 = localSurfaceIndices[k];     // Current edge
                        int idx3 = localSurfaceIndices[k + 1]; // Next edge
                        
                        // Validate triangle
                        if (Pm4CoordinateTransforms.IsValidTriangle(idx1, idx2, idx3) &&
                            Pm4CoordinateTransforms.AreIndicesInBounds(idx1, idx2, idx3, meshData.Vertices.Count))
                        {
                            meshData.TriangleIndices.Add(idx1);
                            meshData.TriangleIndices.Add(idx2);
                            meshData.TriangleIndices.Add(idx3);
                        }
                    }
                }
            }

            if (surfaceVertices.Count > 0)
            {
                meshData.Comments.Add($"Render mesh: {surfaceVertices.Count} MSVT vertices from {surfaceIndices.Count} surfaces");
            }
        }

        /// <summary>
        /// Generates comprehensive metadata about the object for inclusion in OBJ comments
        /// ✨ ENHANCED: Now includes WMO-style analysis and all PM4 data correlations
        /// </summary>
        private void GenerateObjectMetadata(ObjectMeshData meshData, MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                          PM4File pm4File, string sourceFileName, bool renderMeshOnly)
        {
            meshData.Comments.Insert(0, $"MSLK WMO-Style Object Export - Root Node: {objectSegment.RootIndex}");
            meshData.Comments.Insert(1, $"Source File: {sourceFileName}");
            meshData.Comments.Insert(2, $"Generated: {DateTime.Now}");
            meshData.Comments.Insert(3, $"DISCOVERY: PM4 MSLK = Original WMO Group Hierarchy!");
            meshData.Comments.Insert(4, "");
            
            // ✨ WMO-Style Object Summary
            meshData.Comments.Add("=== WMO GROUP ANALYSIS SUMMARY ===");
            foreach (var kvp in meshData.WmoGroupMetadata)
            {
                meshData.Comments.Add($"{kvp.Key}: {kvp.Value}");
            }
            
            // Bounding box information
            if (meshData.BoundingBox.IsValid)
            {
                meshData.Comments.Add("");
                meshData.Comments.Add("=== BOUNDING BOX ===");
                meshData.Comments.Add($"Min: ({meshData.BoundingBox.Min.X:F2}, {meshData.BoundingBox.Min.Y:F2}, {meshData.BoundingBox.Min.Z:F2})");
                meshData.Comments.Add($"Max: ({meshData.BoundingBox.Max.X:F2}, {meshData.BoundingBox.Max.Y:F2}, {meshData.BoundingBox.Max.Z:F2})");
                meshData.Comments.Add($"Size: ({meshData.BoundingBox.Size.X:F2}, {meshData.BoundingBox.Size.Y:F2}, {meshData.BoundingBox.Size.Z:F2})");
            }
            
            // Object hierarchy information
            meshData.Comments.Add("");
            meshData.Comments.Add("=== OBJECT HIERARCHY ===");
            meshData.Comments.Add($"Root Node: {objectSegment.RootIndex}");
            meshData.Comments.Add($"Geometry Nodes: [{string.Join(", ", objectSegment.GeometryNodeIndices)}] ({objectSegment.GeometryNodeIndices.Count} nodes)");
            meshData.Comments.Add($"Anchor/Group Nodes: [{string.Join(", ", objectSegment.DoodadNodeIndices)}] ({objectSegment.DoodadNodeIndices.Count} nodes)");
            
            // ✨ Enhanced Mesh statistics
            meshData.Comments.Add("");
            meshData.Comments.Add("=== COMPREHENSIVE MESH STATISTICS ===");
            meshData.Comments.Add($"Structure Vertices (MSPV): {meshData.VertexCount}");
            meshData.Comments.Add($"Render Vertices (MSVT): {meshData.RenderMeshVertices.Count}");
            meshData.Comments.Add($"Total Vertices: {meshData.VertexCount + meshData.RenderMeshVertices.Count}");
            meshData.Comments.Add($"Structure Triangles: {meshData.TriangleCount}");
            meshData.Comments.Add($"Has Normals: {meshData.Normals.Count > 0}");
            meshData.Comments.Add($"MPRR Render Batches: {meshData.RenderBatches.Count}");
            meshData.Comments.Add($"MSUR Associated Surfaces: {meshData.AssociatedSurfaces.Count}");
            
            // Node details
            if (pm4File.MSLK?.Entries != null && objectSegment.GeometryNodeIndices.Any())
            {
                meshData.Comments.Add("");
                meshData.Comments.Add("=== GEOMETRY NODE DETAILS ===");
                
                foreach (var nodeIndex in objectSegment.GeometryNodeIndices)
                {
                    if (nodeIndex >= 0 && nodeIndex < pm4File.MSLK.Entries.Count)
                    {
                        var entry = pm4File.MSLK.Entries[nodeIndex];
                        meshData.Comments.Add($"Node {nodeIndex}: Flags=0x{entry.Unknown_0x00:X2}, MSPI={entry.MspiFirstIndex}+{entry.MspiIndexCount}, Parent={entry.Unknown_0x04}");
                    }
                }
            }
        }

        /// <summary>
        /// Exports mesh data to OBJ file with comprehensive formatting and metadata
        /// </summary>
        private void ExportToObjFile(ObjectMeshData meshData, string outputPath, int objectRootIndex, bool renderMeshOnly)
        {
            using var writer = new StreamWriter(outputPath);
            
            // Write header comments
            foreach (var comment in meshData.Comments)
            {
                writer.WriteLine($"# {comment}");
            }
            writer.WriteLine();
            
            if (!meshData.HasGeometry)
            {
                writer.WriteLine("# No geometry data available for this object");
                return;
            }
            
            // Write object group
            writer.WriteLine($"o Object_{objectRootIndex}");
            writer.WriteLine();
            
            // Write structure vertices (MSPV-based)
            writer.WriteLine("# Structure Vertices (MSPV via MSPI)");
            for (int i = 0; i < meshData.Vertices.Count; i++)
            {
                var vertex = meshData.Vertices[i];
                var source = i < meshData.VertexSources.Count ? meshData.VertexSources[i] : "Unknown";
                writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}  # {source}");
            }
            
            // ✨ Write render vertices (MSVT-based)
            if (meshData.RenderMeshVertices.Count > 0)
            {
                writer.WriteLine("# Render Vertices (MSVT via MSVI)");
                for (int i = 0; i < meshData.RenderMeshVertices.Count; i++)
                {
                    var vertex = meshData.RenderMeshVertices[i];
                    writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}  # MSVT_Render_{i}");
                }
            }
            writer.WriteLine();
            
            // Write normals if available
            if (meshData.Normals.Count > 0)
            {
                writer.WriteLine("# Vertex Normals");
                foreach (var normal in meshData.Normals)
                {
                    writer.WriteLine($"vn {normal.X:F6} {normal.Y:F6} {normal.Z:F6}");
                }
                writer.WriteLine();
            }
            
            // Write faces
            if (meshData.TriangleCount > 0)
            {
                writer.WriteLine("# Faces");
                bool hasNormals = meshData.Normals.Count == meshData.Vertices.Count;
                
                for (int i = 0; i < meshData.TriangleIndices.Count; i += 3)
                {
                    int idx1 = meshData.TriangleIndices[i] + 1;     // OBJ uses 1-based indexing
                    int idx2 = meshData.TriangleIndices[i + 1] + 1;
                    int idx3 = meshData.TriangleIndices[i + 2] + 1;
                    
                    if (hasNormals)
                    {
                        writer.WriteLine($"f {idx1}//{idx1} {idx2}//{idx2} {idx3}//{idx3}");
                    }
                    else
                    {
                        writer.WriteLine($"f {idx1} {idx2} {idx3}");
                    }
                }
            }
            else
            {
                writer.WriteLine("# No faces generated - geometry consists of points/lines only");
            }
        }

        /// <summary>
        /// Exports all objects with organized folder structure and multiple export strategies
        /// </summary>
        public void ExportAllObjects(List<MslkHierarchyAnalyzer.ObjectSegmentationResult> objectSegments,
                                   PM4File pm4File, string outputDirectory, string baseFileName, bool renderMeshOnly = false)
        {
            Directory.CreateDirectory(outputDirectory);
            
            Console.WriteLine($"🎯 Exporting {objectSegments.Count} objects to separate OBJ files...");
            
            int exportedCount = 0;
            int skippedCount = 0;
            
            foreach (var objectSegment in objectSegments)
            {
                try
                {
                    var objFileName = $"{baseFileName}.object_{objectSegment.RootIndex:D3}.obj";
                    var objPath = Path.Combine(outputDirectory, objFileName);
                    
                    ExportObjectMesh(objectSegment, pm4File, objPath, baseFileName, renderMeshOnly);
                    
                    // Quick validation
                    var fileInfo = new FileInfo(objPath);
                    if (fileInfo.Exists && fileInfo.Length > 100) // Basic check for non-empty file
                    {
                        exportedCount++;
                        Console.WriteLine($"  ✅ Object {objectSegment.RootIndex}: {objFileName}");
                    }
                    else
                    {
                        skippedCount++;
                        Console.WriteLine($"  ⚠️  Object {objectSegment.RootIndex}: {objFileName} (empty/minimal)");
                    }
                }
                catch (Exception ex)
                {
                    skippedCount++;
                    Console.WriteLine($"  ❌ Object {objectSegment.RootIndex}: Export failed - {ex.Message}");
                }
            }
            
            Console.WriteLine($"📊 Export Summary: {exportedCount} exported, {skippedCount} skipped/failed");
        }

        /// <summary>
        /// ✨ NEW: Exports all segmentation strategies with organized folder structure
        /// </summary>
        public void ExportAllStrategiesOrganized(MslkHierarchyAnalyzer.SegmentationStrategiesResult strategies,
                                                PM4File pm4File, string baseOutputDirectory, string baseFileName)
        {
            Console.WriteLine("🗂️  ORGANIZED MULTI-STRATEGY EXPORT");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            
            // Create organized folder structure
            var renderOnlyDir = Path.Combine(baseOutputDirectory, "render_mesh");
            var comprehensiveDir = Path.Combine(baseOutputDirectory, "comprehensive");
            
            Directory.CreateDirectory(renderOnlyDir);
            Directory.CreateDirectory(comprehensiveDir);
            
            Console.WriteLine($"📁 Output Structure:");
            Console.WriteLine($"   📁 render_mesh/     - Clean visual geometry only");
            Console.WriteLine($"   📁 comprehensive/   - All PM4 data (structure + render + collision)");
            Console.WriteLine();
            
            // Export RENDER-ONLY mode (clean visual geometry)
            Console.WriteLine("🎨 RENDER-MESH-ONLY EXPORTS (Clean Visual Geometry)");
            Console.WriteLine(new string('─', 60));
            
            ExportStrategyToFolders(strategies.ByRootHierarchy, pm4File, renderOnlyDir, baseFileName, "by_root", renderMeshOnly: true);
            ExportStrategyToFolders(strategies.ByIndividualGeometry, pm4File, renderOnlyDir, baseFileName, "by_geometry", renderMeshOnly: true);
            ExportStrategyToFolders(strategies.BySubHierarchies, pm4File, renderOnlyDir, baseFileName, "by_subhierarchy", renderMeshOnly: true);
            
            // Export COMPREHENSIVE mode (all PM4 data)
            Console.WriteLine("\n🔍 COMPREHENSIVE EXPORTS (All PM4 Data)");
            Console.WriteLine(new string('─', 60));
            
            ExportStrategyToFolders(strategies.ByRootHierarchy, pm4File, comprehensiveDir, baseFileName, "by_root", renderMeshOnly: false);
            ExportStrategyToFolders(strategies.ByIndividualGeometry, pm4File, comprehensiveDir, baseFileName, "by_geometry", renderMeshOnly: false);
            ExportStrategyToFolders(strategies.BySubHierarchies, pm4File, comprehensiveDir, baseFileName, "by_subhierarchy", renderMeshOnly: false);
            
            // Generate summary
            Console.WriteLine($"\n📊 EXPORT SUMMARY:");
            Console.WriteLine($"   🏗️  Root Hierarchies: {strategies.ByRootHierarchy.Count} objects");
            Console.WriteLine($"   🔧 Individual Geometry: {strategies.ByIndividualGeometry.Count} objects");
            Console.WriteLine($"   🧩 Sub-Hierarchies: {strategies.BySubHierarchies.Count} objects");
            Console.WriteLine($"   📄 Total Objects: {strategies.TotalObjects}");
            Console.WriteLine($"   📁 Folder Structure: {baseOutputDirectory}");
            Console.WriteLine();
            Console.WriteLine("🔍 ANALYSIS WORKFLOW:");
            Console.WriteLine("   1. Start with RENDER_MESH/BY_GEOMETRY for clean individual components");
            Console.WriteLine("   2. Check RENDER_MESH/BY_SUBHIERARCHY for logical groupings");
            Console.WriteLine("   3. Compare with COMPREHENSIVE versions to understand all data");
            Console.WriteLine("   4. Use BY_ROOT for traditional WMO-style full scene objects");
        }

        /// <summary>
        /// ✨ NEW: Exports a specific segmentation strategy to organized subfolders
        /// </summary>
        private void ExportStrategyToFolders(List<MslkHierarchyAnalyzer.ObjectSegmentationResult> objects,
                                           PM4File pm4File, string baseDirectory, string baseFileName, 
                                           string strategyName, bool renderMeshOnly)
        {
            if (!objects.Any())
            {
                Console.WriteLine($"   📁 {strategyName}/: No objects to export");
                return;
            }
            
            var strategyDir = Path.Combine(baseDirectory, strategyName);
            Directory.CreateDirectory(strategyDir);
            
            var modeDesc = renderMeshOnly ? "RENDER" : "COMPREHENSIVE";
            Console.WriteLine($"   📁 {strategyName}/: Exporting {objects.Count} objects ({modeDesc})");
            
            int exportedCount = 0;
            int skippedCount = 0;
            
            foreach (var obj in objects)
            {
                try
                {
                    // Generate strategy-specific filename
                    string objFileName = GenerateStrategyFileName(obj, baseFileName, strategyName);
                    var objPath = Path.Combine(strategyDir, objFileName);
                    
                    ExportObjectMesh(obj, pm4File, objPath, baseFileName, renderMeshOnly);
                    
                    // Validate export
                    var fileInfo = new FileInfo(objPath);
                    if (fileInfo.Exists && fileInfo.Length > 100)
                    {
                        exportedCount++;
                    }
                    else
                    {
                        skippedCount++;
                    }
                }
                catch (Exception ex)
                {
                    skippedCount++;
                    Console.WriteLine($"      ❌ Export failed for object {obj.RootIndex}: {ex.Message}");
                }
            }
            
            Console.WriteLine($"      ✅ {exportedCount} exported, ⚠️  {skippedCount} skipped");
        }

        /// <summary>
        /// ✨ NEW: Generates strategy-specific filenames for better organization
        /// </summary>
        private string GenerateStrategyFileName(MslkHierarchyAnalyzer.ObjectSegmentationResult obj, 
                                              string baseFileName, string strategyName)
        {
            return strategyName switch
            {
                "by_root" => $"{baseFileName}.root_{obj.RootIndex:D3}.obj",
                "by_geometry" => $"{baseFileName}.geom_{obj.RootIndex:D3}.obj",
                "by_subhierarchy" => obj.SegmentationType switch
                {
                    "sub_hierarchy" => $"{baseFileName}.subobj_{obj.RootIndex:D3}.obj",
                    "leaf_geometry" => $"{baseFileName}.leaf_{obj.RootIndex:D3}.obj",
                    _ => $"{baseFileName}.subhier_{obj.RootIndex:D3}.obj"
                },
                _ => $"{baseFileName}.object_{obj.RootIndex:D3}.obj"
            };
        }

        /// <summary>
        /// ✨ ENHANCED: Extracts complete render mesh data from MSUR surfaces with proper validation
        /// </summary>
        private void ExtractCompleteRenderMeshFromSurfaces(List<int> surfaceIndices, PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MSUR?.Entries == null || pm4File.MSVI?.Indices == null || pm4File.MSVT?.Vertices == null)
                return;

            var baseVertexCount = meshData.VertexCount;
            var surfaceVertices = new List<Vector3>();
            var addedVertexIndices = new Dictionary<uint, int>(); // MSVT index -> local mesh index
            
            // Use signature-based duplicate surface elimination (production-ready approach)
            var processedSurfaceSignatures = new HashSet<string>();

            foreach (var surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex < 0 || surfaceIndex >= pm4File.MSUR.Entries.Count)
                    continue;

                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                if (msur.IndexCount < 3 || msur.MsviFirstIndex >= pm4File.MSVI.Indices.Count)
                    continue;

                // Get surface vertex indices
                var surfaceIndices_local = new List<uint>();
                for (int j = 0; j < msur.IndexCount; j++)
                {
                    int msviIdx = (int)(msur.MsviFirstIndex + j);
                    if (msviIdx >= 0 && msviIdx < pm4File.MSVI.Indices.Count)
                    {
                        uint msvtIdx = pm4File.MSVI.Indices[msviIdx];
                        if (msvtIdx < pm4File.MSVT.Vertices.Count)
                        {
                            surfaceIndices_local.Add(msvtIdx);
                        }
                    }
                }

                if (surfaceIndices_local.Count < 3)
                    continue;

                // Create signature for duplicate detection
                var signature = string.Join(",", surfaceIndices_local.OrderBy(x => x));
                if (processedSurfaceSignatures.Contains(signature))
                    continue; // Skip duplicate surface

                processedSurfaceSignatures.Add(signature);

                // Add vertices to mesh
                var localSurfaceIndices = new List<int>();
                foreach (var msvtIdx in surfaceIndices_local)
                {
                    var vertex = pm4File.MSVT.Vertices[(int)msvtIdx];
                    // ✨ FIXED: Use unified spatial alignment transform for MSVT render vertices
                    var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                    
                    meshData.Vertices.Add(transformedVertex);
                    meshData.VertexSources.Add($"MSVT_Surface{surfaceIndex}");
                    localSurfaceIndices.Add(baseVertexCount + surfaceVertices.Count);
                    surfaceVertices.Add(transformedVertex);
                }

                // Generate triangle fan faces
                if (localSurfaceIndices.Count >= 3)
                {
                    for (int k = 1; k < localSurfaceIndices.Count - 1; k++)
                    {
                        int idx1 = localSurfaceIndices[0];     // Fan center
                        int idx2 = localSurfaceIndices[k];     // Current edge
                        int idx3 = localSurfaceIndices[k + 1]; // Next edge
                        
                        // Validate triangle
                        if (Pm4CoordinateTransforms.IsValidTriangle(idx1, idx2, idx3) &&
                            Pm4CoordinateTransforms.AreIndicesInBounds(idx1, idx2, idx3, meshData.Vertices.Count))
                        {
                            meshData.TriangleIndices.Add(idx1);
                            meshData.TriangleIndices.Add(idx2);
                            meshData.TriangleIndices.Add(idx3);
                        }
                    }
                }
            }

            if (surfaceVertices.Count > 0)
            {
                meshData.Comments.Add($"Render mesh: {surfaceVertices.Count} MSVT vertices from {surfaceIndices.Count} surfaces");
            }
        }

        /// <summary>
        /// ✨ ENHANCED: Attempts to resolve what MPRR sequence values reference
        /// </summary>
        private void TryResolveSequenceIndices(MprRenderBatch batch, PM4File pm4File, ObjectMeshData meshData)
        {
            var nonTerminatorValues = batch.Values.Where(v => v != 0xFFFF).ToList();
            if (!nonTerminatorValues.Any()) return;

            var maxValue = nonTerminatorValues.Max();
            var minValue = nonTerminatorValues.Min();

            // Test against various PM4 arrays
            if (pm4File.MSVT?.Vertices != null && maxValue < pm4File.MSVT.Vertices.Count)
            {
                batch.ResolvedIndices = nonTerminatorValues.Select(v => (int)v).ToList();
                meshData.Comments.Add($"  → MPRR[{batch.SequenceIndex}] likely indexes MSVT vertices (max:{maxValue} vs {pm4File.MSVT.Vertices.Count})");
            }
            else if (pm4File.MSVI?.Indices != null && maxValue < pm4File.MSVI.Indices.Count)
            {
                meshData.Comments.Add($"  → MPRR[{batch.SequenceIndex}] might index MSVI indices (max:{maxValue} vs {pm4File.MSVI.Indices.Count})");
            }
            else if (pm4File.MSPV?.Vertices != null && maxValue < pm4File.MSPV.Vertices.Count)
            {
                meshData.Comments.Add($"  → MPRR[{batch.SequenceIndex}] might index MSPV vertices (max:{maxValue} vs {pm4File.MSPV.Vertices.Count})");
            }
            else
            {
                meshData.Comments.Add($"  → MPRR[{batch.SequenceIndex}] indices unclear (max:{maxValue}, range:{minValue}-{maxValue})");
            }
        }

        /// <summary>
        /// ✨ ENHANCED: Finds ALL MSUR surfaces associated with this object (not just overlapping MSVI indices)
        /// </summary>
        private List<int> FindAllAssociatedMsurSurfaces(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment, PM4File pm4File)
        {
            var associatedSurfaces = new List<int>();
            
            if (pm4File.MSUR?.Entries == null || pm4File.MSLK?.Entries == null)
                return associatedSurfaces;

            // Strategy 1: Collect ALL MSVI indices used by this object's geometry nodes
            var usedMsviIndices = new HashSet<uint>();
            foreach (var nodeIndex in objectSegment.GeometryNodeIndices)
            {
                if (nodeIndex < 0 || nodeIndex >= pm4File.MSLK.Entries.Count)
                    continue;
                    
                var mslkEntry = pm4File.MSLK.Entries[nodeIndex];
                if (mslkEntry.MspiFirstIndex >= 0 && pm4File.MSPI?.Indices != null)
                {
                    for (int i = 0; i < mslkEntry.MspiIndexCount && (mslkEntry.MspiFirstIndex + i) < pm4File.MSPI.Indices.Count; i++)
                    {
                        usedMsviIndices.Add(pm4File.MSPI.Indices[mslkEntry.MspiFirstIndex + i]);
                    }
                }
            }

            // Strategy 2: Also include surfaces within spatial proximity of object nodes
            // For now, be more inclusive - we can filter later
            for (int surfaceIndex = 0; surfaceIndex < pm4File.MSUR.Entries.Count; surfaceIndex++)
            {
                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                if (pm4File.MSVI?.Indices != null && msur.MsviFirstIndex < pm4File.MSVI.Indices.Count)
                {
                    // Check if this surface uses any of our MSVI indices OR is in reasonable proximity
                    bool isAssociated = false;
                    
                    for (int i = 0; i < msur.IndexCount && (msur.MsviFirstIndex + i) < pm4File.MSVI.Indices.Count; i++)
                    {
                        var msviIndex = msur.MsviFirstIndex + (uint)i;
                        if (usedMsviIndices.Contains(msviIndex))
                        {
                            isAssociated = true;
                            break;
                        }
                    }
                    
                    // Strategy 3: Include surfaces based on spatial correlation with object root
                    if (!isAssociated && objectSegment.RootIndex >= 0)
                    {
                        // For now, include all surfaces - we can refine this later
                        // In a more sophisticated approach, we'd use spatial distance analysis
                        var rootDistance = Math.Abs(surfaceIndex - objectSegment.RootIndex);
                        if (rootDistance <= 50) // Arbitrary proximity threshold
                        {
                            isAssociated = true;
                        }
                    }
                    
                    if (isAssociated)
                    {
                        associatedSurfaces.Add(surfaceIndex);
                    }
                }
            }

            return associatedSurfaces;
        }

        /// <summary>
        /// ✨ ENHANCED: Generate actual mesh geometry from MPRR sequences (not just analysis)
        /// </summary>
        private void GenerateGeometryFromMprrSequences(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                                      PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MPRR?.Sequences == null || pm4File.MPRR.Sequences.Count == 0)
                return;

            meshData.Comments.Add($"=== MPRR GEOMETRY GENERATION ===");
            var baseVertexCount = meshData.VertexCount;
            int generatedFaces = 0;

            // Strategy: Try to interpret MPRR sequences as actual geometry data
            for (int seqIndex = 0; seqIndex < pm4File.MPRR.Sequences.Count; seqIndex++)
            {
                var sequence = pm4File.MPRR.Sequences[seqIndex];
                if (sequence.Count == 0) continue;

                var batch = new MprRenderBatch
                {
                    SequenceIndex = seqIndex,
                    Values = sequence.ToList()
                };

                // Filter out terminator values
                var validIndices = sequence.Where(v => v != 0xFFFF).ToArray();
                if (validIndices.Length < 3) continue;

                // Try to resolve against MSVT vertices
                if (pm4File.MSVT?.Vertices != null && validIndices.All(v => v < pm4File.MSVT.Vertices.Count))
                {
                    // Generate vertices from MPRR → MSVT mapping
                    var sequenceVertices = new List<int>();
                    foreach (var index in validIndices)
                    {
                        var msvtVertex = pm4File.MSVT.Vertices[index];
                        var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(msvtVertex);
                        
                        meshData.Vertices.Add(transformedVertex);
                        meshData.VertexSources.Add($"MPRR_Seq{seqIndex}_MSVT{index}");
                        sequenceVertices.Add(baseVertexCount + meshData.VertexCount - baseVertexCount - 1);
                    }

                    // Generate faces based on sequence type
                    if (validIndices.Length == 3)
                    {
                        // Triangle
                        meshData.TriangleIndices.AddRange(sequenceVertices);
                        generatedFaces++;
                        batch.BatchType = "Triangle";
                    }
                    else if (validIndices.Length == 4)
                    {
                        // Quad -> two triangles
                        meshData.TriangleIndices.AddRange(new[] { sequenceVertices[0], sequenceVertices[1], sequenceVertices[2] });
                        meshData.TriangleIndices.AddRange(new[] { sequenceVertices[0], sequenceVertices[2], sequenceVertices[3] });
                        generatedFaces += 2;
                        batch.BatchType = "Quad";
                    }
                    else if (validIndices.Length > 4)
                    {
                        // Triangle fan
                        for (int i = 1; i < sequenceVertices.Count - 1; i++)
                        {
                            meshData.TriangleIndices.AddRange(new[] { sequenceVertices[0], sequenceVertices[i], sequenceVertices[i + 1] });
                            generatedFaces++;
                        }
                        batch.BatchType = "TriangleFan";
                    }
                }

                meshData.RenderBatches.Add(batch);
                
                if (meshData.RenderBatches.Count <= 10) // Limit detailed logging
                {
                    meshData.Comments.Add($"MPRR[{seqIndex}]: {batch.BatchType} ({validIndices.Length} indices)");
                }
            }

            meshData.Comments.Add($"Generated {generatedFaces} faces from {meshData.RenderBatches.Count} MPRR sequences");
        }

        /// <summary>
        /// ✨ ENHANCED: Extract ALL relevant MSVT render mesh data for this object
        /// </summary>
        private void ExtractCompleteObjectRenderMesh(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                                   PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null)
                return;

            meshData.Comments.Add($"=== COMPLETE MSVT RENDER MESH EXTRACTION ===");
            
            // Strategy: Extract render mesh vertices that correspond to this object's surfaces AND spatial region
            var extractedVertices = new HashSet<uint>(); // Track to avoid duplicates
            
            foreach (var surfaceData in meshData.AssociatedSurfaces)
            {
                var surface = surfaceData.Surface;
                
                if (surface.IndexCount > 0 && surface.MsviFirstIndex < pm4File.MSVI.Indices.Count)
                {
                    for (int i = 0; i < surface.IndexCount && (surface.MsviFirstIndex + i) < pm4File.MSVI.Indices.Count; i++)
                    {
                        var msviIndex = (int)(surface.MsviFirstIndex + i);
                        var msvtIndex = pm4File.MSVI.Indices[msviIndex];
                        
                        if (msvtIndex < pm4File.MSVT.Vertices.Count && !extractedVertices.Contains(msvtIndex))
                        {
                            var vertex = pm4File.MSVT.Vertices[(int)msvtIndex];
                            var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                            
                            meshData.RenderMeshVertices.Add(transformedVertex);
                            extractedVertices.Add(msvtIndex);
                        }
                    }
                }
            }

            // Also extract additional MSVT vertices based on spatial proximity to this object
            if (objectSegment.GeometryNodeIndices.Any() && pm4File.MSPV?.Vertices != null)
            {
                // Get representative position for this object from its first geometry node
                var firstGeomNode = objectSegment.GeometryNodeIndices.First();
                if (firstGeomNode >= 0 && firstGeomNode < pm4File.MSLK.Entries.Count)
                {
                    var mslkEntry = pm4File.MSLK.Entries[firstGeomNode];
                    if (mslkEntry.MspiFirstIndex >= 0 && pm4File.MSPI?.Indices != null && 
                        mslkEntry.MspiFirstIndex < pm4File.MSPI.Indices.Count)
                    {
                        var firstMspvIndex = pm4File.MSPI.Indices[mslkEntry.MspiFirstIndex];
                        if (firstMspvIndex < pm4File.MSPV.Vertices.Count)
                        {
                            var objectCenter = Pm4CoordinateTransforms.FromMspvVertex(pm4File.MSPV.Vertices[(int)firstMspvIndex]);
                            
                            // Extract nearby MSVT vertices (simplified proximity approach)
                            const float proximityThreshold = 100.0f; // Adjust based on your coordinate system
                            for (uint i = 0; i < Math.Min(pm4File.MSVT.Vertices.Count, 1000); i++) // Limit for performance
                            {
                                if (!extractedVertices.Contains(i))
                                {
                                    var vertex = pm4File.MSVT.Vertices[(int)i];
                                    var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                                    var distance = (transformedVertex - objectCenter).Length();
                                    
                                    if (distance <= proximityThreshold)
                                    {
                                        meshData.RenderMeshVertices.Add(transformedVertex);
                                        extractedVertices.Add(i);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            meshData.Comments.Add($"Extracted {meshData.RenderMeshVertices.Count} total MSVT render vertices");
        }

        /// <summary>
        /// ✨ NEW: Extract MSCN exterior points if relevant to this object
        /// </summary>
        private void ExtractMscnExteriorPoints(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                              PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MSCN?.ExteriorVertices == null || pm4File.MSCN.ExteriorVertices.Count == 0)
                return;

            meshData.Comments.Add($"=== MSCN EXTERIOR POINTS EXTRACTION ===");
            var baseVertexCount = meshData.VertexCount;

            // Strategy: Include MSCN points that are spatially relevant to this object
            // For now, include a reasonable subset - we can refine the correlation later
            var pointsToInclude = Math.Min(pm4File.MSCN.ExteriorVertices.Count, 200); // Limit for performance
            
            for (int i = 0; i < pointsToInclude; i++)
            {
                var mscnVertex = pm4File.MSCN.ExteriorVertices[i];
                var transformedPoint = Pm4CoordinateTransforms.FromMscnVertex(mscnVertex);
                
                meshData.Vertices.Add(transformedPoint);
                meshData.VertexSources.Add($"MSCN_Exterior_{i}");
            }

            meshData.Comments.Add($"Added {pointsToInclude} MSCN exterior points");
        }

        /// <summary>
        /// ✨ NEW: Computes bounding box for all geometry in this object
        /// </summary>
        private void ComputeObjectBoundingBox(ObjectMeshData meshData)
        {
            var allVertices = meshData.Vertices.Concat(meshData.RenderMeshVertices);
            
            foreach (var vertex in allVertices)
            {
                if (vertex.X < meshData.BoundingBox.Min.X) meshData.BoundingBox.Min = meshData.BoundingBox.Min with { X = vertex.X };
                if (vertex.Y < meshData.BoundingBox.Min.Y) meshData.BoundingBox.Min = meshData.BoundingBox.Min with { Y = vertex.Y };
                if (vertex.Z < meshData.BoundingBox.Min.Z) meshData.BoundingBox.Min = meshData.BoundingBox.Min with { Z = vertex.Z };
                
                if (vertex.X > meshData.BoundingBox.Max.X) meshData.BoundingBox.Max = meshData.BoundingBox.Max with { X = vertex.X };
                if (vertex.Y > meshData.BoundingBox.Max.Y) meshData.BoundingBox.Max = meshData.BoundingBox.Max with { Y = vertex.Y };
                if (vertex.Z > meshData.BoundingBox.Max.Z) meshData.BoundingBox.Max = meshData.BoundingBox.Max with { Z = vertex.Z };
            }
        }

        /// <summary>
        /// ✨ NEW: Generates WMO-style metadata for this object
        /// </summary>
        private void GenerateWmoStyleMetadata(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                            PM4File pm4File, ObjectMeshData meshData)
        {
            meshData.WmoGroupMetadata["RootNodeIndex"] = objectSegment.RootIndex;
            meshData.WmoGroupMetadata["GeometryNodeCount"] = objectSegment.GeometryNodeIndices.Count;
            meshData.WmoGroupMetadata["DoodadNodeCount"] = objectSegment.DoodadNodeIndices.Count;
            meshData.WmoGroupMetadata["TotalVertices"] = meshData.VertexCount + meshData.RenderMeshVertices.Count;
            meshData.WmoGroupMetadata["StructureVertices"] = meshData.VertexCount;
            meshData.WmoGroupMetadata["RenderVertices"] = meshData.RenderMeshVertices.Count;
            meshData.WmoGroupMetadata["RenderBatchCount"] = meshData.RenderBatches.Count;
            meshData.WmoGroupMetadata["AssociatedSurfaces"] = meshData.AssociatedSurfaces.Count;
            
            if (meshData.BoundingBox.IsValid)
            {
                meshData.WmoGroupMetadata["BoundingBoxCenter"] = meshData.BoundingBox.Center;
                meshData.WmoGroupMetadata["BoundingBoxSize"] = meshData.BoundingBox.Size;
            }

            // Analyze node types in this object
            var nodeTypeAnalysis = new Dictionary<string, int>();
            foreach (var nodeIndex in objectSegment.GeometryNodeIndices.Concat(objectSegment.DoodadNodeIndices))
            {
                if (nodeIndex >= 0 && nodeIndex < pm4File.MSLK?.Entries?.Count)
                {
                    var entry = pm4File.MSLK.Entries[nodeIndex];
                    var nodeType = entry.MspiFirstIndex >= 0 ? "GeometryNode" : "DoodadNode";
                    nodeTypeAnalysis[nodeType] = nodeTypeAnalysis.GetValueOrDefault(nodeType, 0) + 1;
                }
            }
            meshData.WmoGroupMetadata["NodeTypeAnalysis"] = nodeTypeAnalysis;
        }

        /// <summary>
        /// ✨ NEW: Extracts ONLY clean MSVT render mesh data for visual geometry
        /// </summary>
        private void ExtractCleanRenderMeshOnly(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                              PM4File pm4File, ObjectMeshData meshData)
        {
            if (pm4File.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null || pm4File.MSUR?.Entries == null)
                return;

            meshData.Comments.Add($"=== CLEAN MSVT RENDER MESH EXTRACTION ===");
            
            // Strategy: Extract only MSVT vertices that are part of valid MSUR surfaces
            var validMsurSurfaces = new HashSet<int>();
            var extractedVertices = new HashSet<uint>();

            // Find MSUR surfaces with reasonable triangle counts (likely visual geometry)
            for (int surfaceIndex = 0; surfaceIndex < pm4File.MSUR.Entries.Count; surfaceIndex++)
            {
                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                // Filter for surfaces that look like visual geometry (reasonable triangle counts)
                if (msur.IndexCount >= 3 && msur.IndexCount <= 1000 && // Reasonable polygon count
                    msur.MsviFirstIndex < pm4File.MSVI.Indices.Count)
                {
                    validMsurSurfaces.Add(surfaceIndex);
                }
            }

            meshData.Comments.Add($"Found {validMsurSurfaces.Count} valid MSUR surfaces for visual geometry");

            // Extract vertices from valid surfaces only
            foreach (var surfaceIndex in validMsurSurfaces)
            {
                var msur = pm4File.MSUR.Entries[surfaceIndex];
                
                var surfaceVertices = new List<int>();
                for (int i = 0; i < msur.IndexCount && (msur.MsviFirstIndex + i) < pm4File.MSVI.Indices.Count; i++)
                {
                    var msviIndex = (int)(msur.MsviFirstIndex + i);
                    var msvtIndex = pm4File.MSVI.Indices[msviIndex];
                    
                    if (msvtIndex < pm4File.MSVT.Vertices.Count && !extractedVertices.Contains(msvtIndex))
                    {
                        var vertex = pm4File.MSVT.Vertices[(int)msvtIndex];
                        var transformedVertex = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                        
                        meshData.Vertices.Add(transformedVertex);
                        meshData.VertexSources.Add($"MSVT_Clean_Surface{surfaceIndex}");
                        surfaceVertices.Add(meshData.VertexCount - 1);
                        extractedVertices.Add(msvtIndex);
                    }
                    else if (extractedVertices.Contains(msvtIndex))
                    {
                        // Find existing vertex index
                        var existingIndex = meshData.VertexSources.FindIndex(s => s.Contains($"MSVT") && s.Contains(msvtIndex.ToString()));
                        if (existingIndex >= 0)
                        {
                            surfaceVertices.Add(existingIndex);
                        }
                    }
                }

                // Generate clean triangle faces for this surface
                if (surfaceVertices.Count >= 3)
                {
                    // Use triangle fan approach for clean face generation
                    for (int j = 1; j < surfaceVertices.Count - 1; j++)
                    {
                        if (Pm4CoordinateTransforms.IsValidTriangle(surfaceVertices[0], surfaceVertices[j], surfaceVertices[j + 1]) &&
                            Pm4CoordinateTransforms.AreIndicesInBounds(surfaceVertices[0], surfaceVertices[j], surfaceVertices[j + 1], meshData.VertexCount))
                        {
                            meshData.TriangleIndices.Add(surfaceVertices[0]);
                            meshData.TriangleIndices.Add(surfaceVertices[j]);
                            meshData.TriangleIndices.Add(surfaceVertices[j + 1]);
                        }
                    }
                }
            }

            meshData.Comments.Add($"Extracted {meshData.VertexCount} clean render vertices from {validMsurSurfaces.Count} surfaces");
            meshData.Comments.Add($"Generated {meshData.TriangleCount} clean triangles for visual rendering");
        }

        /// <summary>
        /// ✨ NEW: Extracts ONLY the geometry data directly associated with a specific geometry node
        /// This ensures individual geometry exports are clean and separate
        /// </summary>
        private void ExtractStrictIndividualGeometry(MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
                                                   PM4File pm4File, ObjectMeshData meshData, bool renderMeshOnly)
        {
            if (objectSegment.GeometryNodeIndices.Count != 1)
            {
                meshData.Comments.Add("Error: Individual geometry export expects exactly one geometry node");
                return;
            }

            var nodeIndex = objectSegment.GeometryNodeIndices[0];
            if (nodeIndex < 0 || nodeIndex >= pm4File.MSLK.Entries.Count)
            {
                meshData.Comments.Add($"Error: Invalid node index {nodeIndex}");
                return;
            }

            var mslkEntry = pm4File.MSLK.Entries[nodeIndex];
            meshData.Comments.Add($"=== STRICT INDIVIDUAL GEOMETRY NODE {nodeIndex} ===");
            meshData.Comments.Add($"Mode: {(renderMeshOnly ? "RENDER-ONLY" : "COMPREHENSIVE")}");

            if (renderMeshOnly)
            {
                // RENDER-ONLY: Extract clean MSVT data via direct MSUR correlation
                ExtractDirectRenderMeshForNode(mslkEntry, nodeIndex, pm4File, meshData);
            }
            else
            {
                // COMPREHENSIVE: Extract both structure and render data, but only for this node
                ExtractGeometryNodeMesh(mslkEntry, nodeIndex, pm4File, meshData);
                ExtractDirectRenderMeshForNode(mslkEntry, nodeIndex, pm4File, meshData);
            }

            meshData.Comments.Add($"Individual geometry node {nodeIndex}: {meshData.VertexCount} vertices, {meshData.TriangleCount} triangles");
        }

        /// <summary>
        /// ✨ NEW: Extracts render mesh data directly associated with a specific geometry node
        /// Uses only MSUR surfaces that directly reference this node's MSPI indices
        /// </summary>
        private void ExtractDirectRenderMeshForNode(MSLKEntry mslkEntry, int nodeIndex, PM4File pm4File, ObjectMeshData meshData)
        {
            if (mslkEntry.MspiFirstIndex < 0 || pm4File.MSPI?.Indices == null || 
                pm4File.MSUR?.Entries == null || pm4File.MSVI?.Indices == null || pm4File.MSVT?.Vertices == null)
                return;

            // Get the exact MSPI indices used by this geometry node
            var nodeMspiIndices = new HashSet<uint>();
            for (int i = 0; i < mslkEntry.MspiIndexCount && (mslkEntry.MspiFirstIndex + i) < pm4File.MSPI.Indices.Count; i++)
            {
                nodeMspiIndices.Add(pm4File.MSPI.Indices[mslkEntry.MspiFirstIndex + i]);
            }

            if (!nodeMspiIndices.Any())
            {
                meshData.Comments.Add($"Node {nodeIndex}: No MSPI indices found");
                return;
            }

            meshData.Comments.Add($"Node {nodeIndex}: Using MSPI indices [{string.Join(", ", nodeMspiIndices.Take(5))}{(nodeMspiIndices.Count > 5 ? "..." : "")}]");

            // Find MSUR surfaces that DIRECTLY use these MSPI indices
            var directSurfaces = new List<int>();
            for (int surfaceIndex = 0; surfaceIndex < pm4File.MSUR.Entries.Count; surfaceIndex++)
            {
                var msur = pm4File.MSUR.Entries[surfaceIndex];
                if (msur.IndexCount == 0 || msur.MsviFirstIndex >= pm4File.MSVI.Indices.Count)
                    continue;

                // Check if ANY of this surface's MSVI indices match our node's MSPI indices
                bool isDirectMatch = false;
                for (int i = 0; i < msur.IndexCount && (msur.MsviFirstIndex + i) < pm4File.MSVI.Indices.Count; i++)
                {
                    var msviIndex = msur.MsviFirstIndex + (uint)i;
                    if (nodeMspiIndices.Contains(msviIndex))
                    {
                        isDirectMatch = true;
                        break;
                    }
                }

                if (isDirectMatch)
                {
                    directSurfaces.Add(surfaceIndex);
                }
            }

            meshData.Comments.Add($"Node {nodeIndex}: Found {directSurfaces.Count} directly associated surfaces");

            if (directSurfaces.Any())
            {
                // Extract render mesh from only these directly associated surfaces
                ExtractRenderMeshFromSurfaces(directSurfaces, pm4File, meshData);
            }
            else
            {
                meshData.Comments.Add($"Node {nodeIndex}: No direct MSUR surface matches found");
            }
        }
    }
} 