using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Exports PM4 data as terrain points and full 3D meshes by combining data from multiple chunks.
    /// Special entries contain height/elevation data (Y value) while Position entries contain X,Z coordinates.
    /// Also incorporates vertex, index, and surface data from other chunks for complete 3D model reconstruction.
    /// </summary>
    public class PM4TerrainExporter
    {
        private readonly ILogger? _logger;

        public PM4TerrainExporter(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Represents a 3D terrain point extracted from PM4 data
        /// </summary>
        public class TerrainPoint
        {
            public int SpecialIndex { get; set; }
            public int PositionIndex { get; set; }
            public float X { get; set; } // X coordinate from position entry
            public float Y { get; set; } // Height value from Special entry (elevation)
            public float Z { get; set; } // Z coordinate from position entry (was originally Y)
            public uint SpecialValue { get; set; } // Raw Special value
            
            public override string ToString() => 
                $"({X:F2}, {Y:F2}, {Z:F2}) - Special: 0x{SpecialValue:X8}";
                
            public Vector2 GetXZPosition() => new Vector2(X, Z);
        }

        /// <summary>
        /// Exports terrain and mesh data from a PM4 file
        /// </summary>
        /// <param name="file">The PM4 file to export terrain data from</param>
        /// <param name="outputDir">The directory to save output files to</param>
        /// <returns>An asynchronous task</returns>
        public async Task ExportTerrainDataAsync(PM4File file, string outputDir)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));
            if (string.IsNullOrEmpty(outputDir))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDir));

            // Create output directory if it doesn't exist
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            string baseFileName = Path.GetFileNameWithoutExtension(file.FileName ?? "unknown");
            
            // Export terrain points from MPRL chunk as CSV
            if (file.PositionDataChunk != null && file.PositionDataChunk.Entries.Count > 0)
            {
                // Export terrain points as CSV
                await ExportTerrainPointsAsCsvAsync(file, baseFileName, outputDir);
                
                // Export terrain points as OBJ for 3D visualization
                await ExportTerrainPointsAsObjAsync(file, baseFileName, outputDir);
            }
            else
            {
                _logger?.LogWarning("No position data found in {FileName}", file.FileName ?? "unknown");
            }
            
            // Export the complete mesh using all available chunk data
            if (file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0)
            {
                await ExportCompleteMeshAsObjAsync(file, baseFileName, outputDir);
                
                // Add export of simplified collision hull
                await ExportCollisionHullAsync(file, baseFileName, outputDir);
            }
            else
            {
                _logger?.LogWarning("No vertex position data found in {FileName}", file.FileName ?? "unknown");
            }
        }

        /// <summary>
        /// Extracts terrain points from a PM4 file by combining Special entries with their following Position entries.
        /// </summary>
        /// <param name="file">The PM4 file to extract terrain points from</param>
        /// <returns>A list of terrain points</returns>
        private List<TerrainPoint> ExtractTerrainPoints(PM4File file)
        {
            var terrainPoints = new List<TerrainPoint>();
            var entries = file.PositionDataChunk!.Entries;
            
            // Find Special entries followed by Position entries
            for (int i = 0; i < entries.Count - 1; i++)
            {
                if (entries[i].IsSpecialEntry && !entries[i + 1].IsSpecialEntry)
                {
                    var specialEntry = entries[i];
                    var positionEntry = entries[i + 1];
                    
                    // Extract height from Special entry (interpreted as float)
                    float height = BitConverter.Int32BitsToSingle(specialEntry.SpecialValue);
                    
                    // Create terrain point using X,Y from Position entry and height from Special entry
                    var terrainPoint = new TerrainPoint
                    {
                        SpecialIndex = specialEntry.Index,
                        PositionIndex = positionEntry.Index,
                        X = positionEntry.CoordinateX,
                        Y = height, // Height from Special entry (should be Y coordinate)
                        Z = positionEntry.CoordinateY, // Z coordinate (was originally Y)
                        SpecialValue = (uint)specialEntry.SpecialValue
                    };
                    
                    terrainPoints.Add(terrainPoint);
                }
            }
            
            return terrainPoints;
        }

        /// <summary>
        /// Exports terrain points as CSV for analysis
        /// </summary>
        private async Task ExportTerrainPointsAsCsvAsync(PM4File file, string baseFileName, string outputDir)
        {
            var terrainPoints = ExtractTerrainPoints(file);
            
            if (terrainPoints.Count == 0)
            {
                _logger?.LogWarning("No terrain points found in {FileName}", file.FileName ?? "unknown");
                return;
            }
            
            string csvFileName = $"{baseFileName}_terrain_points.csv";
            string csvFilePath = Path.Combine(outputDir, csvFileName);
            
            try
            {
                using var writer = new StreamWriter(csvFilePath);
                
                // Write header
                await writer.WriteLineAsync("SpecialIndex,PositionIndex,X,Y,Z,SpecialValueHex,SpecialValueDec");
                
                // Write terrain points
                foreach (var point in terrainPoints)
                {
                    await writer.WriteLineAsync(
                        $"{point.SpecialIndex}," +
                        $"{point.PositionIndex}," +
                        $"{point.X.ToString(CultureInfo.InvariantCulture)}," +
                        $"{point.Y.ToString(CultureInfo.InvariantCulture)}," +
                        $"{point.Z.ToString(CultureInfo.InvariantCulture)}," +
                        $"0x{point.SpecialValue:X8}," +
                        $"{point.SpecialValue}");
                }
                
                _logger?.LogInformation("Exported {Count} terrain points to CSV: {FilePath}", 
                    terrainPoints.Count, csvFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting terrain points to CSV: {FilePath}", csvFilePath);
            }
        }
        
        /// <summary>
        /// Exports terrain points as OBJ for 3D visualization, creating a surface mesh
        /// by connecting points into triangles where possible
        /// </summary>
        private async Task ExportTerrainPointsAsObjAsync(PM4File file, string baseFileName, string outputDir)
        {
            var terrainPoints = ExtractTerrainPoints(file);
            
            if (terrainPoints.Count == 0)
            {
                _logger?.LogWarning("No terrain points found in {FileName}", file.FileName ?? "unknown");
                return;
            }
            
            string objFileName = $"{baseFileName}_terrain.obj";
            string objFilePath = Path.Combine(outputDir, objFileName);
            
            try
            {
                using var writer = new StreamWriter(objFilePath);
                
                // Write OBJ header with metadata comments
                await writer.WriteLineAsync($"# Terrain points from {file.FileName}");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer");
                await writer.WriteLineAsync($"# Total points: {terrainPoints.Count}");
                await writer.WriteLineAsync($"# Mesh generation: Delaunay triangulation");
                await writer.WriteLineAsync();
                
                // Write vertices (v x y z)
                foreach (var point in terrainPoints)
                {
                    // Swap X and Z coordinates to match expected output format
                    await writer.WriteLineAsync($"v {point.Z.ToString(CultureInfo.InvariantCulture)} " +
                                              $"{point.Y.ToString(CultureInfo.InvariantCulture)} " +
                                              $"{point.X.ToString(CultureInfo.InvariantCulture)}");
                }

                // Generate faces based on point proximity
                var faces = GenerateTriangleFaces(terrainPoints);
                
                // Write generated faces if any were created
                if (faces.Count > 0)
                {
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"# Generated faces: {faces.Count}");
                    
                    // Write faces (f v1 v2 v3)
                    // Note: OBJ indices are 1-based
                    foreach (var face in faces)
                    {
                        await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                    }
                    
                    _logger?.LogInformation("Created mesh with {VertexCount} vertices and {FaceCount} triangular faces",
                        terrainPoints.Count, faces.Count);
                }
                
                _logger?.LogInformation("Exported terrain mesh to OBJ: {FilePath}", objFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting terrain mesh to OBJ: {FilePath}", objFilePath);
            }
        }
        
        /// <summary>
        /// Exports a complete 3D mesh using all available chunk data
        /// </summary>
        private async Task ExportCompleteMeshAsObjAsync(PM4File file, string baseFileName, string outputDir)
        {
            string objFileName = $"{baseFileName}_complete.obj";
            string objFilePath = Path.Combine(outputDir, objFileName);
            string mtlFileName = $"{baseFileName}_materials.mtl";
            string mtlFilePath = Path.Combine(outputDir, mtlFileName);
            
            try
            {
                using var writer = new StreamWriter(objFilePath);
                
                // Write OBJ header with metadata comments
                await writer.WriteLineAsync($"# Complete 3D mesh from {file.FileName}");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer");
                await writer.WriteLineAsync($"# Using data from multiple PM4 chunks");
                
                // Add material library reference
                await writer.WriteLineAsync($"mtllib {mtlFileName}");
                await writer.WriteLineAsync();
                
                // Start a new object group
                await writer.WriteLineAsync($"o {baseFileName}_mesh");
                
                // Write vertices from MSPV chunk
                if (file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0)
                {
                    _logger?.LogDebug("Writing {Count} vertices from VertexPositionsChunk", 
                        file.VertexPositionsChunk.Vertices.Count);
                    
                    // Extract bounding box information for debugging
                    float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
                    float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
                    
                    foreach (var vertex in file.VertexPositionsChunk.Vertices)
                    {
                        // Track bounding box
                        minX = Math.Min(minX, vertex.X);
                        minY = Math.Min(minY, vertex.Y);
                        minZ = Math.Min(minZ, vertex.Z);
                        maxX = Math.Max(maxX, vertex.X);
                        maxY = Math.Max(maxY, vertex.Y);
                        maxZ = Math.Max(maxZ, vertex.Z);
                        
                        // Write the vertex, swapping X and Z coordinates to match WoW's coordinate system
                        await writer.WriteLineAsync($"v {vertex.Z.ToString(CultureInfo.InvariantCulture)} " +
                                                 $"{vertex.Y.ToString(CultureInfo.InvariantCulture)} " +
                                                 $"{vertex.X.ToString(CultureInfo.InvariantCulture)}");
                    }
                    
                    // Add bounding box information as a comment
                    await writer.WriteLineAsync($"# Bounding Box: Min({minX:F2}, {minY:F2}, {minZ:F2}) Max({maxX:F2}, {maxY:F2}, {maxZ:F2})");
                    await writer.WriteLineAsync($"# Dimensions: W({maxX - minX:F2}) H({maxY - minY:F2}) D({maxZ - minZ:F2})");
                }
                
                // Write normal vectors from MSCN chunk if available
                if (file.NormalCoordinatesChunk != null && file.NormalCoordinatesChunk.Normals.Count > 0)
                {
                    _logger?.LogDebug("Writing {Count} normal vectors from NormalCoordinatesChunk", 
                        file.NormalCoordinatesChunk.Normals.Count);
                    
                    foreach (var normal in file.NormalCoordinatesChunk.Normals)
                    {
                        // Write the normal, swapping X and Z coordinates to match our vertex transformation
                        await writer.WriteLineAsync($"vn {normal.Z.ToString(CultureInfo.InvariantCulture)} " +
                                                 $"{normal.Y.ToString(CultureInfo.InvariantCulture)} " +
                                                 $"{normal.X.ToString(CultureInfo.InvariantCulture)}");
                    }
                }
                
                // Write any surface info or materials
                await writer.WriteLineAsync("usemtl default");
                await writer.WriteLineAsync($"g {baseFileName}_primary");
                
                // Write faces (triangles) from MSPI chunk
                if (file.VertexIndicesChunk != null && file.VertexIndicesChunk.Indices.Count > 0)
                {
                    int triangleCount = file.VertexIndicesChunk.Indices.Count / 3;
                    _logger?.LogDebug("Writing {Count} triangles from VertexIndicesChunk", triangleCount);
                    
                    // In OBJ format, vertex indices start at 1, not 0
                    for (int i = 0; i < triangleCount; i++)
                    {
                        int baseIndex = i * 3;
                        if (baseIndex + 2 < file.VertexIndicesChunk.Indices.Count)
                        {
                            // OBJ uses 1-based indexing, so add 1 to all indices
                            uint v1 = file.VertexIndicesChunk.Indices[baseIndex] + 1;
                            uint v2 = file.VertexIndicesChunk.Indices[baseIndex + 1] + 1;
                            uint v3 = file.VertexIndicesChunk.Indices[baseIndex + 2] + 1;
                            
                            // Determine if we have normal indices to include
                            if (file.NormalCoordinatesChunk != null && file.NormalCoordinatesChunk.Normals.Count > 0)
                            {
                                // Include normal indices (assume same as vertex indices)
                                await writer.WriteLineAsync($"f {v1}//{v1} {v2}//{v2} {v3}//{v3}");
                            }
                            else
                            {
                                // Just vertex indices
                                await writer.WriteLineAsync($"f {v1} {v2} {v3}");
                            }
                        }
                    }
                }
                
                // Add additional data as separate object groups
                
                // Add VertexInfo data if available
                if (file.VertexInfoChunk != null && file.VertexInfoChunk.VertexInfos.Count > 0)
                {
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"o {baseFileName}_vertexInfo");
                    await writer.WriteLineAsync("usemtl vertexInfo");
                    
                    int startIndex = file.VertexPositionsChunk?.Vertices.Count ?? 0;
                    startIndex += 1; // 1-based indexing
                    
                    // Add point markers for vertex info
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"# Vertex Info: {file.VertexInfoChunk.VertexInfos.Count} entries");
                    
                    int pointIndex = 0;
                    foreach (var entry in file.VertexInfoChunk.VertexInfos)
                    {
                        // You can visualize the vertex info points here if needed
                        pointIndex++;
                    }
                }
                
                // Add Surface data as a separate object
                if (file.SurfaceDataChunk != null && file.SurfaceDataChunk.Surfaces.Count > 0)
                {
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"o {baseFileName}_surfaces");
                    await writer.WriteLineAsync("usemtl surfaces");
                    
                    // Add point markers for surface data
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"# Surface Data: {file.SurfaceDataChunk.Surfaces.Count} entries");
                    
                    int surfaceIndex = 0;
                    foreach (var entry in file.SurfaceDataChunk.Surfaces)
                    {
                        // You can visualize the surface data points here if needed
                        surfaceIndex++;
                    }
                }
                
                // Create a simple material file
                using (var mtlWriter = new StreamWriter(mtlFilePath))
                {
                    await mtlWriter.WriteLineAsync("# Material file for PM4 mesh");
                    await mtlWriter.WriteLineAsync("newmtl default");
                    await mtlWriter.WriteLineAsync("Ka 0.5 0.5 0.5");     // Ambient color
                    await mtlWriter.WriteLineAsync("Kd 0.7 0.7 0.7");     // Diffuse color
                    await mtlWriter.WriteLineAsync("Ks 0.2 0.2 0.2");     // Specular color
                    await mtlWriter.WriteLineAsync("Ns 10");              // Specular exponent
                    
                    await mtlWriter.WriteLineAsync("newmtl vertexInfo");
                    await mtlWriter.WriteLineAsync("Ka 0.9 0.1 0.1");     // Red
                    await mtlWriter.WriteLineAsync("Kd 1.0 0.2 0.2");
                    await mtlWriter.WriteLineAsync("Ks 0.8 0.0 0.0");
                    
                    await mtlWriter.WriteLineAsync("newmtl surfaces");
                    await mtlWriter.WriteLineAsync("Ka 0.1 0.1 0.9");     // Blue
                    await mtlWriter.WriteLineAsync("Kd 0.2 0.2 1.0");
                    await mtlWriter.WriteLineAsync("Ks 0.0 0.0 0.8");
                }
                
                _logger?.LogInformation("Exported complete 3D mesh to OBJ: {FilePath}", objFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting complete mesh to OBJ: {FilePath}", objFilePath);
            }
        }
        
        /// <summary>
        /// Exports a simplified collision hull by clustering nearby vertices and creating a more coherent mesh.
        /// This creates a low-poly representation that approximates what might be used for NPC pathing/collision.
        /// </summary>
        private async Task ExportCollisionHullAsync(PM4File file, string baseFileName, string outputDir)
        {
            string objFileName = $"{baseFileName}_collision_hull.obj";
            string objFilePath = Path.Combine(outputDir, objFileName);
            
            try
            {
                if (file.VertexPositionsChunk == null || file.VertexPositionsChunk.Vertices.Count == 0)
                {
                    _logger?.LogWarning("No vertex data found for collision hull export in {FileName}", file.FileName ?? "unknown");
                    return;
                }
                
                // Extract the basic vertices
                var vertices = file.VertexPositionsChunk.Vertices
                    .Select(v => new Vector3(v.Z, v.Y, v.X)) // Swap coordinates to match expected format
                    .ToList();
                
                // Step 1: Perform vertex clustering to reduce vertex count
                var clusteredVertices = ClusterVertices(vertices, distanceThreshold: 2.0f);
                _logger?.LogDebug("Reduced vertex count from {OriginalCount} to {ClusteredCount} through clustering", 
                    vertices.Count, clusteredVertices.Count);
                
                // Step 2: Generate a simplified mesh using convex hull approximation
                var simplifiedFaces = GenerateSimplifiedMesh(clusteredVertices);
                _logger?.LogDebug("Generated {FaceCount} simplified faces", simplifiedFaces.Count);
                
                // Write the OBJ file
                using var writer = new StreamWriter(objFilePath);
                
                // Write OBJ header with metadata comments
                await writer.WriteLineAsync($"# Simplified collision hull from {file.FileName}");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer");
                await writer.WriteLineAsync($"# Original vertices: {vertices.Count}, Clustered: {clusteredVertices.Count}");
                await writer.WriteLineAsync($"# Simplified faces: {simplifiedFaces.Count}");
                await writer.WriteLineAsync();
                
                // Write object name
                await writer.WriteLineAsync($"o {baseFileName}_collision_hull");
                
                // Write vertices (v x y z)
                foreach (var vertex in clusteredVertices)
                {
                    await writer.WriteLineAsync($"v {vertex.X.ToString(CultureInfo.InvariantCulture)} " +
                                              $"{vertex.Y.ToString(CultureInfo.InvariantCulture)} " +
                                              $"{vertex.Z.ToString(CultureInfo.InvariantCulture)}");
                }
                
                // Write faces (f v1 v2 v3)
                foreach (var face in simplifiedFaces)
                {
                    await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                }
                
                _logger?.LogInformation("Exported simplified collision hull to OBJ: {FilePath}", objFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting collision hull to OBJ: {FilePath}", objFilePath);
            }
        }
        
        /// <summary>
        /// Clusters vertices that are close to each other to reduce vertex count
        /// </summary>
        private List<Vector3> ClusterVertices(List<Vector3> vertices, float distanceThreshold)
        {
            if (vertices.Count == 0)
                return new List<Vector3>();
                
            // Create a spatial index for more efficient clustering
            var spatialGrid = new Dictionary<(int, int, int), List<Vector3>>();
            float cellSize = distanceThreshold * 2.0f; // Cell size larger than threshold for neighbor checks
            
            // Build spatial grid
            foreach (var vertex in vertices)
            {
                var cell = ((int)(vertex.X / cellSize), (int)(vertex.Y / cellSize), (int)(vertex.Z / cellSize));
                
                if (!spatialGrid.ContainsKey(cell))
                    spatialGrid[cell] = new List<Vector3>();
                    
                spatialGrid[cell].Add(vertex);
            }
            
            // Create clusters by checking nearby cells
            var processedVertices = new HashSet<Vector3>();
            var clusters = new List<List<Vector3>>();
            
            foreach (var vertex in vertices)
            {
                if (processedVertices.Contains(vertex))
                    continue;
                    
                var cluster = new List<Vector3>();
                var toProcess = new Queue<Vector3>();
                toProcess.Enqueue(vertex);
                processedVertices.Add(vertex);
                
                while (toProcess.Count > 0)
                {
                    var current = toProcess.Dequeue();
                    cluster.Add(current);
                    
                    // Get the current cell and neighboring cells
                    var cell = ((int)(current.X / cellSize), (int)(current.Y / cellSize), (int)(current.Z / cellSize));
                    var neighborCells = new List<(int, int, int)>();
                    
                    // Add current cell and all 26 neighboring cells (3x3x3 grid)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++)
                                neighborCells.Add((cell.Item1 + dx, cell.Item2 + dy, cell.Item3 + dz));
                    
                    // Check vertices in neighboring cells
                    foreach (var neighborCell in neighborCells)
                    {
                        if (!spatialGrid.ContainsKey(neighborCell))
                            continue;
                            
                        foreach (var neighbor in spatialGrid[neighborCell])
                        {
                            if (processedVertices.Contains(neighbor))
                                continue;
                                
                            // Check if this vertex is close enough to the current one
                            float distance = Vector3.Distance(current, neighbor);
                            if (distance <= distanceThreshold)
                            {
                                toProcess.Enqueue(neighbor);
                                processedVertices.Add(neighbor);
                            }
                        }
                    }
                }
                
                if (cluster.Count > 0)
                    clusters.Add(cluster);
            }
            
            // Calculate centroids of each cluster
            var centroids = new List<Vector3>();
            foreach (var cluster in clusters)
            {
                if (cluster.Count == 0)
                    continue;
                    
                // Compute average position (centroid)
                var centroid = Vector3.Zero;
                foreach (var vertex in cluster)
                    centroid += vertex;
                    
                centroid /= cluster.Count;
                centroids.Add(centroid);
            }
            
            return centroids;
        }
        
        /// <summary>
        /// Generates a simplified mesh from the clustered vertices using approximate convex hull
        /// </summary>
        private List<(int, int, int)> GenerateSimplifiedMesh(List<Vector3> vertices)
        {
            var faces = new List<(int, int, int)>();
            if (vertices.Count < 4) // Need at least 4 points for a tetrahedron
                return faces;
                
            // Use a spatial grid for local triangulation
            var spatialGrid = new Dictionary<(int, int, int), List<(int, Vector3)>>();
            float cellSize = 10.0f; // Cell size for spatial partitioning
            
            // Build spatial grid
            for (int i = 0; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                var cell = ((int)(vertex.X / cellSize), (int)(vertex.Y / cellSize), (int)(vertex.Z / cellSize));
                
                if (!spatialGrid.ContainsKey(cell))
                    spatialGrid[cell] = new List<(int, Vector3)>();
                    
                spatialGrid[cell].Add((i, vertex));
            }
            
            // Set of processed edges to avoid duplicate triangles
            var processedEdges = new HashSet<(int, int)>();
            
            // For each vertex, try to build tetrahedra with nearby vertices
            for (int i = 0; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                var cell = ((int)(vertex.X / cellSize), (int)(vertex.Y / cellSize), (int)(vertex.Z / cellSize));
                
                // Get neighboring cells (including current)
                var neighborCells = new List<(int, int, int)>();
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dz = -1; dz <= 1; dz++)
                            neighborCells.Add((cell.Item1 + dx, cell.Item2 + dy, cell.Item3 + dz));
                
                // Get nearby vertices from neighboring cells
                var nearbyVertices = new List<(int, Vector3)>();
                foreach (var neighborCell in neighborCells)
                {
                    if (spatialGrid.ContainsKey(neighborCell))
                        nearbyVertices.AddRange(spatialGrid[neighborCell]);
                }
                
                // Sort by distance to current vertex
                var sortedNearby = nearbyVertices
                    .Where(v => v.Item1 != i) // Exclude self
                    .OrderBy(v => Vector3.Distance(vertex, v.Item2))
                    .Take(15) // Take closest 15 vertices
                    .ToList();
                
                // Create triangles with nearest neighbors
                for (int j = 0; j < sortedNearby.Count - 1; j++)
                {
                    var (idx1, v1) = sortedNearby[j];
                    
                    // Skip if we've already processed edge (i,idx1)
                    var edge1 = (Math.Min(i, idx1), Math.Max(i, idx1));
                    if (processedEdges.Contains(edge1))
                        continue;
                    
                    processedEdges.Add(edge1);
                    
                    for (int k = j + 1; k < sortedNearby.Count; k++)
                    {
                        var (idx2, v2) = sortedNearby[k];
                        
                        // Skip if edge is already processed
                        var edge2 = (Math.Min(i, idx2), Math.Max(i, idx2));
                        var edge3 = (Math.Min(idx1, idx2), Math.Max(idx1, idx2));
                        
                        if (processedEdges.Contains(edge2) || processedEdges.Contains(edge3))
                            continue;
                            
                        // Determine if this is a good triangle
                        var v1v2Dist = Vector3.Distance(v1, v2);
                        var iV1Dist = Vector3.Distance(vertex, v1);
                        var iV2Dist = Vector3.Distance(vertex, v2);
                        
                        // Skip if any edge is too long or triangle is too thin
                        var maxEdge = Math.Max(v1v2Dist, Math.Max(iV1Dist, iV2Dist));
                        var minEdge = Math.Min(v1v2Dist, Math.Min(iV1Dist, iV2Dist));
                        
                        if (maxEdge > 50.0f || minEdge / maxEdge < 0.1f)
                            continue;
                            
                        // Add the triangle
                        faces.Add((i, idx1, idx2));
                        processedEdges.Add(edge2);
                        processedEdges.Add(edge3);
                    }
                }
            }
            
            return faces;
        }
        
        /// <summary>
        /// Generates triangular faces from the terrain points using a distance-based approach
        /// </summary>
        /// <param name="points">The terrain points to create faces from</param>
        /// <returns>A list of triangle indices (each tuple represents indices of 3 points)</returns>
        private List<(int, int, int)> GenerateTriangleFaces(List<TerrainPoint> points)
        {
            var faces = new List<(int, int, int)>();
            if (points.Count < 3)
                return faces;
            
            try
            {
                // Try to detect if points form a grid pattern
                bool isLikelyGrid = TryDetectGridPattern(points);
                
                if (isLikelyGrid)
                {
                    // Sort points by X then Z for grid triangulation
                    var sortedPoints = points
                        .OrderBy(p => Math.Round(p.X / 5) * 5)  // Group by approx 5-unit chunks on X
                        .ThenBy(p => Math.Round(p.Z / 5) * 5)   // Then by approx 5-unit chunks on Z
                        .ToList();
                    
                    // Create a map of sorted indices to original indices
                    var indexMap = new Dictionary<TerrainPoint, int>();
                    for (int i = 0; i < points.Count; i++)
                    {
                        indexMap[points[i]] = i;
                    }
                    
                    // Create triangles based on proximity in the sorted list
                    // This is a simplified approach that will work well for grid-like data
                    for (int i = 0; i < sortedPoints.Count - 2; i++)
                    {
                        var p1 = sortedPoints[i];
                        
                        // Look for nearby points to form triangles
                        for (int j = i + 1; j < sortedPoints.Count; j++)
                        {
                            var p2 = sortedPoints[j];
                            
                            // Skip if points are too far apart
                            if (Distance2D(p1, p2) > 15)
                                continue;
                            
                            for (int k = j + 1; k < sortedPoints.Count; k++)
                            {
                                var p3 = sortedPoints[k];
                                
                                // Check if these three points form a reasonable triangle
                                if (IsValidTriangle(p1, p2, p3, 15))
                                {
                                    // Get original indices
                                    int idx1 = indexMap[p1];
                                    int idx2 = indexMap[p2];
                                    int idx3 = indexMap[p3];
                                    
                                    faces.Add((idx1, idx2, idx3));
                                    
                                    // Break after finding a triangle for each set of points
                                    // to avoid too many overlapping triangles
                                    break;
                                }
                            }
                        }
                    }
                }
                else
                {
                    // For non-grid data, use a simple nearest-neighbor approach
                    // Create a spatial index for faster neighbor queries
                    var spatialIndex = new Dictionary<(int, int), List<(int, TerrainPoint)>>();
                    const int cellSize = 10; // Size of spatial grid cells
                    
                    // Build spatial index
                    for (int i = 0; i < points.Count; i++)
                    {
                        var point = points[i];
                        var cell = ((int)(point.X / cellSize), (int)(point.Z / cellSize));
                        
                        if (!spatialIndex.ContainsKey(cell))
                            spatialIndex[cell] = new List<(int, TerrainPoint)>();
                        
                        spatialIndex[cell].Add((i, point));
                    }
                    
                    // Generate triangles based on proximity
                    var processedEdges = new HashSet<(int, int)>();
                    
                    for (int i = 0; i < points.Count; i++)
                    {
                        var p1 = points[i];
                        var cell = ((int)(p1.X / cellSize), (int)(p1.Z / cellSize));
                        
                        // Get nearby cells for query
                        var nearbyCells = new List<(int, int)> {
                            cell,
                            (cell.Item1 - 1, cell.Item2),
                            (cell.Item1 + 1, cell.Item2),
                            (cell.Item1, cell.Item2 - 1),
                            (cell.Item1, cell.Item2 + 1),
                            (cell.Item1 - 1, cell.Item2 - 1),
                            (cell.Item1 + 1, cell.Item2 + 1),
                            (cell.Item1 - 1, cell.Item2 + 1),
                            (cell.Item1 + 1, cell.Item2 - 1)
                        };
                        
                        // Find all nearby points
                        var nearby = new List<(int, TerrainPoint)>();
                        foreach (var nearCell in nearbyCells)
                        {
                            if (spatialIndex.TryGetValue(nearCell, out var cellPoints))
                            {
                                nearby.AddRange(cellPoints);
                            }
                        }
                        
                        // Sort by distance to current point
                        nearby = nearby
                            .OrderBy(p => Distance2D(p1, p.Item2))
                            .ToList();
                        
                        // Try to form triangles with nearest neighbors
                        for (int j = 0; j < nearby.Count; j++)
                        {
                            var (idx2, p2) = nearby[j];
                            if (idx2 == i || Distance2D(p1, p2) > 20) continue;
                            
                            // Skip if we've already used this edge
                            var edge1 = (Math.Min(i, idx2), Math.Max(i, idx2));
                            if (processedEdges.Contains(edge1)) continue;
                            processedEdges.Add(edge1);
                            
                            for (int k = j + 1; k < nearby.Count; k++)
                            {
                                var (idx3, p3) = nearby[k];
                                if (idx3 == i || idx3 == idx2 || Distance2D(p1, p3) > 20 || Distance2D(p2, p3) > 20) 
                                    continue;
                                
                                // Check other edges
                                var edge2 = (Math.Min(i, idx3), Math.Max(i, idx3));
                                var edge3 = (Math.Min(idx2, idx3), Math.Max(idx2, idx3));
                                
                                if (processedEdges.Contains(edge2) || processedEdges.Contains(edge3)) 
                                    continue;
                                
                                // Check if triangle is valid
                                if (IsValidTriangle(p1, p2, p3, 20))
                                {
                                    faces.Add((i, idx2, idx3));
                                    processedEdges.Add(edge2);
                                    processedEdges.Add(edge3);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Error generating triangle faces, falling back to point cloud");
            }
            
            return faces;
        }
        
        /// <summary>
        /// Attempts to detect if points form a grid pattern
        /// </summary>
        private bool TryDetectGridPattern(List<TerrainPoint> points)
        {
            if (points.Count < 10)
                return false;
                
            // Sample some points and check if they form consistent spacing
            var samplePoints = points.Take(Math.Min(100, points.Count)).ToList();
            
            // Count unique X and Z values (rounded to nearest integer)
            var uniqueX = new HashSet<int>();
            var uniqueZ = new HashSet<int>();
            
            foreach (var point in samplePoints)
            {
                uniqueX.Add((int)Math.Round(point.X));
                uniqueZ.Add((int)Math.Round(point.Z));
            }
            
            // If the number of unique X and Z values is much smaller than the number of points,
            // it's likely that the points form a grid
            double uniqueRatio = (double)(uniqueX.Count * uniqueZ.Count) / samplePoints.Count;
            
            return uniqueRatio < 10.0; // Arbitrary threshold
        }
        
        /// <summary>
        /// Calculates the 2D distance between two terrain points (ignoring Y/height)
        /// </summary>
        private float Distance2D(TerrainPoint p1, TerrainPoint p2)
        {
            float dx = p1.X - p2.X;
            float dz = p1.Z - p2.Z;
            return (float)Math.Sqrt(dx * dx + dz * dz);
        }
        
        /// <summary>
        /// Checks if three points form a valid triangle (not too large, not too small, not too thin)
        /// </summary>
        private bool IsValidTriangle(TerrainPoint p1, TerrainPoint p2, TerrainPoint p3, float maxEdgeLength)
        {
            // Check if any edge is too long
            float d12 = Distance2D(p1, p2);
            float d23 = Distance2D(p2, p3);
            float d31 = Distance2D(p3, p1);
            
            if (d12 > maxEdgeLength || d23 > maxEdgeLength || d31 > maxEdgeLength)
                return false;
                
            // Check if triangle is too thin (avoid degenerate triangles)
            float minEdge = Math.Min(d12, Math.Min(d23, d31));
            float maxEdge = Math.Max(d12, Math.Max(d23, d31));
            
            if (minEdge < maxEdge * 0.1f) // Arbitrary threshold for thinness
                return false;
                
            // Calculate area using Heron's formula
            float s = (d12 + d23 + d31) / 2;
            float area = (float)Math.Sqrt(s * (s - d12) * (s - d23) * (s - d31));
            
            // Reject triangles with very small area
            return area > 0.1f;
        }
    }
} 