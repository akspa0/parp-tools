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
    /// Exports PM4 data as collision/pathing meshes and full 3D models by combining data from multiple chunks.
    /// Special entries contain height/elevation data (Y value) while Position entries contain X,Z coordinates.
    /// Also incorporates vertex, index, and surface data from other chunks for complete 3D model reconstruction.
    /// </summary>
    public class PM4MeshExporter
    {
        private readonly ILogger? _logger;

        public PM4MeshExporter(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Represents a 3D point extracted from PM4 data
        /// </summary>
        public class MeshPoint
        {
            public int SpecialIndex { get; set; }
            public int PositionIndex { get; set; }
            public float X { get; set; } // X coordinate from position entry
            public float Y { get; set; } // Height value from Special entry (elevation)
            public float Z { get; set; } // Z value as a flag/category (0.0 = standard, 2.0 = special)
            public int ZFlag { get; set; } // Z as an integer for easier categorization
            public uint SpecialValue { get; set; } // Raw Special value
            
            // Additional analysis properties
            public int ZAsBits => BitConverter.SingleToInt32Bits(Z);
            public uint ZAsUint => BitConverter.SingleToUInt32Bits(Z);
            public bool IsZVerySmall => Math.Abs(Z) < 0.001f;
            
            // Z value classes for easier categorization
            public int ZClass 
            { 
                get
                {
                    if (Z == 0) return 0;      // Standard waypoint
                    if (Z == 2.0f) return 1;   // Special waypoint
                    return 3;                   // Unknown type
                }
            }
            
            // Get a human-readable waypoint type
            public string WaypointType => ZClass switch
            {
                0 => "Standard",
                1 => "Special",
                _ => "Unknown"
            };
            
            public override string ToString() => 
                $"({X:F2}, {Y:F2}, {Z:F6}) - {WaypointType} Waypoint - Special: 0x{SpecialValue:X8}";
                
            public Vector2 GetXZPosition() => new Vector2(X, Z);
        }

        /// <summary>
        /// Exports mesh data from a PM4 file
        /// </summary>
        /// <param name="file">The PM4 file to export data from</param>
        /// <param name="outputDir">The directory to save output files to</param>
        /// <returns>An asynchronous task</returns>
        public async Task ExportMeshDataAsync(PM4File file, string outputDir)
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
            
            // Export position data points from MPRL chunk as CSV for analysis
            if (file.PositionDataChunk != null && file.PositionDataChunk.Entries.Count > 0)
            {
                // Export position data points as CSV
                await ExportPositionDataAsCsvAsync(file, baseFileName, outputDir);
            }
            
            // Export a unified mesh that combines all data types
            if (file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0)
            {
                // Export the integrated mesh with pathing/collision data
                await ExportIntegratedMeshAsync(file, baseFileName, outputDir);
                
                // Add export of simplified collision hull
                await ExportCollisionHullAsync(file, baseFileName, outputDir);
            }
            else
            {
                _logger?.LogWarning("No vertex position data found in {FileName}", file.FileName ?? "unknown");
            }
        }

        /// <summary>
        /// Extracts waypoint mesh points from a PM4 file by combining Special entries with their following Position entries.
        /// These points represent navigation waypoints used for NPCs, not actual terrain geometry.
        /// </summary>
        /// <param name="file">The PM4 file to extract position data points from</param>
        /// <returns>A list of mesh points representing pathing waypoints</returns>
        private List<MeshPoint> ExtractPositionData(PM4File file)
        {
            var meshPoints = new List<MeshPoint>();
            
            if (file.PositionDataChunk == null || file.PositionDataChunk.Entries.Count == 0)
            {
                return meshPoints;
            }
            
            var entries = file.PositionDataChunk.Entries;
            
            // Extract bounding box information for coordinate analysis
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            
            // Track various Z values to understand their meaning
            var zValues = new List<float>();
            var uniqueZValues = new HashSet<float>();
            
            // Find Special entries followed by Position entries
            for (int i = 0; i < entries.Count - 1; i++)
            {
                if (entries[i].IsSpecialEntry && !entries[i + 1].IsSpecialEntry)
                {
                    var specialEntry = entries[i];
                    var positionEntry = entries[i + 1];
                    
                    // Extract height from Special entry (interpreted as float)
                    float height = BitConverter.Int32BitsToSingle(specialEntry.SpecialValue);
                    
                    // Store original Z value for analysis
                    float zValue = positionEntry.CoordinateZ;
                    zValues.Add(zValue);
                    uniqueZValues.Add(zValue);
                    
                    // Create a waypoint mesh point - these are navigation pathing points
                    // The Z value appears to be a flag/category rather than a coordinate
                    // Z = 0.0 seems to be standard waypoints
                    // Z = 2.0 might represent special waypoints (e.g., object interaction points)
                    var meshPoint = new MeshPoint
                    {
                        SpecialIndex = specialEntry.Index,
                        PositionIndex = positionEntry.Index,
                        X = positionEntry.CoordinateX,     // Original X coordinate 
                        Y = height,                        // Height from special entry
                        Z = positionEntry.CoordinateZ,     // Z value as a flag/category (0.0 or 2.0)
                        ZFlag = (int)positionEntry.CoordinateZ,  // For easier categorization 
                        SpecialValue = (uint)specialEntry.SpecialValue
                    };
                    
                    // Track bounding box
                    minX = Math.Min(minX, meshPoint.X);
                    minY = Math.Min(minY, meshPoint.Y);
                    minZ = Math.Min(minZ, meshPoint.Z);
                    maxX = Math.Max(maxX, meshPoint.X);
                    maxY = Math.Max(maxY, meshPoint.Y);
                    maxZ = Math.Max(maxZ, meshPoint.Z);
                    
                    meshPoints.Add(meshPoint);
                }
            }
            
            // Analyze Z values to understand their purpose
            if (meshPoints.Count > 0)
            {
                _logger?.LogDebug("Position data bounding box: Min({MinX:F2}, {MinY:F2}, {MinZ:F6}) Max({MaxX:F2}, {MaxY:F2}, {MaxZ:F6})",
                    minX, minY, minZ, maxX, maxY, maxZ);
                
                // Log unique Z values for analysis
                _logger?.LogDebug("Found {Count} unique Z values out of {Total} points", uniqueZValues.Count, zValues.Count);
                
                if (uniqueZValues.Count <= 10)
                {
                    _logger?.LogDebug("Small number of unique Z values suggests they are flags or categories: {Values}",
                        string.Join(", ", uniqueZValues.OrderBy(z => z).Select(z => z.ToString("F6"))));
                    
                    // Count occurrences of each Z value
                    var zValueCounts = zValues.GroupBy(z => z)
                                              .Select(g => new { Value = g.Key, Count = g.Count() })
                                              .OrderBy(x => x.Value)
                                              .ToList();
                    
                    foreach (var zvc in zValueCounts)
                    {
                        string category = zvc.Value < 0.1f ? "Standard waypoint" : "Special waypoint";
                        _logger?.LogDebug("Z value {Value:F6} ({Category}): {Count} occurrences ({Percent:F2}%)",
                            zvc.Value, category, zvc.Count, (100.0 * zvc.Count / zValues.Count));
                    }
                }
                
                // Check for potential interpretations of Z values
                if (uniqueZValues.Contains(0.0f) && uniqueZValues.Contains(2.0f))
                {
                    _logger?.LogDebug("Z values appear to be flags/categories: 0.0 = standard waypoint, 2.0 = special waypoint");
                }
                
                // Map float values to integers to see if they might represent indices
                var asIntegers = zValues.Take(20).Select(z => BitConverter.SingleToInt32Bits(z)).ToList();
                _logger?.LogDebug("First few Z values as integer bits: {Integers}", 
                    string.Join(", ", asIntegers.Select(i => $"0x{i:X8}")));
            }
            
            return meshPoints;
        }

        /// <summary>
        /// Exports position data points as CSV for analysis. 
        /// These points represent navigation waypoints used for NPC pathing, not actual terrain geometry.
        /// </summary>
        private async Task ExportPositionDataAsCsvAsync(PM4File file, string baseFileName, string outputDir)
        {
            var meshPoints = ExtractPositionData(file);
            
            if (meshPoints.Count == 0)
            {
                _logger?.LogWarning("No position data points found in {FileName}", file.FileName ?? "unknown");
                return;
            }
            
            string csvFileName = $"{baseFileName}_position_data.csv";
            string csvFilePath = Path.Combine(outputDir, csvFileName);
            
            try
            {
                using var writer = new StreamWriter(csvFilePath);
                
                // Write header with detailed coordinate information
                await writer.WriteLineAsync("# Coordinate system transformation: (-Z, Y, X)");
                await writer.WriteLineAsync("# Position data represents NPC navigation waypoints");
                await writer.WriteLineAsync("# These points define where characters can move within the scene");
                await writer.WriteLineAsync("# Z value seems to be a flag/category: 0.0 = standard waypoint, 2.0 = special waypoint");
                await writer.WriteLineAsync("# Original coordinates are stored and then transformed for OBJ output");
                await writer.WriteLineAsync();
                await writer.WriteLineAsync("SpecialIndex,PositionIndex,X,Y,Z,WaypointType,TransformedX,TransformedY,TransformedZ,SpecialValueHex,SpecialValueDec");
                
                // Write mesh points
                foreach (var point in meshPoints)
                {
                    // Calculate the transformed coordinates as they will appear in the OBJ file
                    // Use F6 format to ensure we don't get scientific notation for small values
                    string transformedX = (-point.Z).ToString("F6", CultureInfo.InvariantCulture);
                    string transformedY = point.Y.ToString("F6", CultureInfo.InvariantCulture);
                    string transformedZ = point.X.ToString("F6", CultureInfo.InvariantCulture);
                    
                    // Interpret the Z value as a waypoint type
                    string waypointType = point.Z < 0.1f ? "Standard" : "Special";
                    
                    await writer.WriteLineAsync(
                        $"{point.SpecialIndex}," +
                        $"{point.PositionIndex}," +
                        $"{point.X.ToString("F6", CultureInfo.InvariantCulture)}," +
                        $"{point.Y.ToString("F6", CultureInfo.InvariantCulture)}," +
                        $"{point.Z.ToString("F6", CultureInfo.InvariantCulture)}," +
                        $"{waypointType}," +
                        $"{transformedX}," +
                        $"{transformedY}," +
                        $"{transformedZ}," +
                        $"0x{point.SpecialValue:X8}," +
                        $"{point.SpecialValue}");
                }
                
                _logger?.LogInformation("Exported {Count} navigation waypoints to CSV: {FilePath}", 
                    meshPoints.Count, csvFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting navigation waypoints to CSV: {FilePath}", csvFilePath);
            }
        }
        
        /// <summary>
        /// Exports position data points as OBJ for 3D visualization, creating a surface mesh
        /// by connecting points into triangles where possible
        /// </summary>
        private async Task ExportPositionDataAsObjAsync(PM4File file, string baseFileName, string outputDir)
        {
            var meshPoints = ExtractPositionData(file);
            
            if (meshPoints.Count == 0)
            {
                _logger?.LogWarning("No position data points found in {FileName}", file.FileName ?? "unknown");
                return;
            }
            
            string objFileName = $"{baseFileName}_position_mesh.obj";
            string objFilePath = Path.Combine(outputDir, objFileName);
            
            try
            {
                using var writer = new StreamWriter(objFilePath);
                
                // Write OBJ header with detailed metadata comments
                await writer.WriteLineAsync($"# Integrated mesh data from {file.FileName}");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer");
                await writer.WriteLineAsync($"# Coordinate system transformation: (-Z, Y, X)");
                await writer.WriteLineAsync($"# Position data points are integrated with hull vertices to form a complete mesh");
                await writer.WriteLineAsync($"# Total position points: {meshPoints.Count}");
                
                // Extract hull data if available
                List<Vector3> hullVertices = new List<Vector3>();
                float minHullX = float.MaxValue, minHullY = float.MaxValue, minHullZ = float.MaxValue;
                float maxHullX = float.MinValue, maxHullY = float.MinValue, maxHullZ = float.MinValue;
                
                if (file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0)
                {
                    // Extract hull vertices and track bounding box
                    foreach (var vertex in file.VertexPositionsChunk.Vertices)
                    {
                        var v = new Vector3(vertex.X, vertex.Y, vertex.Z);
                        hullVertices.Add(v);
                        
                        minHullX = Math.Min(minHullX, v.X);
                        minHullY = Math.Min(minHullY, v.Y);
                        minHullZ = Math.Min(minHullZ, v.Z);
                        maxHullX = Math.Max(maxHullX, v.X);
                        maxHullY = Math.Max(maxHullY, v.Y);
                        maxHullZ = Math.Max(maxHullZ, v.Z);
                    }
                    
                    await writer.WriteLineAsync($"# Total hull vertices: {hullVertices.Count}");
                    await writer.WriteLineAsync($"# Hull bounding box: Min({minHullX:F2}, {minHullY:F2}, {minHullZ:F2}) Max({maxHullX:F2}, {maxHullY:F2}, {maxHullZ:F2})");
                    
                    // Extract position data bounding box for comparison
                    float minPosX = meshPoints.Min(p => p.X);
                    float minPosY = meshPoints.Min(p => p.Y);
                    float minPosZ = meshPoints.Min(p => p.Z);
                    float maxPosX = meshPoints.Max(p => p.X);
                    float maxPosY = meshPoints.Max(p => p.Y);
                    float maxPosZ = meshPoints.Max(p => p.Z);
                    
                    await writer.WriteLineAsync($"# Position data bounding box: Min({minPosX:F2}, {minPosY:F2}, {minPosZ:F2}) Max({maxPosX:F2}, {maxPosY:F2}, {maxPosZ:F2})");
                    
                    // Try to detect elevation offset if needed for vertical alignment
                    float positionMidY = (minPosY + maxPosY) / 2;
                    float hullMidY = (minHullY + maxHullY) / 2;
                    float elevationOffset = hullMidY - positionMidY;
                    
                    await writer.WriteLineAsync($"# Detected elevation offset: {elevationOffset:F2}");
                }

                await writer.WriteLineAsync($"# Integrated mesh generation: Combined position and hull data");
                await writer.WriteLineAsync($"# Position data appears to represent boundary pathing points");
                await writer.WriteLineAsync();
                
                // Create a single integrated object
                await writer.WriteLineAsync($"o {baseFileName}_integrated_mesh");
                
                // Track vertex indices for face generation
                int vertexCount = 0;
                
                // Write position data vertices (v x y z) first
                await writer.WriteLineAsync("# Position data vertices (pathing/boundary points)");
                foreach (var point in meshPoints)
                {
                    // Use the same coordinate transformation as the complete mesh
                    await writer.WriteLineAsync($"v {(-point.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                              $"{point.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                              $"{point.X.ToString("F6", CultureInfo.InvariantCulture)}");
                    vertexCount++;
                }

                // Generate position-only triangles
                var positionFaces = GenerateTriangleFaces(meshPoints);
                
                // Now add hull vertices
                if (hullVertices.Count > 0)
                {
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync("# Hull mesh vertices");
                    
                    int hullBaseIndex = vertexCount;
                    
                    // Write hull vertices
                    foreach (var vertex in hullVertices)
                    {
                        // Use the same coordinate transformation as the position data
                        await writer.WriteLineAsync($"v {(-vertex.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{vertex.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{vertex.X.ToString("F6", CultureInfo.InvariantCulture)}");
                        vertexCount++;
                    }
                    
                    // Add object groups for clarity
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync("g position_triangles");
                    
                    // Write position-only triangular faces if any were created
                    if (positionFaces.Count > 0)
                    {
                        await writer.WriteLineAsync($"# Generated faces from position data: {positionFaces.Count}");
                        
                        // Write faces (f v1 v2 v3)
                        // Note: OBJ indices are 1-based
                        foreach (var face in positionFaces)
                        {
                            await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                        }
                        
                        _logger?.LogInformation("Created position data mesh with {VertexCount} vertices and {FaceCount} triangular faces",
                            meshPoints.Count, positionFaces.Count);
                    }
                    
                    // Add hull mesh faces
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync("g hull_triangles");
                    
                    // Try to get triangle indices from the file if available
                    if (file.VertexIndicesChunk != null && file.VertexIndicesChunk.Indices.Count > 0)
                    {
                        int triangleCount = file.VertexIndicesChunk.Indices.Count / 3;
                        await writer.WriteLineAsync($"# Hull triangles from vertex indices: {triangleCount}");
                        
                        // In OBJ format, vertex indices start at 1, not 0
                        for (int i = 0; i < triangleCount; i++)
                        {
                            int baseIndex = i * 3;
                            if (baseIndex + 2 < file.VertexIndicesChunk.Indices.Count)
                            {
                                // OBJ uses 1-based indexing, add hull base index and then add 1
                                uint v1 = (uint)(file.VertexIndicesChunk.Indices[baseIndex] + hullBaseIndex + 1);
                                uint v2 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 1] + hullBaseIndex + 1);
                                uint v3 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 2] + hullBaseIndex + 1);
                                
                                await writer.WriteLineAsync($"f {v1} {v2} {v3}");
                            }
                        }
                    }
                    else
                    {
                        // Generate simplified hull faces
                        var hullFaces = GenerateSimplifiedMesh(hullVertices);
                        
                        if (hullFaces.Count > 0)
                        {
                            await writer.WriteLineAsync($"# Generated faces from hull data: {hullFaces.Count}");
                            
                            // Write hull faces with adjusted indices
                            foreach (var face in hullFaces)
                            {
                                int v1 = hullBaseIndex + face.Item1 + 1;
                                int v2 = hullBaseIndex + face.Item2 + 1;
                                int v3 = hullBaseIndex + face.Item3 + 1;
                                await writer.WriteLineAsync($"f {v1} {v2} {v3}");
                            }
                            
                            _logger?.LogInformation("Added hull mesh with {VertexCount} vertices and {FaceCount} triangular faces",
                                hullVertices.Count, hullFaces.Count);
                        }
                    }
                    
                    // Now create connection triangles between position data and hull vertices
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync("g connection_triangles");
                    
                    // Generate connection faces between position data and hull vertices
                    var connectionFaces = GenerateConnectionFaces(meshPoints, hullVertices, hullBaseIndex);
                    
                    if (connectionFaces.Count > 0)
                    {
                        await writer.WriteLineAsync($"# Connection faces between position and hull data: {connectionFaces.Count}");
                        
                        foreach (var face in connectionFaces)
                        {
                            await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                        }
                        
                        _logger?.LogInformation("Generated {Count} connection triangles between position and hull data",
                            connectionFaces.Count);
                    }
                }
                else
                {
                    // If no hull data is available, just write position faces
                    if (positionFaces.Count > 0)
                    {
                        await writer.WriteLineAsync($"# Generated faces from position data: {positionFaces.Count}");
                        
                        foreach (var face in positionFaces)
                        {
                            await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                        }
                        
                        _logger?.LogInformation("Created position data mesh with {VertexCount} vertices and {FaceCount} triangular faces",
                            meshPoints.Count, positionFaces.Count);
                    }
                }
                
                _logger?.LogInformation("Exported integrated mesh to OBJ: {FilePath}", objFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting integrated mesh to OBJ: {FilePath}", objFilePath);
            }
        }
        
        /// <summary>
        /// Generates faces that connect position data points to hull vertices, creating a unified mesh surface
        /// </summary>
        private List<(int, int, int)> GenerateConnectionFaces(List<MeshPoint> positionPoints, List<Vector3> hullVertices, int hullBaseIndex)
        {
            var connectionFaces = new List<(int, int, int)>();
            
            try
            {
                // Build spatial index for hull vertices to speed up proximity searches
                var hullSpatialIndex = new Dictionary<(int, int), List<(int, Vector3)>>();
                const int cellSize = 10;
                
                for (int i = 0; i < hullVertices.Count; i++)
                {
                    var vertex = hullVertices[i];
                    
                    // Use X and Z for 2D grid (ignoring Y/height)
                    var cell = ((int)(vertex.X / cellSize), (int)(vertex.Z / cellSize));
                    
                    if (!hullSpatialIndex.ContainsKey(cell))
                        hullSpatialIndex[cell] = new List<(int, Vector3)>();
                    
                    hullSpatialIndex[cell].Add((i, vertex));
                }
                
                // Track edges that have already been connected to avoid duplicate connections
                var connectedEdges = new HashSet<(int, int)>();
                
                // For each position point, find the nearest hull vertices and connect them
                for (int i = 0; i < positionPoints.Count; i++)
                {
                    var point = positionPoints[i];
                    
                    // Convert MeshPoint to Vector3 using the same coordinate system as hull vertices
                    var pointVector = new Vector3(point.X, point.Y, point.Z);
                    
                    // Determine the cell for this point
                    var cell = ((int)(pointVector.X / cellSize), (int)(pointVector.Z / cellSize));
                    
                    // Get nearby cells for hull vertex lookup
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
                    
                    // Find nearby hull vertices
                    var nearbyHullVertices = new List<(int, Vector3, float)>(); // index, vertex, distance
                    
                    foreach (var nearCell in nearbyCells)
                    {
                        if (hullSpatialIndex.TryGetValue(nearCell, out var vertices))
                        {
                            foreach (var (idx, vertex) in vertices)
                            {
                                // Calculate 2D distance (ignoring Y/height)
                                float dx = vertex.X - pointVector.X;
                                float dz = vertex.Z - pointVector.Z;
                                float distance = (float)Math.Sqrt(dx * dx + dz * dz);
                                
                                // Only consider vertices within a reasonable distance
                                if (distance < 50f)
                                {
                                    nearbyHullVertices.Add((idx, vertex, distance));
                                }
                            }
                        }
                    }
                    
                    // Sort by distance and take the closest few
                    var closestHullVertices = nearbyHullVertices
                        .OrderBy(v => v.Item3)
                        .Take(5)
                        .ToList();
                    
                    // Connect to the closest hull vertices
                    if (closestHullVertices.Count >= 2)
                    {
                        // Get the closest vertex
                        var (idx1, vertex1, _) = closestHullVertices[0];
                        
                        // Connect to the other nearby vertices
                        for (int j = 1; j < closestHullVertices.Count; j++)
                        {
                            var (idx2, vertex2, _) = closestHullVertices[j];
                            
                            // Check if these hull vertices are reasonably close to each other
                            float hullDx = vertex1.X - vertex2.X;
                            float hullDz = vertex1.Z - vertex2.Z;
                            float hullDistance = (float)Math.Sqrt(hullDx * hullDx + hullDz * hullDz);
                            
                            if (hullDistance < 30f) // Don't connect hull vertices that are too far apart
                            {
                                // Create a face from position point to two hull vertices
                                // OBJ is 1-based, so add 1 for actual indices in the file
                                // Hull indices also need hullBaseIndex added
                                int positionIndex = i;
                                int hullIndex1 = hullBaseIndex + idx1;
                                int hullIndex2 = hullBaseIndex + idx2;
                                
                                // Check if this edge has already been connected
                                var edge = (Math.Min(hullIndex1, hullIndex2), Math.Max(hullIndex1, hullIndex2));
                                
                                if (!connectedEdges.Contains(edge))
                                {
                                    connectionFaces.Add((positionIndex, hullIndex1, hullIndex2));
                                    connectedEdges.Add(edge);
                                }
                            }
                        }
                    }
                }
                
                _logger?.LogDebug("Generated {Count} connection faces between position data and hull", connectionFaces.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Error generating connection faces, falling back to separate meshes");
            }
            
            return connectionFaces;
        }
        
        /// <summary>
        /// Exports a fully integrated mesh combining all relevant data from the PM4 file.
        /// This combines vertex data, position (pathing) data, and preserves proper chunk relationships.
        /// </summary>
        private async Task ExportIntegratedMeshAsync(PM4File file, string baseFileName, string outputDir)
        {
            string objFileName = $"{baseFileName}_integrated.obj";
            string objFilePath = Path.Combine(outputDir, objFileName);
            string mtlFileName = $"{baseFileName}_materials.mtl";
            string mtlFilePath = Path.Combine(outputDir, mtlFileName);
            
            try
            {
                using var writer = new StreamWriter(objFilePath);
                
                // Write OBJ header with detailed metadata
                await writer.WriteLineAsync($"# Integrated PM4 mesh from {file.FileName}");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer");
                await writer.WriteLineAsync($"# File represents a combined Pathing & Models (PM4) dataset");
                await writer.WriteLineAsync($"# Coordinate system transformation: (-Z, Y, X)");
                
                // Get all data sources
                var positionPoints = ExtractPositionData(file);
                bool hasPositionData = positionPoints.Count > 0;
                
                bool hasVertexData = file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0;
                bool hasIndices = file.VertexIndicesChunk != null && file.VertexIndicesChunk.Indices.Count > 0;
                bool hasNormals = file.NormalCoordinatesChunk != null && file.NormalCoordinatesChunk.Normals.Count > 0;
                
                // Create simplified collision hull vertices for inclusion in the integrated mesh
                List<Vector3> collisionHullVertices = new List<Vector3>();
                List<(int, int, int)> collisionHullFaces = new List<(int, int, int)>();
                
                if (hasVertexData)
                {
                    // Create simplified hull for integration
                    var vertices = file.VertexPositionsChunk.Vertices
                        .Select(v => new Vector3(v.X, v.Y, v.Z)) // Keep original order
                        .ToList();
                    
                    collisionHullVertices = ClusterVertices(vertices, distanceThreshold: 2.0f);
                    collisionHullFaces = GenerateSimplifiedMesh(collisionHullVertices);
                }
                
                // Display info about what data we have
                await writer.WriteLineAsync("# Contains the following data:");
                if (hasPositionData) await writer.WriteLineAsync($"#  - Position (pathing) points: {positionPoints.Count}");
                if (hasVertexData) 
                {
                    await writer.WriteLineAsync($"#  - Model vertices: {file.VertexPositionsChunk.Vertices.Count}");
                    await writer.WriteLineAsync($"#  - Collision hull vertices: {collisionHullVertices.Count}");
                }
                if (hasIndices) await writer.WriteLineAsync($"#  - Model faces: {file.VertexIndicesChunk.Indices.Count / 3}");
                if (hasNormals) await writer.WriteLineAsync($"#  - Normal vectors: {file.NormalCoordinatesChunk.Normals.Count}");
                
                // Add material library reference
                await writer.WriteLineAsync($"mtllib {mtlFileName}");
                await writer.WriteLineAsync();
                
                // Define a single unified object
                await writer.WriteLineAsync($"o {baseFileName}");
                
                // Write position data vertices first with a pathing material
                if (hasPositionData)
                {
                    await writer.WriteLineAsync($"g {baseFileName}_pathing");
                    await writer.WriteLineAsync("usemtl pathing");
                    
                    // Write position data vertices (v x y z)
                    await writer.WriteLineAsync("# Navigation waypoints for NPC pathing");
                    int positionVertexCount = 0; // Track how many we write for face indexing
                    
                    foreach (var point in positionPoints)
                    {
                        // Apply consistent coordinate transformation
                        await writer.WriteLineAsync($"v {(-point.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{point.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{point.X.ToString("F6", CultureInfo.InvariantCulture)}");
                        positionVertexCount++;
                    }
                    
                    // Generate and write position data faces - for pathing visualization
                    // We'll prefer a sparser set of triangles for navigation visualization
                    var positionFaces = GenerateTriangleFaces(positionPoints, maxEdgeLength: 15.0f, sparse: true);
                    if (positionFaces.Count > 0)
                    {
                        await writer.WriteLineAsync();
                        await writer.WriteLineAsync($"# Navigation mesh triangular faces: {positionFaces.Count}");
                        
                        // Position vertices start at index 1 (OBJ is 1-indexed)
                        foreach (var face in positionFaces)
                        {
                            await writer.WriteLineAsync($"f {face.Item1 + 1} {face.Item2 + 1} {face.Item3 + 1}");
                        }
                        
                        _logger?.LogInformation("Created navigation waypoint mesh with {VertexCount} points and {FaceCount} triangular faces",
                            positionPoints.Count, positionFaces.Count);
                    }
                }
                
                // Next write model vertices with a different material
                if (hasVertexData)
                {
                    await writer.WriteLineAsync();
                    await writer.WriteLineAsync($"g {baseFileName}_model");
                    await writer.WriteLineAsync("usemtl model");
                    
                    // Write model vertices
                    await writer.WriteLineAsync("# Model vertices");
                    int modelVertexStart = hasPositionData ? positionPoints.Count + 1 : 1; // 1-indexed for OBJ
                    
                    foreach (var vertex in file.VertexPositionsChunk.Vertices)
                    {
                        // Apply consistent coordinate transformation
                        await writer.WriteLineAsync($"v {(-vertex.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{vertex.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                                  $"{vertex.X.ToString("F6", CultureInfo.InvariantCulture)}");
                    }
                    
                    // Write normals if available
                    if (hasNormals)
                    {
                        await writer.WriteLineAsync();
                        await writer.WriteLineAsync("# Normal vectors");
                        
                        foreach (var normal in file.NormalCoordinatesChunk.Normals)
                        {
                            // Apply consistent transformation to normals
                            await writer.WriteLineAsync($"vn {(-normal.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                                     $"{normal.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                                     $"{normal.X.ToString("F6", CultureInfo.InvariantCulture)}");
                        }
                    }
                    
                    // Write model faces using indices if available
                    if (hasIndices)
                    {
                        await writer.WriteLineAsync();
                        await writer.WriteLineAsync($"# Model triangular faces");
                        int triangleCount = file.VertexIndicesChunk.Indices.Count / 3;
                        
                        for (int i = 0; i < triangleCount; i++)
                        {
                            int baseIndex = i * 3;
                            if (baseIndex + 2 < file.VertexIndicesChunk.Indices.Count)
                            {
                                // Get vertex indices from the chunk and add appropriate offset
                                // We need to offset indices by positionVertexCount and add 1 for OBJ format (1-indexed)
                                uint idx1 = (uint)(file.VertexIndicesChunk.Indices[baseIndex] + modelVertexStart);
                                uint idx2 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 1] + modelVertexStart);
                                uint idx3 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 2] + modelVertexStart);
                                
                                // Write the face
                                if (hasNormals)
                                {
                                    // Include normal indices (assuming they match vertex indices)
                                    uint n1 = (uint)(file.VertexIndicesChunk.Indices[baseIndex] + 1);
                                    uint n2 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 1] + 1);
                                    uint n3 = (uint)(file.VertexIndicesChunk.Indices[baseIndex + 2] + 1);
                                    
                                    await writer.WriteLineAsync($"f {idx1}//{n1} {idx2}//{n2} {idx3}//{n3}");
                                }
                                else
                                {
                                    await writer.WriteLineAsync($"f {idx1} {idx2} {idx3}");
                                }
                            }
                        }
                        
                        _logger?.LogInformation("Added model mesh with {VertexCount} vertices and {TriangleCount} triangular faces",
                            file.VertexPositionsChunk.Vertices.Count, triangleCount);
                    }
                    
                    // Add collision hull to the integrated mesh with its own material
                    if (collisionHullVertices.Count > 0 && collisionHullFaces.Count > 0)
                    {
                        await writer.WriteLineAsync();
                        await writer.WriteLineAsync($"g {baseFileName}_collision_hull");
                        await writer.WriteLineAsync("usemtl collision");
                        
                        // Write collision hull vertices
                        await writer.WriteLineAsync("# Collision hull vertices");
                        int hullVertexStart = modelVertexStart + file.VertexPositionsChunk.Vertices.Count;
                        
                        foreach (var vertex in collisionHullVertices)
                        {
                            // Apply consistent coordinate transformation
                            await writer.WriteLineAsync($"v {(-vertex.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                                      $"{vertex.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                                      $"{vertex.X.ToString("F6", CultureInfo.InvariantCulture)}");
                        }
                        
                        // Write collision hull faces
                        await writer.WriteLineAsync();
                        await writer.WriteLineAsync($"# Collision hull faces");
                        
                        foreach (var face in collisionHullFaces)
                        {
                            int idx1 = hullVertexStart + face.Item1;
                            int idx2 = hullVertexStart + face.Item2;
                            int idx3 = hullVertexStart + face.Item3;
                            await writer.WriteLineAsync($"f {idx1} {idx2} {idx3}");
                        }
                        
                        _logger?.LogInformation("Added collision hull with {VertexCount} vertices and {FaceCount} triangular faces",
                            collisionHullVertices.Count, collisionHullFaces.Count);
                    }
                }
                
                // Create a material file with added collision hull material
                using (var mtlWriter = new StreamWriter(mtlFilePath))
                {
                    await mtlWriter.WriteLineAsync("# Material file for PM4 mesh");
                    
                    await mtlWriter.WriteLineAsync("newmtl model");
                    await mtlWriter.WriteLineAsync("Ka 0.7 0.7 0.7");     // Gray
                    await mtlWriter.WriteLineAsync("Kd 0.8 0.8 0.8");
                    await mtlWriter.WriteLineAsync("Ks 0.3 0.3 0.3");
                    await mtlWriter.WriteLineAsync("Ns 10");
                    
                    await mtlWriter.WriteLineAsync("newmtl pathing");
                    await mtlWriter.WriteLineAsync("Ka 0.9 0.5 0.2");     // Orange
                    await mtlWriter.WriteLineAsync("Kd 1.0 0.6 0.3");
                    await mtlWriter.WriteLineAsync("Ks 0.8 0.4 0.2");
                    await mtlWriter.WriteLineAsync("Ns 10");
                    
                    await mtlWriter.WriteLineAsync("newmtl collision");
                    await mtlWriter.WriteLineAsync("Ka 0.2 0.8 0.3");     // Green
                    await mtlWriter.WriteLineAsync("Kd 0.3 0.9 0.4");
                    await mtlWriter.WriteLineAsync("Ks 0.1 0.6 0.2");
                    await mtlWriter.WriteLineAsync("Ns 10");
                    await mtlWriter.WriteLineAsync("d 0.7");              // Some transparency
                }
                
                _logger?.LogInformation("Exported integrated mesh to: {FilePath}", objFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting integrated mesh to OBJ: {FilePath}", objFilePath);
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
                
                // Extract the basic vertices - fix the extraction to be consistent with other transformations
                var vertices = file.VertexPositionsChunk.Vertices
                    .Select(v => new Vector3(v.X, v.Y, v.Z)) // Keep original order to apply transformation uniformly
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
                
                // Write vertices (v x y z) - apply the standard coordinate transformation here
                foreach (var vertex in clusteredVertices)
                {
                    // Apply the same coordinate transformation as other exports: (-Z, Y, X)
                    await writer.WriteLineAsync($"v {(-vertex.Z).ToString("F6", CultureInfo.InvariantCulture)} " +
                                              $"{vertex.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                                              $"{vertex.X.ToString("F6", CultureInfo.InvariantCulture)}");
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
        /// Generates triangular faces from the position data points using a distance-based approach
        /// </summary>
        /// <param name="points">The position data points to create faces from</param>
        /// <returns>A list of triangle indices (each tuple represents indices of 3 points)</returns>
        private List<(int, int, int)> GenerateTriangleFaces(List<MeshPoint> points, float maxEdgeLength = 20.0f, bool sparse = false)
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
                    var indexMap = new Dictionary<MeshPoint, int>();
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
                            if (Distance2D(p1, p2) > maxEdgeLength)
                                continue;
                            
                            for (int k = j + 1; k < sortedPoints.Count; k++)
                            {
                                var p3 = sortedPoints[k];
                                
                                // Check if these three points form a reasonable triangle
                                if (IsValidTriangle(p1, p2, p3, maxEdgeLength))
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
                    var spatialIndex = new Dictionary<(int, int), List<(int, MeshPoint)>>();
                    const int cellSize = 10; // Size of spatial grid cells
                    
                    // Build spatial index
                    for (int i = 0; i < points.Count; i++)
                    {
                        var point = points[i];
                        var cell = ((int)(point.X / cellSize), (int)(point.Z / cellSize));
                        
                        if (!spatialIndex.ContainsKey(cell))
                            spatialIndex[cell] = new List<(int, MeshPoint)>();
                        
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
                        var nearby = new List<(int, MeshPoint)>();
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
                            if (idx2 == i || Distance2D(p1, p2) > maxEdgeLength) continue;
                            
                            // Skip if we've already used this edge
                            var edge1 = (Math.Min(i, idx2), Math.Max(i, idx2));
                            if (processedEdges.Contains(edge1)) continue;
                            processedEdges.Add(edge1);
                            
                            for (int k = j + 1; k < nearby.Count; k++)
                            {
                                var (idx3, p3) = nearby[k];
                                if (idx3 == i || idx3 == idx2 || Distance2D(p1, p3) > maxEdgeLength || Distance2D(p2, p3) > maxEdgeLength) 
                                    continue;
                                
                                // Check other edges
                                var edge2 = (Math.Min(i, idx3), Math.Max(i, idx3));
                                var edge3 = (Math.Min(idx2, idx3), Math.Max(idx2, idx3));
                                
                                if (processedEdges.Contains(edge2) || processedEdges.Contains(edge3)) 
                                    continue;
                                
                                // Check if triangle is valid
                                if (IsValidTriangle(p1, p2, p3, maxEdgeLength))
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
        private bool TryDetectGridPattern(List<MeshPoint> points)
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
        /// Calculates the 2D distance between two mesh points (ignoring Y/height)
        /// </summary>
        private float Distance2D(MeshPoint p1, MeshPoint p2)
        {
            float dx = p1.X - p2.X;
            float dz = p1.Z - p2.Z;
            return (float)Math.Sqrt(dx * dx + dz * dz);
        }
        
        /// <summary>
        /// Checks if three points form a valid triangle (not too large, not too small, not too thin)
        /// </summary>
        private bool IsValidTriangle(MeshPoint p1, MeshPoint p2, MeshPoint p3, float maxEdgeLength)
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