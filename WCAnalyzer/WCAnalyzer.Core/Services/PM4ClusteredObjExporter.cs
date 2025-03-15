using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Statistics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for exporting PM4 mesh data to Wavefront OBJ format with vertex clustering.
    /// This exporter groups vertices by proximity to create more meaningful object components.
    /// </summary>
    public class PM4ClusteredObjExporter
    {
        private readonly ILogger<PM4ClusteredObjExporter>? _logger;
        private readonly string _outputDirectory;
        private readonly VertexClusteringService _clusteringService;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4ClusteredObjExporter"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Output directory for the OBJ files</param>
        public PM4ClusteredObjExporter(ILogger<PM4ClusteredObjExporter>? logger = null, string? outputDirectory = null)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? Path.Combine(Directory.GetCurrentDirectory(), "output");
            _clusteringService = new VertexClusteringService();
        }

        /// <summary>
        /// Exports a PM4 file to clustered OBJ format.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="targetClusterCount">Optional target number of clusters (default: auto-determined)</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportToClusteredObjAsync(PM4AnalysisResult result, int? targetClusterCount = null)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (result.PM4File == null)
            {
                _logger?.LogWarning("Cannot export {FileName} to clustered OBJ: PM4File is null", result.FileName);
                return;
            }

            // Check if we have the necessary data to generate an OBJ file
            if (result.PM4File.VertexPositionsChunk == null || result.PM4File.VertexPositionsChunk.Vertices.Count == 0)
            {
                _logger?.LogWarning("Cannot export {FileName} to clustered OBJ: No vertices found", result.FileName);
                return;
            }

            try
            {
                // Create output directory if it doesn't exist
                var objDir = Path.Combine(_outputDirectory, "pm4_clustered_obj");
                Directory.CreateDirectory(objDir);

                string baseFileName = Path.GetFileNameWithoutExtension(result.FileName ?? "unknown");
                string objFilePath = Path.Combine(objDir, $"{baseFileName}_clustered.obj");

                _logger?.LogInformation("Exporting {FileName} to clustered OBJ format: {ObjFilePath}", result.FileName, objFilePath);

                // Extract terrain coordinates from filename
                var coordinates = PM4ObjExporter.ExtractCoordinatesFromFilename(result.FileName ?? string.Empty);
                _logger?.LogDebug("Extracted coordinates: ({X}, {Y})", coordinates.x, coordinates.y);

                // Build the OBJ content with clustering
                var objContent = GenerateClusteredObjContent(result.PM4File, targetClusterCount, coordinates);

                // Write the OBJ file
                await File.WriteAllTextAsync(objFilePath, objContent);

                _logger?.LogInformation("Successfully exported {FileName} to clustered OBJ format", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting {FileName} to clustered OBJ format", result.FileName);
            }
        }

        /// <summary>
        /// Exports a collection of PM4 files to clustered OBJ format.
        /// </summary>
        /// <param name="results">The PM4 analysis results to export.</param>
        /// <param name="targetClusterCount">Optional target number of clusters (default: auto-determined)</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportAllToClusteredObjAsync(IEnumerable<PM4AnalysisResult> results, int? targetClusterCount = null)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            _logger?.LogInformation("Starting batch export of PM4 files to clustered OBJ format");

            foreach (var result in results)
            {
                await ExportToClusteredObjAsync(result, targetClusterCount);
            }

            _logger?.LogInformation("Batch export of PM4 files to clustered OBJ format completed");
        }

        /// <summary>
        /// Generates clustered OBJ content from a PM4 file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to convert to OBJ format.</param>
        /// <param name="targetClusterCount">Optional target number of clusters (default: auto-determined)</param>
        /// <param name="coordinates">Optional coordinates for additional context.</param>
        /// <returns>A string containing the OBJ file content.</returns>
        private string GenerateClusteredObjContent(PM4File pm4File, int? targetClusterCount = null, (int x, int y) coordinates = default)
        {
            if (pm4File == null)
                throw new ArgumentNullException(nameof(pm4File));

            if (pm4File.VertexPositionsChunk == null || pm4File.VertexPositionsChunk.Vertices.Count == 0)
            {
                _logger?.LogWarning("No vertex data found in PM4 file");
                return string.Empty;
            }

            if (pm4File.VertexIndicesChunk == null || pm4File.VertexIndicesChunk.Indices.Count == 0)
            {
                _logger?.LogWarning("No index data found in PM4 file");
                return string.Empty;
            }

            // Validate that indices are within bounds
            if (pm4File.VertexIndicesChunk.Indices.Any(i => i >= pm4File.VertexPositionsChunk.Vertices.Count))
            {
                _logger?.LogError("Invalid vertex indices found: some indices are out of range");
                return string.Empty;
            }

            var sb = new StringBuilder();

            // Add a header comment
            sb.AppendLine($"# Clustered OBJ file generated from PM4 file: {pm4File.FileName}");
            sb.AppendLine($"# Generated by WCAnalyzer on {DateTime.Now}");
            if (coordinates.x >= 0 && coordinates.y >= 0)
            {
                sb.AppendLine($"# Terrain coordinates: {coordinates.x}, {coordinates.y}");
            }
            sb.AppendLine();

            // Add material library reference
            sb.AppendLine("mtllib pm4_clustered_materials.mtl");
            sb.AppendLine();

            // Start a new object group
            string objName = Path.GetFileNameWithoutExtension(pm4File.FileName ?? "unknown");
            sb.AppendLine($"o {objName}");

            // Get vertices for clustering
            var vertices = pm4File.VertexPositionsChunk.Vertices;
            
            // Perform vertex clustering directly with Vector3
            var vertexClusters = _clusteringService.ClusterVertices(vertices, targetClusterCount);
            
            // Convert indices to uint list
            var indicesList = pm4File.VertexIndicesChunk.Indices.ToList();
            
            // Validate triangle indices are multiples of 3
            if (indicesList.Count % 3 != 0)
            {
                _logger?.LogError("Invalid triangle data: index count is not a multiple of 3");
                return string.Empty;
            }
            
            var triangleClusters = _clusteringService.GroupTrianglesByCluster(indicesList, vertexClusters);

            if (triangleClusters == null || triangleClusters.Count == 0)
            {
                _logger?.LogWarning("No valid triangle clusters generated");
                return string.Empty;
            }

            _logger?.LogInformation("Grouped vertices into {ClusterCount} clusters", triangleClusters.Count);

            // Add vertices
            _logger?.LogDebug("Writing {Count} vertices", vertices.Count);
            
            // Extract bounding box information for debugging
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            
            foreach (var vertex in pm4File.VertexPositionsChunk.Vertices)
            {
                // Track bounding box
                minX = Math.Min(minX, vertex.X);
                minY = Math.Min(minY, vertex.Y);
                minZ = Math.Min(minZ, vertex.Z);
                maxX = Math.Max(maxX, vertex.X);
                maxY = Math.Max(maxY, vertex.Y);
                maxZ = Math.Max(maxZ, vertex.Z);
                
                // Use invariant culture to ensure consistent decimal format
                sb.AppendLine($"v {vertex.X.ToString(CultureInfo.InvariantCulture)} {vertex.Y.ToString(CultureInfo.InvariantCulture)} {vertex.Z.ToString(CultureInfo.InvariantCulture)}");
            }
            
            // Add bounding box information as a comment
            sb.AppendLine($"# Bounding Box: Min({minX:F2}, {minY:F2}, {minZ:F2}) Max({maxX:F2}, {maxY:F2}, {maxZ:F2})");
            sb.AppendLine($"# Dimensions: W({maxX - minX:F2}) H({maxY - minY:F2}) D({maxZ - minZ:F2})");
            sb.AppendLine();

            // Add faces (triangles) grouped by cluster
            foreach (var clusterGroup in triangleClusters)
            {
                int clusterId = clusterGroup.Key;
                var triangleIndices = clusterGroup.Value;
                
                if (triangleIndices.Count == 0)
                    continue;
                
                // Create a group and material for this cluster
                string groupName = $"{objName}_cluster_{clusterId}";
                string materialName = $"cluster_{clusterId}";
                
                sb.AppendLine($"g {groupName}");
                sb.AppendLine($"usemtl {materialName}");
                
                _logger?.LogDebug("Writing {Count} triangles for cluster {ClusterId}", 
                    triangleIndices.Count, clusterId);
                
                // Write faces for this cluster
                foreach (uint triangleIndex in triangleIndices)
                {
                    int baseIndex = (int)triangleIndex * 3;
                    
                    // Validate indices are within bounds
                    if (baseIndex + 2 >= pm4File.VertexIndicesChunk.Indices.Count)
                    {
                        _logger?.LogWarning("Triangle index {Index} is out of bounds, skipping", triangleIndex);
                        continue;
                    }
                    
                    int v1Index = (int)pm4File.VertexIndicesChunk.Indices[baseIndex];
                    int v2Index = (int)pm4File.VertexIndicesChunk.Indices[baseIndex + 1];
                    int v3Index = (int)pm4File.VertexIndicesChunk.Indices[baseIndex + 2];
                    
                    // Validate vertex indices
                    if (v1Index >= pm4File.VertexPositionsChunk.Vertices.Count ||
                        v2Index >= pm4File.VertexPositionsChunk.Vertices.Count ||
                        v3Index >= pm4File.VertexPositionsChunk.Vertices.Count)
                    {
                        _logger?.LogWarning("Invalid vertex indices for triangle {Index}, skipping", triangleIndex);
                        continue;
                    }
                    
                    // OBJ uses 1-based indices
                    v1Index++;
                    v2Index++;
                    v3Index++;
                    
                    sb.AppendLine($"f {v1Index} {v2Index} {v3Index}");
                }
                
                sb.AppendLine();
            }

            // Add position data as a separate object if available
            if (pm4File.PositionDataChunk != null)
            {
                var positionRecords = pm4File.PositionDataChunk.Entries.Where(e => !e.IsControlRecord).ToList();
                if (positionRecords.Count > 0)
                {
                    sb.AppendLine();
                    sb.AppendLine("# Server position data points (not part of the mesh)");
                    sb.AppendLine($"o {objName}_PositionData");
                    sb.AppendLine("usemtl positionData");
                    
                    _logger?.LogDebug("Writing {Count} position data points", positionRecords.Count);
                    
                    int positionVertexStart = pm4File.VertexPositionsChunk.Vertices.Count;
                    positionVertexStart += 1; // 1-based indexing
                    
                    foreach (var pos in positionRecords)
                    {
                        sb.AppendLine($"v {pos.CoordinateX.ToString(CultureInfo.InvariantCulture)} {pos.CoordinateY.ToString(CultureInfo.InvariantCulture)} {pos.CoordinateZ.ToString(CultureInfo.InvariantCulture)}");
                    }
                    
                    // Draw position points
                    int pointCount = positionRecords.Count;
                    
                    sb.Append("p");
                    for (int i = 0; i < pointCount; i++)
                    {
                        sb.Append($" {positionVertexStart + i}");
                    }
                    sb.AppendLine();
                }
            }

            // Generate a material file with colors for each cluster
            GenerateClusteredMaterialFile(triangleClusters);

            return sb.ToString();
        }

        /// <summary>
        /// Generates a material file with distinct colors for each cluster.
        /// </summary>
        /// <param name="triangleClusters">Dictionary mapping cluster IDs to triangle indices.</param>
        private void GenerateClusteredMaterialFile(Dictionary<int, List<uint>> triangleClusters)
        {
            try
            {
                var objDir = Path.Combine(_outputDirectory, "pm4_clustered_obj");
                Directory.CreateDirectory(objDir);
                string mtlFilePath = Path.Combine(objDir, "pm4_clustered_materials.mtl");
                
                var mtlContent = new StringBuilder();
                mtlContent.AppendLine("# PM4 Clustered Materials Library");
                mtlContent.AppendLine("# Generated by WCAnalyzer");
                mtlContent.AppendLine();
                
                // Generate a unique color for each cluster
                foreach (int clusterId in triangleClusters.Keys)
                {
                    // Generate a deterministic but varied color based on cluster ID
                    double hue = (clusterId * 0.618033988749895) % 1.0; // Golden ratio conjugate
                    double saturation = 0.7;
                    double value = 0.95;
                    
                    // Convert HSV to RGB
                    var (r, g, b) = HsvToRgb(hue, saturation, value);
                    
                    mtlContent.AppendLine($"newmtl cluster_{clusterId}");
                    mtlContent.AppendLine($"Ka {r * 0.3:F6} {g * 0.3:F6} {b * 0.3:F6}");  // Ambient color
                    mtlContent.AppendLine($"Kd {r:F6} {g:F6} {b:F6}");                    // Diffuse color
                    mtlContent.AppendLine($"Ks 0.2 0.2 0.2");                             // Specular color
                    mtlContent.AppendLine("d 1.0");                                        // Transparency
                    mtlContent.AppendLine("Ns 10.0");                                      // Shininess
                    mtlContent.AppendLine();
                }
                
                // Position data material (blue)
                mtlContent.AppendLine("newmtl positionData");
                mtlContent.AppendLine("Ka 0.0 0.0 0.2");
                mtlContent.AppendLine("Kd 0.0 0.0 1.0");
                mtlContent.AppendLine("Ks 0.2 0.2 0.8");
                mtlContent.AppendLine("d 1.0");
                mtlContent.AppendLine("Ns 10.0");
                
                File.WriteAllText(mtlFilePath, mtlContent.ToString());
                _logger?.LogDebug("Generated clustered material file: {FilePath}", mtlFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Failed to generate clustered material file");
            }
        }

        /// <summary>
        /// Converts HSV color values to RGB.
        /// </summary>
        /// <param name="hue">Hue value (0.0 to 1.0)</param>
        /// <param name="saturation">Saturation value (0.0 to 1.0)</param>
        /// <param name="value">Value/brightness (0.0 to 1.0)</param>
        /// <returns>RGB color values (0.0 to 1.0)</returns>
        private static (double r, double g, double b) HsvToRgb(double hue, double saturation, double value)
        {
            int hi = (int)(hue * 6) % 6;
            double f = hue * 6 - hi;
            double p = value * (1 - saturation);
            double q = value * (1 - f * saturation);
            double t = value * (1 - (1 - f) * saturation);

            return hi switch
            {
                0 => (value, t, p),
                1 => (q, value, p),
                2 => (p, value, t),
                3 => (p, q, value),
                4 => (t, p, value),
                _ => (value, p, q)
            };
        }
    }
} 