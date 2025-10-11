using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Enhanced service for exporting PM4 mesh data to Wavefront OBJ format with proper coordinate transformation
    /// and position data categorization.
    /// </summary>
    public class PM4EnhancedObjExporter
    {
        private readonly ILogger<PM4EnhancedObjExporter>? _logger;
        private readonly string _outputDirectory;
        private static readonly Regex CoordinatePattern = new Regex(@"_(\d+)_(\d+)", RegexOptions.Compiled);

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4EnhancedObjExporter"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Output directory for the OBJ files</param>
        public PM4EnhancedObjExporter(ILogger<PM4EnhancedObjExporter>? logger, string outputDirectory)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
        }

        /// <summary>
        /// Exports a collection of PM4 files to OBJ format.
        /// </summary>
        /// <param name="results">The PM4 analysis results to export.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportAllToObjAsync(IEnumerable<PM4AnalysisResult> results)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            _logger?.LogInformation("Starting enhanced batch export of PM4 files to OBJ format");

            foreach (var result in results)
            {
                await ExportToObjAsync(result);
            }

            _logger?.LogInformation("Enhanced batch export of PM4 files to OBJ format completed");
        }

        /// <summary>
        /// Exports a PM4 file to OBJ format with enhanced coordinate handling.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportToObjAsync(PM4AnalysisResult result)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (result.PM4File == null)
            {
                _logger?.LogWarning("Cannot export {FileName} to enhanced OBJ: PM4File is null", result.FileName);
                return;
            }

            // Check if we have the necessary data to generate an OBJ file
            if (result.PM4File.VertexPositionsChunk == null || result.PM4File.VertexPositionsChunk.Vertices.Count == 0)
            {
                _logger?.LogWarning("Cannot export {FileName} to enhanced OBJ: No vertices found", result.FileName);
                return;
            }

            try
            {
                // Create output directory if it doesn't exist
                var objDir = Path.Combine(_outputDirectory, "pm4_obj");
                Directory.CreateDirectory(objDir);

                string baseFileName = Path.GetFileNameWithoutExtension(result.FileName);
                string objFilePath = Path.Combine(objDir, $"{baseFileName}.obj");

                _logger?.LogInformation("Exporting {FileName} to enhanced OBJ format: {ObjFilePath}", result.FileName, objFilePath);

                // Extract terrain coordinates from filename
                var coordinates = ExtractCoordinatesFromFilename(result.FileName);
                _logger?.LogDebug("Extracted coordinates: ({X}, {Y})", coordinates.x, coordinates.y);

                // Build the OBJ content with proper coordinate handling
                var objContent = GenerateObjContent(result.PM4File, coordinates);

                // Write the OBJ file
                await File.WriteAllTextAsync(objFilePath, objContent);

                _logger?.LogInformation("Successfully exported {FileName} to enhanced OBJ format", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting {FileName} to enhanced OBJ format", result.FileName);
            }
        }

        /// <summary>
        /// Exports all PM4 files to a single consolidated OBJ file with proper coordinate handling
        /// and position categorization.
        /// </summary>
        /// <param name="results">The PM4 analysis results to export.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportToConsolidatedEnhancedObjAsync(IEnumerable<PM4AnalysisResult> results)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            var resultsList = results.Where(r => r.PM4File != null).ToList();

            if (resultsList.Count == 0)
            {
                _logger?.LogWarning("No PM4 files to export to consolidated enhanced OBJ format");
                return;
            }

            try
            {
                // Create output directory if it doesn't exist
                var objDir = Path.Combine(_outputDirectory, "pm4_obj");
                Directory.CreateDirectory(objDir);

                string objFilePath = Path.Combine(objDir, "consolidated_pm4.obj");
                string mtlFilePath = Path.Combine(objDir, "pm4_materials.mtl");

                _logger?.LogInformation("Exporting {Count} PM4 files to consolidated enhanced OBJ format: {ObjFilePath}", resultsList.Count, objFilePath);

                // Build the OBJ content with proper coordinate handling
                var objContent = GenerateConsolidatedObjContent(resultsList);

                // Write the OBJ file
                await File.WriteAllTextAsync(objFilePath, objContent);
                
                // Generate and write the MTL file
                await GenerateEnhancedMaterialFileAsync(mtlFilePath);

                _logger?.LogInformation("Successfully exported {Count} PM4 files to consolidated enhanced OBJ file", resultsList.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting PM4 files to consolidated enhanced OBJ format");
            }
        }

        /// <summary>
        /// Extracts X and Y coordinates from a PM4 or ADT filename.
        /// </summary>
        /// <param name="filename">The filename to extract coordinates from.</param>
        /// <returns>A tuple containing the X and Y coordinates.</returns>
        public static (int x, int y) ExtractCoordinatesFromFilename(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                return (-1, -1);

            // Extract using regex pattern
            var match = CoordinatePattern.Match(filename);
            if (match.Success && match.Groups.Count >= 3)
            {
                if (int.TryParse(match.Groups[1].Value, out int x) &&
                    int.TryParse(match.Groups[2].Value, out int y))
                {
                    return (x, y);
                }
            }

            // Fallback to manual parsing
            string[] parts = Path.GetFileNameWithoutExtension(filename).Split('_');
            if (parts.Length >= 2)
            {
                // Try to get the last two parts which should be coordinates
                if (int.TryParse(parts[parts.Length - 2], out int x) &&
                    int.TryParse(parts[parts.Length - 1], out int y))
                {
                    return (x, y);
                }
            }

            return (-1, -1); // Invalid/not found
        }

        /// <summary>
        /// Categorizes a position data entry based on its special value.
        /// </summary>
        /// <param name="specialValue">The special value to categorize.</param>
        /// <returns>A string representing the category.</returns>
        private string GetPositionCategory(int specialValue)
        {
            // Check for known category values
            if (specialValue == 0)
                return "Standard";
            else if (specialValue == 2)
                return "Special";
            else if ((specialValue & 0x1) != 0)
                return "Flagged";
            else if ((specialValue & 0xFF) == 4)
                return "PathNode";
            else if ((specialValue & 0xFF) == 8)
                return "WaypointNode";
            else if ((specialValue & 0xFF) == 16)
                return "SpawnPoint";
            else
                return $"Category_{specialValue & 0xFF:X2}";
        }

        /// <summary>
        /// Generates OBJ content from a PM4 file with proper coordinate handling.
        /// </summary>
        /// <param name="pm4File">The PM4 file to convert to OBJ format.</param>
        /// <param name="coordinates">Optional coordinates for additional context.</param>
        /// <returns>A string containing the OBJ file content.</returns>
        private string GenerateObjContent(PM4File pm4File, (int x, int y) coordinates = default)
        {
            var sb = new StringBuilder();

            // Add a header comment
            sb.AppendLine($"# Enhanced OBJ file generated from PM4 file: {pm4File.FileName}");
            sb.AppendLine($"# Generated by WCAnalyzer Enhanced OBJ Exporter on {DateTime.Now}");
            if (coordinates.x >= 0 && coordinates.y >= 0)
            {
                sb.AppendLine($"# Terrain coordinates: {coordinates.x}, {coordinates.y}");
            }
            sb.AppendLine($"# Coordinate system: X and Z are preserved, Y is up");
            sb.AppendLine();

            // Add material library reference
            sb.AppendLine("mtllib pm4_materials.mtl");
            sb.AppendLine();

            // Start a new object group
            string objName = Path.GetFileNameWithoutExtension(pm4File.FileName ?? "unknown");
            sb.AppendLine($"o {objName}");

            // Add vertices
            if (pm4File.VertexPositionsChunk != null)
            {
                _logger?.LogDebug("Writing {Count} vertices", pm4File.VertexPositionsChunk.Vertices.Count);
                
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
                    // Preserve X and Z, keep Y as up
                    sb.AppendLine($"v {vertex.X.ToString(CultureInfo.InvariantCulture)} {vertex.Y.ToString(CultureInfo.InvariantCulture)} {vertex.Z.ToString(CultureInfo.InvariantCulture)}");
                }
                
                // Add bounding box information as a comment
                sb.AppendLine($"# Bounding Box: Min({minX:F2}, {minY:F2}, {minZ:F2}) Max({maxX:F2}, {maxY:F2}, {maxZ:F2})");
                sb.AppendLine($"# Dimensions: W({maxX - minX:F2}) H({maxY - minY:F2}) D({maxZ - minZ:F2})");
            }

            sb.AppendLine();

            // Add faces (triangles) if available
            if (pm4File.VertexIndicesChunk != null && pm4File.VertexIndicesChunk.Indices.Count > 0)
            {
                int triangleCount = pm4File.VertexIndicesChunk.Indices.Count / 3;
                _logger?.LogDebug("Writing {Count} triangles", triangleCount);

                // Apply material
                sb.AppendLine("usemtl default");
                sb.AppendLine("g mesh");

                for (int i = 0; i < triangleCount; i++)
                {
                    int baseIdx = i * 3;
                    uint v1 = pm4File.VertexIndicesChunk.Indices[baseIdx] + 1; // +1 because OBJ uses 1-based indexing
                    uint v2 = pm4File.VertexIndicesChunk.Indices[baseIdx + 1] + 1;
                    uint v3 = pm4File.VertexIndicesChunk.Indices[baseIdx + 2] + 1;
                    sb.AppendLine($"f {v1} {v2} {v3}");
                }
            }

            // Add position data as separate objects, sorted by SpecialValueDec
            if (pm4File.PositionDataChunk != null)
            {
                var positionEntries = pm4File.PositionDataChunk.Entries;
                
                // Group position records by their associated special values
                var positionDataByCategory = new Dictionary<string, List<MPRLChunk.ServerPositionData>>();
                
                for (int i = 0; i < positionEntries.Count; i++)
                {
                    var entry = positionEntries[i];
                    
                    if (!entry.IsSpecialEntry)
                    {
                        int specialValue = 0;
                        
                        // Check if the previous entry is a special entry to get its value
                        if (i > 0 && positionEntries[i - 1].IsSpecialEntry)
                        {
                            specialValue = positionEntries[i - 1].SpecialValue;
                        }
                        
                        string category = GetPositionCategory(specialValue);
                        
                        if (!positionDataByCategory.ContainsKey(category))
                        {
                            positionDataByCategory[category] = new List<MPRLChunk.ServerPositionData>();
                        }
                        
                        positionDataByCategory[category].Add(entry);
                    }
                }
                
                // Add each category as a separate object with its own material
                foreach (var categoryGroup in positionDataByCategory)
                {
                    sb.AppendLine();
                    sb.AppendLine($"# Position data points - Category: {categoryGroup.Key}");
                    sb.AppendLine($"o {objName}_PositionData_{categoryGroup.Key}");
                    sb.AppendLine($"usemtl {categoryGroup.Key.ToLowerInvariant()}");
                    
                    _logger?.LogDebug("Writing {Count} position data points with category {Category}", 
                        categoryGroup.Value.Count, categoryGroup.Key);
                    
                    int vertexStart = pm4File.VertexPositionsChunk?.Vertices.Count ?? 0;
                    vertexStart += 1; // 1-based indexing
                    
                    // Sort positions by SpecialValueDec within each category
                    var sortedPositions = categoryGroup.Value
                        .OrderBy(p => {
                            // Find the special value associated with this position
                            int index = positionEntries.IndexOf(p);
                            if (index > 0 && positionEntries[index - 1].IsSpecialEntry)
                                return positionEntries[index - 1].SpecialValue;
                            return 0;
                        })
                        .ToList();
                    
                    foreach (var pos in sortedPositions)
                    {
                        // Write vertex with proper coordinate ordering (X and Z preserved, Y is up)
                        sb.AppendLine($"v {pos.CoordinateX.ToString(CultureInfo.InvariantCulture)} {pos.CoordinateY.ToString(CultureInfo.InvariantCulture)} {pos.CoordinateZ.ToString(CultureInfo.InvariantCulture)}");
                    }
                    
                    // Draw position points
                    int pointCount = sortedPositions.Count;
                    
                    sb.Append("p");
                    for (int i = 0; i < pointCount; i++)
                    {
                        sb.Append($" {vertexStart + i}");
                    }
                    sb.AppendLine();
                    
                    // Update vertex start for the next category
                    vertexStart += pointCount;
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates OBJ content from multiple PM4 files into one consolidated file with enhanced
        /// coordinate handling and position categorization.
        /// </summary>
        /// <param name="results">The PM4 analysis results to convert to OBJ format.</param>
        /// <returns>A string containing the consolidated OBJ file content.</returns>
        private string GenerateConsolidatedObjContent(List<PM4AnalysisResult> results)
        {
            var sb = new StringBuilder();

            // Add a header comment
            sb.AppendLine($"# Enhanced consolidated OBJ file generated from {results.Count} PM4 files");
            sb.AppendLine($"# Generated by WCAnalyzer Enhanced OBJ Exporter on {DateTime.Now}");
            sb.AppendLine($"# This file contains all geometry from multiple PM4 files");
            sb.AppendLine($"# Each PM4 file is represented as a separate object with its own vertices and faces");
            sb.AppendLine($"# Coordinate system: X and Z are preserved, Y is up");
            sb.AppendLine($"# Position data is categorized by SpecialValueDec and sorted within each category");
            sb.AppendLine();

            // Add a material library reference
            sb.AppendLine("mtllib pm4_materials.mtl");
            sb.AppendLine();

            int vertexOffset = 1; // OBJ uses 1-based indexing

            // Dictionary to collect all position data by category across all files
            var allPositionDataByCategory = new Dictionary<string, List<(string fileName, MPRLChunk.ServerPositionData position, int specialValue)>>();

            // First pass: collect and process mesh data
            foreach (var result in results)
            {
                if (result.PM4File == null || string.IsNullOrEmpty(result.FileName))
                    continue;

                string objName = Path.GetFileNameWithoutExtension(result.FileName);
                
                // Extract coordinates
                var coordinates = ExtractCoordinatesFromFilename(result.FileName);
                
                // Add a new object for this file
                sb.AppendLine($"o {objName}");
                
                // Add comments with extra file information
                sb.AppendLine($"# File: {result.FileName}");
                sb.AppendLine($"# Path: {result.FilePath}");
                if (coordinates.x >= 0 && coordinates.y >= 0)
                {
                    sb.AppendLine($"# Terrain coordinates: {coordinates.x}, {coordinates.y}");
                }
                
                bool hasVertices = false;
                int vertexCount = 0;
                
                // Add vertices from this file
                if (result.PM4File.VertexPositionsChunk != null)
                {
                    var vertices = result.PM4File.VertexPositionsChunk.Vertices;
                    vertexCount = vertices.Count;
                    hasVertices = vertexCount > 0;
                    
                    if (hasVertices)
                    {
                        _logger?.LogDebug("Adding {Count} vertices from {File}", vertexCount, result.FileName);
                        
                        // Extract bounding box information for context
                        float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
                        float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
                        
                        foreach (var vertex in vertices)
                        {
                            // Track bounding box
                            minX = Math.Min(minX, vertex.X);
                            minY = Math.Min(minY, vertex.Y);
                            minZ = Math.Min(minZ, vertex.Z);
                            maxX = Math.Max(maxX, vertex.X);
                            maxY = Math.Max(maxY, vertex.Y);
                            maxZ = Math.Max(maxZ, vertex.Z);
                            
                            // Preserve X and Z, keep Y as up
                            sb.AppendLine($"v {vertex.X.ToString(CultureInfo.InvariantCulture)} " +
                                         $"{vertex.Y.ToString(CultureInfo.InvariantCulture)} " +
                                         $"{vertex.Z.ToString(CultureInfo.InvariantCulture)}");
                        }
                        
                        sb.AppendLine($"# Bounding Box: Min({minX:F2}, {minY:F2}, {minZ:F2}) Max({maxX:F2}, {maxY:F2}, {maxZ:F2})");
                    }
                }
                
                // Add indices (faces) if available
                if (hasVertices && result.PM4File.VertexIndicesChunk != null)
                {
                    int triangleCount = result.PM4File.VertexIndicesChunk.Indices.Count / 3;
                    
                    if (triangleCount > 0)
                    {
                        _logger?.LogDebug("Adding {Count} triangles from {File}", triangleCount, result.FileName);

                        // Group faces by material if available
                        string materialName = $"material_{objName}";
                        sb.AppendLine($"usemtl {materialName}");
                        sb.AppendLine($"g {objName}_mesh");
                        
                        // In OBJ format, vertex indices start at 1, not 0
                        for (int i = 0; i < triangleCount; i++)
                        {
                            int baseIndex = i * 3;
                            if (baseIndex + 2 < result.PM4File.VertexIndicesChunk.Indices.Count)
                            {
                                // OBJ uses 1-based indexing, so add vertex offset to all indices
                                uint v1 = result.PM4File.VertexIndicesChunk.Indices[baseIndex] + (uint)vertexOffset;
                                uint v2 = result.PM4File.VertexIndicesChunk.Indices[baseIndex + 1] + (uint)vertexOffset;
                                uint v3 = result.PM4File.VertexIndicesChunk.Indices[baseIndex + 2] + (uint)vertexOffset;
                                
                                // Format: f v1 v2 v3
                                sb.AppendLine($"f {v1} {v2} {v3}");
                            }
                        }
                    }
                }
                
                // Collect position data for categorization (not writing it yet)
                if (result.PM4File.PositionDataChunk != null)
                {
                    var positionEntries = result.PM4File.PositionDataChunk.Entries;
                    
                    for (int i = 0; i < positionEntries.Count; i++)
                    {
                        var entry = positionEntries[i];
                        
                        if (!entry.IsSpecialEntry)
                        {
                            int specialValue = 0;
                            
                            // Check if the previous entry is a special entry to get its value
                            if (i > 0 && positionEntries[i - 1].IsSpecialEntry)
                            {
                                specialValue = positionEntries[i - 1].SpecialValue;
                            }
                            
                            string category = GetPositionCategory(specialValue);
                            
                            if (!allPositionDataByCategory.ContainsKey(category))
                            {
                                allPositionDataByCategory[category] = new List<(string, MPRLChunk.ServerPositionData, int)>();
                            }
                            
                            allPositionDataByCategory[category].Add((result.FileName, entry, specialValue));
                        }
                    }
                }
                
                // Update the vertex offset for the next file
                vertexOffset += vertexCount;
            }

            // Second pass: add all position data organized by category
            sb.AppendLine();
            sb.AppendLine("# Position data organized by category and sorted by SpecialValueDec");
            
            foreach (var categoryGroup in allPositionDataByCategory)
            {
                sb.AppendLine();
                sb.AppendLine($"# Position data points - Category: {categoryGroup.Key}");
                sb.AppendLine($"o PositionData_{categoryGroup.Key}");
                sb.AppendLine($"usemtl {categoryGroup.Key.ToLowerInvariant()}");
                
                _logger?.LogDebug("Writing {Count} consolidated position data points with category {Category}", 
                    categoryGroup.Value.Count, categoryGroup.Key);
                
                // Sort positions by SpecialValueDec within each category
                var sortedPositions = categoryGroup.Value
                    .OrderBy(p => p.specialValue)
                    .ThenBy(p => p.fileName)
                    .ToList();
                
                foreach (var (fileName, pos, _) in sortedPositions)
                {
                    // Write vertex with proper coordinate ordering (X and Z preserved, Y is up)
                    sb.AppendLine($"v {pos.CoordinateX.ToString(CultureInfo.InvariantCulture)} " +
                                $"{pos.CoordinateY.ToString(CultureInfo.InvariantCulture)} " +
                                $"{pos.CoordinateZ.ToString(CultureInfo.InvariantCulture)}");
                }
                
                // Draw position points
                int pointCount = sortedPositions.Count;
                
                sb.Append("p");
                for (int i = 0; i < pointCount; i++)
                {
                    sb.Append($" {vertexOffset + i}");
                }
                sb.AppendLine();
                
                // Update vertex offset for the next category
                vertexOffset += pointCount;
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates and writes an enhanced material file for the OBJ files.
        /// </summary>
        /// <param name="mtlPath">The path where to write the MTL file.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        private async Task GenerateEnhancedMaterialFileAsync(string mtlPath)
        {
            try
            {
                using var writer = new StreamWriter(mtlPath);
                
                // Write MTL header
                await writer.WriteLineAsync("# Material definitions for enhanced PM4 OBJ files");
                await writer.WriteLineAsync($"# Generated by WCAnalyzer on {DateTime.Now}");
                await writer.WriteLineAsync();
                
                // Default material
                await writer.WriteLineAsync("newmtl default");
                await writer.WriteLineAsync("Ka 0.8 0.8 0.8");
                await writer.WriteLineAsync("Kd 0.8 0.8 0.8");
                await writer.WriteLineAsync("Ks 0.1 0.1 0.1");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Standard waypoint material
                await writer.WriteLineAsync("newmtl standard");
                await writer.WriteLineAsync("Ka 0.2 0.6 0.2");
                await writer.WriteLineAsync("Kd 0.3 0.8 0.3");
                await writer.WriteLineAsync("Ks 0.1 0.3 0.1");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Special waypoint material
                await writer.WriteLineAsync("newmtl special");
                await writer.WriteLineAsync("Ka 0.6 0.2 0.2");
                await writer.WriteLineAsync("Kd 0.8 0.3 0.3");
                await writer.WriteLineAsync("Ks 0.3 0.1 0.1");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Flagged waypoint material
                await writer.WriteLineAsync("newmtl flagged");
                await writer.WriteLineAsync("Ka 0.6 0.6 0.2");
                await writer.WriteLineAsync("Kd 0.8 0.8 0.3");
                await writer.WriteLineAsync("Ks 0.3 0.3 0.1");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Path node waypoint material
                await writer.WriteLineAsync("newmtl pathnode");
                await writer.WriteLineAsync("Ka 0.2 0.2 0.6");
                await writer.WriteLineAsync("Kd 0.3 0.3 0.8");
                await writer.WriteLineAsync("Ks 0.1 0.1 0.3");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Waypoint node material
                await writer.WriteLineAsync("newmtl waypointnode");
                await writer.WriteLineAsync("Ka 0.6 0.2 0.6");
                await writer.WriteLineAsync("Kd 0.8 0.3 0.8");
                await writer.WriteLineAsync("Ks 0.3 0.1 0.3");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Spawn point material
                await writer.WriteLineAsync("newmtl spawnpoint");
                await writer.WriteLineAsync("Ka 0.2 0.6 0.6");
                await writer.WriteLineAsync("Kd 0.3 0.8 0.8");
                await writer.WriteLineAsync("Ks 0.1 0.3 0.3");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                await writer.WriteLineAsync();
                
                // Generic category material
                await writer.WriteLineAsync("newmtl category_");
                await writer.WriteLineAsync("Ka 0.5 0.5 0.5");
                await writer.WriteLineAsync("Kd 0.7 0.7 0.7");
                await writer.WriteLineAsync("Ks 0.2 0.2 0.2");
                await writer.WriteLineAsync("Ns 10");
                await writer.WriteLineAsync("d 1.0");
                await writer.WriteLineAsync("illum 2");
                
                _logger?.LogInformation("Generated enhanced material file: {MtlPath}", mtlPath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating enhanced material file: {MtlPath}", mtlPath);
            }
        }
    }
} 