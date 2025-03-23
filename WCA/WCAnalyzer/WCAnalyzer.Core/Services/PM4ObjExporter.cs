using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Numerics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for exporting PM4 mesh data (vertices and triangles) to Wavefront OBJ format.
    /// </summary>
    public class PM4ObjExporter
    {
        private readonly ILogger<PM4ObjExporter>? _logger;
        private readonly string _outputDirectory;
        private static readonly Regex CoordinatePattern = new Regex(@"_(\d+)_(\d+)", RegexOptions.Compiled);

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4ObjExporter"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Output directory for the OBJ files</param>
        public PM4ObjExporter(ILogger<PM4ObjExporter>? logger, string outputDirectory)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
        }

        /// <summary>
        /// Exports a PM4 file to OBJ format.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportToObjAsync(PM4AnalysisResult result)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (result.PM4File == null)
            {
                _logger?.LogWarning("Cannot export {FileName} to OBJ: PM4File is null", result.FileName);
                return;
            }

            // Check if we have the necessary data to generate an OBJ file
            if (result.PM4File.VertexPositionsChunk == null || result.PM4File.VertexPositionsChunk.Vertices.Count == 0)
            {
                _logger?.LogWarning("Cannot export {FileName} to OBJ: No vertices found", result.FileName);
                return;
            }

            try
            {
                // Create output directory if it doesn't exist
                var objDir = Path.Combine(_outputDirectory, "pm4_obj");
                Directory.CreateDirectory(objDir);

                string baseFileName = Path.GetFileNameWithoutExtension(result.FileName);
                string objFilePath = Path.Combine(objDir, $"{baseFileName}.obj");

                _logger?.LogInformation("Exporting {FileName} to OBJ format: {ObjFilePath}", result.FileName, objFilePath);

                // Extract terrain coordinates from filename
                var coordinates = ExtractCoordinatesFromFilename(result.FileName);
                _logger?.LogDebug("Extracted coordinates: ({X}, {Y})", coordinates.x, coordinates.y);

                // Build the OBJ content
                var objContent = GenerateObjContent(result.PM4File, coordinates);

                // Write the OBJ file
                await File.WriteAllTextAsync(objFilePath, objContent);

                _logger?.LogInformation("Successfully exported {FileName} to OBJ format", result.FileName);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting {FileName} to OBJ format", result.FileName);
            }
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

            _logger?.LogInformation("Starting batch export of PM4 files to OBJ format");

            foreach (var result in results)
            {
                await ExportToObjAsync(result);
            }

            _logger?.LogInformation("Batch export of PM4 files to OBJ format completed");
        }

        /// <summary>
        /// Exports all PM4 files to a single consolidated OBJ file.
        /// </summary>
        /// <param name="results">The PM4 analysis results to export.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task ExportToConsolidatedObjAsync(IEnumerable<PM4AnalysisResult> results)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            var resultsList = results.Where(r => r.PM4File != null).ToList();

            if (resultsList.Count == 0)
            {
                _logger?.LogWarning("No PM4 files to export to consolidated OBJ format");
                return;
            }

            try
            {
                // Create output directory if it doesn't exist
                var objDir = Path.Combine(_outputDirectory, "pm4_obj");
                Directory.CreateDirectory(objDir);

                string objFilePath = Path.Combine(objDir, "consolidated_pm4.obj");
                string mtlFilePath = Path.Combine(objDir, "pm4_materials.mtl");

                _logger?.LogInformation("Exporting {Count} PM4 files to consolidated OBJ format: {ObjFilePath}", resultsList.Count, objFilePath);

                // Build the OBJ content
                var objContent = GenerateConsolidatedObjContent(resultsList);

                // Write the OBJ file
                await File.WriteAllTextAsync(objFilePath, objContent);
                
                // Generate and write the MTL file
                await GenerateConsolidatedMaterialFileAsync(mtlFilePath);

                _logger?.LogInformation("Successfully exported {Count} PM4 files to consolidated OBJ file", resultsList.Count);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error exporting PM4 files to consolidated OBJ format");
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
        /// Generates OBJ content from a PM4 file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to convert to OBJ format.</param>
        /// <param name="coordinates">Optional coordinates for additional context.</param>
        /// <returns>A string containing the OBJ file content.</returns>
        private string GenerateObjContent(PM4File pm4File, (int x, int y) coordinates = default)
        {
            var sb = new StringBuilder();

            // Add a header comment
            sb.AppendLine($"# OBJ file generated from PM4 file: {pm4File.FileName}");
            sb.AppendLine($"# Generated by WCAnalyzer on {DateTime.Now}");
            if (coordinates.x >= 0 && coordinates.y >= 0)
            {
                sb.AppendLine($"# Terrain coordinates: {coordinates.x}, {coordinates.y}");
            }
            sb.AppendLine();

            // Add material library reference for future use
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
                sb.AppendLine($"g {objName}_mesh");

                // In OBJ format, vertex indices start at 1, not 0
                for (int i = 0; i < triangleCount; i++)
                {
                    int baseIndex = i * 3;
                    if (baseIndex + 2 < pm4File.VertexIndicesChunk.Indices.Count)
                    {
                        // OBJ uses 1-based indexing, so add 1 to all indices
                        uint v1 = pm4File.VertexIndicesChunk.Indices[baseIndex] + 1;
                        uint v2 = pm4File.VertexIndicesChunk.Indices[baseIndex + 1] + 1;
                        uint v3 = pm4File.VertexIndicesChunk.Indices[baseIndex + 2] + 1;
                        
                        sb.AppendLine($"f {v1} {v2} {v3}");
                    }
                }
            }
            else
            {
                _logger?.LogWarning("No triangle data available in {FileName}, OBJ will only contain vertices", pm4File.FileName);
            }

            // Add position data as a separate object if available
            if (pm4File.PositionDataChunk != null)
            {
                var positionRecords = pm4File.PositionDataChunk.Entries.Where(e => !e.IsSpecialEntry).ToList();
                if (positionRecords.Count > 0)
                {
                    sb.AppendLine();
                    sb.AppendLine("# Server position data points (not part of the mesh)");
                    sb.AppendLine($"o {objName}_PositionData");
                    sb.AppendLine("usemtl positionData");
                    
                    _logger?.LogDebug("Writing {Count} position data points", positionRecords.Count);
                    
                    int positionVertexStart = pm4File.VertexPositionsChunk?.Vertices.Count ?? 0;
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
                
                // Add command records as a separate object
                var commandRecords = pm4File.PositionDataChunk.Entries.Where(e => e.IsSpecialEntry).ToList();
                if (commandRecords.Count > 0)
                {
                    sb.AppendLine();
                    sb.AppendLine("# Command data points");
                    sb.AppendLine($"o {objName}_CommandData");
                    sb.AppendLine("usemtl commandData");
                    
                    _logger?.LogDebug("Writing {Count} command data points", commandRecords.Count);
                    
                    int commandVertexStart = pm4File.VertexPositionsChunk?.Vertices.Count ?? 0;
                    commandVertexStart += positionRecords.Count;
                    commandVertexStart += 1; // 1-based indexing
                    
                    int validCommandPoints = 0;
                    List<int> validIndices = new List<int>();
                    
                    foreach (var cmd in commandRecords)
                    {
                        float x = cmd.Value1;
                        float y = cmd.CoordinateY;
                        float z = cmd.Value3;
                        
                        if (!float.IsNaN(x) && !float.IsNaN(y) && !float.IsNaN(z))
                        {
                            sb.AppendLine($"v {x.ToString(CultureInfo.InvariantCulture)} {y.ToString(CultureInfo.InvariantCulture)} {z.ToString(CultureInfo.InvariantCulture)}");
                            validIndices.Add(commandVertexStart + validCommandPoints);
                            validCommandPoints++;
                        }
                    }
                    
                    // Add point drawing commands if we have valid points
                    if (validCommandPoints > 0)
                    {
                        sb.Append("p");
                        foreach (int idx in validIndices)
                        {
                            sb.Append($" {idx}");
                        }
                        sb.AppendLine();
                    }
                    
                    _logger?.LogDebug("Added {Count} valid command points from {Total} total commands", validCommandPoints, commandRecords.Count);
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates OBJ content from multiple PM4 files into one consolidated file.
        /// </summary>
        /// <param name="results">The PM4 analysis results to convert to OBJ format.</param>
        /// <returns>A string containing the consolidated OBJ file content.</returns>
        private string GenerateConsolidatedObjContent(List<PM4AnalysisResult> results)
        {
            var sb = new StringBuilder();

            // Add a header comment
            sb.AppendLine($"# Consolidated OBJ file generated from {results.Count} PM4 files");
            sb.AppendLine($"# Generated by WCAnalyzer on {DateTime.Now}");
            sb.AppendLine($"# This file contains all geometry from multiple PM4 files");
            sb.AppendLine($"# Each PM4 file is represented as a separate object with its own vertices and faces");
            sb.AppendLine($"# Coordinate system: Using original X, Y, Z coordinates (not transformed)");
            sb.AppendLine();

            // Add a material library reference
            sb.AppendLine("mtllib pm4_materials.mtl");
            sb.AppendLine();

            int vertexOffset = 1; // OBJ uses 1-based indexing

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
                            
                            // Use original coordinates (no transformation)
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
                        
                        sb.AppendLine();
                    }
                }
                
                // Update offsets for the next file
                vertexOffset += vertexCount;
                
                // Add position data as a separate object if available
                if (result.PM4File.PositionDataChunk != null)
                {
                    var positionRecords = result.PM4File.PositionDataChunk.Entries.Where(e => !e.IsSpecialEntry).ToList();
                    if (positionRecords.Count > 0)
                    {
                        // Group position data by special value category for better visualization
                        var categorizedPositions = positionRecords
                            .Select(p => {
                                // Get special value from preceding entry if available
                                int specialValue = 0;
                                int specialIndex = -1;
                                
                                if (p.Index > 0 && result.PM4File.PositionDataChunk.Entries.Count > p.Index - 1)
                                {
                                    var prevEntry = result.PM4File.PositionDataChunk.Entries[p.Index - 1];
                                    if (prevEntry.IsSpecialEntry)
                                    {
                                        specialValue = prevEntry.SpecialValue;
                                        specialIndex = prevEntry.Index;
                                    }
                                }
                                
                                // Determine category
                                string category = "Standard";
                                if (specialValue == 2) category = "Special";
                                else if ((specialValue & 0x1) != 0) category = "Flagged";
                                else if (specialValue != 0) category = $"Category_{specialValue:X}";
                                
                                return new { 
                                    Position = p, 
                                    SpecialValue = specialValue,
                                    SpecialIndex = specialIndex,
                                    Category = category 
                                };
                            })
                            .GroupBy(p => p.Category)
                            .OrderBy(g => g.Key)
                            .ToList();
                        
                        // Process each category separately
                        foreach (var category in categorizedPositions)
                        {
                            sb.AppendLine($"o {objName}_PositionData_{category.Key}");
                            sb.AppendLine($"usemtl {category.Key.ToLowerInvariant()}");
                            
                            _logger?.LogDebug("Adding {Count} {Category} position data points from {File}", 
                                category.Count(), category.Key, result.FileName);
                            
                            // Add position data points using original coordinates (no transformation)
                            foreach (var item in category)
                            {
                                var pos = item.Position;
                                sb.AppendLine($"v {pos.CoordinateX.ToString(CultureInfo.InvariantCulture)} " +
                                            $"{pos.CoordinateY.ToString(CultureInfo.InvariantCulture)} " +
                                            $"{pos.CoordinateZ.ToString(CultureInfo.InvariantCulture)}");
                            }
                            
                            // Add point drawing commands
                            sb.Append("p");
                            for (int i = 0; i < category.Count(); i++)
                            {
                                sb.Append($" {vertexOffset + i}");
                            }
                            sb.AppendLine();
                            
                            // Update vertex offset for this category
                            vertexOffset += category.Count();
                            
                            sb.AppendLine();
                        }
                    }
                    
                    // Add command records as points (optional, can skip if preferred)
                    var commandRecords = result.PM4File.PositionDataChunk.Entries.Where(e => e.IsSpecialEntry).ToList();
                    if (commandRecords.Count > 0)
                    {
                        sb.AppendLine($"o {objName}_CommandRecords");
                        sb.AppendLine("usemtl commands");
                        
                        _logger?.LogDebug("Adding {Count} command records from {File}", commandRecords.Count, result.FileName);
                        
                        foreach (var cmd in commandRecords)
                        {
                            // For command records, use Y as the height and make a flat point at (0,Y,0)
                            sb.AppendLine($"v 0 {cmd.CoordinateY.ToString(CultureInfo.InvariantCulture)} 0");
                        }
                        
                        // Add point drawing commands
                        sb.Append("p");
                        for (int i = 0; i < commandRecords.Count; i++)
                        {
                            sb.Append($" {vertexOffset + i}");
                        }
                        sb.AppendLine();
                        
                        // Update vertex offset
                        vertexOffset += commandRecords.Count;
                        
                        sb.AppendLine();
                    }
                }
            }
            
            // Add material definitions
            sb.AppendLine("# Define materials");
            sb.AppendLine("usemtl standard");
            sb.AppendLine("  Kd 0.8 0.8 0.8");
            sb.AppendLine("usemtl special");
            sb.AppendLine("  Kd 1.0 0.5 0.5");
            sb.AppendLine("usemtl flagged");
            sb.AppendLine("  Kd 0.5 0.5 1.0");
            sb.AppendLine("usemtl commands");
            sb.AppendLine("  Kd 1.0 1.0 0.5");

            return sb.ToString();
        }

        /// <summary>
        /// Generates a material library file (MTL) for use with consolidated OBJ exports.
        /// </summary>
        /// <param name="mtlPath">The path to write the MTL file to.</param>
        private async Task GenerateConsolidatedMaterialFileAsync(string mtlPath)
        {
            try
            {
                using (var writer = new StreamWriter(mtlPath))
                {
                    await writer.WriteLineAsync("# Material library for PM4 visualization");
                    await writer.WriteLineAsync("# Generated by WCAnalyzer");
                    await writer.WriteLineAsync();
                    
                    // Standard waypoints (category 0)
                    await writer.WriteLineAsync("newmtl standard");
                    await writer.WriteLineAsync("Ka 0.3 0.8 0.3");
                    await writer.WriteLineAsync("Kd 0.3 1.0 0.3");
                    await writer.WriteLineAsync("Ks 0.5 0.5 0.5");
                    await writer.WriteLineAsync("Ns 32.0");
                    await writer.WriteLineAsync();
                    
                    // Special waypoints (category 2)
                    await writer.WriteLineAsync("newmtl special");
                    await writer.WriteLineAsync("Ka 0.8 0.3 0.3");
                    await writer.WriteLineAsync("Kd 1.0 0.3 0.3");
                    await writer.WriteLineAsync("Ks 0.5 0.5 0.5");
                    await writer.WriteLineAsync("Ns 32.0");
                    await writer.WriteLineAsync();
                    
                    // Flagged waypoints (bit 0x1 set)
                    await writer.WriteLineAsync("newmtl flagged");
                    await writer.WriteLineAsync("Ka 0.3 0.3 0.8");
                    await writer.WriteLineAsync("Kd 0.3 0.3 1.0");
                    await writer.WriteLineAsync("Ks 0.5 0.5 0.5");
                    await writer.WriteLineAsync("Ns 32.0");
                    await writer.WriteLineAsync();
                    
                    // Command data points
                    await writer.WriteLineAsync("newmtl commands");
                    await writer.WriteLineAsync("Ka 0.8 0.8 0.3");
                    await writer.WriteLineAsync("Kd 1.0 1.0 0.3");
                    await writer.WriteLineAsync("Ks 0.5 0.5 0.5");
                    await writer.WriteLineAsync("Ns 32.0");
                    await writer.WriteLineAsync();
                    
                    // Other categories (generic)
                    for (int i = 1; i <= 10; i++)
                    {
                        if (i != 2) // Skip 2 which is already defined as "special"
                        {
                            float r = (i % 3) * 0.33f + 0.34f;
                            float g = ((i+1) % 3) * 0.33f + 0.34f;
                            float b = ((i+2) % 3) * 0.33f + 0.34f;
                            
                            await writer.WriteLineAsync($"newmtl category_{i:X}");
                            await writer.WriteLineAsync($"Ka {r:F2} {g:F2} {b:F2}");
                            await writer.WriteLineAsync($"Kd {r:F2} {g:F2} {b:F2}");
                            await writer.WriteLineAsync("Ks 0.5 0.5 0.5");
                            await writer.WriteLineAsync("Ns 32.0");
                            await writer.WriteLineAsync();
                        }
                    }
                }
                
                _logger?.LogInformation("Generated material library file: {FilePath}", mtlPath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating material library file: {FilePath}", mtlPath);
            }
        }

        /// <summary>
        /// Exports terrain position data to a point-cloud visualization file in OBJ format.
        /// Each point will be represented as a small sphere for better visibility.
        /// </summary>
        /// <param name="positions">List of 2D positions (x,y)</param>
        /// <param name="outputPath">Path to save the OBJ file</param>
        /// <returns>True if export was successful</returns>
        public bool ExportTerrainMeshFromPositions(List<Vector2> positions, string outputPath)
        {
            try
            {
                // Create material file first
                string materialFilePath = Path.ChangeExtension(outputPath, ".mtl");
                using (var mtlWriter = new StreamWriter(materialFilePath))
                {
                    mtlWriter.WriteLine("# Material file for terrain point cloud visualization");
                    mtlWriter.WriteLine("newmtl pointMaterial");
                    mtlWriter.WriteLine("Ka 0.2 0.2 0.8");
                    mtlWriter.WriteLine("Kd 0.4 0.4 1.0");
                    mtlWriter.WriteLine("Ks 0.7 0.7 1.0");
                    mtlWriter.WriteLine("d 1.0");
                    mtlWriter.WriteLine("illum 2");
                    mtlWriter.WriteLine("Ns 80");
                }

                using var writer = new StreamWriter(outputPath);
                var sb = new StringBuilder();

                // Write OBJ header
                sb.AppendLine("# Terrain point cloud from 2D positions (Azeroth map, March 2002)");
                sb.AppendLine("# Generated by WCAnalyzer");
                sb.AppendLine($"# Total points: {positions.Count}");
                sb.AppendLine();
                
                // Reference the material file
                sb.AppendLine($"mtllib {Path.GetFileName(materialFilePath)}");
                sb.AppendLine();

                // Set the point radius based on data density
                float pointRadius = CalculateOptimalPointRadius(positions);
                _logger?.LogInformation("Using point radius of {Radius:F4} for visualization", pointRadius);

                // Generate the point cloud
                int pointIndex = 1; // OBJ indices start at 1
                
                foreach (var position in positions)
                {
                    // Create a named object for each point
                    sb.AppendLine($"o point_{pointIndex}");
                    sb.AppendLine("usemtl pointMaterial");
                    
                    // Create a small sphere for each point
                    // Base vertex position
                    float x = position.X;
                    float y = 0; // Use 0 for the elevation since this is a 2D map
                    float z = position.Y;
                    
                    // Generate a small sphere (simplified with 8 vertices)
                    sb.AppendLine($"v {x.ToString("F6", CultureInfo.InvariantCulture)} {y.ToString("F6", CultureInfo.InvariantCulture)} {z.ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {(x+pointRadius).ToString("F6", CultureInfo.InvariantCulture)} {y.ToString("F6", CultureInfo.InvariantCulture)} {z.ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {x.ToString("F6", CultureInfo.InvariantCulture)} {(y+pointRadius).ToString("F6", CultureInfo.InvariantCulture)} {z.ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {x.ToString("F6", CultureInfo.InvariantCulture)} {y.ToString("F6", CultureInfo.InvariantCulture)} {(z+pointRadius).ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {(x-pointRadius).ToString("F6", CultureInfo.InvariantCulture)} {y.ToString("F6", CultureInfo.InvariantCulture)} {z.ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {x.ToString("F6", CultureInfo.InvariantCulture)} {(y-pointRadius).ToString("F6", CultureInfo.InvariantCulture)} {z.ToString("F6", CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"v {x.ToString("F6", CultureInfo.InvariantCulture)} {y.ToString("F6", CultureInfo.InvariantCulture)} {(z-pointRadius).ToString("F6", CultureInfo.InvariantCulture)}");
                    
                    // Create triangles to form a basic octahedron
                    int baseIndex = (pointIndex - 1) * 7 + 1;
                    sb.AppendLine($"f {baseIndex} {baseIndex+1} {baseIndex+2}");
                    sb.AppendLine($"f {baseIndex} {baseIndex+2} {baseIndex+3}");
                    sb.AppendLine($"f {baseIndex} {baseIndex+3} {baseIndex+4}");
                    sb.AppendLine($"f {baseIndex} {baseIndex+4} {baseIndex+5}");
                    sb.AppendLine($"f {baseIndex} {baseIndex+5} {baseIndex+6}");
                    sb.AppendLine($"f {baseIndex} {baseIndex+6} {baseIndex+1}");
                    
                    pointIndex++;
                }

                writer.Write(sb.ToString());
                
                // Also generate a simpler visualization that can be loaded faster
                GenerateSimplePointMapVisualization(positions, Path.ChangeExtension(outputPath, ".simple.obj"));
                
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to export terrain point cloud: {Message}", ex.Message);
                return false;
            }
        }
        
        /// <summary>
        /// Generates a simpler visualization with just dots for faster loading with large datasets
        /// </summary>
        private bool GenerateSimplePointMapVisualization(List<Vector2> positions, string outputPath)
        {
            try
            {
                // Calculate bounding box
                var stats = AnalyzePositionGrid(positions);
                float width = stats.maxX - stats.minX;
                float height = stats.maxY - stats.minY;
                
                using var writer = new StreamWriter(outputPath);
                writer.WriteLine("# Simple point map visualization");
                writer.WriteLine($"# Total points: {positions.Count}");
                writer.WriteLine();
                
                // Export all vertices first
                foreach (var position in positions)
                {
                    writer.WriteLine($"v {position.X.ToString("F6", CultureInfo.InvariantCulture)} 0.0 {position.Y.ToString("F6", CultureInfo.InvariantCulture)}");
                }
                
                writer.WriteLine();
                writer.WriteLine("# Point visualization as vertices only (use with vertex display enabled in 3D viewer)");
                writer.WriteLine("o PointMap");
                writer.WriteLine("g points");
                
                // Create a bounding box to help with visualization
                writer.WriteLine();
                writer.WriteLine("# Bounding box outline");
                writer.WriteLine("v {0} 0.0 {1}", stats.minX.ToString("F6", CultureInfo.InvariantCulture), stats.minY.ToString("F6", CultureInfo.InvariantCulture));
                writer.WriteLine("v {0} 0.0 {1}", stats.maxX.ToString("F6", CultureInfo.InvariantCulture), stats.minY.ToString("F6", CultureInfo.InvariantCulture));
                writer.WriteLine("v {0} 0.0 {1}", stats.maxX.ToString("F6", CultureInfo.InvariantCulture), stats.maxY.ToString("F6", CultureInfo.InvariantCulture));
                writer.WriteLine("v {0} 0.0 {1}", stats.minX.ToString("F6", CultureInfo.InvariantCulture), stats.maxY.ToString("F6", CultureInfo.InvariantCulture));
                
                int baseIndex = positions.Count + 1;
                writer.WriteLine("l {0} {1} {2} {3} {0}", baseIndex, baseIndex+1, baseIndex+2, baseIndex+3);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to export simple point map: {Message}", ex.Message);
                return false;
            }
        }
        
        /// <summary>
        /// Calculates an optimal point radius based on the density and distribution of points
        /// </summary>
        private float CalculateOptimalPointRadius(List<Vector2> positions)
        {
            if (positions.Count <= 1) return 0.1f;
            
            var stats = AnalyzePositionGrid(positions);
            
            // Calculate average spacing between points
            float avgSpacing = (stats.avgSpacingX + stats.avgSpacingY) / 2.0f;
            
            // Use a fraction of the average spacing for the radius
            float radius = avgSpacing * 0.2f;
            
            // Ensure the radius is reasonable (not too small or large)
            return Math.Max(0.01f, Math.Min(radius, 1.0f));
        }

        /// <summary>
        /// Extracts position data from PM4 files and exports it as a point cloud visualization.
        /// </summary>
        /// <param name="results">The PM4 analysis results to process.</param>
        /// <param name="outputPath">Path to save the OBJ file.</param>
        /// <returns>True if export was successful.</returns>
        public bool ExportTerrainFromPositionData(IEnumerable<PM4AnalysisResult> results, string outputPath)
        {
            try
            {
                var allPositions = new List<Vector2>();

                foreach (var result in results)
                {
                    if (result.PM4File?.PositionDataChunk == null) continue;

                    var positions = result.PM4File.PositionDataChunk.Entries
                        .Where(e => !e.IsSpecialEntry)  // Skip control records
                        .Select(p => new Vector2(
                            float.Parse(p.CoordinateX.ToString("F6", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture),
                            float.Parse(p.CoordinateZ.ToString("F6", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)))
                        .Where(p => !float.IsNaN(p.X) && !float.IsNaN(p.Y))  // Filter out invalid coordinates
                        .ToList();

                    allPositions.AddRange(positions);

                    _logger?.LogInformation("Extracted {Count} valid positions from {File}", 
                        positions.Count, result.FileName);
                }

                if (!allPositions.Any())
                {
                    _logger?.LogWarning("No valid position data found in any PM4 files");
                    return false;
                }

                // Remove duplicate positions to ensure clean visualization
                allPositions = allPositions.Distinct().ToList();

                // Calculate grid statistics
                var stats = AnalyzePositionGrid(allPositions);
                _logger?.LogInformation("Position Data Analysis:");
                _logger?.LogInformation("Total Points: {Count}", allPositions.Count);
                _logger?.LogInformation("Bounding Box: ({MinX:F6}, {MinY:F6}) to ({MaxX:F6}, {MaxY:F6})",
                    stats.minX, stats.minY, stats.maxX, stats.maxY);
                _logger?.LogInformation("Average Point Spacing: X={AvgSpacingX:F6}, Y={AvgSpacingY:F6}",
                    stats.avgSpacingX, stats.avgSpacingY);
                
                string pointCloudPath = Path.Combine(Path.GetDirectoryName(outputPath) ?? string.Empty,
                                                  "azeroth_march2002_map_points.obj");
                
                _logger?.LogInformation("Generating point cloud visualization at: {Path}", pointCloudPath);

                // Export the point cloud
                bool success = ExportTerrainMeshFromPositions(allPositions, pointCloudPath);
                
                if (success)
                {
                    _logger?.LogInformation("Successfully created point cloud visualization with {Count} points", 
                                         allPositions.Count);
                }
                
                return success;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to export terrain from position data: {Message}", ex.Message);
                return false;
            }
        }

        /// <summary>
        /// Analyzes position data to determine grid characteristics.
        /// </summary>
        private (float minX, float maxX, float minY, float maxY, float avgSpacingX, float avgSpacingY) 
            AnalyzePositionGrid(List<Vector2> positions)
        {
            if (!positions.Any())
                return (0, 0, 0, 0, 0, 0);

            // Calculate bounds
            float minX = positions.Min(p => p.X);
            float maxX = positions.Max(p => p.X);
            float minY = positions.Min(p => p.Y);
            float maxY = positions.Max(p => p.Y);

            // Sort points to analyze spacing
            var sortedX = positions.OrderBy(p => p.X).Select(p => p.X).ToList();
            var sortedY = positions.OrderBy(p => p.Y).Select(p => p.Y).ToList();

            // Calculate average spacing by looking at consecutive points
            float avgSpacingX = 0, avgSpacingY = 0;
            int spacingCountX = 0, spacingCountY = 0;

            for (int i = 1; i < sortedX.Count; i++)
            {
                float spacing = sortedX[i] - sortedX[i - 1];
                if (spacing > 0.001f) // Ignore duplicate positions
                {
                    avgSpacingX += spacing;
                    spacingCountX++;
                }
            }

            for (int i = 1; i < sortedY.Count; i++)
            {
                float spacing = sortedY[i] - sortedY[i - 1];
                if (spacing > 0.001f) // Ignore duplicate positions
                {
                    avgSpacingY += spacing;
                    spacingCountY++;
                }
            }

            avgSpacingX = spacingCountX > 0 ? avgSpacingX / spacingCountX : 0;
            avgSpacingY = spacingCountY > 0 ? avgSpacingY / spacingCountY : 0;

            return (minX, maxX, minY, maxY, avgSpacingX, avgSpacingY);
        }
    }
} 