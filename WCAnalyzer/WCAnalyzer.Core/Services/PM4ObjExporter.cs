using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PM4;
using System.Globalization;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for exporting PM4 data to Wavefront OBJ format.
    /// </summary>
    public class PM4ObjExporter
    {
        private readonly ILogger<PM4ObjExporter> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4ObjExporter"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance.</param>
        public PM4ObjExporter(ILogger<PM4ObjExporter>? logger = null)
        {
            _logger = logger ?? NullLogger<PM4ObjExporter>.Instance;
        }

        /// <summary>
        /// Exports a PM4 analysis result to the specified directory, creating separate OBJ files for
        /// visual data (MSPV) and navigation mesh data (MSVT) when both are available.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task ExportToDirectoryAsync(PM4AnalysisResult result, string outputDirectory)
        {
            var baseName = Path.GetFileNameWithoutExtension(result.FileName);
            
            // Check if we have both data types
            bool hasMsvtData = result.HasVertexData && result.PM4Data.VertexData.Count > 0;
            bool hasMspvData = result.HasVertexPositions && result.PM4Data.VertexPositions.Count > 0;
            
            // Always export the standard 3D visualization using MSPV data when available
            if (hasMspvData)
            {
                var visualObjPath = Path.Combine(outputDirectory, $"{baseName}_visual.obj");
                await ExportVisualObjAsync(result, visualObjPath);
            }
            
            // Export navmesh/collision grid using MSVT data when available
            if (hasMsvtData)
            {
                var navmeshObjPath = Path.Combine(outputDirectory, $"{baseName}_navmesh.obj");
                await ExportNavmeshObjAsync(result, navmeshObjPath);
            }
            
            // Create a merged model if both data types are available
            if (hasMspvData && hasMsvtData)
            {
                var mergedObjPath = Path.Combine(outputDirectory, $"{baseName}_merged.obj");
                await ExportMergedObjAsync(result, mergedObjPath);
            }
            
            // Also create a combined/default OBJ for backward compatibility
            var defaultObjPath = Path.Combine(outputDirectory, $"{baseName}.obj");
            if (hasMsvtData && hasMspvData)
            {
                await ExportMergedObjAsync(result, defaultObjPath);
            }
            else if (hasMspvData)
            {
                await ExportVisualObjAsync(result, defaultObjPath);
            }
            else if (hasMsvtData)
            {
                await ExportNavmeshObjAsync(result, defaultObjPath);
            }
        }
        
        /// <summary>
        /// Exports the 3D visual representation using MSPV data.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputPath">The output file path.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task ExportVisualObjAsync(PM4AnalysisResult result, string outputPath)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentNullException(nameof(outputPath));

            // Ensure the directory exists
            var directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Validate that we have vertex data to export
            if (!result.HasVertexPositions)
            {
                _logger.LogWarning("No MSPV vertex positions in PM4 file {FileName}, cannot export visual OBJ", result.FileName);
                return;
            }

            var objBuilder = new StringBuilder();
            
            // Add file header
            objBuilder.AppendLine($"# OBJ file exported from WCAnalyzer - 3D Visual Model");
            objBuilder.AppendLine($"# Source: {result.FileName}");
            objBuilder.AppendLine($"# Export Date: {DateTime.Now}");
            objBuilder.AppendLine($"# Using MSPV vertex positions data for 3D visualization");
            objBuilder.AppendLine($"# Vertices: {result.PM4Data.VertexPositions.Count}");
            
            if (result.HasVertexIndices)
            {
                objBuilder.AppendLine($"# Triangles: {result.PM4Data.VertexIndices.Count / 3}");
            }
            objBuilder.AppendLine();
            
            // Write object name
            objBuilder.AppendLine($"o {Path.GetFileNameWithoutExtension(result.FileName)}_visual");
            objBuilder.AppendLine();
            
            // Write vertex positions (v x y z)
            _logger.LogInformation("Exporting {Count} MSPV vertices for visual model", result.PM4Data.VertexPositions.Count);
            
            // Apply the same coordinate transformation as for MSVT vertices
            const float worldConstant = 17066.0f;
            
            foreach (var vertex in result.PM4Data.VertexPositions)
            {
                // Apply the transformation to align with world coordinates
                float worldX = worldConstant - vertex.X;
                float worldY = worldConstant - vertex.Y;
                float worldZ = vertex.Z;  // Z is already in the correct scale
                
                objBuilder.AppendLine($"v {worldX.ToString("F8", CultureInfo.InvariantCulture)} {worldY.ToString("F8", CultureInfo.InvariantCulture)} {worldZ.ToString("F8", CultureInfo.InvariantCulture)}");
            }
            objBuilder.AppendLine();
            
            // Write faces (f v1 v2 v3) if we have vertex indices
            if (result.HasVertexIndices)
            {
                _logger.LogInformation("Exporting {Count} triangles for visual model", result.PM4Data.VertexIndices.Count / 3);
                
                // Get the maximum vertex index (for validation)
                int maxVertexIndex = result.PM4Data.VertexPositions.Count;
                int validTrianglesExported = 0;
                int invalidTrianglesSkipped = 0;
                
                // Check if there are any indices with value 1 (which seems to be causing issues)
                bool hasIndex1 = false;
                foreach (var index in result.PM4Data.VertexIndices)
                {
                    if (index == 0) // 0-based index that becomes 1 in OBJ
                    {
                        hasIndex1 = true;
                        break;
                    }
                }
                
                if (hasIndex1)
                {
                    _logger.LogWarning("Found vertex index 0 (becomes 1 in OBJ) which may cause issues with face references");
                }
                
                for (int i = 0; i < result.PM4Data.VertexIndices.Count; i += 3)
                {
                    // OBJ indices are 1-based, so we need to add 1
                    // Also, make sure we have enough indices for a triangle
                    if (i + 2 < result.PM4Data.VertexIndices.Count)
                    {
                        var index1 = result.PM4Data.VertexIndices[i] + 1;
                        var index2 = result.PM4Data.VertexIndices[i + 1] + 1;
                        var index3 = result.PM4Data.VertexIndices[i + 2] + 1;
                        
                        // Validate indices - must be positive and not exceed vertex count
                        bool validIndices = 
                            index1 > 0 && index1 <= maxVertexIndex &&
                            index2 > 0 && index2 <= maxVertexIndex &&
                            index3 > 0 && index3 <= maxVertexIndex &&
                            !(index1 == index2 || index1 == index3 || index2 == index3); // No degenerate triangles
                            
                        if (validIndices)
                        {
                            objBuilder.AppendLine($"f {index1} {index2} {index3}");
                            validTrianglesExported++;
                        }
                        else
                        {
                            // Log the first few invalid triangles for debugging
                            if (invalidTrianglesSkipped < 10)
                            {
                                _logger.LogDebug("Skipping invalid triangle: [{Index1}, {Index2}, {Index3}] (max valid index: {MaxIndex})",
                                    index1, index2, index3, maxVertexIndex);
                            }
                            invalidTrianglesSkipped++;
                        }
                    }
                }
                
                _logger.LogInformation("Exported {ValidCount} valid triangles, skipped {InvalidCount} invalid triangles", 
                    validTrianglesExported, invalidTrianglesSkipped);
            }
            
            // Write the OBJ file
            await File.WriteAllTextAsync(outputPath, objBuilder.ToString());
            
            _logger.LogInformation("Successfully exported 3D visual model to OBJ: {OutputPath}", outputPath);
        }
        
        /// <summary>
        /// Exports the navigation mesh/collision grid using MSVT data.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputPath">The output file path.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task ExportNavmeshObjAsync(PM4AnalysisResult result, string outputPath)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentNullException(nameof(outputPath));

            // Ensure the directory exists
            var directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Validate that we have vertex data to export
            if (!result.HasVertexData)
            {
                _logger.LogWarning("No MSVT vertex data in PM4 file {FileName}, cannot export interior OBJ", result.FileName);
                return;
            }

            var objBuilder = new StringBuilder();
            
            // Add file header
            objBuilder.AppendLine($"# OBJ file exported from WCAnalyzer - Interior Model");
            objBuilder.AppendLine($"# Source: {result.FileName}");
            objBuilder.AppendLine($"# Export Date: {DateTime.Now}");
            objBuilder.AppendLine($"# Using MSVT interior vertex data");
            objBuilder.AppendLine($"# Vertices: {result.PM4Data.VertexData.Count}");
            
            // Add info about MSVI if available
            if (result.PM4Data.MsviIndices != null && result.PM4Data.MsviIndices.Count > 0)
            {
                objBuilder.AppendLine($"# MSVI Indices: {result.PM4Data.MsviIndices.Count}");
            }
            
            objBuilder.AppendLine();
            
            // Write object name
            objBuilder.AppendLine($"o {Path.GetFileNameWithoutExtension(result.FileName)}_interior");
            objBuilder.AppendLine();
            
            // Write vertex positions (v x y z)
            _logger.LogInformation("Exporting {Count} MSVT vertices for interior model", result.PM4Data.VertexData.Count);
            
            // The X and Y in MSVT are reversed compared to MSPV (YXZ vs XYZ)
            // Use exact value 17066 as requested
            const float worldConstant = 17066.0f;
            
            foreach (var vertex in result.PM4Data.VertexData)
            {
                // Inverted transformation to align with MSPV data
                // Since MSVT is stored as YXZ and MSPV is XYZ, we need to swap X and Y
                float worldX = worldConstant - vertex.RawY;  // Swap Y to X
                float worldY = worldConstant - vertex.RawX;  // Swap X to Y
                float worldZ = vertex.RawZ;  // Use raw Z value without transformation
                
                objBuilder.AppendLine($"v {worldX.ToString("F8", CultureInfo.InvariantCulture)} {worldY.ToString("F8", CultureInfo.InvariantCulture)} {worldZ.ToString("F8", CultureInfo.InvariantCulture)}");
            }
            objBuilder.AppendLine();
            
            // Export MSVI indices as faces if available (they may define quads for the interior model)
            if (result.PM4Data.MsviIndices != null && result.PM4Data.MsviIndices.Count > 0)
            {
                _logger.LogInformation("Exporting MSVI indices as faces for interior model");
                
                // Assuming MSVI contains quad indices (groups of 4) or triangles (groups of 3)
                int validFacesExported = 0;
                int invalidFacesSkipped = 0;
                int maxVertexIndex = result.PM4Data.VertexData.Count;
                
                // Check if indices appear to be quads (divisible by 4) or triangles (divisible by 3)
                bool appearsToBeQuads = result.PM4Data.MsviIndices.Count % 4 == 0;
                bool appearsToBeTriangles = result.PM4Data.MsviIndices.Count % 3 == 0;
                
                _logger.LogInformation("MSVI indices count: {Count}, appears to be quads: {IsQuads}, appears to be triangles: {IsTriangles}", 
                    result.PM4Data.MsviIndices.Count, appearsToBeQuads, appearsToBeTriangles);
                
                if (appearsToBeQuads)
                {
                    // Process as quads (groups of 4 indices)
                    for (int i = 0; i < result.PM4Data.MsviIndices.Count; i += 4)
                    {
                        if (i + 3 < result.PM4Data.MsviIndices.Count)
                        {
                            var index1 = result.PM4Data.MsviIndices[i] + 1; // OBJ indices are 1-based
                            var index2 = result.PM4Data.MsviIndices[i + 1] + 1;
                            var index3 = result.PM4Data.MsviIndices[i + 2] + 1;
                            var index4 = result.PM4Data.MsviIndices[i + 3] + 1;
                            
                            // Validate indices
                            bool validIndices = 
                                index1 > 0 && index1 <= maxVertexIndex &&
                                index2 > 0 && index2 <= maxVertexIndex &&
                                index3 > 0 && index3 <= maxVertexIndex &&
                                index4 > 0 && index4 <= maxVertexIndex;
                                
                            if (validIndices)
                            {
                                objBuilder.AppendLine($"f {index1} {index2} {index3} {index4}");
                                validFacesExported++;
                            }
                            else
                            {
                                invalidFacesSkipped++;
                            }
                        }
                    }
                }
                else if (appearsToBeTriangles)
                {
                    // Process as triangles (groups of 3 indices)
                    for (int i = 0; i < result.PM4Data.MsviIndices.Count; i += 3)
                    {
                        if (i + 2 < result.PM4Data.MsviIndices.Count)
                        {
                            var index1 = result.PM4Data.MsviIndices[i] + 1; // OBJ indices are 1-based
                            var index2 = result.PM4Data.MsviIndices[i + 1] + 1;
                            var index3 = result.PM4Data.MsviIndices[i + 2] + 1;
                            
                            // Validate indices
                            bool validIndices = 
                                index1 > 0 && index1 <= maxVertexIndex &&
                                index2 > 0 && index2 <= maxVertexIndex &&
                                index3 > 0 && index3 <= maxVertexIndex;
                                
                            if (validIndices)
                            {
                                objBuilder.AppendLine($"f {index1} {index2} {index3}");
                                validFacesExported++;
                            }
                            else
                            {
                                invalidFacesSkipped++;
                            }
                        }
                    }
                }
                
                _logger.LogInformation("Exported {ValidCount} valid faces, skipped {InvalidCount} invalid faces", 
                    validFacesExported, invalidFacesSkipped);
            }
            
            // Write the OBJ file
            await File.WriteAllTextAsync(outputPath, objBuilder.ToString());
            
            _logger.LogInformation("Successfully exported interior model to OBJ: {OutputPath}", outputPath);
        }

        /// <summary>
        /// Exports a merged model with both MSPV and MSVT vertices.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputPath">The output file path.</param>
        private async Task ExportMergedObjAsync(PM4AnalysisResult result, string outputPath)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentNullException(nameof(outputPath));

            // Ensure the directory exists
            var directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Validate that we have both vertex data types to export
            if (!result.HasVertexPositions || !result.HasVertexData)
            {
                _logger.LogWarning("Missing required vertex data in PM4 file {FileName}, need both MSPV and MSVT data for merged export", result.FileName);
                return;
            }

            var objBuilder = new StringBuilder();
            
            // Add file header
            objBuilder.AppendLine($"# OBJ file exported from WCAnalyzer - Complete Merged Model");
            objBuilder.AppendLine($"# Source: {result.FileName}");
            objBuilder.AppendLine($"# Export Date: {DateTime.Now}");
            objBuilder.AppendLine($"# Using both MSPV vertices (exterior) and MSVT vertices (interior)");
            objBuilder.AppendLine($"# MSPV Vertices: {result.PM4Data.VertexPositions.Count}");
            objBuilder.AppendLine($"# MSVT Vertices: {result.PM4Data.VertexData.Count}");
            objBuilder.AppendLine($"# Total Vertices: {result.PM4Data.VertexPositions.Count + result.PM4Data.VertexData.Count}");
            
            if (result.HasVertexIndices)
            {
                objBuilder.AppendLine($"# MSPV Triangles: {result.PM4Data.VertexIndices.Count / 3}");
            }
            
            if (result.PM4Data.MsviIndices != null && result.PM4Data.MsviIndices.Count > 0)
            {
                objBuilder.AppendLine($"# MSVI Indices: {result.PM4Data.MsviIndices.Count}");
            }
            
            objBuilder.AppendLine();
            
            // Write object name
            objBuilder.AppendLine($"o {Path.GetFileNameWithoutExtension(result.FileName)}_complete");
            objBuilder.AppendLine();
            
            // Log info for debugging
            _logger.LogInformation("MSPV count: {Count}", result.PM4Data.VertexPositions.Count);
            _logger.LogInformation("MSVT count: {Count}", result.PM4Data.VertexData.Count);
            
            // First write MSPV vertices with proper coordinate transformation
            _logger.LogInformation("Exporting {Count} MSPV vertices for merged model", result.PM4Data.VertexPositions.Count);
            
            // Apply the same coordinate transformation as for MSVT vertices
            const float worldConstant = 17066.0f;
            
            foreach (var vertex in result.PM4Data.VertexPositions)
            {
                // Apply the transformation to align with world coordinates
                float worldX = worldConstant - vertex.X;
                float worldY = worldConstant - vertex.Y;
                float worldZ = vertex.Z;  // Z is already in the correct scale
                
                objBuilder.AppendLine($"v {worldX.ToString("F8", CultureInfo.InvariantCulture)} {worldY.ToString("F8", CultureInfo.InvariantCulture)} {worldZ.ToString("F8", CultureInfo.InvariantCulture)}");
            }
            
            // Then append MSVT vertices with proper coordinate transformation
            _logger.LogInformation("Appending {Count} MSVT vertices for merged model", result.PM4Data.VertexData.Count);
            
            foreach (var vertex in result.PM4Data.VertexData)
            {
                // Inverted transformation to align with MSPV data
                // Since MSVT is stored as YXZ and MSPV is XYZ, we need to swap X and Y
                float worldX = worldConstant - vertex.RawY;  // Swap Y to X
                float worldY = worldConstant - vertex.RawX;  // Swap X to Y
                float worldZ = vertex.RawZ;  // Use raw Z value without transformation
                
                objBuilder.AppendLine($"v {worldX.ToString("F8", CultureInfo.InvariantCulture)} {worldY.ToString("F8", CultureInfo.InvariantCulture)} {worldZ.ToString("F8", CultureInfo.InvariantCulture)}");
            }
            objBuilder.AppendLine();
            
            // Write MSPV faces (f v1 v2 v3) if we have vertex indices
            int mspvVertexCount = result.PM4Data.VertexPositions.Count;
            
            if (result.HasVertexIndices)
            {
                _logger.LogInformation("Exporting {Count} triangles for MSPV data", result.PM4Data.VertexIndices.Count / 3);
                
                // Get the maximum vertex index (for validation)
                int maxVertexIndex = mspvVertexCount;
                int validTrianglesExported = 0;
                int invalidTrianglesSkipped = 0;
                
                for (int i = 0; i < result.PM4Data.VertexIndices.Count; i += 3)
                {
                    // OBJ indices are 1-based, so we need to add 1
                    // Also, make sure we have enough indices for a triangle
                    if (i + 2 < result.PM4Data.VertexIndices.Count)
                    {
                        var index1 = result.PM4Data.VertexIndices[i] + 1;
                        var index2 = result.PM4Data.VertexIndices[i + 1] + 1;
                        var index3 = result.PM4Data.VertexIndices[i + 2] + 1;
                        
                        // Validate indices - must be positive and not exceed vertex count
                        bool validIndices = 
                            index1 > 0 && index1 <= maxVertexIndex &&
                            index2 > 0 && index2 <= maxVertexIndex &&
                            index3 > 0 && index3 <= maxVertexIndex &&
                            !(index1 == index2 || index1 == index3 || index2 == index3); // No degenerate triangles
                            
                        if (validIndices)
                        {
                            objBuilder.AppendLine($"f {index1} {index2} {index3}");
                            validTrianglesExported++;
                        }
                        else
                        {
                            invalidTrianglesSkipped++;
                        }
                    }
                }
                
                _logger.LogInformation("Exported {ValidCount} valid MSPV triangles, skipped {InvalidCount} invalid triangles", 
                    validTrianglesExported, invalidTrianglesSkipped);
            }
            
            // Export MSVI indices as faces if available (they may define quads for the interior model)
            if (result.PM4Data.MsviIndices != null && result.PM4Data.MsviIndices.Count > 0)
            {
                _logger.LogInformation("Exporting MSVI indices as faces for interior model");
                
                // Assuming MSVI contains quad indices (groups of 4) or triangles (groups of 3)
                int validFacesExported = 0;
                int invalidFacesSkipped = 0;
                int maxVertexIndex = result.PM4Data.VertexData.Count;
                
                // Check if indices appear to be quads (divisible by 4) or triangles (divisible by 3)
                bool appearsToBeQuads = result.PM4Data.MsviIndices.Count % 4 == 0;
                bool appearsToBeTriangles = result.PM4Data.MsviIndices.Count % 3 == 0;
                
                _logger.LogInformation("MSVI indices count: {Count}, appears to be quads: {IsQuads}, appears to be triangles: {IsTriangles}", 
                    result.PM4Data.MsviIndices.Count, appearsToBeQuads, appearsToBeTriangles);
                
                // Offset for MSVT vertices (they come after MSPV vertices in the merged file)
                int msvtVertexOffset = mspvVertexCount;
                
                if (appearsToBeQuads)
                {
                    // Process as quads (groups of 4 indices)
                    for (int i = 0; i < result.PM4Data.MsviIndices.Count; i += 4)
                    {
                        if (i + 3 < result.PM4Data.MsviIndices.Count)
                        {
                            // Add the offset to get the correct vertex indices for the merged file
                            // Also add 1 for OBJ 1-based indexing
                            var index1 = result.PM4Data.MsviIndices[i] + 1 + msvtVertexOffset;
                            var index2 = result.PM4Data.MsviIndices[i + 1] + 1 + msvtVertexOffset;
                            var index3 = result.PM4Data.MsviIndices[i + 2] + 1 + msvtVertexOffset;
                            var index4 = result.PM4Data.MsviIndices[i + 3] + 1 + msvtVertexOffset;
                            
                            // Validate indices - must not exceed total vertex count
                            bool validIndices = 
                                index1 > msvtVertexOffset && index1 <= msvtVertexOffset + maxVertexIndex &&
                                index2 > msvtVertexOffset && index2 <= msvtVertexOffset + maxVertexIndex &&
                                index3 > msvtVertexOffset && index3 <= msvtVertexOffset + maxVertexIndex &&
                                index4 > msvtVertexOffset && index4 <= msvtVertexOffset + maxVertexIndex;
                                
                            if (validIndices)
                            {
                                objBuilder.AppendLine($"f {index1} {index2} {index3} {index4}");
                                validFacesExported++;
                            }
                            else
                            {
                                invalidFacesSkipped++;
                            }
                        }
                    }
                }
                else if (appearsToBeTriangles)
                {
                    // Process as triangles (groups of 3 indices)
                    for (int i = 0; i < result.PM4Data.MsviIndices.Count; i += 3)
                    {
                        if (i + 2 < result.PM4Data.MsviIndices.Count)
                        {
                            // Add the offset to get the correct vertex indices for the merged file
                            // Also add 1 for OBJ 1-based indexing
                            var index1 = result.PM4Data.MsviIndices[i] + 1 + msvtVertexOffset;
                            var index2 = result.PM4Data.MsviIndices[i + 1] + 1 + msvtVertexOffset;
                            var index3 = result.PM4Data.MsviIndices[i + 2] + 1 + msvtVertexOffset;
                            
                            // Validate indices - must not exceed total vertex count
                            bool validIndices = 
                                index1 > msvtVertexOffset && index1 <= msvtVertexOffset + maxVertexIndex &&
                                index2 > msvtVertexOffset && index2 <= msvtVertexOffset + maxVertexIndex &&
                                index3 > msvtVertexOffset && index3 <= msvtVertexOffset + maxVertexIndex;
                                
                            if (validIndices)
                            {
                                objBuilder.AppendLine($"f {index1} {index2} {index3}");
                                validFacesExported++;
                            }
                            else
                            {
                                invalidFacesSkipped++;
                            }
                        }
                    }
                }
                
                _logger.LogInformation("Exported {ValidCount} valid MSVI faces, skipped {InvalidCount} invalid faces", 
                    validFacesExported, invalidFacesSkipped);
            }
            
            // Write the OBJ file
            await File.WriteAllTextAsync(outputPath, objBuilder.ToString());
            
            _logger.LogInformation("Successfully exported merged model to OBJ: {OutputPath}", outputPath);
        }
    }
} 