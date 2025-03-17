using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PM4;

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
        /// Exports a PM4 analysis result to OBJ format.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputPath">The output file path (should end with .obj).</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task ExportToObjAsync(PM4AnalysisResult result, string outputPath)
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
                _logger.LogWarning("No vertex positions in PM4 file {FileName}, cannot export to OBJ", result.FileName);
                return;
            }

            var objBuilder = new StringBuilder();
            
            // Add file header
            objBuilder.AppendLine($"# OBJ file exported from WCAnalyzer");
            objBuilder.AppendLine($"# Source: {result.FileName}");
            objBuilder.AppendLine($"# Export Date: {DateTime.Now}");
            objBuilder.AppendLine($"# Vertices: {result.PM4Data.VertexPositions.Count}");
            if (result.HasVertexIndices)
            {
                objBuilder.AppendLine($"# Triangles: {result.PM4Data.VertexIndices.Count / 3}");
            }
            objBuilder.AppendLine();
            
            // Write object name
            objBuilder.AppendLine($"o {Path.GetFileNameWithoutExtension(result.FileName)}");
            objBuilder.AppendLine();
            
            // Write vertex positions (v x y z)
            _logger.LogInformation("Exporting {Count} vertices", result.PM4Data.VertexPositions.Count);
            foreach (var vertex in result.PM4Data.VertexPositions)
            {
                objBuilder.AppendLine($"v {vertex.X} {vertex.Y} {vertex.Z}");
            }
            objBuilder.AppendLine();
            
            // Write faces (f v1 v2 v3) if we have vertex indices
            if (result.HasVertexIndices)
            {
                _logger.LogInformation("Exporting {Count} triangles", result.PM4Data.VertexIndices.Count / 3);
                
                for (int i = 0; i < result.PM4Data.VertexIndices.Count; i += 3)
                {
                    // OBJ indices are 1-based, so we need to add 1
                    // Also, make sure we have enough indices for a triangle
                    if (i + 2 < result.PM4Data.VertexIndices.Count)
                    {
                        var index1 = result.PM4Data.VertexIndices[i] + 1;
                        var index2 = result.PM4Data.VertexIndices[i + 1] + 1;
                        var index3 = result.PM4Data.VertexIndices[i + 2] + 1;
                        
                        objBuilder.AppendLine($"f {index1} {index2} {index3}");
                    }
                }
            }
            
            // Write the OBJ file
            await File.WriteAllTextAsync(outputPath, objBuilder.ToString());
            
            _logger.LogInformation("Successfully exported PM4 data to OBJ: {OutputPath}", outputPath);
        }
        
        /// <summary>
        /// Adds a command to the PM4CsvGenerator to export PM4 data to OBJ format.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task ExportToObjAsync(PM4AnalysisResult result, string outputDirectory)
        {
            var objFileName = Path.GetFileNameWithoutExtension(result.FileName) + ".obj";
            var outputPath = Path.Combine(outputDirectory, objFileName);
            
            await ExportToObjAsync(result, outputPath);
        }
    }
} 