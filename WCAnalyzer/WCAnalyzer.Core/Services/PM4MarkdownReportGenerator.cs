using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Generates Markdown reports for PM4 file analysis.
    /// </summary>
    public class PM4MarkdownReportGenerator
    {
        private readonly ILogger? _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4MarkdownReportGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PM4MarkdownReportGenerator(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generates a comprehensive Markdown report for a PM4 file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <param name="outputPath">The path to write the report to</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task GenerateComprehensiveReportAsync(PM4File pm4File, string outputPath)
        {
            if (pm4File == null)
                throw new ArgumentNullException(nameof(pm4File));
            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));

            var report = new StringBuilder();

            // File Information
            report.AppendLine("# PM4 File Analysis Report");
            report.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine($"\n## File Information");
            report.AppendLine($"- File Name: {pm4File.FileName ?? "Unknown"}");
            report.AppendLine($"- File Path: {pm4File.FilePath ?? "Unknown"}");
            report.AppendLine($"- File Size: {pm4File.FileSize:N0} bytes");
            report.AppendLine($"- Version: {pm4File.Version?.Version ?? 0}");

            // Position Data Analysis
            if (pm4File.PositionDataChunk != null)
            {
                report.AppendLine("\n## Position Data Analysis");
                await GeneratePositionDataReportAsync(pm4File.PositionDataChunk, report);
            }

            // Vertex Data Analysis
            if (pm4File.VertexPositionsChunk != null)
            {
                report.AppendLine("\n## Vertex Data Analysis");
                await GenerateVertexDataReportAsync(pm4File.VertexPositionsChunk, report);
            }

            // Write the report to file
            await File.WriteAllTextAsync(outputPath, report.ToString());
            _logger?.LogInformation("Generated comprehensive Markdown report: {FilePath}", outputPath);
        }

        /// <summary>
        /// Generates the position data section of the report.
        /// </summary>
        private async Task GeneratePositionDataReportAsync(MPRLChunk positionChunk, StringBuilder report)
        {
            var entries = positionChunk.Entries;
            int totalEntries = entries.Count;
            var positionRecords = entries.Where(e => !e.IsSpecialEntry).ToList();
            var specialRecords = entries.Where(e => e.IsSpecialEntry).ToList();

            report.AppendLine($"### Overview");
            report.AppendLine($"- Total Entries: {totalEntries:N0}");
            report.AppendLine($"- Position Records: {positionRecords.Count:N0}");
            report.AppendLine($"- Special Records: {specialRecords.Count:N0}");

            if (positionRecords.Count > 0)
            {
                report.AppendLine("\n### Position Records Analysis");
                
                // Calculate coordinate ranges
                var minX = positionRecords.Min(p => p.CoordinateX);
                var maxX = positionRecords.Max(p => p.CoordinateX);
                var minY = positionRecords.Min(p => p.CoordinateY);
                var maxY = positionRecords.Max(p => p.CoordinateY);
                var minZ = positionRecords.Min(p => p.CoordinateZ);
                var maxZ = positionRecords.Max(p => p.CoordinateZ);

                report.AppendLine("\n#### Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|--------|");
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");

                report.AppendLine("\n#### Sample Position Records");
                report.AppendLine("| Index | X | Y | Z |");
                report.AppendLine("|-------|-----|-----|-----|");
                foreach (var pos in positionRecords.Take(5))
                {
                    report.AppendLine($"| {pos.Index} | {pos.CoordinateX:F2} | {pos.CoordinateY:F2} | {pos.CoordinateZ:F2} |");
                }
            }

            if (specialRecords.Count > 0)
            {
                report.AppendLine("\n### Special Records Analysis");
                report.AppendLine("| Index | Command (Hex) | Y Coordinate | Special Value |");
                report.AppendLine("|-------|---------------|--------------|---------------|");
                foreach (var cmd in specialRecords.Take(10))
                {
                    report.AppendLine($"| {cmd.Index} | 0x{cmd.SpecialValue:X8} | {cmd.CoordinateY:F2} | {cmd.SpecialValue} |");
                }

                if (specialRecords.Count > 10)
                {
                    report.AppendLine($"\n_Showing top 10 of {specialRecords.Count} special records_");
                }
            }
        }

        /// <summary>
        /// Generates the vertex data section of the report.
        /// </summary>
        private async Task GenerateVertexDataReportAsync(MSPVChunk vertexChunk, StringBuilder report)
        {
            var vertices = vertexChunk.Vertices;
            
            report.AppendLine($"### Overview");
            report.AppendLine($"- Total Vertices: {vertices.Count:N0}");

            if (vertices.Count > 0)
            {
                // Calculate coordinate ranges
                var minX = vertices.Min(v => v.X);
                var maxX = vertices.Max(v => v.X);
                var minY = vertices.Min(v => v.Y);
                var maxY = vertices.Max(v => v.Y);
                var minZ = vertices.Min(v => v.Z);
                var maxZ = vertices.Max(v => v.Z);

                report.AppendLine("\n#### Coordinate Ranges");
                report.AppendLine("| Axis | Minimum | Maximum | Range |");
                report.AppendLine("|------|---------|---------|--------|");
                report.AppendLine($"| X | {minX:F2} | {maxX:F2} | {maxX - minX:F2} |");
                report.AppendLine($"| Y | {minY:F2} | {maxY:F2} | {maxY - minY:F2} |");
                report.AppendLine($"| Z | {minZ:F2} | {maxZ:F2} | {maxZ - minZ:F2} |");

                report.AppendLine("\n#### Sample Vertices");
                report.AppendLine("| Index | X | Y | Z |");
                report.AppendLine("|-------|-----|-----|-----|");
                for (int i = 0; i < Math.Min(5, vertices.Count); i++)
                {
                    var vertex = vertices[i];
                    report.AppendLine($"| {i} | {vertex.X:F2} | {vertex.Y:F2} | {vertex.Z:F2} |");
                }
            }
        }
    }
} 