using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Linq;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Represents the results of a PM4 file analysis.
    /// </summary>
    public class PM4AnalysisResult
    {
        /// <summary>
        /// Gets or sets the name of the analyzed file.
        /// </summary>
        public string? FileName { get; set; }

        /// <summary>
        /// Gets or sets the path of the analyzed file.
        /// </summary>
        public string? FilePath { get; set; }

        /// <summary>
        /// Gets or sets the errors encountered during analysis.
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the timestamp when the analysis was performed.
        /// </summary>
        public DateTime AnalysisTime { get; set; } = DateTime.Now;

        /// <summary>
        /// Gets or sets the file version.
        /// </summary>
        public int Version { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether shadow data is present.
        /// </summary>
        public bool HasShadowData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether vertex positions are present.
        /// </summary>
        public bool HasVertexPositions { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether vertex indices are present.
        /// </summary>
        public bool HasVertexIndices { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether normal coordinates are present.
        /// </summary>
        public bool HasNormalCoordinates { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether links are present.
        /// </summary>
        public bool HasLinks { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether vertex data is present.
        /// </summary>
        public bool HasVertexData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the second vertex indices are present.
        /// </summary>
        public bool HasVertexIndices2 { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether surface data is present.
        /// </summary>
        public bool HasSurfaceData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether position data is present.
        /// </summary>
        public bool HasPositionData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether value pairs are present.
        /// </summary>
        public bool HasValuePairs { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether building data is present.
        /// </summary>
        public bool HasBuildingData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether simple data is present.
        /// </summary>
        public bool HasSimpleData { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether final data is present.
        /// </summary>
        public bool HasFinalData { get; set; }

        /// <summary>
        /// Gets or sets additional metadata or context for the analysis.
        /// </summary>
        [JsonExtensionData]
        public Dictionary<string, object>? AdditionalData { get; set; }

        /// <summary>
        /// Gets or sets the referenced PM4 file instance.
        /// </summary>
        [JsonIgnore]
        public PM4File? PM4File { get; set; }

        /// <summary>
        /// Gets or sets the FileDataID to file name mapping for model references.
        /// </summary>
        public Dictionary<uint, string> ResolvedFileNames { get; set; } = new Dictionary<uint, string>();

        /// <summary>
        /// Creates a PM4AnalysisResult from a PM4File instance.
        /// </summary>
        /// <param name="pm4File">The PM4File to analyze.</param>
        /// <returns>A PM4AnalysisResult containing the analysis data.</returns>
        public static PM4AnalysisResult FromPM4File(PM4File pm4File)
        {
            if (pm4File == null)
                throw new ArgumentNullException(nameof(pm4File));

            var result = new PM4AnalysisResult
            {
                FileName = pm4File.FileName,
                FilePath = pm4File.FilePath,
                Errors = new List<string>(pm4File.Errors),
                Version = pm4File.Version?.Version != null ? (int)pm4File.Version.Version : 0,
                HasShadowData = pm4File.ShadowData != null,
                HasVertexPositions = pm4File.VertexPositions != null,
                HasVertexIndices = pm4File.VertexIndices != null,
                HasNormalCoordinates = pm4File.NormalCoordinates != null,
                HasLinks = pm4File.Links != null,
                HasVertexData = pm4File.VertexData != null,
                HasVertexIndices2 = pm4File.VertexIndices2 != null,
                HasSurfaceData = pm4File.SurfaceData != null,
                HasPositionData = pm4File.PositionData != null,
                HasValuePairs = pm4File.ValuePairs != null,
                HasBuildingData = pm4File.BuildingData != null,
                HasSimpleData = pm4File.SimpleData != null,
                HasFinalData = pm4File.FinalData != null,
                PM4File = pm4File
            };

            return result;
        }

        /// <summary>
        /// Gets a summary string representing the analysis results.
        /// </summary>
        /// <returns>A summary string.</returns>
        public string GetSummary()
        {
            var summary = new System.Text.StringBuilder();
            summary.AppendLine($"# PM4 Analysis Result for: {FileName}");
            summary.AppendLine($"Path: {FilePath}");
            summary.AppendLine($"Analysis Time: {AnalysisTime}");
            summary.AppendLine($"Version: {Version}");
            
            summary.AppendLine("\n## Chunks present:");
            if (HasShadowData) summary.AppendLine("- MSHD (Shadow Data)");
            if (HasVertexPositions) summary.AppendLine("- MSPV (Vertex Positions)");
            if (HasVertexIndices) summary.AppendLine("- MSPI (Vertex Indices)");
            if (HasNormalCoordinates) summary.AppendLine("- MSCN (Normal Coordinates)");
            if (HasLinks) summary.AppendLine("- MSLK (Links)");
            if (HasVertexData) summary.AppendLine("- MSVT (Vertex Data)");
            if (HasVertexIndices2) summary.AppendLine("- MSVI (Vertex Indices)");
            if (HasSurfaceData) summary.AppendLine("- MSUR (Surface Data)");
            if (HasPositionData) summary.AppendLine("- MPRL (Position Data)");
            if (HasValuePairs) summary.AppendLine("- MPRR (Value Pairs)");
            if (HasBuildingData) summary.AppendLine("- MDBH (Building Data)");
            if (HasSimpleData) summary.AppendLine("- MDOS (Simple Data)");
            if (HasFinalData) summary.AppendLine("- MDSF (Final Data)");

            // Include detailed information from the PM4File if available
            if (PM4File != null)
            {
                if (PM4File.VertexPositionsChunk != null && PM4File.VertexPositionsChunk.Vertices.Count > 0)
                {
                    summary.AppendLine("\n## Vertex Data:");
                    summary.AppendLine($"Total Vertices: {PM4File.VertexPositionsChunk.Vertices.Count}");
                    
                    // Show sample vertices
                    summary.AppendLine("\n### Sample Vertices:");
                    int sampleCount = Math.Min(5, PM4File.VertexPositionsChunk.Vertices.Count);
                    for (int i = 0; i < sampleCount; i++)
                    {
                        var vertex = PM4File.VertexPositionsChunk.Vertices[i];
                        summary.AppendLine($"- Vertex {i}: ({vertex.X:F2}, {vertex.Y:F2}, {vertex.Z:F2})");
                    }
                }
                
                if (PM4File.VertexIndicesChunk != null && PM4File.VertexIndicesChunk.Indices.Count > 0)
                {
                    summary.AppendLine("\n## Triangle Data:");
                    int triangleCount = PM4File.VertexIndicesChunk.Indices.Count / 3;
                    summary.AppendLine($"Total Triangles: {triangleCount}");
                    
                    // Show sample triangles
                    summary.AppendLine("\n### Sample Triangles:");
                    int sampleCount = Math.Min(3, triangleCount);
                    for (int i = 0; i < sampleCount; i++)
                    {
                        int baseIndex = i * 3;
                        if (baseIndex + 2 < PM4File.VertexIndicesChunk.Indices.Count)
                        {
                            summary.AppendLine($"- Triangle {i}: Vertices [{PM4File.VertexIndicesChunk.Indices[baseIndex]}, " +
                                $"{PM4File.VertexIndicesChunk.Indices[baseIndex + 1]}, {PM4File.VertexIndicesChunk.Indices[baseIndex + 2]}]");
                        }
                    }
                }
                
                if (PM4File.PositionDataChunk != null && PM4File.PositionDataChunk.Entries.Count > 0)
                {
                    summary.AppendLine("\n## Position Data:");
                    summary.AppendLine($"Total Entries: {PM4File.PositionDataChunk.Entries.Count}");
                    
                    // Count position vs command records
                    int positionRecords = PM4File.PositionDataChunk.Entries.Count(e => !e.IsControlRecord);
                    int commandRecords = PM4File.PositionDataChunk.Entries.Count(e => e.IsControlRecord);
                    summary.AppendLine($"Position Records: {positionRecords}");
                    summary.AppendLine($"Command Records: {commandRecords}");
                    
                    // Show sample position records
                    if (positionRecords > 0)
                    {
                        summary.AppendLine("\n### Sample Position Records:");
                        int sampleCount = Math.Min(5, positionRecords);
                        var positions = PM4File.PositionDataChunk.Entries.Where(e => !e.IsControlRecord).Take(sampleCount).ToList();
                        foreach (var pos in positions)
                        {
                            summary.AppendLine($"- Index {pos.Index}: ({pos.CoordinateX:F2}, {pos.CoordinateY:F2}, {pos.CoordinateZ:F2})");
                        }
                    }
                    
                    // Show sample command records
                    if (commandRecords > 0)
                    {
                        summary.AppendLine("\n### Sample Command Records:");
                        int sampleCount = Math.Min(5, commandRecords);
                        var commands = PM4File.PositionDataChunk.Entries.Where(e => e.IsControlRecord).Take(sampleCount).ToList();
                        foreach (var cmd in commands)
                        {
                            summary.AppendLine($"- Index {cmd.Index}: Command=0x{cmd.CommandValue:X8}, Y={cmd.CoordinateY:F2}");
                        }
                    }
                    
                    // Show the sequence pattern
                    summary.AppendLine("\n### Entry Sequence Pattern:");
                    int sequenceSample = Math.Min(10, PM4File.PositionDataChunk.Entries.Count);
                    for (int i = 0; i < sequenceSample; i++)
                    {
                        var entry = PM4File.PositionDataChunk.Entries[i];
                        string type = entry.IsControlRecord ? "Command" : "Position";
                        string details = entry.IsControlRecord ? 
                            $"Command=0x{entry.CommandValue:X8}, Y={entry.CoordinateY:F2}" : 
                            $"({entry.CoordinateX:F2}, {entry.CoordinateY:F2}, {entry.CoordinateZ:F2})";
                        summary.AppendLine($"- Entry {i}: {type} - {details}");
                    }
                }
            }

            if (Errors.Count > 0)
            {
                summary.AppendLine("\n## Errors encountered:");
                foreach (var error in Errors)
                {
                    summary.AppendLine($"- {error}");
                }
            }

            return summary.ToString();
        }
    }
} 