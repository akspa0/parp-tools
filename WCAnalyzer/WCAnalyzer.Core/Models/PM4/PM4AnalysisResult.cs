using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

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
                HasFinalData = pm4File.FinalData != null
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
            summary.AppendLine($"PM4 Analysis Result for: {FileName}");
            summary.AppendLine($"Path: {FilePath}");
            summary.AppendLine($"Analysis Time: {AnalysisTime}");
            summary.AppendLine($"Version: {Version}");
            
            summary.AppendLine("\nChunks present:");
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

            if (Errors.Count > 0)
            {
                summary.AppendLine("\nErrors encountered:");
                foreach (var error in Errors)
                {
                    summary.AppendLine($"- {error}");
                }
            }

            return summary.ToString();
        }
    }
} 