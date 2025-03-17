using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Linq;
using System.Text;
using System.Numerics;
using WCAnalyzer.Core.Models.PM4.Chunks;
using WCAnalyzer.Core.Utilities;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Services;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Analysis result for a PM4 file.
    /// </summary>
    public class PM4AnalysisResult
    {
        /// <summary>
        /// Gets or sets the file name.
        /// </summary>
        public string FileName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file path.
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the list of errors.
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the parsed PM4 data.
        /// </summary>
        public PM4Data PM4Data { get; set; } = new PM4Data();

        /// <summary>
        /// Gets or sets a value indicating whether the analysis was successful.
        /// </summary>
        public bool Success { get; set; } = true;

        /// <summary>
        /// Gets the PM4 file for backward compatibility.
        /// </summary>
        public PM4File PM4File => null;

        /// <summary>
        /// Gets a value indicating whether the analysis has errors.
        /// </summary>
        public bool HasErrors => Errors.Count > 0 || (PM4Data?.Errors.Count > 0);

        /// <summary>
        /// Gets a value indicating whether the file has vertex positions.
        /// </summary>
        public bool HasVertexPositions => PM4Data?.VertexPositions?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has vertex indices.
        /// </summary>
        public bool HasVertexIndices => PM4Data?.VertexIndices?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has links.
        /// </summary>
        public bool HasLinks => PM4Data?.Links?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has position data.
        /// </summary>
        public bool HasPositionData => PM4Data?.PositionData?.Count > 0 || PM4Data?.Positions?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has position references.
        /// </summary>
        public bool HasPositionReferences => PM4Data?.PositionReferences?.Count > 0;

        // Backward compatibility properties
        public bool HasShadowData => false;
        public bool HasNormalCoordinates => false;
        public bool HasVertexData => HasVertexPositions;
        public bool HasVertexInfo => HasVertexIndices;
        public bool HasSurfaceData => false;
        public bool HasPositionReference => HasPositionReferences;
        public bool HasDestructibleBuildingHeader => false;
        public bool HasObjectData => false;
        public bool HasServerFlagData => false;
        public bool HasVersion => Version > 0;
        public bool HasCRC => false;
        public int Version => PM4Data?.Version ?? 0;
        public List<string> ResolvedFileNames { get; set; } = new List<string>();
        
        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"File: {FileName}");
            
            if (HasErrors)
            {
                sb.AppendLine($"Errors: {Errors.Count + (PM4Data?.Errors.Count ?? 0)}");
                foreach (var error in Errors)
                {
                    sb.AppendLine($"- {error}");
                }
                
                if (PM4Data?.Errors != null)
                {
                    foreach (var error in PM4Data.Errors)
                    {
                        sb.AppendLine($"- {error}");
                    }
                }
            }
            
            if (PM4Data != null)
            {
                sb.AppendLine($"Version: {PM4Data.Version}");
                sb.AppendLine($"Vertex Positions: {PM4Data.VertexPositions?.Count ?? 0}");
                sb.AppendLine($"Vertex Indices: {PM4Data.VertexIndices?.Count ?? 0}");
                sb.AppendLine($"Links: {PM4Data.Links?.Count ?? 0}");
                sb.AppendLine($"Position Data: {PM4Data.PositionData?.Count ?? 0}");
                sb.AppendLine($"Position References: {PM4Data.PositionReferences?.Count ?? 0}");
            }
            
            return sb.ToString();
        }
        
        /// <summary>
        /// Gets a summary of the analysis result.
        /// </summary>
        /// <returns>A string containing a summary of the analysis result.</returns>
        public string GetSummary()
        {
            return ToString();
        }
        
        /// <summary>
        /// Gets a detailed report of the analysis result.
        /// </summary>
        /// <returns>A string containing a detailed report of the analysis result.</returns>
        public string GetDetailedReport()
        {
            var sb = new StringBuilder(ToString());
            
            if (PM4Data != null)
            {
                // Add more detailed information about the file content
                if (HasVertexPositions && PM4Data.VertexPositions != null)
                {
                    sb.AppendLine("\nVertex Positions (first 10):");
                    for (int i = 0; i < Math.Min(10, PM4Data.VertexPositions.Count); i++)
                    {
                        sb.AppendLine($"  {i}: {PM4Data.VertexPositions[i]}");
                    }
                }
                
                if (HasVertexIndices && PM4Data.VertexIndices != null)
                {
                    sb.AppendLine("\nVertex Indices (first 10):");
                    for (int i = 0; i < Math.Min(10, PM4Data.VertexIndices.Count); i++)
                    {
                        sb.AppendLine($"  {i}: {PM4Data.VertexIndices[i]}");
                    }
                }
                
                if (HasLinks && PM4Data.Links != null)
                {
                    sb.AppendLine("\nLinks (first 10):");
                    for (int i = 0; i < Math.Min(10, PM4Data.Links.Count); i++)
                    {
                        sb.AppendLine($"  {i}: {PM4Data.Links[i].SourceIndex} -> {PM4Data.Links[i].TargetIndex}");
                    }
                }
            }
            
            return sb.ToString();
        }
    }
    
    /// <summary>
    /// Represents an unknown chunk in a PM4 file.
    /// </summary>
    public class UnknownChunk
    {
        /// <summary>
        /// Gets or sets the chunk name (reversed from file).
        /// </summary>
        public string ChunkName { get; set; }
        
        /// <summary>
        /// Gets or sets the original chunk name as read from file.
        /// </summary>
        public string OriginalChunkName { get; set; }
        
        /// <summary>
        /// Gets or sets the chunk size in bytes.
        /// </summary>
        public int Size { get; set; }
        
        /// <summary>
        /// Gets or sets the position in the file where the chunk data starts.
        /// </summary>
        public long Position { get; set; }
        
        /// <summary>
        /// Gets or sets a hexadecimal preview of the first few bytes of the chunk.
        /// </summary>
        public string HexPreview { get; set; }
    }
    
    /// <summary>
    /// Represents the parsed data from a PM4 file.
    /// </summary>
    public class PM4Data
    {
        /// <summary>
        /// Gets or sets the file version.
        /// </summary>
        public int Version { get; set; }
        
        /// <summary>
        /// Gets or sets the list of errors.
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the list of unknown chunks encountered during parsing.
        /// </summary>
        public List<UnknownChunk> UnknownChunks { get; set; } = new List<UnknownChunk>();
        
        /// <summary>
        /// Gets or sets the list of vertex positions.
        /// </summary>
        public List<Services.Vector3> VertexPositions { get; set; } = new List<Services.Vector3>();
        
        /// <summary>
        /// Gets or sets the list of vertex indices.
        /// </summary>
        public List<int> VertexIndices { get; set; } = new List<int>();
        
        /// <summary>
        /// Gets or sets the list of links.
        /// </summary>
        public List<LinkData> Links { get; set; } = new List<LinkData>();
        
        /// <summary>
        /// Gets or sets the list of position data.
        /// </summary>
        public List<PositionData> PositionData { get; set; } = new List<PositionData>();
        
        /// <summary>
        /// Gets or sets the list of position data (alias for PositionData for compatibility).
        /// </summary>
        public List<PositionData> Positions 
        { 
            get => PositionData; 
            set => PositionData = value; 
        }
        
        /// <summary>
        /// Gets or sets the list of position references.
        /// </summary>
        public List<PositionReference> PositionReferences { get; set; } = new List<PositionReference>();
    }
} 