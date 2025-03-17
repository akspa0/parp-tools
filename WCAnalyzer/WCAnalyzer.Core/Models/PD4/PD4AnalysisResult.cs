using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WCAnalyzer.Core.Services;
using WCAnalyzer.Core.Models.PD4.Chunks;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Analysis result for a PD4 file.
    /// </summary>
    public class PD4AnalysisResult
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
        /// Gets or sets the parsed PD4 data.
        /// </summary>
        public PD4Data PD4Data { get; set; } = new PD4Data();

        /// <summary>
        /// Gets or sets a value indicating whether the analysis was successful.
        /// </summary>
        public bool Success { get; set; } = true;

        /// <summary>
        /// Gets the PD4 file for backward compatibility.
        /// </summary>
        public PD4File PD4File => null;

        /// <summary>
        /// Gets a value indicating whether the analysis has errors.
        /// </summary>
        public bool HasErrors => Errors.Count > 0 || (PD4Data?.Errors.Count > 0);

        /// <summary>
        /// Gets a value indicating whether the file has vertex positions.
        /// </summary>
        public bool HasVertexPositions => PD4Data?.VertexPositions?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has vertex indices.
        /// </summary>
        public bool HasVertexIndices => PD4Data?.VertexIndices?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has texture names.
        /// </summary>
        public bool HasTextureNames => PD4Data?.TextureNames?.Count > 0;

        /// <summary>
        /// Gets a value indicating whether the file has material data.
        /// </summary>
        public bool HasMaterialData => PD4Data?.MaterialData?.Count > 0;

        // Backward compatibility properties
        public bool HasVersion => Version > 0;
        public bool HasCRC => false;
        public bool HasShadowData => false;
        public bool HasNormalCoordinates => false;
        public bool HasLinks => false;
        public bool HasVertexData => HasVertexPositions;
        public bool HasVertexInfo => HasVertexIndices;
        public bool HasSurfaceData => false;
        public int Version => PD4Data?.Version ?? 0;
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
                sb.AppendLine($"Errors: {Errors.Count + (PD4Data?.Errors.Count ?? 0)}");
                foreach (var error in Errors)
                {
                    sb.AppendLine($"- {error}");
                }
                
                if (PD4Data?.Errors != null)
                {
                    foreach (var error in PD4Data.Errors)
                    {
                        sb.AppendLine($"- {error}");
                    }
                }
            }
            
            if (PD4Data != null)
            {
                sb.AppendLine($"Version: {PD4Data.Version}");
                sb.AppendLine($"Vertex Positions: {PD4Data.VertexPositions?.Count ?? 0}");
                sb.AppendLine($"Vertex Indices: {PD4Data.VertexIndices?.Count ?? 0}");
                sb.AppendLine($"Texture Names: {PD4Data.TextureNames?.Count ?? 0}");
                sb.AppendLine($"Material Data: {PD4Data.MaterialData?.Count ?? 0}");
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
            
            if (PD4Data != null)
            {
                // Add more detailed information about the file content
                if (HasVertexPositions && PD4Data.VertexPositions != null)
                {
                    sb.AppendLine("\nVertex Positions (first 10):");
                    for (int i = 0; i < Math.Min(10, PD4Data.VertexPositions.Count); i++)
                    {
                        sb.AppendLine($"  {i}: {PD4Data.VertexPositions[i]}");
                    }
                }
                
                if (HasVertexIndices && PD4Data.VertexIndices != null)
                {
                    sb.AppendLine("\nVertex Indices (first 10):");
                    for (int i = 0; i < Math.Min(10, PD4Data.VertexIndices.Count); i++)
                    {
                        sb.AppendLine($"  {i}: {PD4Data.VertexIndices[i]}");
                    }
                }
                
                if (HasTextureNames && PD4Data.TextureNames != null)
                {
                    sb.AppendLine("\nTexture Names:");
                    for (int i = 0; i < PD4Data.TextureNames.Count; i++)
                    {
                        sb.AppendLine($"  {i}: {PD4Data.TextureNames[i]}");
                    }
                }
                
                if (HasMaterialData && PD4Data.MaterialData != null)
                {
                    sb.AppendLine("\nMaterial Data (first 10):");
                    for (int i = 0; i < Math.Min(10, PD4Data.MaterialData.Count); i++)
                    {
                        var material = PD4Data.MaterialData[i];
                        sb.AppendLine($"  {i}: TextureIndex={material.TextureIndex}, Flags={material.Flags:X8}, Value1={material.Value1}, Value2={material.Value2}");
                    }
                }
            }
            
            return sb.ToString();
        }
    }
    
    /// <summary>
    /// Represents the parsed data from a PD4 file.
    /// </summary>
    public class PD4Data
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
        /// Gets or sets the list of vertex positions.
        /// </summary>
        public List<Services.Vector3> VertexPositions { get; set; } = new List<Services.Vector3>();
        
        /// <summary>
        /// Gets or sets the list of vertex indices.
        /// </summary>
        public List<int> VertexIndices { get; set; } = new List<int>();
        
        /// <summary>
        /// Gets or sets the list of texture names.
        /// </summary>
        public List<string> TextureNames { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the list of material data.
        /// </summary>
        public List<MaterialData> MaterialData { get; set; } = new List<MaterialData>();
    }
} 