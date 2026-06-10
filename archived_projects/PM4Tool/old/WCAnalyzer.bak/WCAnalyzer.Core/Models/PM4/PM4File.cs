using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.Extensions.Logging;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using MVER = Warcraft.NET.Files.ADT.Chunks.MVER;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Represents a PM4 file, which is a server-side supplementary file to ADTs.
    /// These files are not shipped to the client and are used by the server only.
    /// </summary>
    public class PM4File
    {
        private readonly ILogger<PM4File>? _logger;
        private object? _chunkedFile;

        /// <summary>
        /// Gets or sets the file size in bytes
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// Gets or sets the version chunk.
        /// </summary>
        public MVER? Version { get; set; }

        /// <summary>
        /// Gets or sets the shadow data chunk.
        /// </summary>
        public MSHDChunk? ShadowDataChunk { get; set; }

        /// <summary>
        /// Gets or sets the shadow data chunk (alias for ShadowDataChunk).
        /// </summary>
        public IIFFChunk? ShadowData { get; set; }

        /// <summary>
        /// Gets or sets the vertex positions chunk.
        /// </summary>
        public MSPVChunk? VertexPositionsChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the vertex positions chunk (alias for VertexPositionsChunk).
        /// </summary>
        public IIFFChunk? VertexPositions { get; set; }

        /// <summary>
        /// Gets or sets the vertex indices chunk.
        /// </summary>
        public MSPIChunk? VertexIndicesChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the vertex indices chunk (alias for VertexIndicesChunk).
        /// </summary>
        public IIFFChunk? VertexIndices { get; set; }

        /// <summary>
        /// Gets or sets the normal coordinates chunk.
        /// </summary>
        public MSCNChunk? NormalCoordinatesChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the normal coordinates chunk (alias for NormalCoordinatesChunk).
        /// </summary>
        public IIFFChunk? NormalCoordinates { get; set; }

        /// <summary>
        /// Gets or sets the links chunk.
        /// </summary>
        public MSLKChunk? LinksChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the links chunk (alias for LinksChunk).
        /// </summary>
        public IIFFChunk? Links { get; set; }

        /// <summary>
        /// Gets or sets the vertex info chunk.
        /// </summary>
        public MSVIChunk? VertexInfoChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the vertex info chunk (alias for VertexInfoChunk).
        /// </summary>
        public IIFFChunk? VertexInfo { get; set; }

        /// <summary>
        /// Gets or sets the vertex data chunk.
        /// </summary>
        public MSVTChunk? VertexDataChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the vertex data chunk (alias for VertexDataChunk).
        /// </summary>
        public IIFFChunk? VertexData { get; set; }

        /// <summary>
        /// Gets or sets the surface data chunk.
        /// </summary>
        public MSURChunk? SurfaceDataChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the surface data chunk (alias for SurfaceDataChunk).
        /// </summary>
        public IIFFChunk? SurfaceData { get; set; }

        /// <summary>
        /// Gets or sets the position data chunk.
        /// </summary>
        public MPRLChunk? PositionDataChunk { get; set; }
        
        /// <summary>
        /// Gets or sets the position data chunk (alias for PositionDataChunk).
        /// </summary>
        public IIFFChunk? PositionData { get; set; }

        /// <summary>
        /// Gets or sets the value pairs chunk.
        /// </summary>
        public IIFFChunk? ValuePairs { get; set; }

        /// <summary>
        /// Gets or sets the building data chunk.
        /// </summary>
        public IIFFChunk? BuildingData { get; set; }

        /// <summary>
        /// Gets or sets the simple data chunk.
        /// </summary>
        public IIFFChunk? SimpleData { get; set; }

        /// <summary>
        /// Gets or sets the final data chunk.
        /// </summary>
        public IIFFChunk? FinalData { get; set; }

        /// <summary>
        /// Gets the list of errors encountered during parsing.
        /// </summary>
        public List<string> Errors { get; private set; } = new List<string>();

        /// <summary>
        /// Gets the file name.
        /// </summary>
        public string? FileName { get; private set; }

        /// <summary>
        /// Gets the file path.
        /// </summary>
        public string? FilePath { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4File"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PM4File(ILogger<PM4File>? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4File"/> class.
        /// </summary>
        /// <param name="inData">The binary data to load from.</param>
        /// <param name="fileName">Optional file name for reference.</param>
        /// <param name="filePath">Optional file path for reference.</param>
        /// <param name="logger">Optional logger instance</param>
        public PM4File(byte[] inData, string? fileName = null, string? filePath = null, ILogger<PM4File>? logger = null)
        {
            if (inData == null)
                throw new ArgumentNullException(nameof(inData));

            _logger = logger;
            FileName = fileName;
            FilePath = filePath;

            try
            {
                _logger?.LogDebug("Processing PM4 file: {FileName}", fileName);
                
                // Skip the reflection approach as ChunkedFile is abstract
                // Process the file directly with manual chunk reading
                ProcessManually(inData);
            }
            catch (Exception ex)
            {
                string errorMsg = $"Error parsing PM4 file: {ex.Message}";
                _logger?.LogError(ex, errorMsg);
                Errors.Add(errorMsg);
                
                if (ex.InnerException != null)
                {
                    string innerMsg = $"Inner exception: {ex.InnerException.Message}";
                    _logger?.LogError(ex.InnerException, innerMsg);
                    Errors.Add(innerMsg);
                }
            }
        }

        /// <summary>
        /// Load a PM4 file from disk.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file.</param>
        /// <param name="logger">Optional logger instance</param>
        /// <returns>A new PM4File instance.</returns>
        public static PM4File FromFile(string filePath, ILogger<PM4File>? logger = null)
        {
            try
            {
                byte[] fileData = File.ReadAllBytes(filePath);
                string fileName = Path.GetFileName(filePath);
                
                logger?.LogInformation("Loading PM4 file: {FilePath}", filePath);
                var pm4File = new PM4File(fileData, fileName, filePath, logger);
                pm4File.FileSize = fileData.Length;
                return pm4File;
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Failed to load PM4 file: {FilePath}", filePath);
                var pm4File = new PM4File(logger);
                pm4File.Errors.Add($"Failed to load PM4 file: {ex.Message}");
                return pm4File;
            }
        }

        /// <summary>
        /// Processes the file manually if the reflection approach fails.
        /// </summary>
        /// <param name="inData">The binary data to process.</param>
        private void ProcessManually(byte[] inData)
        {
            try
            {
                _logger?.LogInformation("Starting manual processing of PM4 file");
                using (var ms = new MemoryStream(inData))
                using (var br = new BinaryReader(ms))
                {
                    while (ms.Position < ms.Length)
                    {
                        try
                        {
                            // Read chunk signature (4 bytes)
                            string signature = new string(br.ReadChars(4));

                            // Read chunk size (4 bytes)
                            uint size = br.ReadUInt32();

                            _logger?.LogTrace("Found chunk: {Signature}, Size: {Size} bytes", signature, size);

                            // Read chunk data
                            byte[] data = br.ReadBytes((int)size);

                            // Process the chunk
                            ProcessChunk(signature, data);
                        }
                        catch (EndOfStreamException)
                        {
                            // End of file reached
                            _logger?.LogDebug("End of file reached");
                            break;
                        }
                    }
                }
                
                _logger?.LogInformation("Manual processing of PM4 file completed");
            }
            catch (Exception ex)
            {
                string errorMsg = $"Error in manual processing: {ex.Message}";
                _logger?.LogError(ex, errorMsg);
                Errors.Add(errorMsg);
                if (ex.InnerException != null)
                {
                    string innerMsg = $"Inner exception: {ex.InnerException.Message}";
                    _logger?.LogError(ex.InnerException, innerMsg);
                    Errors.Add(innerMsg);
                }
            }
        }

        /// <summary>
        /// Processes a chunk of data.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The chunk data.</param>
        protected void ProcessChunk(string signature, byte[] data)
        {
            _logger?.LogDebug("Processing chunk with signature: {Signature}", signature);
            
            switch (signature)
            {
                case "MVER":
                case "REVM":
                    Version = new MVER(data);
                    break;

                case "MSHD":
                case "DHSM":
                    ShadowDataChunk = new MSHDChunk(data);
                    ShadowData = ShadowDataChunk;
                    break;

                case "MSPV":
                case "VPSM":
                    VertexPositionsChunk = new MSPVChunk(data);
                    VertexPositions = VertexPositionsChunk;
                    break;

                case "MSPI":
                case "IPSM":
                    VertexIndicesChunk = new MSPIChunk(data);
                    VertexIndices = VertexIndicesChunk;
                    break;

                case "MSCN":
                case "NCSM":
                    NormalCoordinatesChunk = new MSCNChunk(data);
                    NormalCoordinates = NormalCoordinatesChunk;
                    break;

                case "MSLK":
                case "KLSM":
                    LinksChunk = new MSLKChunk(data);
                    Links = LinksChunk;
                    break;

                case "MSVT":
                case "TVSM":
                    VertexDataChunk = new MSVTChunk(data);
                    VertexData = VertexDataChunk;
                    break;

                case "MSVI":
                case "IVSM":
                    VertexInfoChunk = new MSVIChunk(data);
                    VertexInfo = VertexInfoChunk;
                    break;

                case "MSUR":
                case "RUSM":
                    SurfaceDataChunk = new MSURChunk(data);
                    SurfaceData = SurfaceDataChunk;
                    break;

                case "MPRL":
                case "LRPM":
                    PositionDataChunk = new MPRLChunk(data);
                    PositionData = PositionDataChunk;
                    break;

                case "MPRR":
                case "RRPM":
                    ValuePairs = new GenericChunk(signature, data);
                    break;

                case "MDBH":
                case "HBDM":
                    BuildingData = new GenericChunk(signature, data);
                    break;

                case "MDOS":
                case "SODM":
                    SimpleData = new GenericChunk(signature, data);
                    break;

                case "MDSF":
                case "FSDM":
                    FinalData = new GenericChunk(signature, data);
                    break;

                // For other chunks, we still use the generic chunk class
                default:
                    var rawChunk = new GenericChunk(signature, data);
                    _logger?.LogWarning("Used generic chunk for signature: {Signature}", signature);
                    AssignChunkBySignature(signature, rawChunk);
                    break;
            }
        }

        /// <summary>
        /// Assigns a chunk to the appropriate property based on its signature.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="chunk">The chunk to assign.</param>
        private void AssignChunkBySignature(string signature, IIFFChunk chunk)
        {
            switch (signature)
            {
                case "MVER":
                case "REVM":
                    Version = chunk as MVER;
                    break;

                case "MSHD":
                case "DHSM":
                    ShadowDataChunk = chunk as MSHDChunk;
                    ShadowData = ShadowDataChunk;
                    break;

                case "MSPV":
                case "VPSM":
                    VertexPositionsChunk = chunk as MSPVChunk;
                    VertexPositions = VertexPositionsChunk;
                    break;

                case "MSPI":
                case "IPSM":
                    VertexIndicesChunk = chunk as MSPIChunk;
                    VertexIndices = VertexIndicesChunk;
                    break;

                case "MSCN":
                case "NCSM":
                    NormalCoordinatesChunk = chunk as MSCNChunk;
                    NormalCoordinates = NormalCoordinatesChunk;
                    break;

                case "MSLK":
                case "KLSM":
                    LinksChunk = chunk as MSLKChunk;
                    Links = LinksChunk;
                    break;

                case "MSVT":
                case "TVSM":
                    VertexDataChunk = chunk as MSVTChunk;
                    VertexData = VertexDataChunk;
                    break;

                case "MSVI":
                case "IVSM":
                    VertexInfoChunk = chunk as MSVIChunk;
                    VertexInfo = VertexInfoChunk;
                    break;

                case "MSUR":
                case "RUSM":
                    SurfaceDataChunk = chunk as MSURChunk;
                    SurfaceData = SurfaceDataChunk;
                    break;

                case "MPRL":
                case "LRPM":
                    PositionDataChunk = chunk as MPRLChunk;
                    PositionData = PositionDataChunk;
                    break;

                case "MPRR":
                case "RRPM":
                    ValuePairs = chunk;
                    break;

                case "MDBH":
                case "HBDM":
                    BuildingData = chunk;
                    break;

                case "MDOS":
                case "SODM":
                    SimpleData = chunk;
                    break;

                case "MDSF":
                case "FSDM":
                    FinalData = chunk;
                    break;
            }
        }

        /// <summary>
        /// Provides a simple summary of the PM4 file.
        /// </summary>
        /// <returns>A string containing basic information about the PM4 file.</returns>
        public string GetSummary()
        {
            var summary = new System.Text.StringBuilder();
            
            summary.AppendLine($"# PM4 File: {FileName}");
            summary.AppendLine($"## File Information");
            summary.AppendLine($"- Path: {FilePath}");
            summary.AppendLine($"- Version: {Version?.Version ?? 0}");
            
            // Summarize which chunks are present
            summary.AppendLine("\n## Chunks Present");
            summary.AppendLine("| Chunk | Present | Size (bytes) |");
            summary.AppendLine("|-------|---------|-------------|");
            summary.AppendLine($"| MSHD (Shadow Data) | {(ShadowData != null ? "Yes" : "No")} | {GetChunkSize(ShadowData)} |");
            summary.AppendLine($"| MSPV (Vertex Positions) | {(VertexPositions != null ? "Yes" : "No")} | {GetChunkSize(VertexPositions)} |");
            summary.AppendLine($"| MSPI (Vertex Indices) | {(VertexIndices != null ? "Yes" : "No")} | {GetChunkSize(VertexIndices)} |");
            summary.AppendLine($"| MSCN (Normal Coordinates) | {(NormalCoordinates != null ? "Yes" : "No")} | {GetChunkSize(NormalCoordinates)} |");
            summary.AppendLine($"| MSLK (Links) | {(Links != null ? "Yes" : "No")} | {GetChunkSize(Links)} |");
            summary.AppendLine($"| MSVT (Vertex Data) | {(VertexData != null ? "Yes" : "No")} | {GetChunkSize(VertexData)} |");
            summary.AppendLine($"| MSVI (Vertex Indices 2) | {(VertexInfo != null ? "Yes" : "No")} | {GetChunkSize(VertexInfo)} |");
            summary.AppendLine($"| MSUR (Surface Data) | {(SurfaceData != null ? "Yes" : "No")} | {GetChunkSize(SurfaceData)} |");
            summary.AppendLine($"| MPRL (Position Data) | {(PositionData != null ? "Yes" : "No")} | {GetChunkSize(PositionData)} |");
            summary.AppendLine($"| MPRR (Value Pairs) | {(ValuePairs != null ? "Yes" : "No")} | {GetChunkSize(ValuePairs)} |");
            summary.AppendLine($"| MDBH (Building Data) | {(BuildingData != null ? "Yes" : "No")} | {GetChunkSize(BuildingData)} |");
            summary.AppendLine($"| MDOS (Simple Data) | {(SimpleData != null ? "Yes" : "No")} | {GetChunkSize(SimpleData)} |");
            summary.AppendLine($"| MDSF (Final Data) | {(FinalData != null ? "Yes" : "No")} | {GetChunkSize(FinalData)} |");
            
            // Add detailed information for specific chunks
            if (ShadowDataChunk != null)
            {
                summary.AppendLine("\nShadow Data Details:");
                summary.AppendLine("-----------------");
                summary.AppendLine($"Count: {ShadowDataChunk.ShadowEntries.Count} entries");
                
                // Display a sample of the shadow data
                if (ShadowDataChunk.ShadowEntries.Count > 0)
                {
                    summary.AppendLine("\nSample Shadow Entries:");
                    for (int i = 0; i < Math.Min(5, ShadowDataChunk.ShadowEntries.Count); i++)
                    {
                        var entry = ShadowDataChunk.ShadowEntries[i];
                        summary.AppendLine($"  Entry {i}: Value1={entry.Value1}, Value2={entry.Value2}, Value3={entry.Value3}, Value4={entry.Value4}");
                    }
                }
            }

            if (VertexPositionsChunk != null)
            {
                summary.AppendLine("\nVertex Positions Details:");
                summary.AppendLine("------------------------");
                summary.AppendLine($"Count: {VertexPositionsChunk.Vertices.Count} vertices");
                
                // Display a sample of the vertex positions
                if (VertexPositionsChunk.Vertices.Count > 0)
                {
                    summary.AppendLine("\nSample Vertex Positions:");
                    for (int i = 0; i < Math.Min(5, VertexPositionsChunk.Vertices.Count); i++)
                    {
                        var vertex = VertexPositionsChunk.Vertices[i];
                        summary.AppendLine($"  Vertex {i}: X={vertex.X}, Y={vertex.Y}, Z={vertex.Z}");
                    }
                }
            }

            if (VertexIndicesChunk != null)
            {
                summary.AppendLine("\nVertex Indices Details:");
                summary.AppendLine("----------------------");
                summary.AppendLine($"Count: {VertexIndicesChunk.Indices.Count} indices");
                
                // Display a sample of the vertex indices
                if (VertexIndicesChunk.Indices.Count > 0)
                {
                    summary.AppendLine("\nSample Vertex Indices:");
                    for (int i = 0; i < Math.Min(15, VertexIndicesChunk.Indices.Count); i += 3)
                    {
                        if (i + 2 < VertexIndicesChunk.Indices.Count)
                        {
                            summary.AppendLine($"  Triangle {i/3}: [{VertexIndicesChunk.Indices[i]}, {VertexIndicesChunk.Indices[i+1]}, {VertexIndicesChunk.Indices[i+2]}]");
                        }
                    }
                }
            }

            if (NormalCoordinatesChunk != null)
            {
                summary.AppendLine("\nNormal Coordinates Details:");
                summary.AppendLine("--------------------------");
                summary.AppendLine($"Count: {NormalCoordinatesChunk.Normals.Count} normals");
                
                // Display a sample of the normals
                if (NormalCoordinatesChunk.Normals.Count > 0)
                {
                    summary.AppendLine("\nSample Normal Vectors:");
                    for (int i = 0; i < Math.Min(5, NormalCoordinatesChunk.Normals.Count); i++)
                    {
                        var normal = NormalCoordinatesChunk.Normals[i];
                        summary.AppendLine($"  Normal {i}: X={normal.X}, Y={normal.Y}, Z={normal.Z}");
                    }
                }
            }

            if (LinksChunk != null)
            {
                summary.AppendLine("\nLinks Details:");
                summary.AppendLine("--------------");
                summary.AppendLine($"Count: {LinksChunk.Links.Count} links");
                
                // Display a sample of the links
                if (LinksChunk.Links.Count > 0)
                {
                    summary.AppendLine("\nSample Links:");
                    for (int i = 0; i < Math.Min(5, LinksChunk.Links.Count); i++)
                    {
                        var link = LinksChunk.Links[i];
                        summary.AppendLine($"  Link {i}: Source={link.SourceIndex}, Target={link.TargetIndex}");
                    }
                }
            }

            if (VertexInfoChunk != null)
            {
                summary.AppendLine("\nVertex Info Details:");
                summary.AppendLine("-------------------");
                summary.AppendLine($"Count: {VertexInfoChunk.VertexInfos.Count} entries");
                
                // Display a sample of the vertex info entries
                if (VertexInfoChunk.VertexInfos.Count > 0)
                {
                    summary.AppendLine("\nSample Vertex Info Entries:");
                    for (int i = 0; i < Math.Min(5, VertexInfoChunk.VertexInfos.Count); i++)
                    {
                        var info = VertexInfoChunk.VertexInfos[i];
                        summary.AppendLine($"  Info {i}: Value1={info.Value1}, Value2={info.Value2}");
                    }
                }
            }

            if (VertexDataChunk != null)
            {
                summary.AppendLine("\nVertex Data Details:");
                summary.AppendLine("-------------------");
                summary.AppendLine($"Count: {VertexDataChunk.Vertices.Count} entries");
                
                // Display a sample of the vertex data entries
                if (VertexDataChunk.Vertices.Count > 0)
                {
                    summary.AppendLine("\nSample Vertex Data Entries:");
                    for (int i = 0; i < Math.Min(5, VertexDataChunk.Vertices.Count); i++)
                    {
                        var vertex = VertexDataChunk.Vertices[i];
                        summary.AppendLine($"  Vertex {i}: X={vertex.X}, Y={vertex.Y}, Z={vertex.Z}, Flag1={vertex.Flag1}, Flag2={vertex.Flag2}");
                    }
                }
            }

            if (SurfaceDataChunk != null)
            {
                summary.AppendLine("\nSurface Data Details:");
                summary.AppendLine("---------------------");
                summary.AppendLine($"Count: {SurfaceDataChunk.Surfaces.Count} surfaces");
                
                // Display a sample of the surface data entries
                if (SurfaceDataChunk.Surfaces.Count > 0)
                {
                    summary.AppendLine("\nSample Surface Data Entries:");
                    for (int i = 0; i < Math.Min(5, SurfaceDataChunk.Surfaces.Count); i++)
                    {
                        var surface = SurfaceDataChunk.Surfaces[i];
                        summary.AppendLine($"  Surface {i}: Index1={surface.Index1}, Index2={surface.Index2}, Index3={surface.Index3}, Flags={surface.Flags}");
                    }
                }
            }
            
            if (Errors.Count > 0)
            {
                summary.AppendLine("\n## Errors");
                foreach (var error in Errors)
                {
                    summary.AppendLine($"- {error}");
                }
            }
            
            return summary.ToString();
        }

        /// <summary>
        /// Gets the size of a chunk safely
        /// </summary>
        private uint GetChunkSize(IIFFChunk? chunk)
        {
            if (chunk == null) return 0;
            
            // Try to cast to PM4Chunk which has a Size property
            if (chunk is PM4Chunk pm4Chunk)
            {
                return pm4Chunk.Size;
            }
            
            // For other IIFFChunk implementations, try to get the Data property via reflection
            var dataProperty = chunk.GetType().GetProperty("Data");
            if (dataProperty != null)
            {
                var data = dataProperty.GetValue(chunk) as byte[];
                if (data != null)
                {
                    return (uint)data.Length;
                }
            }
            
            return 0;
        }
    }
} 