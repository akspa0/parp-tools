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
        /// Gets or sets the version chunk.
        /// </summary>
        public MVER? Version { get; set; }

        /// <summary>
        /// Gets or sets the shadow data chunk.
        /// </summary>
        public MSHDChunk? ShadowDataChunk { get; set; }

        /// <summary>
        /// Gets or sets the vertex positions chunk.
        /// </summary>
        public MSPVChunk? VertexPositionsChunk { get; set; }

        /// <summary>
        /// Gets or sets the vertex indices chunk.
        /// </summary>
        public MSPIChunk? VertexIndicesChunk { get; set; }

        /// <summary>
        /// Gets or sets the normal coordinates chunk.
        /// </summary>
        public IIFFChunk? NormalCoordinates { get; set; }

        /// <summary>
        /// Gets or sets the links chunk.
        /// </summary>
        public IIFFChunk? Links { get; set; }

        /// <summary>
        /// Gets or sets the vertex data chunk.
        /// </summary>
        public IIFFChunk? VertexData { get; set; }

        /// <summary>
        /// Gets or sets the vertex indices chunk.
        /// </summary>
        public IIFFChunk? VertexIndices2 { get; set; }

        /// <summary>
        /// Gets or sets the surface data chunk.
        /// </summary>
        public IIFFChunk? SurfaceData { get; set; }

        /// <summary>
        /// Gets or sets the position data chunk.
        /// </summary>
        public MPRLChunk? PositionDataChunk { get; set; }

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
                return new PM4File(fileData, fileName, filePath, logger);
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
                    NormalCoordinates = new GenericChunk(signature, data);
                    break;

                case "MSLK":
                case "KLSM":
                    Links = new GenericChunk(signature, data);
                    break;

                case "MSVT":
                case "TVSM":
                    VertexData = new GenericChunk(signature, data);
                    break;

                case "MSVI":
                case "IVSM":
                    VertexIndices2 = new GenericChunk(signature, data);
                    break;

                case "MSUR":
                case "RUSM":
                    SurfaceData = new GenericChunk(signature, data);
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
                    NormalCoordinates = chunk;
                    break;

                case "MSLK":
                case "KLSM":
                    Links = chunk;
                    break;

                case "MSVT":
                case "TVSM":
                    VertexData = chunk;
                    break;

                case "MSVI":
                case "IVSM":
                    VertexIndices2 = chunk;
                    break;

                case "MSUR":
                case "RUSM":
                    SurfaceData = chunk;
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
            summary.AppendLine($"PM4 File: {FileName}");
            summary.AppendLine($"Path: {FilePath}");
            summary.AppendLine($"Version: {(Version != null ? Version.Version.ToString() : "Unknown")}");
            
            if (Errors.Count > 0)
            {
                summary.AppendLine("\nErrors encountered:");
                foreach (var error in Errors)
                {
                    summary.AppendLine($"- {error}");
                }
            }

            summary.AppendLine("\nChunks present:");
            if (Version != null) summary.AppendLine("- MVER (Version)");
            if (ShadowData != null) summary.AppendLine("- MSHD (Shadow Data)");
            if (VertexPositions != null) summary.AppendLine("- MSPV (Vertex Positions)");
            if (VertexIndices != null) summary.AppendLine("- MSPI (Vertex Indices)");
            if (NormalCoordinates != null) summary.AppendLine("- MSCN (Normal Coordinates)");
            if (Links != null) summary.AppendLine("- MSLK (Links)");
            if (VertexData != null) summary.AppendLine("- MSVT (Vertex Data)");
            if (VertexIndices2 != null) summary.AppendLine("- MSVI (Vertex Indices)");
            if (SurfaceData != null) summary.AppendLine("- MSUR (Surface Data)");
            if (PositionData != null) summary.AppendLine("- MPRL (Position Data)");
            if (ValuePairs != null) summary.AppendLine("- MPRR (Value Pairs)");
            if (BuildingData != null) summary.AppendLine("- MDBH (Building Data)");
            if (SimpleData != null) summary.AppendLine("- MDOS (Simple Data)");
            if (FinalData != null) summary.AppendLine("- MDSF (Final Data)");

            // Add detailed chunk information
            summary.AppendLine("\nDetailed Chunk Information:");
            
            if (VertexPositionsChunk != null)
            {
                summary.AppendLine($"\nMSPV (Vertex Positions):");
                summary.AppendLine($"- Vertex Count: {VertexPositionsChunk.Vertices.Count}");
                if (VertexPositionsChunk.Vertices.Count > 0)
                {
                    summary.AppendLine($"- First 5 Vertices:");
                    for (int i = 0; i < Math.Min(5, VertexPositionsChunk.Vertices.Count); i++)
                    {
                        var v = VertexPositionsChunk.Vertices[i];
                        summary.AppendLine($"  {i}: ({v.X}, {v.Y}, {v.Z})");
                    }
                }
            }
            
            if (VertexIndicesChunk != null)
            {
                summary.AppendLine($"\nMSPI (Vertex Indices):");
                summary.AppendLine($"- Index Count: {VertexIndicesChunk.Indices.Count}");
                if (VertexIndicesChunk.Indices.Count > 0)
                {
                    summary.AppendLine($"- First 10 Indices:");
                    for (int i = 0; i < Math.Min(10, VertexIndicesChunk.Indices.Count); i++)
                    {
                        summary.AppendLine($"  {i}: {VertexIndicesChunk.Indices[i]}");
                    }
                }
            }
            
            if (PositionDataChunk != null)
            {
                summary.AppendLine($"\nMPRL (Position Data):");
                summary.AppendLine($"- Entry Count: {PositionDataChunk.Entries.Count}");
                if (PositionDataChunk.Entries.Count > 0)
                {
                    summary.AppendLine($"- First 5 Entries:");
                    for (int i = 0; i < Math.Min(5, PositionDataChunk.Entries.Count); i++)
                    {
                        var entry = PositionDataChunk.Entries[i];
                        summary.AppendLine($"  {i}: FileDataID={entry.FileDataID}, Pos=({entry.Position.X}, {entry.Position.Y}, {entry.Position.Z}), Rot=({entry.Rotation.X}, {entry.Rotation.Y}, {entry.Rotation.Z}), Scale={entry.Scale}");
                    }
                }
            }
            
            return summary.ToString();
        }
    }
} 