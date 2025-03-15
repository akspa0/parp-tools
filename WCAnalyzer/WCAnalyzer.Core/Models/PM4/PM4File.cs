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
            summary.AppendLine($"| MSVI (Vertex Indices 2) | {(VertexIndices2 != null ? "Yes" : "No")} | {GetChunkSize(VertexIndices2)} |");
            summary.AppendLine($"| MSUR (Surface Data) | {(SurfaceData != null ? "Yes" : "No")} | {GetChunkSize(SurfaceData)} |");
            summary.AppendLine($"| MPRL (Position Data) | {(PositionData != null ? "Yes" : "No")} | {GetChunkSize(PositionData)} |");
            summary.AppendLine($"| MPRR (Value Pairs) | {(ValuePairs != null ? "Yes" : "No")} | {GetChunkSize(ValuePairs)} |");
            summary.AppendLine($"| MDBH (Building Data) | {(BuildingData != null ? "Yes" : "No")} | {GetChunkSize(BuildingData)} |");
            summary.AppendLine($"| MDOS (Simple Data) | {(SimpleData != null ? "Yes" : "No")} | {GetChunkSize(SimpleData)} |");
            summary.AppendLine($"| MDSF (Final Data) | {(FinalData != null ? "Yes" : "No")} | {GetChunkSize(FinalData)} |");
            
            // Add detailed information for specific chunks
            if (VertexPositionsChunk != null)
            {
                summary.AppendLine("\n## Vertex Positions (MSPV)");
                summary.AppendLine($"- Vertex Count: {VertexPositionsChunk.Vertices.Count}");
                
                if (VertexPositionsChunk.Vertices.Count > 0)
                {
                    summary.AppendLine("\n### Sample Vertices");
                    summary.AppendLine("| Index | X | Y | Z |");
                    summary.AppendLine("|-------|-----|-----|-----|");
                    
                    // Show first 10 vertices as a sample
                    for (int i = 0; i < Math.Min(10, VertexPositionsChunk.Vertices.Count); i++)
                    {
                        var vertex = VertexPositionsChunk.Vertices[i];
                        summary.AppendLine($"| {i} | {vertex.X:F2} | {vertex.Y:F2} | {vertex.Z:F2} |");
                    }
                    
                    if (VertexPositionsChunk.Vertices.Count > 10)
                    {
                        summary.AppendLine($"_Showing 10 of {VertexPositionsChunk.Vertices.Count} vertices_");
                    }
                }
            }
            
            if (VertexIndicesChunk != null)
            {
                summary.AppendLine("\n## Vertex Indices (MSPI)");
                summary.AppendLine($"- Index Count: {VertexIndicesChunk.Indices.Count}");
                
                if (VertexIndicesChunk.Indices.Count > 0)
                {
                    // Calculate triangle count (assuming triangles)
                    int triangleCount = VertexIndicesChunk.Indices.Count / 3;
                    summary.AppendLine($"- Triangle Count: {triangleCount}");
                    
                    // Show first few triangles
                    summary.AppendLine("\n### Sample Triangle Indices");
                    summary.AppendLine("| Triangle | Vertex 1 | Vertex 2 | Vertex 3 |");
                    summary.AppendLine("|----------|----------|----------|----------|");
                    
                    for (int i = 0; i < Math.Min(10, triangleCount); i++)
                    {
                        int baseIndex = i * 3;
                        if (baseIndex + 2 < VertexIndicesChunk.Indices.Count)
                        {
                            summary.AppendLine($"| {i} | {VertexIndicesChunk.Indices[baseIndex]} | " +
                                $"{VertexIndicesChunk.Indices[baseIndex + 1]} | {VertexIndicesChunk.Indices[baseIndex + 2]} |");
                        }
                    }
                    
                    if (triangleCount > 10)
                    {
                        summary.AppendLine($"_Showing 10 of {triangleCount} triangles_");
                    }
                }
            }
            
            if (PositionDataChunk != null)
            {
                summary.AppendLine("\n## Position Data (MPRL)");
                summary.AppendLine($"- Position Entry Count: {PositionDataChunk.Entries.Count}");
                
                if (PositionDataChunk.Entries.Count > 0)
                {
                    // Count position records vs command records
                    int positionRecords = PositionDataChunk.Entries.Count(e => !e.IsControlRecord);
                    int commandRecords = PositionDataChunk.Entries.Count(e => e.IsControlRecord);
                    summary.AppendLine($"- Position Records: {positionRecords}");
                    summary.AppendLine($"- Command/Control Records: {commandRecords}");
                    
                    // Group by CommandValue to show distribution
                    var groupedByCommand = PositionDataChunk.Entries
                        .Where(e => e.IsControlRecord)
                        .GroupBy(e => e.CommandValue)
                        .OrderByDescending(g => g.Count())
                        .ToList();
                        
                    if (groupedByCommand.Any())
                    {
                        summary.AppendLine($"\n### Command Value Distribution");
                        summary.AppendLine("| Command (Hex) | Command (Dec) | Count | Sample Y Value |");
                        summary.AppendLine("|--------------|--------------|-------|----------------|");
                        
                        foreach (var group in groupedByCommand.Take(10))
                        {
                            var sample = group.First();
                            summary.AppendLine($"| 0x{group.Key:X8} | {group.Key} | {group.Count()} | {sample.CoordinateY:F2} |");
                        }
                        
                        if (groupedByCommand.Count > 10)
                        {
                            summary.AppendLine($"_Showing 10 of {groupedByCommand.Count} command values_");
                        }
                    }
                    
                    // Show commands and positions with pattern analysis
                    summary.AppendLine("\n### Command-Position Pattern Analysis");
                    
                    // Find the most common patterns of control records followed by position records
                    var entryPairs = new List<(MPRLChunk.ServerPositionData Command, MPRLChunk.ServerPositionData Position)>();
                    for (int i = 0; i < PositionDataChunk.Entries.Count - 1; i++)
                    {
                        var current = PositionDataChunk.Entries[i];
                        var next = PositionDataChunk.Entries[i + 1];
                        
                        if (current.IsControlRecord && !next.IsControlRecord)
                        {
                            entryPairs.Add((current, next));
                        }
                    }
                    
                    if (entryPairs.Any())
                    {
                        summary.AppendLine($"- Command-Position Pairs: {entryPairs.Count}");
                        
                        var groupedPairs = entryPairs
                            .GroupBy(pair => pair.Command.CommandValue)
                            .OrderByDescending(g => g.Count())
                            .Take(5)
                            .ToList();
                            
                        summary.AppendLine("\n#### Most Common Command-Position Pairs");
                        summary.AppendLine("| Command (Hex) | Count | Example Position |");
                        summary.AppendLine("|--------------|-------|------------------|");
                        
                        foreach (var group in groupedPairs)
                        {
                            var sample = group.First();
                            string posValue = $"({sample.Position.CoordinateX:F2}, {sample.Position.CoordinateY:F2}, {sample.Position.CoordinateZ:F2})";
                            summary.AppendLine($"| 0x{group.Key:X8} | {group.Count()} | {posValue} |");
                        }
                    }
                    
                    // Show position records
                    if (positionRecords > 0)
                    {
                        summary.AppendLine("\n### Position Records");
                        summary.AppendLine("| Index | X | Y | Z |");
                        summary.AppendLine("|-------|------------|----------|----------|");
                        
                        var positions = PositionDataChunk.Entries.Where(e => !e.IsControlRecord).Take(15).ToList();
                        foreach (var entry in positions)
                        {
                            summary.AppendLine($"| {entry.Index} | {entry.CoordinateX:F2} | {entry.CoordinateY:F2} | {entry.CoordinateZ:F2} |");
                        }
                        
                        if (positionRecords > 15)
                        {
                            summary.AppendLine($"_Showing 15 of {positionRecords} position records_");
                        }
                    }
                    
                    // Show command records
                    if (commandRecords > 0)
                    {
                        summary.AppendLine("\n### Command Records");
                        summary.AppendLine("| Index | Command (Hex) | Y Value |");
                        summary.AppendLine("|-------|--------------|---------|");
                        
                        var commands = PositionDataChunk.Entries.Where(e => e.IsControlRecord).Take(15).ToList();
                        foreach (var entry in commands)
                        {
                            summary.AppendLine($"| {entry.Index} | 0x{entry.CommandValue:X8} | {entry.CoordinateY:F2} |");
                        }
                        
                        if (commandRecords > 15)
                        {
                            summary.AppendLine($"_Showing 15 of {commandRecords} command records_");
                        }
                    }
                    
                    // Show sample sequence of entries to demonstrate the pattern
                    summary.AppendLine("\n### Entry Sequence Example");
                    summary.AppendLine("| Index | Type | Value 1 | Value 2 | Value 3 | Command (Hex) |");
                    summary.AppendLine("|-------|------|---------|---------|---------|--------------|");
                    
                    int maxSample = Math.Min(20, PositionDataChunk.Entries.Count);
                    for (int i = 0; i < maxSample; i++)
                    {
                        var entry = PositionDataChunk.Entries[i];
                        string value1 = float.IsNaN(entry.Value1) ? "NaN" : entry.Value1.ToString("F2");
                        string value3 = float.IsNaN(entry.Value3) ? "NaN" : entry.Value3.ToString("F2");
                        string type = entry.IsControlRecord ? "Command" : "Position";
                        string command = entry.IsControlRecord ? $"0x{entry.CommandValue:X8}" : "-";
                        
                        summary.AppendLine($"| {entry.Index} | {type} | {value1} | {entry.Value2:F2} | {value3} | {command} |");
                    }
                    
                    if (PositionDataChunk.Entries.Count > 20)
                    {
                        summary.AppendLine($"_Showing 20 of {PositionDataChunk.Entries.Count} entries_");
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