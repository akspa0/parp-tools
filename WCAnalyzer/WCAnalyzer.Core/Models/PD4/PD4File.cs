using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Models.PD4.Chunks;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Represents a PD4 file, which is a server-side supplementary file to WMOs.
    /// These files are not shipped to the client and are used by the server only.
    /// </summary>
    public class PD4File
    {
        private readonly ILogger<PD4File>? _logger;

        /// <summary>
        /// Gets or sets the file size in bytes
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// Gets the version chunk.
        /// </summary>
        public MVERChunk? VersionChunk { get; private set; }

        /// <summary>
        /// Gets the version chunk (alias for VersionChunk).
        /// </summary>
        public IIFFChunk? Version => VersionChunk;

        /// <summary>
        /// Gets the CRC chunk.
        /// </summary>
        public MCRCChunk? CrcChunk { get; private set; }

        /// <summary>
        /// Gets the header chunk.
        /// </summary>
        public MSHDChunk? HeaderChunk { get; private set; }
        
        /// <summary>
        /// Gets the shadow data chunk (alias for HeaderChunk).
        /// </summary>
        public MSHDChunk? ShadowDataChunk => HeaderChunk;

        /// <summary>
        /// Gets the vertex positions chunk.
        /// </summary>
        public MSPVChunk? VertexPositionsChunk { get; private set; }

        /// <summary>
        /// Gets the vertex indices chunk.
        /// </summary>
        public MSPIChunk? VertexIndicesChunk { get; private set; }

        /// <summary>
        /// Gets the normal coordinates chunk.
        /// </summary>
        public MSCNChunk? NormalCoordinatesChunk { get; private set; }

        /// <summary>
        /// Gets the link data chunk.
        /// </summary>
        public MSLKChunk? LinkDataChunk { get; private set; }

        /// <summary>
        /// Gets the link data chunk (alias for LinkDataChunk).
        /// </summary>
        public MSLKChunk? LinksChunk => LinkDataChunk;

        /// <summary>
        /// Gets the vertex data chunk.
        /// </summary>
        public MSVTChunk? VertexDataChunk { get; private set; }

        /// <summary>
        /// Gets the vertex indices data chunk.
        /// </summary>
        public MSVIChunk? VertexIndicesDataChunk { get; private set; }
        
        /// <summary>
        /// Gets the vertex info chunk (alias for VertexIndicesDataChunk).
        /// </summary>
        public MSVIChunk? VertexInfoChunk => VertexIndicesDataChunk;

        /// <summary>
        /// Gets the surface data chunk.
        /// </summary>
        public MSURChunk? SurfaceDataChunk { get; private set; }

        /// <summary>
        /// Gets all chunks in the file.
        /// </summary>
        public Dictionary<string, IIFFChunk> Chunks { get; private set; } = new Dictionary<string, IIFFChunk>();

        /// <summary>
        /// Gets the list of errors encountered during parsing.
        /// </summary>
        public List<string> Errors { get; } = new List<string>();

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4File"/> class.
        /// </summary>
        /// <param name="logger">Optional logger for logging operations.</param>
        public PD4File(ILogger<PD4File>? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Creates a new instance of the <see cref="PD4File"/> class from the specified file.
        /// </summary>
        /// <param name="filePath">Path to the PD4 file.</param>
        /// <param name="logger">Optional logger for logging operations.</param>
        /// <returns>A new instance of the <see cref="PD4File"/> class.</returns>
        public static PD4File FromFile(string filePath, ILogger<PD4File>? logger = null)
        {
            var pd4File = new PD4File(logger);
            
            try
            {
                // Read the file
                byte[] fileData = File.ReadAllBytes(filePath);
                pd4File.FileSize = fileData.Length;
                
                // Parse the chunks
                using (var ms = new MemoryStream(fileData))
                {
                    pd4File.Chunks = ChunkParsingUtility.ParsePD4Chunks(ms, logger);
                    
                    // Process the chunks
                    foreach (var chunk in pd4File.Chunks)
                    {
                        pd4File.AssignChunk(chunk.Key, chunk.Value);
                    }
                }
            }
            catch (Exception ex)
            {
                pd4File.Errors.Add($"Error opening file: {ex.Message}");
                logger?.LogError(ex, "Error opening file {FilePath}", filePath);
            }
            
            return pd4File;
        }

        /// <summary>
        /// Assigns a chunk to the appropriate property.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="chunk">The chunk object.</param>
        private void AssignChunk(string signature, IIFFChunk chunk)
        {
            try
            {
                switch (signature)
                {
                    case "MVER":
                        if (chunk is MVERChunk mver)
                            VersionChunk = mver;
                        break;
                        
                    case "MCRC":
                        if (chunk is MCRCChunk mcrc)
                            CrcChunk = mcrc;
                        break;
                        
                    case "MSHD":
                        if (chunk is MSHDChunk mshd)
                            HeaderChunk = mshd;
                        break;
                        
                    case "MSPV":
                        if (chunk is MSPVChunk mspv)
                            VertexPositionsChunk = mspv;
                        break;
                        
                    case "MSPI":
                        if (chunk is MSPIChunk mspi)
                            VertexIndicesChunk = mspi;
                        break;
                        
                    case "MSCN":
                        if (chunk is MSCNChunk mscn)
                            NormalCoordinatesChunk = mscn;
                        break;
                        
                    case "MSLK":
                        if (chunk is MSLKChunk mslk)
                            LinkDataChunk = mslk;
                        break;
                        
                    case "MSVT":
                        if (chunk is MSVTChunk msvt)
                            VertexDataChunk = msvt;
                        break;
                        
                    case "MSVI":
                        if (chunk is MSVIChunk msvi)
                            VertexIndicesDataChunk = msvi;
                        break;
                        
                    case "MSUR":
                        if (chunk is MSURChunk msur)
                            SurfaceDataChunk = msur;
                        break;
                        
                    default:
                        _logger?.LogDebug("Unknown chunk type: {Signature}", signature);
                        break;
                }
            }
            catch (Exception ex)
            {
                Errors.Add($"Error processing chunk {signature}: {ex.Message}");
                _logger?.LogError(ex, "Error processing chunk {Signature}", signature);
            }
        }

        /// <summary>
        /// Gets a summary of the file.
        /// </summary>
        /// <returns>A string containing a summary of the file.</returns>
        public string GetSummary()
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("PD4 File Summary:");
            sb.AppendLine($"File Size: {FileSize:N0} bytes");
            sb.AppendLine($"Version: {VersionChunk?.Version ?? 0}");
            sb.AppendLine($"CRC: {CrcChunk?.Crc().ToString("X8") ?? "N/A"}");
            
            if (HeaderChunk != null)
            {
                sb.AppendLine($"Header Info:");
                sb.AppendLine($"  Flags: 0x{HeaderChunk.Flags():X8}");
                sb.AppendLine($"  Unknown Values: [{HeaderChunk.Value0x04()}, {HeaderChunk.Value0x08()}, {HeaderChunk.Value0x0c()}]");
            }
            
            sb.AppendLine($"Vertex Positions: {VertexPositionsChunk?.Positions().Count ?? 0}");
            sb.AppendLine($"Vertex Indices: {VertexIndicesChunk?.Indices.Count ?? 0}");
            sb.AppendLine($"Normal Coordinates: {NormalCoordinatesChunk?.Normals.Count ?? 0}");
            sb.AppendLine($"Link Entries: {LinkDataChunk?.Entries.Count ?? 0}");
            sb.AppendLine($"Transformed Vertices: {VertexDataChunk?.Vertices.Count ?? 0}");
            sb.AppendLine($"Vertex Indices Data: {VertexIndicesDataChunk?.Entries.Count ?? 0}");
            sb.AppendLine($"Surface Data Entries: {SurfaceDataChunk?.Entries.Count ?? 0}");
            
            if (Errors.Count > 0)
            {
                sb.AppendLine("\nErrors:");
                foreach (var error in Errors)
                {
                    sb.AppendLine($"- {error}");
                }
            }
            
            return sb.ToString();
        }
    }
} 