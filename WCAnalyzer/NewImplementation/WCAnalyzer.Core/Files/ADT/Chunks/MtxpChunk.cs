using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MTXP chunk - Contains texture paths
    /// </summary>
    public class MtxpChunk : ADTChunk
    {
        /// <summary>
        /// The MTXP chunk signature
        /// </summary>
        public const string SIGNATURE = "MTXP";

        /// <summary>
        /// Gets the list of texture paths
        /// </summary>
        public List<string> TexturePaths { get; } = new List<string>();

        /// <summary>
        /// Gets the raw data for the paths section
        /// </summary>
        public byte[] PathsData { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MtxpChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MtxpChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MTXP chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MTXP chunk has no data");
                    return;
                }

                // Store the raw data for later use with offsets
                PathsData = new byte[Data.Length];
                Array.Copy(Data, PathsData, Data.Length);
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Read paths (null-terminated strings)
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    var path = ReadNullTerminatedString(reader);
                    if (!string.IsNullOrEmpty(path))
                    {
                        TexturePaths.Add(path);
                    }
                }
                
                Logger?.LogDebug($"MTXP: Read {TexturePaths.Count} texture paths");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MTXP chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MTXP chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the total size of all paths with null terminators
                int totalSize = 0;
                foreach (var path in TexturePaths)
                {
                    totalSize += Encoding.ASCII.GetByteCount(path) + 1; // +1 for null terminator
                }
                
                writer.Write(totalSize);
                
                // Write each path with a null terminator
                foreach (var path in TexturePaths)
                {
                    byte[] pathBytes = Encoding.ASCII.GetBytes(path);
                    writer.Write(pathBytes);
                    writer.Write((byte)0); // Null terminator
                }
                
                Logger?.LogDebug($"MTXP: Wrote {TexturePaths.Count} texture paths");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MTXP chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the path at the specified offset in the raw data
        /// </summary>
        /// <param name="offset">The offset into the raw data</param>
        /// <returns>The path at the offset, or null if the offset is invalid</returns>
        public string GetPathAtOffset(int offset)
        {
            if (PathsData == null || offset < 0 || offset >= PathsData.Length)
            {
                return null;
            }
            
            try
            {
                List<byte> pathBytes = new List<byte>();
                int i = offset;
                
                // Read bytes until we hit a null terminator or the end of the data
                while (i < PathsData.Length && PathsData[i] != 0)
                {
                    pathBytes.Add(PathsData[i]);
                    i++;
                }
                
                return Encoding.ASCII.GetString(pathBytes.ToArray());
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error getting path at offset {offset}: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Reads a null-terminated string from the binary reader
        /// </summary>
        /// <param name="reader">The binary reader to read from</param>
        /// <returns>The string that was read</returns>
        private static string ReadNullTerminatedString(BinaryReader reader)
        {
            var bytes = new List<byte>();
            byte b;
            
            // Read bytes until we hit a null terminator or the end of the stream
            while (reader.BaseStream.Position < reader.BaseStream.Length && (b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
    }
} 