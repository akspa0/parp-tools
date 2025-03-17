using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Models.PD4;
using WCAnalyzer.Core.Models.PM4;

namespace WCAnalyzer.Core.Utilities
{
    /// <summary>
    /// Provides utility methods for parsing and decoding chunk data from PM4 and PD4 files.
    /// </summary>
    public static class ChunkParsingUtility
    {
        // These constants are used for the coordinate transformations
        private const float COORDINATE_OFFSET = 17066.666f;
        private const float HEIGHT_CONVERSION = 36.0f;

        /// <summary>
        /// Safely converts a uint to int with overflow checking.
        /// </summary>
        /// <param name="value">The uint value to convert.</param>
        /// <returns>The converted int value.</returns>
        /// <exception cref="InvalidDataException">Thrown when the uint value exceeds int.MaxValue.</exception>
        public static int SafeUIntToInt(uint value)
        {
            return SafeConversions.UIntToInt(value);
        }
        
        /// <summary>
        /// Reads a chunk from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>A tuple containing the chunk signature and data.</returns>
        private static (string, byte[]) ReadChunk(BinaryReader reader, ILogger? logger = null)
        {
            try
            {
                // Read chunk signature and size
                string signature = new string(reader.ReadChars(4));
                uint size = reader.ReadUInt32();
                
                // Safely convert uint to int
                int safeSize = SafeConversions.UIntToInt(size);
                
                // Read chunk data
                byte[] data = reader.ReadBytes(safeSize);
                
                logger?.LogTrace("Read chunk: {Signature}, Size: {Size} bytes", signature, size);
                
                return (signature, data);
            }
            catch (Exception ex)
            {
                logger?.LogError(ex, "Error reading chunk at position {Position}", reader.BaseStream.Position);
                throw;
            }
        }
        
        /// <summary>
        /// Reads a signed 24-bit integer (int24_t) from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <returns>The 24-bit signed integer as an int.</returns>
        public static int ReadInt24(BinaryReader reader)
        {
            return SafeConversions.ReadInt24(reader);
        }
        
        /// <summary>
        /// Reads a null-terminated string from a binary reader with a maximum length.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="maxLength">The maximum length of the string in bytes.</param>
        /// <returns>The string.</returns>
        public static string ReadNullTerminatedString(BinaryReader reader, int maxLength)
        {
            return SafeConversions.ReadNullTerminatedString(reader, maxLength);
        }
        
        /// <summary>
        /// Builds a hex dump of byte data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="maxBytes">The maximum number of bytes to include. Use 0 for all bytes.</param>
        /// <returns>A formatted hex dump.</returns>
        public static string BuildHexDump(byte[] data, int maxBytes = 128)
        {
            if (data == null || data.Length == 0)
            {
                return "(empty)";
            }

            var sb = new StringBuilder();
            int length = maxBytes > 0 && maxBytes < data.Length ? maxBytes : data.Length;

            for (int i = 0; i < length; i++)
            {
                if (i > 0 && i % 16 == 0)
                {
                    sb.AppendLine();
                }
                else if (i > 0)
                {
                    sb.Append(' ');
                }

                sb.Append(data[i].ToString("X2"));
            }

            if (length < data.Length)
            {
                sb.Append($"... ({data.Length - length} more bytes)");
            }

            return sb.ToString();
        }
        
        /// <summary>
        /// Transforms vertex coordinates from file format to world coordinates.
        /// </summary>
        /// <param name="rawX">The raw X coordinate.</param>
        /// <param name="rawY">The raw Y coordinate.</param>
        /// <param name="rawZ">The raw Z coordinate.</param>
        /// <returns>A Vector3 containing the world coordinates.</returns>
        /// <remarks>
        /// Applies the transformation:
        /// worldPos.y = 17066.666 - position.y;
        /// worldPos.x = 17066.666 - position.x;
        /// worldPos.z = position.z / 36.0f; (to convert internal inch height to yards)
        /// </remarks>
        public static Vector3 TransformToWorldCoordinates(float rawX, float rawY, float rawZ)
        {
            return new Vector3(
                COORDINATE_OFFSET - rawX,
                COORDINATE_OFFSET - rawY,
                rawZ / HEIGHT_CONVERSION
            );
        }
        
        /// <summary>
        /// Parses PM4 chunks from a stream.
        /// </summary>
        /// <param name="stream">The stream containing PM4 chunks.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>A dictionary of chunk signatures and their corresponding IIFFChunk objects.</returns>
        public static Dictionary<string, IIFFChunk> ParsePM4Chunks(Stream stream, ILogger? logger = null)
        {
            var chunks = new Dictionary<string, IIFFChunk>();
            
            using (var reader = new BinaryReader(stream, Encoding.ASCII, true))
            {
                while (stream.Position < stream.Length)
                {
                    try
                    {
                        var (signature, data) = ReadChunk(reader, logger);
                        var chunk = Models.PM4.PM4ChunkFactory.CreateChunk(signature, data);
                        chunks[signature] = chunk;
                        
                        logger?.LogDebug("Parsed PM4 chunk: {Signature}, Size: {Size} bytes", signature, data.Length);
                    }
                    catch (Exception ex)
                    {
                        logger?.LogError(ex, "Error parsing PM4 chunk at position {Position}", stream.Position);
                        break;
                    }
                }
            }
            
            return chunks;
        }
        
        /// <summary>
        /// Parses PD4 chunks from a stream.
        /// </summary>
        /// <param name="stream">The stream containing PD4 chunks.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>A dictionary of chunk signatures and their corresponding IIFFChunk objects.</returns>
        public static Dictionary<string, IIFFChunk> ParsePD4Chunks(Stream stream, ILogger? logger = null)
        {
            var chunks = new Dictionary<string, IIFFChunk>();
            
            using (var reader = new BinaryReader(stream, Encoding.ASCII, true))
            {
                while (stream.Position < stream.Length)
                {
                    try
                    {
                        var (signature, data) = ReadChunk(reader, logger);
                        var chunk = Models.PD4.PD4ChunkFactory.CreateChunk(signature, data);
                        chunks[signature] = chunk;
                        
                        logger?.LogDebug("Parsed PD4 chunk: {Signature}, Size: {Size} bytes", signature, data.Length);
                    }
                    catch (Exception ex)
                    {
                        logger?.LogError(ex, "Error parsing PD4 chunk at position {Position}", stream.Position);
                        break;
                    }
                }
            }
            
            return chunks;
        }
        
        /// <summary>
        /// Creates a memory stream from a file path.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns>A memory stream containing the file contents.</returns>
        public static MemoryStream CreateMemoryStreamFromFile(string filePath)
        {
            // Use a more robust method to ensure proper disposal of the file stream
            var memoryStream = new MemoryStream();
            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                fileStream.CopyTo(memoryStream);
            }
            
            // Reset position to beginning for subsequent reads
            memoryStream.Position = 0;
            return memoryStream;
        }
    }
} 