using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.Utilities
{
    /// <summary>
    /// Utility class for parsing chunk-based file formats.
    /// </summary>
    public static class ChunkFileParser
    {
        /// <summary>
        /// Parses all chunks from a stream using a factory method to create the appropriate chunk types.
        /// </summary>
        /// <typeparam name="T">The base chunk type.</typeparam>
        /// <param name="stream">The stream to read from.</param>
        /// <param name="chunkFactory">A function that creates a chunk instance from a signature and data.</param>
        /// <param name="logger">Optional logger.</param>
        /// <returns>A dictionary of chunks indexed by signature.</returns>
        public static Dictionary<string, T> ParseChunks<T>(
            Stream stream,
            Func<string, byte[], T> chunkFactory,
            ILogger? logger = null) where T : IIFFChunk
        {
            var chunks = new Dictionary<string, T>();
            
            using var reader = new BinaryReader(stream, Encoding.ASCII, true);
            while (stream.Position < stream.Length)
            {
                try
                {
                    // Read the chunk signature and reverse it (as per WoW file format)
                    var signatureBytes = reader.ReadBytes(4);
                    Array.Reverse(signatureBytes);
                    var signature = Encoding.ASCII.GetString(signatureBytes);
                    
                    // Read the chunk size
                    var size = reader.ReadUInt32();
                    
                    // Read the chunk data
                    var data = reader.ReadBytes((int)size);
                    
                    // Create and add the chunk
                    var chunk = chunkFactory(signature, data);
                    if (!chunks.ContainsKey(signature))
                    {
                        chunks[signature] = chunk;
                        logger?.LogDebug("Parsed chunk: {Signature}, Size: {Size} bytes", signature, data.Length);
                    }
                    else
                    {
                        logger?.LogWarning("Duplicate chunk: {Signature}, skipping", signature);
                    }
                }
                catch (EndOfStreamException)
                {
                    // End of file reached
                    break;
                }
                catch (Exception ex)
                {
                    // Error reading chunk
                    logger?.LogError(ex, "Error parsing chunk at position {Position}", stream.Position);
                    break;
                }
            }
            
            return chunks;
        }
        
        /// <summary>
        /// Parses all chunks from a byte array using a factory method to create the appropriate chunk types.
        /// </summary>
        /// <typeparam name="T">The base chunk type.</typeparam>
        /// <param name="data">The byte array to read from.</param>
        /// <param name="chunkFactory">A function that creates a chunk instance from a signature and data.</param>
        /// <param name="logger">Optional logger.</param>
        /// <returns>A dictionary of chunks indexed by signature.</returns>
        public static Dictionary<string, T> ParseChunks<T>(
            byte[] data,
            Func<string, byte[], T> chunkFactory,
            ILogger? logger = null) where T : IIFFChunk
        {
            using var stream = new MemoryStream(data);
            return ParseChunks(stream, chunkFactory, logger);
        }
    }
} 