using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;
using WCAnalyzer.Core.Common.Utils;

namespace WCAnalyzer.Core.Files
{
    /// <summary>
    /// Factory for creating and parsing chunk objects
    /// </summary>
    public abstract class ChunkFactory
    {
        /// <summary>
        /// Logger instance
        /// </summary>
        protected readonly ILogger? Logger;
        
        /// <summary>
        /// Creates a new ChunkFactory with optional logger
        /// </summary>
        /// <param name="logger">Optional logger</param>
        protected ChunkFactory(ILogger? logger = null)
        {
            Logger = logger;
        }
        
        /// <summary>
        /// Read a chunk from a BinaryReader
        /// </summary>
        /// <param name="reader">BinaryReader to read from</param>
        /// <returns>A chunk object or null if reading failed</returns>
        public IChunk? ReadChunk(BinaryReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }
            
            try
            {
                // Check if we have enough data for a chunk header
                if (reader.BaseStream.Position + 8 > reader.BaseStream.Length)
                {
                    return null; // Not enough data left for a chunk
                }
                
                string signature = reader.ReadChunkSignature();
                uint size = reader.ReadUInt32();
                
                // Check if we have enough data for the chunk content
                if (reader.BaseStream.Position + size > reader.BaseStream.Length)
                {
                    Logger?.LogError($"Chunk {signature} claims size {size} but not enough data remains in stream");
                    return null;
                }
                
                // Read the chunk data
                byte[] data = reader.ReadBytes((int)size);
                
                // Create a chunk based on the signature
                Interfaces.IChunk chunk = CreateChunk(signature, data);
                
                // Try to parse the chunk
                if (chunk is IBinarySerializable serializable)
                {
                    serializable.Parse();
                }
                
                // Both IChunk interfaces are implemented by BaseChunk, so this is safe
                return (Common.Interfaces.IChunk)chunk;
            }
            catch (Exception ex)
            {
                Logger?.LogError(ex, "Error reading chunk");
                return null;
            }
        }
        
        /// <summary>
        /// Read all chunks from a BinaryReader until the end of the stream
        /// </summary>
        /// <param name="reader">BinaryReader to read from</param>
        /// <returns>List of chunks read from the stream</returns>
        public List<IChunk> ReadAllChunks(BinaryReader reader)
        {
            List<IChunk> chunks = new List<IChunk>();
            
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }
            
            try
            {
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    IChunk? chunk = ReadChunk(reader);
                    if (chunk != null)
                    {
                        chunks.Add(chunk);
                    }
                    else
                    {
                        // If we failed to read a chunk, skip ahead to try to find the next valid chunk
                        reader.BaseStream.Position += 1;
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError(ex, "Error reading chunks");
            }
            
            return chunks;
        }
        
        /// <summary>
        /// Create a chunk object for the given signature and data
        /// </summary>
        /// <param name="signature">The four-character chunk signature</param>
        /// <param name="data">Raw binary data</param>
        /// <returns>A chunk object appropriate for the signature</returns>
        public abstract Interfaces.IChunk CreateChunk(string signature, byte[] data);
    }
} 