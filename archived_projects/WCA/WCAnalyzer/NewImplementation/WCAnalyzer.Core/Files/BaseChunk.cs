using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;
using WCAnalyzer.Core.Common.Utils;

namespace WCAnalyzer.Core.Files
{
    /// <summary>
    /// Base implementation for all chunk types
    /// </summary>
    public abstract class BaseChunk : Common.Interfaces.IChunk, Interfaces.IChunk, IBinarySerializable
    {
        /// <summary>
        /// Logger instance
        /// </summary>
        protected readonly ILogger? Logger;
        
        /// <summary>
        /// Gets the four-character chunk signature
        /// </summary>
        public string Signature { get; }
        
        /// <summary>
        /// Gets the raw binary data of this chunk
        /// </summary>
        public byte[] Data { get; protected set; }
        
        /// <summary>
        /// Gets the size of the chunk data in bytes
        /// </summary>
        public uint Size => Data != null ? (uint)Data.Length : 0;
        
        /// <summary>
        /// List of errors encountered during parsing
        /// </summary>
        public List<string> Errors { get; } = new List<string>();
        
        /// <summary>
        /// Indicates whether the chunk was successfully parsed
        /// </summary>
        public bool IsParsed { get; protected set; }
        
        /// <summary>
        /// Creates a new chunk with the given signature and data
        /// </summary>
        /// <param name="signature">The four-character signature</param>
        /// <param name="data">Raw binary data</param>
        /// <param name="logger">Optional logger</param>
        protected BaseChunk(string signature, byte[] data, ILogger? logger = null)
        {
            if (signature == null || signature.Length != 4)
            {
                throw new ArgumentException("Chunk signature must be a 4-character string", nameof(signature));
            }
            
            Signature = signature;
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Logger = logger;
            IsParsed = false;
        }
        
        /// <summary>
        /// Parse the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public abstract bool Parse();
        
        /// <summary>
        /// Write the chunk data to a binary format
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public abstract byte[] Write();
        
        /// <summary>
        /// Log an error message
        /// </summary>
        /// <param name="message">The error message</param>
        protected void LogError(string message)
        {
            Errors.Add(message);
            Logger?.LogError(message);
        }
        
        /// <summary>
        /// Log a warning message
        /// </summary>
        /// <param name="message">The warning message</param>
        protected void LogWarning(string message)
        {
            Logger?.LogWarning(message);
        }
        
        /// <summary>
        /// Write this chunk to a BinaryWriter
        /// </summary>
        /// <param name="writer">BinaryWriter to write to</param>
        public void WriteTo(BinaryWriter writer)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }
            
            writer.WriteChunkSignature(Signature);
            
            byte[] data = Write();
            writer.Write((uint)data.Length);
            writer.Write(data);
        }
        
        /// <summary>
        /// Returns a string representation of this chunk
        /// </summary>
        public override string ToString()
        {
            return $"{Signature} ({Size} bytes)";
        }
        
        /// <summary>
        /// Get parsing errors
        /// </summary>
        /// <returns>List of error messages</returns>
        public IEnumerable<string> GetErrors()
        {
            return Errors;
        }
    }
} 