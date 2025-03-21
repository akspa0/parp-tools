using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files
{
    /// <summary>
    /// Represents a chunk with an unknown signature
    /// </summary>
    public class UnknownChunk : BaseChunk
    {
        /// <summary>
        /// Creates a new UnknownChunk with the specified signature and data
        /// </summary>
        /// <param name="signature">Chunk signature (4 characters)</param>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public UnknownChunk(string signature, byte[] data, ILogger? logger = null)
            : base(signature, data, logger)
        {
            LogWarning($"Created unknown chunk with signature '{signature}'");
        }
        
        /// <summary>
        /// Parse the chunk data
        /// </summary>
        /// <returns>Always returns false since we don't know how to parse this chunk</returns>
        public override bool Parse()
        {
            // We don't know how to parse this, so just log a warning and return false
            LogWarning($"Cannot parse unknown chunk type '{Signature}'");
            return false;
        }
        
        /// <summary>
        /// Write the chunk data
        /// </summary>
        /// <returns>The original data bytes</returns>
        public override byte[] Write()
        {
            // Just return the original data since we don't know how to modify it
            return Data;
        }
    }
} 