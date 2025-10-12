using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PD4
{
    /// <summary>
    /// Base class for PD4 file chunks
    /// </summary>
    public abstract class PD4Chunk : BaseChunk, IPD4Chunk
    {
        /// <summary>
        /// Creates a new PD4 chunk with the specified signature and data
        /// </summary>
        /// <param name="signature">Chunk signature (4 characters)</param>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        protected PD4Chunk(string signature, byte[] data, ILogger? logger = null)
            : base(signature, data, logger)
        {
        }
    }
} 