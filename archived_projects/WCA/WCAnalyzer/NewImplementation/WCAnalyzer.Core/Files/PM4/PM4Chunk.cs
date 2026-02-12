using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4
{
    /// <summary>
    /// Base class for PM4 file chunks
    /// </summary>
    public abstract class PM4Chunk : BaseChunk, IPM4Chunk
    {
        /// <summary>
        /// Creates a new PM4 chunk with the specified signature and data
        /// </summary>
        /// <param name="signature">Chunk signature (4 characters)</param>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        protected PM4Chunk(string signature, byte[] data, ILogger? logger = null)
            : base(signature, data, logger)
        {
        }
    }
} 