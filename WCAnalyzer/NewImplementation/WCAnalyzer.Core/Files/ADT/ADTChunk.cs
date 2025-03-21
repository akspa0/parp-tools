using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.ADT
{
    /// <summary>
    /// Base class for all ADT (terrain) file chunks
    /// </summary>
    public abstract class ADTChunk : BaseChunk
    {
        /// <summary>
        /// Creates a new ADTChunk with the given signature and data
        /// </summary>
        /// <param name="signature">The four-character signature</param>
        /// <param name="data">Raw binary data</param>
        /// <param name="logger">Optional logger</param>
        protected ADTChunk(string signature, byte[] data, ILogger? logger = null)
            : base(signature, data, logger)
        {
        }
    }
} 