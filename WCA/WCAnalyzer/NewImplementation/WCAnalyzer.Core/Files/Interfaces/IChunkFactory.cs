using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.Interfaces
{
    /// <summary>
    /// Interface for chunk factory implementations
    /// </summary>
    public interface IChunkFactory
    {
        /// <summary>
        /// Create a chunk object for the given signature and data
        /// </summary>
        /// <param name="signature">The four-character chunk signature</param>
        /// <param name="data">Raw binary data</param>
        /// <returns>A chunk object appropriate for the signature</returns>
        IChunk CreateChunk(string signature, byte[] data);
    }
} 