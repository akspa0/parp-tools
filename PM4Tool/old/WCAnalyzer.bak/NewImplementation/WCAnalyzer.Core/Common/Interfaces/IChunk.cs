using System.Collections.Generic;

namespace WCAnalyzer.Core.Common.Interfaces
{
    /// <summary>
    /// Interface for all file chunks
    /// </summary>
    public interface IChunk
    {
        /// <summary>
        /// Gets the chunk signature (four-character code)
        /// </summary>
        string Signature { get; }
        
        /// <summary>
        /// Gets the size of the chunk data in bytes
        /// </summary>
        uint Size { get; }
        
        /// <summary>
        /// Gets the raw data bytes
        /// </summary>
        byte[] Data { get; }
        
        /// <summary>
        /// Gets a list of errors encountered during parsing
        /// </summary>
        List<string> Errors { get; }
        
        /// <summary>
        /// Gets whether this chunk was successfully parsed
        /// </summary>
        bool IsParsed { get; }
    }
} 