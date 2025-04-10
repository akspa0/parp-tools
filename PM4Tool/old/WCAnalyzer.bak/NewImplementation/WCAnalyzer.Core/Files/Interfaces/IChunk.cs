using System.Collections.Generic;

namespace WCAnalyzer.Core.Files.Interfaces
{
    /// <summary>
    /// Interface for chunk-based file components
    /// </summary>
    public interface IChunk
    {
        /// <summary>
        /// Chunk signature (4 characters)
        /// </summary>
        string Signature { get; }
        
        /// <summary>
        /// Raw chunk data
        /// </summary>
        byte[] Data { get; }
        
        /// <summary>
        /// Whether the chunk has been parsed
        /// </summary>
        bool IsParsed { get; }
        
        /// <summary>
        /// Parse the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        bool Parse();
        
        /// <summary>
        /// Write the chunk data
        /// </summary>
        /// <returns>Binary data for this chunk</returns>
        byte[] Write();
        
        /// <summary>
        /// Get parsing errors
        /// </summary>
        /// <returns>List of error messages</returns>
        IEnumerable<string> GetErrors();
    }
} 