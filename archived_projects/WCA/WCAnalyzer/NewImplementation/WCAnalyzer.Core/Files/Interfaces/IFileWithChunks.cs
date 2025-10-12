using System.Collections.Generic;

namespace WCAnalyzer.Core.Files.Interfaces
{
    /// <summary>
    /// Interface for files that consist of chunks
    /// </summary>
    public interface IFileWithChunks
    {
        /// <summary>
        /// Gets the list of chunks in this file
        /// </summary>
        IReadOnlyList<IChunk> Chunks { get; }
        
        /// <summary>
        /// Gets whether the file was successfully loaded
        /// </summary>
        bool IsLoaded { get; }
        
        /// <summary>
        /// Gets any errors encountered during loading
        /// </summary>
        IReadOnlyList<string> Errors { get; }
        
        /// <summary>
        /// Loads the file from the specified path
        /// </summary>
        /// <param name="path">Path to the file</param>
        /// <returns>True if loaded successfully, false otherwise</returns>
        bool Load(string path);
        
        /// <summary>
        /// Gets a chunk by signature
        /// </summary>
        /// <typeparam name="T">Type of chunk to get</typeparam>
        /// <param name="signature">Chunk signature to look for</param>
        /// <returns>The first chunk with the matching signature, or null if not found</returns>
        T? GetChunk<T>(string signature) where T : IChunk;
    }
} 