using System.Collections.Generic;

namespace WCAnalyzer.Core.Common.Interfaces
{
    /// <summary>
    /// Generic interface for file parsers
    /// </summary>
    /// <typeparam name="TResult">The parsed file result type</typeparam>
    public interface IFileParser<TResult> where TResult : class
    {
        /// <summary>
        /// Parses a file from the specified path
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>The parsed result</returns>
        TResult Parse(string filePath);
        
        /// <summary>
        /// Parses a file from binary data
        /// </summary>
        /// <param name="data">The raw file data</param>
        /// <returns>The parsed result</returns>
        TResult Parse(byte[] data);
        
        /// <summary>
        /// Gets a list of errors encountered during parsing
        /// </summary>
        List<string> GetErrors();
    }
} 