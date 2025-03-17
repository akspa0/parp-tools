using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Represents a generic chunk with unknown format.
    /// </summary>
    public class GenericChunk : PM4Chunk
    {
        private readonly string _signature;

        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => _signature;

        /// <summary>
        /// Initializes a new instance of the <see cref="GenericChunk"/> class.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The chunk data.</param>
        public GenericChunk(string signature, byte[] data) : base(data)
        {
            if (string.IsNullOrEmpty(signature))
                throw new ArgumentNullException(nameof(signature));

            _signature = signature;
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            // No specific parsing for generic chunks
        }

        /// <summary>
        /// Returns a string representation of this chunk.
        /// </summary>
        /// <returns>A string representation of this chunk.</returns>
        public override string ToString()
        {
            return $"Generic Chunk: {Signature}, Size: {Size} bytes";
        }

        /// <summary>
        /// Gets a hexadecimal representation of the chunk data.
        /// </summary>
        /// <param name="maxBytes">The maximum number of bytes to include in the representation.</param>
        /// <returns>A hexadecimal representation of the chunk data.</returns>
        public string GetHexDump(int maxBytes = 64)
        {
            if (Data == null || Data.Length == 0)
                return "Empty";

            int bytesToShow = Math.Min(maxBytes, Data.Length);
            var hexDump = BitConverter.ToString(Data, 0, bytesToShow).Replace("-", " ");
            
            if (bytesToShow < Data.Length)
                hexDump += " ...";
                
            return hexDump;
        }
    }
} 