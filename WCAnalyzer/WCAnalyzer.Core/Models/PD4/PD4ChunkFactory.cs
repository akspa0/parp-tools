using System;
using System.Collections.Generic;
using System.Text;
using Warcraft.NET.Files.Interfaces;
using WCAnalyzer.Core.Models.PD4.Chunks;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Factory class for creating PD4 chunks.
    /// </summary>
    public static class PD4ChunkFactory
    {
        /// <summary>
        /// Creates a chunk for the specified signature and data.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The chunk data.</param>
        /// <returns>The created chunk.</returns>
        public static IIFFChunk CreateChunk(string signature, byte[] data)
        {
            if (string.IsNullOrEmpty(signature))
            {
                throw new ArgumentException("Signature cannot be null or empty.", nameof(signature));
            }

            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            switch (signature)
            {
                case "MVER":
                    return new MVERChunk(data);
                case "MCRC":
                    return new MCRCChunk(data);
                case "MSHD":
                    return new MSHDChunk(data);
                case "MSPV":
                    return new MSPVChunk(data);
                case "MSPI":
                    return new MSPIChunk(data);
                case "MSCN":
                    return new MSCNChunk(data);
                case "MSLK":
                    return new MSLKChunk(data);
                case "MSVT":
                    return new MSVTChunk(data);
                case "MSVI":
                    return new MSVIChunk(data);
                case "MSUR":
                    return new MSURChunk(data);
                default:
                    // For unknown chunks, use GenericChunk
                    return new GenericChunk(signature, data);
            }
        }
    }
} 