using System;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Factory class for creating PM4 chunk instances based on signature.
    /// </summary>
    public static class PM4ChunkFactory
    {
        /// <summary>
        /// Creates a PM4 chunk instance based on the provided signature and data.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The chunk data.</param>
        /// <returns>A new instance of a PM4 chunk.</returns>
        public static PM4Chunk CreateChunk(string signature, byte[] data)
        {
            if (string.IsNullOrEmpty(signature))
                throw new ArgumentNullException(nameof(signature));

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return signature switch
            {
                "MVER" => new MVERChunk(data),
                "MCRC" => new MCRCChunk(data),
                "MSHD" => new MSHDChunk(data),
                "MSPV" => new MSPVChunk(data),
                "MSPI" => new MSPIChunk(data),
                "MSCN" => new MSCNChunk(data),
                "MSLK" => new MSLKChunk(data),
                "MSVT" => new MSVTChunk(data),
                "MSVI" => new MSVIChunk(data),
                "MSUR" => new MSURChunk(data),
                "MPRL" => new MPRLChunk(data),
                "MPRR" => new MPRRChunk(data),
                "MDBH" => new MDBHChunk(data),
                "MDOS" => new MDOSChunk(data),
                "MDSF" => new MDSFChunk(data),
                _ => throw new ArgumentException($"Unknown chunk signature: {signature}", nameof(signature))
            };
        }
    }
} 