using Microsoft.Extensions.Logging;
using System;
using System.Text;
using WCAnalyzer.Core.Files.Interfaces;
using WCAnalyzer.Core.Files.ADT.Chunks;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT
{
    /// <summary>
    /// Factory class for creating ADT chunk instances based on chunk signature.
    /// </summary>
    public class ADTChunkFactory : IChunkFactory
    {
        private readonly ILogger _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ADTChunkFactory"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public ADTChunkFactory(ILogger logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Creates a chunk based on the provided signature and data.
        /// </summary>
        /// <param name="signature">The signature of the chunk.</param>
        /// <param name="data">The chunk data.</param>
        /// <returns>An instance of <see cref="IChunk"/> or null if the signature is not recognized.</returns>
        public IChunk CreateChunk(string signature, byte[] data)
        {
            if (string.IsNullOrEmpty(signature))
            {
                _logger.LogWarning("ADTChunkFactory: Null or empty signature provided to CreateChunk");
                return null;
            }

            if (data == null)
            {
                _logger.LogWarning($"ADTChunkFactory: Null data provided for chunk with signature '{signature}'");
                return null;
            }

            // Handle special cases for chunks that need additional parsing parameters
            if (signature == McrfChunk.SIGNATURE)
            {
                // For MCRF, we need to extract m2 and wmo reference counts
                // For the basic implementation, let's estimate these as a function of the data length
                uint m2Count = (uint)(data.Length / 4); // Assuming each reference is 4 bytes
                uint wmoCount = 0; // Default to zero for now unless we can determine from context
                return new McrfChunk(data, m2Count, wmoCount, _logger);
            }

            switch (signature)
            {
                case MverChunk.SIGNATURE:
                    return new MverChunk(data, _logger);
                case MhdrChunk.SIGNATURE:
                    return new MhdrChunk(data, _logger);
                case McnkChunk.SIGNATURE:
                    return new McnkChunk(data, _logger);
                case McinChunk.SIGNATURE:
                    return new McinChunk(data, _logger);
                case MtexChunk.SIGNATURE:
                    return new MtexChunk(data, _logger);
                case MmdxChunk.SIGNATURE:
                    return new MmdxChunk(data, _logger);
                case MmidChunk.SIGNATURE:
                    return new MmidChunk(data, _logger);
                case MwmoChunk.SIGNATURE:
                    return new MwmoChunk(data, _logger);
                case MwidChunk.SIGNATURE:
                    return new MwidChunk(data, _logger);
                case MddfChunk.SIGNATURE:
                    return new MddfChunk(data, _logger);
                case ModfChunk.SIGNATURE:
                    return new ModfChunk(data, _logger);
                case McvtChunk.SIGNATURE:
                    return new McvtChunk(data, _logger);
                case McnrChunk.SIGNATURE:
                    return new McnrChunk(data, _logger);
                case McalChunk.SIGNATURE:
                    return new McalChunk(data, _logger);
                case Mh2oChunk.SIGNATURE:
                    return new Mh2oChunk(data, _logger);
                case MccvChunk.SIGNATURE:
                    return new MccvChunk(data, _logger);
                case McshChunk.SIGNATURE:
                    return new McshChunk(data, _logger);
                case MclyChunk.SIGNATURE:
                    return new MclyChunk(data, _logger);
                case MfboChunk.SIGNATURE:
                    return new MfboChunk(data, _logger);
                case MtfxChunk.SIGNATURE:
                    // MTFX chunk contains texture file extensions data
                    return new MtfxChunk(data, _logger);
                case McrdChunk.SIGNATURE:
                    // MCRD chunk contains terrain hole data
                    return new McrdChunk(data, _logger);
                case McseChunk.SIGNATURE:
                    // MCSE chunk contains sound emitter data
                    // Ideally, we would get the emitter count from the parent MCNK
                    // For now, we'll let the chunk estimate it from the data size
                    return new McseChunk(data, _logger);
                case MampChunk.SIGNATURE:
                    // MAMP chunk - Map objects data
                    return new MampChunk(data, _logger);
                case MbmhChunk.SIGNATURE:
                    // MBMH chunk - Blend map header
                    return new MbmhChunk(data, _logger);
                case MbmiChunk.SIGNATURE:
                    // MBMI chunk - Blend map information
                    return new MbmiChunk(data, _logger);
                case MbbbChunk.SIGNATURE:
                    // MBBB chunk - Bounding box data
                    return new MbbbChunk(data, _logger);
                case McbbChunk.SIGNATURE:
                    // MCBB chunk - Collision bounding box data
                    return new McbbChunk(data, _logger);
                case MclqChunk.SIGNATURE:
                    return new MclqChunk(data, _logger);
                case MhidChunk.SIGNATURE:
                    // MHID chunk - Height texture file data IDs
                    return new MhidChunk(data, _logger);
                case MfogChunk.SIGNATURE:
                    // MFOG chunk - Fog information
                    return new MfogChunk(data, _logger);
                case MclvChunk.SIGNATURE:
                    // MCLV chunk - Vertex colors
                    return new MclvChunk(data, _logger);
                case McrwChunk.SIGNATURE:
                    // MCRW chunk - Render water
                    return new McrwChunk(data, _logger);
                case McmtChunk.SIGNATURE:
                    // MCMT chunk - Material information
                    return new McmtChunk(data, _logger);
                case McddChunk.SIGNATURE:
                    // MCDD chunk - Detail doodads
                    return new McddChunk(data, _logger);
                case MwdrChunk.SIGNATURE:
                    // MWDR chunk - WMO doodad references
                    return new MwdrChunk(data, _logger);
                case MdidChunk.SIGNATURE:
                    // MDID chunk - Doodad file data IDs
                    return new MdidChunk(data, _logger);
                case MbnvChunk.SIGNATURE:
                    // MBNV chunk - Normal vectors for terrain blending
                    return new MbnvChunk(data, _logger);
                case MtxpChunk.SIGNATURE:
                    // MTXP chunk - Texture paths
                    return new MtxpChunk(data, _logger);
                case MlhdChunk.SIGNATURE:
                    // MLHD chunk - Legion Header Data (Legion+)
                    return new MlhdChunk(data, _logger);
                case MlvhChunk.SIGNATURE:
                    // MLVH chunk - Legion Vertex Height Data (Legion+)
                    return new MlvhChunk(data, _logger);
                case MlviChunk.SIGNATURE:
                    return new MlviChunk(data, _logger);
                case MlllChunk.SIGNATURE:
                    return new MlllChunk(data, _logger);
                case MlndChunk.SIGNATURE:
                    return new MlndChunk(data, _logger);
                case MlsiChunk.SIGNATURE:
                    return new MlsiChunk(data, _logger);
                case MldxChunk.SIGNATURE:
                    return new MldxChunk(data, _logger);
                case MlmxChunk.SIGNATURE:
                    return new MlmxChunk(data, _logger);
                case MlmdChunk.SIGNATURE:
                    return new MlmdChunk(data, _logger);
                default:
                    _logger.LogWarning($"ADTChunkFactory: Unknown chunk signature '{signature}'");
                    return new UnknownChunk(signature, data, _logger);
            }
        }
    }
} 