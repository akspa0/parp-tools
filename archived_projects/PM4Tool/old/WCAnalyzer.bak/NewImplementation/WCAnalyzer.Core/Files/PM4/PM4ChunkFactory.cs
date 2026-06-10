using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Files.PM4
{
    /// <summary>
    /// Factory for creating PM4 file chunks
    /// </summary>
    public class PM4ChunkFactory : ChunkFactory
    {
        /// <summary>
        /// Creates a new PM4 chunk factory
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public PM4ChunkFactory(ILogger? logger = null)
            : base(logger)
        {
        }
        
        /// <summary>
        /// Creates a chunk based on the signature and data
        /// </summary>
        /// <param name="signature">Chunk signature (4 characters)</param>
        /// <param name="data">Raw chunk data</param>
        /// <returns>Appropriate chunk object for the signature</returns>
        public override Interfaces.IChunk CreateChunk(string signature, byte[] data)
        {
            if (signature == null)
            {
                Logger?.LogWarning("Null signature passed to PM4ChunkFactory.CreateChunk");
                return new UnknownChunk("????", data, Logger);
            }
            
            if (data == null)
            {
                Logger?.LogWarning($"Null data passed to PM4ChunkFactory.CreateChunk for {signature}");
                return new UnknownChunk(signature, new byte[0], Logger);
            }
            
            switch (signature)
            {
                case MverChunk.SIGNATURE:
                    return new MverChunk(data, Logger);
                
                case MshdChunk.SIGNATURE:
                    return new MshdChunk(data, Logger);
                
                case MspvChunk.SIGNATURE:
                    return new MspvChunk(data, Logger);
                
                case MspiChunk.SIGNATURE:
                    return new MspiChunk(data, Logger);
                
                case MprlChunk.SIGNATURE:
                    return new MprlChunk(data, Logger);
                
                case MslkChunk.SIGNATURE:
                    return new MslkChunk(data, Logger);
                
                case MsurChunk.SIGNATURE:
                    return new MsurChunk(data, Logger);
                
                case MsvtChunk.SIGNATURE:
                    return new MsvtChunk(data, Logger);
                
                case "MSCN":
                    return new MscnChunk(data);
                
                case "MSVI":
                    return new MsviChunk(data);
                
                case "MPRR":
                    return new MprrChunk(data);
                
                case "MDBH":
                    return new MdbhChunk(data);
                
                case "MDOS":
                    return new MdosChunk(data);
                
                case "MDSF":
                    return new MdsfChunk(data);
                
                default:
                    Logger?.LogWarning($"Unknown chunk signature '{signature}'");
                    return new UnknownChunk(signature, data, Logger);
            }
        }

        public override IChunk CreateChunk(string chunkId)
        {
            switch (chunkId)
            {
                case "MVER":
                    return new MverChunk();
                case "MSHD":
                    return new MshdChunk();
                case "MSPV":
                    return new MspvChunk();
                case "MSPI":
                    return new MspiChunk();
                case "MPRL":
                    return new MprlChunk();
                case "MSLK":
                    return new MslkChunk();
                case "MSUR":
                    return new MsurChunk();
                case "MSVT":
                    return new MsvtChunk();
                case "MSCN":
                    return new MscnChunk();
                case "MSVI":
                    return new MsviChunk();
                case "MPRR":
                    return new MprrChunk();
                case "MDBH":
                    return new MdbhChunk();
                case "MDOS":
                    return new MdosChunk();
                case "MDSF":
                    return new MdsfChunk();
                default:
                    return CreateUnknownChunk(chunkId);
            }
        }
    }
} 