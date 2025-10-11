using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;
using WCAnalyzer.Core.Files.PD4.Chunks;

namespace WCAnalyzer.Core.Files.PD4
{
    /// <summary>
    /// Factory for creating PD4 file chunks
    /// </summary>
    public class PD4ChunkFactory : ChunkFactory
    {
        /// <summary>
        /// Creates a new PD4 chunk factory
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public PD4ChunkFactory(ILogger? logger = null)
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
                Logger?.LogWarning("Null signature passed to PD4ChunkFactory.CreateChunk");
                return new UnknownChunk("????", data, Logger);
            }
            
            if (data == null)
            {
                Logger?.LogWarning($"Null data passed to PD4ChunkFactory.CreateChunk for {signature}");
                return new UnknownChunk(signature, new byte[0], Logger);
            }
            
            switch (signature)
            {
                case "MVER":
                    return new MverChunk(data);
                
                case "MCRC":
                    return new McrcChunk(BitConverter.ToUInt32(data, 0));
                
                case "MSHD":
                    return new MshdChunk(data);
                
                case "MSPV":
                    return new MspvChunk(data);
                
                case "MSPI":
                    return new MspiChunk(data);
                
                case "MSCN":
                    return new MscnChunk(data);
                
                case "MSLK":
                    return new MslkChunk(data);
                
                case "MSVT":
                    return new MsvtChunk(data);
                
                case "MSVI":
                    return new MsviChunk(data);
                
                case "MSUR":
                    return new MsurChunk(data);
                
                default:
                    Logger?.LogWarning($"Unknown chunk signature '{signature}'");
                    return new UnknownChunk(signature, data, Logger);
            }
        }
        
        /// <summary>
        /// Creates a chunk based on the chunk ID
        /// </summary>
        /// <param name="chunkId">The chunk ID (4 characters)</param>
        /// <returns>An appropriate chunk object for the ID</returns>
        public override IChunk CreateChunk(string chunkId)
        {
            switch (chunkId)
            {
                case "MVER":
                    return new MverChunk();
                
                case "MCRC":
                    return new McrcChunk();
                
                case "MSHD":
                    return new MshdChunk();
                
                case "MSPV":
                    return new MspvChunk();
                
                case "MSPI":
                    return new MspiChunk();
                
                case "MSCN":
                    return new MscnChunk();
                
                case "MSLK":
                    return new MslkChunk();
                
                case "MSVT":
                    return new MsvtChunk();
                
                case "MSVI":
                    return new MsviChunk();
                
                case "MSUR":
                    return new MsurChunk();
                
                default:
                    return CreateUnknownChunk(chunkId);
            }
        }
    }
} 