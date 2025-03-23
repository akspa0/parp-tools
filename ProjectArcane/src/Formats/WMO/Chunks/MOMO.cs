using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Interfaces;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOMO chunk - Container chunk for v14 (Alpha) WMO data
    /// Only present in v14 WMO files
    /// </summary>
    public class MOMO : ContainerChunkBase
    {
        public override string ChunkId => "MOMO";

        public MOMO() : base() { }

        public override void Read(BinaryReader reader, uint size)
        {
            // Read all contained chunks until we reach the end of the MOMO chunk
            long endPosition = reader.BaseStream.Position + size;
            
            while (reader.BaseStream.Position < endPosition)
            {
                IChunk chunk = ChunkFactory.CreateChunk(reader);
                if (chunk != null)
                {
                    SubChunks[chunk.ChunkId] = chunk;
                }
            }
        }
    }
} 