using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSPI chunk - Contains vertex indices.
    /// According to documentation: uint32_t msp_indices[]; // index into #MSPV
    /// </summary>
    public class MSPIChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSPI";

        /// <summary>
        /// Gets the vertex indices.
        /// </summary>
        public List<uint> Indices { get; private set; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPIChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSPIChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Indices.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each index is 4 bytes (uint32_t)
                int entrySize = 4;
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    // Read the index
                    uint index = reader.ReadUInt32();
                    Indices.Add(index);
                }
            }
        }
    }
} 