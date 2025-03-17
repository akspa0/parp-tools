using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSVI chunk - Contains vertex information.
    /// </summary>
    public class MSVIChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSVI";

        /// <summary>
        /// Gets the vertex information entries.
        /// </summary>
        public List<VertexInfo> Entries { get; private set; } = new List<VertexInfo>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSVIChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSVIChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Entries.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each entry is 12 bytes (0x0C)
                int entrySize = 12;
                int entryCount = Data.Length / entrySize;

                for (int i = 0; i < entryCount; i++)
                {
                    var entry = new VertexInfo
                    {
                        Index = i,
                        Value0x00 = reader.ReadUInt32(),
                        Value0x04 = reader.ReadUInt32(),
                        Value0x08 = reader.ReadUInt32()
                    };
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a vertex information entry.
        /// </summary>
        public class VertexInfo
        {
            /// <summary>
            /// Gets or sets the index of this entry in the chunk.
            /// </summary>
            public int Index { get; set; }

            /// <summary>
            /// Gets or sets the value at offset 0x00.
            /// </summary>
            public uint Value0x00 { get; set; }

            /// <summary>
            /// Gets or sets the value at offset 0x04.
            /// </summary>
            public uint Value0x04 { get; set; }

            /// <summary>
            /// Gets or sets the value at offset 0x08.
            /// </summary>
            public uint Value0x08 { get; set; }

            /// <summary>
            /// Returns a string representation of this entry.
            /// </summary>
            public override string ToString()
            {
                return $"VertexInfo[{Index}]: 0x00={Value0x00}, 0x04={Value0x04}, 0x08={Value0x08}";
            }
        }
    }
} 