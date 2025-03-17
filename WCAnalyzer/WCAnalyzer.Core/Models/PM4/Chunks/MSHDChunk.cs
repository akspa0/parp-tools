using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSHD chunk - Contains shadow data.
    /// </summary>
    public class MSHDChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSHD";

        /// <summary>
        /// Gets the shadow data entries.
        /// </summary>
        public List<ShadowEntry> ShadowEntries { get; private set; } = new List<ShadowEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSHDChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSHDChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            ShadowEntries.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each entry is typically a fixed size structure
                while (ms.Position < ms.Length)
                {
                    var entry = new ShadowEntry
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32(),
                        Value3 = reader.ReadUInt32(),
                        Value4 = reader.ReadUInt32()
                    };
                    ShadowEntries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a shadow data entry in the MSHD chunk.
        /// </summary>
        public class ShadowEntry
        {
            /// <summary>
            /// Gets or sets the first value.
            /// </summary>
            public uint Value1 { get; set; }

            /// <summary>
            /// Gets or sets the second value.
            /// </summary>
            public uint Value2 { get; set; }

            /// <summary>
            /// Gets or sets the third value.
            /// </summary>
            public uint Value3 { get; set; }

            /// <summary>
            /// Gets or sets the fourth value.
            /// </summary>
            public uint Value4 { get; set; }
        }
    }
} 