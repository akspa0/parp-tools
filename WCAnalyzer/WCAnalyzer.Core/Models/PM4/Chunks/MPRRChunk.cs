using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MPRR chunk - Contains position-related reference data.
    /// </summary>
    public class MPRRChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MPRR";

        /// <summary>
        /// Gets the position reference entries.
        /// </summary>
        public List<PositionReference> Entries { get; private set; } = new List<PositionReference>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MPRRChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MPRRChunk(byte[] data) : base(data)
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
                // The MPRR chunk contains entries with the following structure:
                // struct { 
                //   uint16_t _0x00; 
                //   uint16_t _0x02; 
                // } mprr[];
                
                // Each entry is 4 bytes
                int entrySize = 4;
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    var entry = new PositionReference
                    {
                        Index = i,
                        Value0x00 = reader.ReadUInt16(),
                        Value0x02 = reader.ReadUInt16()
                    };
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a position reference entry.
        /// </summary>
        public class PositionReference
        {
            /// <summary>
            /// Gets or sets the sequential index of this entry in the chunk
            /// </summary>
            public int Index { get; set; }
            
            /// <summary>
            /// Value at offset 0x00 (uint16_t)
            /// </summary>
            public ushort Value0x00 { get; set; }

            /// <summary>
            /// Value at offset 0x02 (uint16_t)
            /// </summary>
            public ushort Value0x02 { get; set; }
            
            /// <summary>
            /// Returns a string representation of this object.
            /// </summary>
            public override string ToString()
            {
                return $"PositionReference[{Index}]: ({Value0x00}, {Value0x02})";
            }
        }
    }
} 