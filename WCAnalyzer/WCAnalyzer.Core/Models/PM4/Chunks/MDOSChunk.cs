using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MDOS chunk - Contains object data for server-side processing.
    /// </summary>
    public class MDOSChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MDOS";

        /// <summary>
        /// Gets the object data entries.
        /// </summary>
        public List<ObjectData> Entries { get; private set; } = new List<ObjectData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MDOSChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MDOSChunk(byte[] data) : base(data)
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
                while (ms.Position < ms.Length)
                {
                    var entry = new ObjectData
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32()
                    };
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents server-side object data.
        /// </summary>
        public class ObjectData
        {
            /// <summary>
            /// Gets or sets the first value.
            /// </summary>
            public uint Value1 { get; set; }

            /// <summary>
            /// Gets or sets the second value.
            /// </summary>
            public uint Value2 { get; set; }
        }
    }
} 