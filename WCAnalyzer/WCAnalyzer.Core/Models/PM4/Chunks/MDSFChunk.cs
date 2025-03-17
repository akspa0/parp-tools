using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MDSF chunk - Contains server-side flag data.
    /// </summary>
    public class MDSFChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MDSF";

        /// <summary>
        /// Gets the server flag data entries.
        /// </summary>
        public List<ServerFlagData> Entries { get; private set; } = new List<ServerFlagData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MDSFChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MDSFChunk(byte[] data) : base(data)
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
                    var entry = new ServerFlagData
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32()
                    };
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents server-side flag data.
        /// </summary>
        public class ServerFlagData
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