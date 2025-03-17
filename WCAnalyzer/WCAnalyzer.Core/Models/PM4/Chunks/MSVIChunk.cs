using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSVI chunk - Contains vertex information data.
    /// </summary>
    public class MSVIChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSVI";

        /// <summary>
        /// Gets the vertex information entries.
        /// </summary>
        public List<VertexInfo> VertexInfos { get; private set; } = new List<VertexInfo>();

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
            VertexInfos.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var info = new VertexInfo
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32()
                    };
                    VertexInfos.Add(info);
                }
            }
        }

        /// <summary>
        /// Represents vertex information data.
        /// </summary>
        public class VertexInfo
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