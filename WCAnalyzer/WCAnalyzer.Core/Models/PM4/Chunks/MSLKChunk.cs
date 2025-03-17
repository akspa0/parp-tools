using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSLK chunk - Contains links data between vertices.
    /// </summary>
    public class MSLKChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSLK";

        /// <summary>
        /// Gets the links data.
        /// </summary>
        public List<LinkData> Links { get; private set; } = new List<LinkData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLKChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSLKChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Links.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var link = new LinkData
                    {
                        SourceIndex = reader.ReadUInt32(),
                        TargetIndex = reader.ReadUInt32()
                    };
                    Links.Add(link);
                }
            }
        }

        /// <summary>
        /// Represents a link between two vertices.
        /// </summary>
        public class LinkData
        {
            /// <summary>
            /// Gets or sets the source vertex index.
            /// </summary>
            public uint SourceIndex { get; set; }

            /// <summary>
            /// Gets or sets the target vertex index.
            /// </summary>
            public uint TargetIndex { get; set; }
        }
    }
} 