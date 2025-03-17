using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSUR chunk - Contains surface data.
    /// </summary>
    public class MSURChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSUR";

        /// <summary>
        /// Gets the surface data entries.
        /// </summary>
        public List<SurfaceData> Surfaces { get; private set; } = new List<SurfaceData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSURChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSURChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Surfaces.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var surface = new SurfaceData
                    {
                        Index1 = reader.ReadUInt32(),
                        Index2 = reader.ReadUInt32(),
                        Index3 = reader.ReadUInt32(),
                        Flags = reader.ReadUInt32()
                    };
                    Surfaces.Add(surface);
                }
            }
        }

        /// <summary>
        /// Represents surface data.
        /// </summary>
        public class SurfaceData
        {
            /// <summary>
            /// Gets or sets the first vertex index.
            /// </summary>
            public uint Index1 { get; set; }

            /// <summary>
            /// Gets or sets the second vertex index.
            /// </summary>
            public uint Index2 { get; set; }

            /// <summary>
            /// Gets or sets the third vertex index.
            /// </summary>
            public uint Index3 { get; set; }

            /// <summary>
            /// Gets or sets the flags.
            /// </summary>
            public uint Flags { get; set; }
        }
    }
} 