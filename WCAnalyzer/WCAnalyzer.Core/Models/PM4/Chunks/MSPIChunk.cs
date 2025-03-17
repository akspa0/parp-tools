using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSPI chunk - Contains vertex indices.
    /// </summary>
    public class MSPIChunk : PM4Chunk
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
                // Each index is a uint (4 bytes)
                while (ms.Position < ms.Length)
                {
                    Indices.Add(reader.ReadUInt32());
                }
            }
        }
    }
} 