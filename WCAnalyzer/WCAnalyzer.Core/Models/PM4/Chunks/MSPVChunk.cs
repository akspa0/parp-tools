using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSPV chunk - Contains vertex positions.
    /// </summary>
    public class MSPVChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSPV";

        /// <summary>
        /// Gets the vertex positions.
        /// </summary>
        public List<Vector3> Vertices { get; private set; } = new List<Vector3>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPVChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSPVChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Vertices.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each vertex is 12 bytes (3 floats for X, Y, Z)
                while (ms.Position < ms.Length)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    Vertices.Add(new Vector3(x, y, z));
                }
            }
        }
    }
} 