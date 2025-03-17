using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSCN chunk - Contains normal vector data for meshes.
    /// </summary>
    public class MSCNChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSCN";

        /// <summary>
        /// Gets the normal vectors.
        /// </summary>
        public List<Vector3> Normals { get; private set; } = new List<Vector3>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSCNChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSCNChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Normals.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each normal is 12 bytes (3 floats for X, Y, Z)
                while (ms.Position < ms.Length)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    Normals.Add(new Vector3(x, y, z));
                }
            }
        }
    }
} 