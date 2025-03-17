using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSPV chunk - Contains vertex positions.
    /// According to documentation: C3Vectori msp_vertices[];
    /// </summary>
    public class MSPVChunk : PD4Chunk
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
                // Each vertex is 12 bytes (3 * 4 bytes for float x, y, z)
                int entrySize = 12;
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    // Read the x, y, z coordinates of the vertex
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    
                    Vertices.Add(new Vector3(x, y, z));
                }
            }
        }
    }
} 