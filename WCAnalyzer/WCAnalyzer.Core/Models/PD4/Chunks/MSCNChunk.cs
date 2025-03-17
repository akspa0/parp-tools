using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSCN chunk - Contains normal coordinate data.
    /// According to documentation: C3Vectori mscn[]; // n ≠ normals.u
    /// </summary>
    public class MSCNChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSCN";

        /// <summary>
        /// Gets the normal coordinates.
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
                // Each normal is 12 bytes (3 * 4 bytes for float x, y, z)
                int entrySize = 12;
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    // Read the x, y, z coordinates of the normal
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    
                    Normals.Add(new Vector3(x, y, z));
                }
            }
        }
    }
} 