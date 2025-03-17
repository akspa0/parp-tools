using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MSVT chunk - Contains vertex data.
    /// According to documentation, values are ordered YXZ and require specific transformations.
    /// </summary>
    public class MSVTChunk : PM4Chunk
    {
        // Constants for coordinate transformations as per documentation
        private const float COORDINATE_OFFSET = 17066.666f;
        private const float HEIGHT_CONVERSION = 36.0f; // Convert internal inch height to yards

        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSVT";

        /// <summary>
        /// Gets the vertex data entries.
        /// </summary>
        public List<VertexData> Vertices { get; private set; } = new List<VertexData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSVTChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSVTChunk(byte[] data) : base(data)
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
                // The MSVT chunk contains C3Vectori values
                // For some reason the values are ordered YXZ
                int entrySize = 12; // 3 floats
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    // Read the raw values in YXZ order as per documentation
                    float rawY = reader.ReadSingle();
                    float rawX = reader.ReadSingle();
                    float rawZ = reader.ReadSingle();
                    
                    // Apply the transformation formulas from the documentation
                    float worldY = COORDINATE_OFFSET - rawY;
                    float worldX = COORDINATE_OFFSET - rawX;
                    float worldZ = rawZ / HEIGHT_CONVERSION;
                    
                    var vertex = new VertexData
                    {
                        Index = i,
                        // Store both raw and transformed values
                        RawY = rawY,
                        RawX = rawX,
                        RawZ = rawZ,
                        // Transformed world coordinates
                        WorldX = worldX,
                        WorldY = worldY,
                        WorldZ = worldZ
                    };
                    Vertices.Add(vertex);
                }
            }
        }

        /// <summary>
        /// Represents vertex data with both raw and transformed world coordinates.
        /// </summary>
        public class VertexData
        {
            /// <summary>
            /// Gets or sets the index of this vertex in the chunk.
            /// </summary>
            public int Index { get; set; }
            
            /// <summary>
            /// Gets or sets the raw Y value as read from the file.
            /// </summary>
            public float RawY { get; set; }

            /// <summary>
            /// Gets or sets the raw X value as read from the file.
            /// </summary>
            public float RawX { get; set; }

            /// <summary>
            /// Gets or sets the raw Z value as read from the file.
            /// </summary>
            public float RawZ { get; set; }

            /// <summary>
            /// Gets or sets the transformed world X coordinate.
            /// Formula: worldPos.x = 17066.666 - position.x
            /// </summary>
            public float WorldX { get; set; }

            /// <summary>
            /// Gets or sets the transformed world Y coordinate.
            /// Formula: worldPos.y = 17066.666 - position.y
            /// </summary>
            public float WorldY { get; set; }

            /// <summary>
            /// Gets or sets the transformed world Z coordinate.
            /// Formula: worldPos.z = position.z / 36.0f (converts internal inch height to yards)
            /// </summary>
            public float WorldZ { get; set; }
            
            /// <summary>
            /// Gets the Vector3 representation of the world position.
            /// </summary>
            public Vector3 WorldPosition => new Vector3(WorldX, WorldY, WorldZ);
            
            /// <summary>
            /// Returns a string representation of this vertex.
            /// </summary>
            public override string ToString()
            {
                return $"Vertex[{Index}]: World({WorldX:F2}, {WorldY:F2}, {WorldZ:F2}), Raw({RawX:F2}, {RawY:F2}, {RawZ:F2})";
            }
        }
    }
} 