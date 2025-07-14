using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// TVSM chunk (MSVT reversed)
    /// </summary>
    public class TVSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "TVSM";
        
        /// <summary>
        /// Raw vertex data for the TVSM chunk.
        /// According to documentation, these values are ordered YXZ and require specific transformation.
        /// </summary>
        public List<Vector3> RawVertices { get; } = new List<Vector3>();
        
        /// <summary>
        /// Gets transformed vertices according to documentation formulas:
        /// worldPos.y = 17066.666 - position.y
        /// worldPos.x = 17066.666 - position.x
        /// worldPos.z = position.z / 36.0f
        /// </summary>
        public IEnumerable<Vector3> TransformedVertices
        {
            get
            {
                const float transformConstant = 17066.666f;
                const float zScale = 36.0f;
                
                foreach (var vertex in RawVertices)
                {
                    yield return new Vector3(
                        transformConstant - vertex.Y, // X from Y
                        transformConstant - vertex.X, // Y from X
                        vertex.Z / zScale            // Z scaled
                    );
                }
            }
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TVSM"/> class.
        /// </summary>
        public TVSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TVSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public TVSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the TVSM data.
        /// According to documentation, values are ordered YXZ.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            RawVertices.Clear();
            
            // Each vertex is 12 bytes (3 floats or 3 ints)
            int bytesAvailable = (int)reader.BaseStream.Length;
            int vertexCount = bytesAvailable / 12;
            
            for (int i = 0; i < vertexCount; i++)
            {
                // Read in YXZ order as specified in documentation
                float y = reader.ReadSingle();
                float x = reader.ReadSingle();
                float z = reader.ReadSingle();
                RawVertices.Add(new Vector3(x, y, z));
            }
        }
        
        /// <summary>
        /// Gets a detailed description of the chunk contents.
        /// </summary>
        /// <returns>A detailed description string.</returns>
        public override string GetDetailedDescription()
        {
            if (!IsParsed)
            {
                return $"TVSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"TVSM vertex count: {RawVertices.Count}");
            sb.AppendLine("Values are stored in YXZ order and need transformation for in-game coordinates");
            
            // Display the first few raw vertices
            int displayCount = Math.Min(RawVertices.Count, 10);
            for (int i = 0; i < displayCount; i++)
            {
                var vertex = RawVertices[i];
                sb.AppendLine($"  Raw Vertex {i}: ({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3})");
                
                // Also show the transformed vertex
                var transformedVertex = new Vector3(
                    17066.666f - vertex.Y,
                    17066.666f - vertex.X,
                    vertex.Z / 36.0f
                );
                sb.AppendLine($"  Transformed {i}: ({transformedVertex.X:F3}, {transformedVertex.Y:F3}, {transformedVertex.Z:F3})");
            }
            
            if (RawVertices.Count > displayCount)
            {
                sb.AppendLine($"  ... and {RawVertices.Count - displayCount} more vertices");
            }
            
            return sb.ToString();
        }
    }
} 