using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// VVSM chunk
    /// </summary>
    public class VVSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "VVSM";
        
        /// <summary>
        /// Vertices for the VVSM chunk.
        /// </summary>
        public List<Vector3> Vertices { get; } = new List<Vector3>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="VVSM"/> class.
        /// </summary>
        public VVSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="VVSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public VVSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the VVSM data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Vertices.Clear();
            
            // Each vertex is 12 bytes (3 floats - X, Y, Z)
            int bytesAvailable = (int)reader.BaseStream.Length;
            int vertexCount = bytesAvailable / 12;
            
            for (int i = 0; i < vertexCount; i++)
            {
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                Vertices.Add(new Vector3(x, y, z));
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
                return $"VVSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"VVSM vertex count: {Vertices.Count}");
            
            // Display the first few vertices
            int displayCount = Math.Min(Vertices.Count, 20);
            for (int i = 0; i < displayCount; i += 2)
            {
                sb.Append($"  Vertex {i}-{Math.Min(i + 1, displayCount - 1)}: ");
                
                for (int j = 0; j < 2 && i + j < displayCount; j++)
                {
                    var vertex = Vertices[i + j];
                    sb.Append($"({vertex.X:F6}, {vertex.Y:F6}, {vertex.Z:F6}) ");
                }
                
                sb.AppendLine();
            }
            
            if (Vertices.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Vertices.Count - displayCount} more vertices");
            }
            
            return sb.ToString();
        }
    }
} 