using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the VPSM (MSPV reversed) chunk which contains vertex position data.
    /// </summary>
    public class VPSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "VPSM";
        
        /// <summary>
        /// Gets the list of vertex positions.
        /// </summary>
        public List<Vector3> Vertices { get; } = new List<Vector3>();
        
        /// <summary>
        /// Gets or sets a value indicating whether the data has been parsed.
        /// </summary>
        public bool HasBeenParsed => IsParsed;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="VPSM"/> class.
        /// </summary>
        public VPSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="VPSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public VPSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the vertex data from the raw binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Vertices.Clear();
            
            // Determine how many vertices we have
            int bytesAvailable = (int)reader.BaseStream.Length;
            int vertexCount = bytesAvailable / 12; // Each vertex is 3 floats of 4 bytes each
            
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
                return $"Vertex data: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"Vertex count: {Vertices.Count}");
            
            // Show first few vertices
            int displayCount = Math.Min(Vertices.Count, 5);
            for (int i = 0; i < displayCount; i++)
            {
                var v = Vertices[i];
                sb.AppendLine($"  Vertex {i}: ({v.X}, {v.Y}, {v.Z})");
            }
            
            if (Vertices.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Vertices.Count - displayCount} more vertices");
            }
            
            return sb.ToString();
        }
    }
} 