using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// NVSM chunk (MSCN reversed)
    /// According to documentation, this is not actually related to normals despite the name.
    /// </summary>
    public class NVSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "NVSM";
        
        /// <summary>
        /// Raw vector data for the NVSM chunk.
        /// Documentation notes that despite the name, this is not related to normals.
        /// </summary>
        public List<Vector3> Vectors { get; } = new List<Vector3>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="NVSM"/> class.
        /// </summary>
        public NVSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="NVSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public NVSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the NVSM data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Vectors.Clear();
            
            // Each vector is 12 bytes (3 floats or 3 ints)
            int bytesAvailable = (int)reader.BaseStream.Length;
            int vectorCount = bytesAvailable / 12;
            
            for (int i = 0; i < vectorCount; i++)
            {
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                Vectors.Add(new Vector3(x, y, z));
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
                return $"NVSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"NVSM vector count: {Vectors.Count}");
            sb.AppendLine("Note: Documentation states these are not normal vectors despite the name");
            
            // Display the first few vectors
            int displayCount = Math.Min(Vectors.Count, 10);
            for (int i = 0; i < displayCount; i++)
            {
                var vector = Vectors[i];
                sb.AppendLine($"  Vector {i}: ({vector.X:F6}, {vector.Y:F6}, {vector.Z:F6})");
            }
            
            if (Vectors.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Vectors.Count - displayCount} more vectors");
            }
            
            return sb.ToString();
        }
    }
} 