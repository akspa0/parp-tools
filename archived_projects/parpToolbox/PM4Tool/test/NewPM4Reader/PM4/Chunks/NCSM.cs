using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the NCSM (MSCN reversed) chunk which contains vertex normal data.
    /// </summary>
    public class NCSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "NCSM";
        
        /// <summary>
        /// Gets the list of vertex normals.
        /// </summary>
        public List<Vector3> Normals { get; } = new List<Vector3>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="NCSM"/> class.
        /// </summary>
        public NCSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="NCSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public NCSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the normal data from the raw binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Normals.Clear();
            
            // Determine how many normals we have
            int bytesAvailable = (int)reader.BaseStream.Length;
            int normalCount = bytesAvailable / 12; // Each normal is 3 floats of 4 bytes each
            
            for (int i = 0; i < normalCount; i++)
            {
                float x = reader.ReadSingle();
                float y = reader.ReadSingle();
                float z = reader.ReadSingle();
                
                // Normals should be normalized (length = 1)
                Vector3 normal = new Vector3(x, y, z);
                if (normal != Vector3.Zero)
                {
                    normal = Vector3.Normalize(normal);
                }
                
                Normals.Add(normal);
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
                return $"Normal data: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"Normal count: {Normals.Count}");
            
            // Show first few normals
            int displayCount = Math.Min(Normals.Count, 5);
            for (int i = 0; i < displayCount; i++)
            {
                var n = Normals[i];
                sb.AppendLine($"  Normal {i}: ({n.X:F6}, {n.Y:F6}, {n.Z:F6})");
            }
            
            if (Normals.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Normals.Count - displayCount} more normals");
            }
            
            return sb.ToString();
        }
    }
} 