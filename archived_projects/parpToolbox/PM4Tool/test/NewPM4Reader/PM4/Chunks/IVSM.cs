using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// IVSM chunk (MSVI reversed)
    /// According to documentation, these might be indices for quads or n-gons, not just triangles.
    /// </summary>
    public class IVSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "IVSM";
        
        /// <summary>
        /// Raw indices for the IVSM chunk.
        /// According to documentation, these might be indices for quads or n-gons, not just triangles.
        /// The n-gon count and first index might be defined in MSUR where _0x01 is count and _0x14 is offset.
        /// </summary>
        public List<uint> Indices { get; } = new List<uint>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="IVSM"/> class.
        /// </summary>
        public IVSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="IVSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public IVSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the IVSM data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Indices.Clear();
            
            // Each index is 4 bytes (uint32_t)
            int bytesAvailable = (int)reader.BaseStream.Length;
            int indexCount = bytesAvailable / 4;
            
            for (int i = 0; i < indexCount; i++)
            {
                uint index = reader.ReadUInt32();
                Indices.Add(index);
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
                return $"IVSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"IVSM index count: {Indices.Count}");
            sb.AppendLine("Note: Documentation suggests these may be quad or n-gon indices, not triangles");
            sb.AppendLine("      Related MSUR chunk may define index counts and offsets");
            
            // Display the first few indices
            int displayCount = Math.Min(Indices.Count, 20);
            for (int i = 0; i < displayCount; i += 4)
            {
                sb.Append($"  Indices {i}-{Math.Min(i + 3, displayCount - 1)}: ");
                
                for (int j = 0; j < 4 && i + j < displayCount; j++)
                {
                    sb.Append($"{Indices[i + j]} ");
                }
                
                sb.AppendLine();
            }
            
            if (Indices.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Indices.Count - displayCount} more indices");
            }
            
            return sb.ToString();
        }
    }
} 