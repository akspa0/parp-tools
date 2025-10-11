using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the IPSM (MSPI reversed) chunk which contains index data.
    /// </summary>
    public class IPSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "IPSM";
        
        /// <summary>
        /// Gets the list of indices.
        /// </summary>
        public List<uint> Indices { get; } = new List<uint>();
        
        /// <summary>
        /// Gets a value indicating whether the chunk data has been parsed.
        /// </summary>
        public bool HasBeenParsed => IsParsed;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="IPSM"/> class.
        /// </summary>
        public IPSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="IPSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public IPSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the index data from the raw binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Indices.Clear();
            
            // Determine how many indices we have
            int bytesAvailable = (int)reader.BaseStream.Length;
            int indexCount = bytesAvailable / 4; // Each index is 4 bytes (uint)
            
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
                return $"Index data: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"Index count: {Indices.Count}");
            
            // Show the first few indices
            int displayCount = Math.Min(Indices.Count, 15);
            for (int i = 0; i < displayCount; i++)
            {
                sb.Append($"{Indices[i]} ");
                
                // Line break every 5 indices for readability
                if ((i + 1) % 5 == 0)
                {
                    sb.AppendLine();
                }
            }
            
            if (Indices.Count > displayCount)
            {
                sb.AppendLine($"\n  ... and {Indices.Count - displayCount} more indices");
            }
            
            return sb.ToString();
        }
    }
} 