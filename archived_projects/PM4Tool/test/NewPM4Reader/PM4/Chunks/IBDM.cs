using System;
using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the IBDM (MDBI reversed) chunk which contains index data.
    /// </summary>
    public class IBDM : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "IBDM";

        /// <summary>
        /// Gets or sets the index or ID value.
        /// </summary>
        public uint Index { get; set; }

        /// <summary>
        /// Gets or sets the raw data of the chunk.
        /// </summary>
        public byte[] RawData { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="IBDM"/> class.
        /// </summary>
        public IBDM()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IBDM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public IBDM(BinaryReader reader)
        {
            ReadBinary(reader);
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            // Save the initial position
            long startPos = reader.BaseStream.Position;
            
            // Based on the sample data, IBDM chunks appear to be 4 bytes containing an ID or index
            if (reader.BaseStream.Length - reader.BaseStream.Position >= 4)
            {
                Index = reader.ReadUInt32();
                
                // Save raw data for any additional bytes
                long bytesRead = reader.BaseStream.Position - startPos;
                reader.BaseStream.Position = startPos;
                RawData = reader.ReadBytes((int)bytesRead);
            }
            else
            {
                // Read whatever is available
                RawData = reader.ReadBytes((int)(reader.BaseStream.Length - reader.BaseStream.Position));
            }
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteBinary(BinaryWriter writer)
        {
            if (RawData.Length > 0)
            {
                writer.Write(RawData);
            }
            else
            {
                writer.Write(Index);
            }
        }
        
        /// <summary>
        /// Returns a human-readable representation of the chunk data.
        /// </summary>
        /// <returns>A string representing the chunk content.</returns>
        public override string ToString()
        {
            return $"Index: 0x{Index:X8} ({Index})";
        }
    }
} 