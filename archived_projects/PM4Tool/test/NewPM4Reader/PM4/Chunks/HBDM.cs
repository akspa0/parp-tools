using System;
using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the HBDM (MDBH reversed) chunk which contains header data.
    /// </summary>
    public class HBDM : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "HBDM";

        /// <summary>
        /// Gets or sets the header value.
        /// </summary>
        public uint Value { get; set; }

        /// <summary>
        /// Gets or sets the raw data of the chunk.
        /// </summary>
        public byte[] RawData { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="HBDM"/> class.
        /// </summary>
        public HBDM()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="HBDM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public HBDM(BinaryReader reader)
        {
            ReadBinary(reader);
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            // Based on the sample data, HBDM chunks appear to be 4 bytes
            if (reader.BaseStream.Length - reader.BaseStream.Position >= 4)
            {
                Value = reader.ReadUInt32();
                
                // For consistency, always store the raw data
                reader.BaseStream.Position -= 4;
                RawData = reader.ReadBytes(4);
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
                writer.Write(Value);
            }
        }
        
        /// <summary>
        /// Returns a human-readable representation of the chunk data.
        /// </summary>
        /// <returns>A string representing the chunk content.</returns>
        public override string ToString()
        {
            return $"Header Value: 0x{Value:X8} ({Value})";
        }
    }
} 