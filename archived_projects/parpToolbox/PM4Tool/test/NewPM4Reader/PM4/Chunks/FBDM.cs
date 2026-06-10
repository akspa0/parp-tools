using System;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the FBDM (MDBF reversed) chunk which contains file path data.
    /// </summary>
    public class FBDM : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "FBDM";

        /// <summary>
        /// Gets or sets the file path.
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the raw data of the chunk.
        /// </summary>
        public byte[] RawData { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="FBDM"/> class.
        /// </summary>
        public FBDM()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FBDM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public FBDM(BinaryReader reader)
        {
            ReadBinary(reader);
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            // Store current position to calculate total size
            long startPos = reader.BaseStream.Position;
            
            // Read data until we find a null terminator or end of chunk
            using var memoryStream = new MemoryStream();
            byte b;
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                b = reader.ReadByte();
                memoryStream.WriteByte(b);
                
                // Check if we've reached the end of the string
                if (b == 0)
                    break;
            }
            
            // Store the raw data
            RawData = memoryStream.ToArray();
            
            // Try to interpret as a string
            if (RawData.Length > 0)
            {
                try
                {
                    // The data appears to be a null-terminated string
                    FilePath = Encoding.ASCII.GetString(RawData).TrimEnd('\0');
                }
                catch
                {
                    // If we can't interpret as a string, just leave the raw data
                    FilePath = $"<Binary data of {RawData.Length} bytes>";
                }
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
            else if (!string.IsNullOrEmpty(FilePath))
            {
                // Write file path as null-terminated string
                writer.Write(Encoding.ASCII.GetBytes(FilePath));
                writer.Write((byte)0); // Null terminator
            }
        }
        
        /// <summary>
        /// Returns a human-readable representation of the chunk data.
        /// </summary>
        /// <returns>A string representing the chunk content.</returns>
        public override string ToString()
        {
            if (!string.IsNullOrEmpty(FilePath))
            {
                return $"File Path: {FilePath}";
            }
            else
            {
                return $"Raw Data: {RawData.Length} bytes";
            }
        }
    }
} 