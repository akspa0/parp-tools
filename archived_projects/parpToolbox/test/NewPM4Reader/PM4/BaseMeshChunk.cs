using System;
using System.IO;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4
{
    /// <summary>
    /// Base class for mesh-related chunks (VPSM, IPSM, etc.).
    /// </summary>
    public abstract class BaseMeshChunk : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public abstract string Signature { get; }

        /// <summary>
        /// Gets or sets the raw data of the chunk.
        /// </summary>
        public byte[] RawData { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Gets the size of the raw data in bytes.
        /// </summary>
        public int DataSize => RawData.Length;

        /// <summary>
        /// Gets a value indicating whether the chunk has been fully parsed.
        /// </summary>
        public bool IsParsed { get; protected set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseMeshChunk"/> class.
        /// </summary>
        protected BaseMeshChunk()
        {
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public virtual void ReadBinary(BinaryReader reader)
        {
            // Store the starting position
            long startPos = reader.BaseStream.Position;
            long availableBytes = reader.BaseStream.Length - startPos;
            
            // Read the raw data
            RawData = reader.ReadBytes((int)availableBytes);
            
            // Parse the data
            try
            {
                using var memReader = new BinaryReader(new MemoryStream(RawData));
                ParseData(memReader);
                IsParsed = true;
            }
            catch (Exception)
            {
                IsParsed = false;
            }
        }

        /// <summary>
        /// Parses the chunk data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected abstract void ParseData(BinaryReader reader);

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public virtual void WriteBinary(BinaryWriter writer)
        {
            if (RawData.Length > 0)
            {
                writer.Write(RawData);
            }
        }
        
        /// <summary>
        /// Returns a human-readable summary of the chunk.
        /// </summary>
        /// <returns>A string representation of the chunk.</returns>
        public override string ToString()
        {
            return $"{GetType().Name} data: {DataSize} bytes, Parsed: {IsParsed}";
        }
        
        /// <summary>
        /// Gets a detailed description of the chunk contents.
        /// </summary>
        /// <returns>A detailed description string.</returns>
        public abstract string GetDetailedDescription();
    }
} 