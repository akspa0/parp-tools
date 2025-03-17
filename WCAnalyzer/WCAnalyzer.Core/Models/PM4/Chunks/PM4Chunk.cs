using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// Base class for PM4 chunks containing common functionality.
    /// </summary>
    public abstract class PM4Chunk : IIFFChunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public abstract string Signature { get; }

        /// <summary>
        /// Gets the binary data of the chunk.
        /// </summary>
        public byte[] Data { get; protected set; } = Array.Empty<byte>();

        /// <summary>
        /// Gets the size of the data in bytes.
        /// </summary>
        public uint Size => (uint)Data.Length;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Chunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        protected PM4Chunk(byte[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            ReadData();
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected abstract void ReadData();

        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        /// <returns>The chunk's signature.</returns>
        public string GetSignature() => Signature;

        /// <summary>
        /// Loads the chunk data from a byte array.
        /// </summary>
        /// <param name="data">The binary data to load from.</param>
        public void LoadBinaryData(byte[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            ReadData();
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public void ReadData(BinaryReader reader, uint size)
        {
            if (reader == null)
                throw new ArgumentNullException(nameof(reader));

            Data = reader.ReadBytes((int)size);
            ReadData();
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteData(BinaryWriter writer)
        {
            if (writer == null)
                throw new ArgumentNullException(nameof(writer));

            writer.Write(Data);
        }
    }
} 