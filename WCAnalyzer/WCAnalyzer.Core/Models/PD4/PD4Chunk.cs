using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PD4
{
    /// <summary>
    /// Base class for PD4 chunks.
    /// </summary>
    public abstract class PD4Chunk : IIFFChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
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
        /// Initializes a new instance of the <see cref="PD4Chunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        protected PD4Chunk(byte[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            ReadData();
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected abstract void ReadData();

        /// <summary>
        /// Gets the bytes that make up this chunk.
        /// </summary>
        /// <returns>The bytes that make up this chunk.</returns>
        public byte[] GetBytes()
        {
            return Data;
        }

        /// <summary>
        /// Gets the bytes that make up this chunk, including the header.
        /// </summary>
        /// <returns>The bytes that make up this chunk, including the header.</returns>
        public byte[] GetBytesWithHeader()
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write the signature
                writer.Write(Signature.ToCharArray());

                // Write the chunk size
                writer.Write(Size);

                // Write the chunk data
                writer.Write(Data);

                return ms.ToArray();
            }
        }

        /// <summary>
        /// Deserialzes the provided binary data of the object. This is the full data block which follows the data
        /// signature and data block length.
        /// </summary>
        /// <param name="inData">The binary data containing the object.</param>
        public void LoadBinaryData(byte[] inData)
        {
            Data = inData ?? throw new ArgumentNullException(nameof(inData));
            ReadData();
        }

        /// <summary>
        /// Gets the static data signature of this data block type.
        /// </summary>
        /// <returns>A string representing the block signature.</returns>
        public string GetSignature()
        {
            return Signature;
        }
    }
} 