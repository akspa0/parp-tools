using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PM4
{
    /// <summary>
    /// Generic chunk implementation for storing raw chunk data that does not have
    /// a specific implementation. This is used as a fallback during PM4 parsing.
    /// </summary>
    public class GenericChunk : IIFFChunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public string Signature { get; private set; }

        /// <summary>
        /// Gets the raw binary data of the chunk.
        /// </summary>
        public byte[] Data { get; private set; }

        /// <summary>
        /// Gets the size of the chunk data.
        /// </summary>
        public uint Size => (uint)(Data?.Length ?? 0);

        /// <summary>
        /// Initializes a new instance of the <see cref="GenericChunk"/> class.
        /// </summary>
        /// <param name="signature">The chunk signature.</param>
        /// <param name="data">The raw binary data of the chunk.</param>
        public GenericChunk(string signature, byte[] data)
        {
            Signature = signature ?? throw new ArgumentNullException(nameof(signature));
            Data = data ?? throw new ArgumentNullException(nameof(data));
        }

        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        /// <returns>The chunk's signature.</returns>
        public string GetSignature()
        {
            return Signature;
        }

        /// <summary>
        /// Loads the chunk data from a byte array.
        /// </summary>
        /// <param name="data">The binary data to load from.</param>
        public void LoadBinaryData(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            Data = data;
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