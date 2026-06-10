using System.IO;
using Warcraft.NET;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Files;

namespace WoWToolbox.Core.Legacy.Liquid
{
    /// <summary>
    /// Helper class for reading MCLQ chunks from ADT files
    /// </summary>
    public static class MCLQReader
    {
        /// <summary>
        /// Reads an MCLQ chunk from the given binary reader
        /// </summary>
        /// <param name="reader">The binary reader to read from</param>
        /// <returns>The parsed MCLQ chunk</returns>
        public static MCLQChunk ReadChunk(BinaryReader reader)
        {
            var chunk = new MCLQChunk();
            chunk.Deserialize(reader);
            return chunk;
        }

        /// <summary>
        /// Reads an MCLQ chunk from the given IFF chunk
        /// </summary>
        /// <param name="chunk">The IFF chunk to read from</param>
        /// <returns>The parsed MCLQ chunk</returns>
        public static MCLQChunk ReadChunk(IIFFChunk chunk)
        {
            if (chunk.GetSignature() != MCLQChunk.Signature)
            {
                throw new InvalidDataException($"Invalid chunk signature. Expected {MCLQChunk.Signature}, got {chunk.GetSignature()}");
            }

            if (chunk is IBinarySerializable serializableChunk)
            {
                return new MCLQChunk(serializableChunk.Serialize());
            }
            else
            {
                throw new InvalidOperationException("Chunk does not implement IBinarySerializable and cannot provide raw data.");
            }
        }

        /// <summary>
        /// Reads an MCLQ chunk from the given byte array
        /// </summary>
        /// <param name="data">The byte array to read from</param>
        /// <returns>The parsed MCLQ chunk</returns>
        public static MCLQChunk ReadChunk(byte[] data)
        {
            return new MCLQChunk(data);
        }
    }
} 