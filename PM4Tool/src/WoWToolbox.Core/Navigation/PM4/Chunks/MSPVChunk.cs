using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Files.Structures; // Need this for C3Vector
using WoWToolbox.Core.Vectors; // May not be needed if C3Vectori isn't used

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSPV chunk containing path vertices.
    /// Reverting to C3Vector (float) based on analysis of raw bytes read as int.
    /// </summary>
    public class MSPVChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSPV";
        public string GetSignature() => ExpectedSignature;

        // Using C3Vector (float) based on data analysis
        public List<C3Vector> Vertices { get; private set; } = new List<C3Vector>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            // Size is 12 bytes per vertex (3 * float)
            return (uint)(Vertices.Count * 12);
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));

            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long size = br.BaseStream.Length - startPosition;
            int vertexSize = 12; // sizeof(float) * 3

            if (size < 0) throw new InvalidDataException("Stream size is negative.");
            if (size % vertexSize != 0)
            {
                Console.WriteLine($"Warning: MSPV chunk size {size} is not a multiple of vertex size {vertexSize}. Possible padding or corruption.");
                size -= (size % vertexSize); 
            }

            int vertexCount = (int)(size / vertexSize);
            Vertices = new List<C3Vector>(vertexCount);

            for (int i = 0; i < vertexCount; i++)
            {
                 if (br.BaseStream.Position + vertexSize > br.BaseStream.Length)
                {
                    Console.WriteLine($"Warning: MSPV chunk unexpected end of stream at vertex {i}. Read {Vertices.Count} vertices out of expected {vertexCount}.");
                    break;
                }
                // Read as floats
                var vertex = new C3Vector
                {
                    X = br.ReadSingle(),
                    Y = br.ReadSingle(),
                    Z = br.ReadSingle()
                };
                Vertices.Add(vertex);
            }

            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size && size > 0)
            {
                 Console.WriteLine($"Warning: MSPV chunk read {bytesRead} bytes, expected to process {size} bytes based on multiples of {vertexSize}.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var vertex in Vertices)
            {
                bw.Write(vertex.X);
                bw.Write(vertex.Y);
                bw.Write(vertex.Z);
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MSPV Chunk [{Vertices.Count} Vertices] (Reading as Float)";
        }
    }
} 