using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics; // Assuming Vector3 will be needed later for coordinate transformations
using Warcraft.NET.Files.Interfaces;
using System.Runtime.InteropServices; // Required for BitConverter

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents a vertex entry in the MSVT chunk.
    /// Note the YXZ ordering from the file.
    /// Z component is treated as a float based on analysis.
    /// </summary>
    public struct MsvtVertex
    {
        public int Y { get; set; } // Read first
        public int X { get; set; } // Read second
        public int Z { get; set; } // Read third as int

        // Size in file is 12 bytes (int, int, int)
        public const int StructSize = 12;

        public override string ToString() => $"(X:{X}, Y:{Y}, Z:{Z})";

        // Constants for coordinate transformation (from documentation PD4.md)
        private const float CoordinateOffset = 17066.666f;
        private const float HeightScaleFactor = 36.0f; // Factor to divide Z by

        /// <summary>
        /// Converts the internal file coordinates (YXZ) to world coordinates (XYZ)
        /// according to PD4.md documentation (using integer Z).
        /// </summary>
        /// <returns>Vector3 representing world coordinates.</returns>
        public Vector3 ToWorldCoordinates()
        {
            // Apply YXZ -> XYZ swap, offset, and Z scaling based on PD4.md
            // Use integer Z directly in the calculation.
            return new Vector3(
                CoordinateOffset - (float)X, // World X = Offset - File X
                CoordinateOffset - (float)Y, // World Y = Offset - File Y
                (float)Z / HeightScaleFactor // World Z = Z / 36.0f
            );
        }

        /// <summary>
        /// Creates an MsvtVertex from standard world coordinates (XYZ).
        /// </summary>
        /// <param name="worldPos">Vector3 representing world coordinates.</param>
        /// <returns>MsvtVertex with internal file coordinates (YXZ).</returns>
        public static MsvtVertex FromWorldCoordinates(Vector3 worldPos)
        {
            return new MsvtVertex
            {
                 X = (int)(CoordinateOffset - worldPos.X),
                 Y = (int)(CoordinateOffset - worldPos.Y),
                 // Reverse the Z calculation: WorldZ = FileZ / Scale => FileZ = WorldZ * Scale
                 Z = (int)(worldPos.Z * HeightScaleFactor)
            };
        }
    }

    /// <summary>
    /// Represents the MSVT chunk containing geometry vertices.
    /// </summary>
    public class MSVTChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVT";
        public string GetSignature() => ExpectedSignature;

        public List<MsvtVertex> Vertices { get; private set; } = new List<MsvtVertex>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            // Size of MsvtVertex in file is 12 bytes
            return (uint)(Vertices.Count * MsvtVertex.StructSize);
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
            long endPosition = br.BaseStream.Length; // Assuming the stream contains only this chunk's data initially
            long size = endPosition - startPosition;

            if (size % MsvtVertex.StructSize != 0)
            {
                // Handle error: size is not a multiple of vertex size
                // For now, let's clear and potentially log an error or throw
                Vertices.Clear();
                // Consider adding logging here
                Console.WriteLine($"Warning: MSVT chunk size {size} is not a multiple of {MsvtVertex.StructSize} bytes. Vertex data might be corrupt.");
                return; // Or throw an exception
            }

            int vertexCount = (int)(size / MsvtVertex.StructSize);
            Vertices = new List<MsvtVertex>(vertexCount);

            for (int i = 0; i < vertexCount; i++)
            {
                if (br.BaseStream.Position + MsvtVertex.StructSize > br.BaseStream.Length)
                {
                    // Log error or throw exception
                    Console.WriteLine($"Warning: MSVT chunk unexpected end of stream at vertex {i}. Expected {vertexCount} vertices.");
                    break;
                }
                // Read in YXZ order, Z as int
                var vertex = new MsvtVertex
                {
                    Y = br.ReadInt32(),
                    X = br.ReadInt32(),
                    Z = br.ReadInt32() // Read Z as int (was ReadSingle)
                };
                Vertices.Add(vertex);
            }

            // Ensure we read exactly the expected number of bytes
            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size)
            {
                // Handle error: Did not read the entire chunk data
                 Console.WriteLine($"Warning: MSVT chunk read {bytesRead} bytes, expected {size} bytes.");
                 // Decide how to handle this - potentially throw, log, or adjust position.
                 // For robustness, ensure the stream position is at the expected end if possible.
                 // br.BaseStream.Position = startPosition + size; // Use with caution
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var vertex in Vertices)
            {
                // Write in YXZ order, Z as int
                bw.Write(vertex.Y);
                bw.Write(vertex.X);
                bw.Write(vertex.Z); // Write Z as int (was float)
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MSVT Chunk [{Vertices.Count} Vertices]";
        }
    }
} 