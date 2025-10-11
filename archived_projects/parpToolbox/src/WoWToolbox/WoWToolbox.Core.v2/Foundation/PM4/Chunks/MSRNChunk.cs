using System;
using System.Collections.Generic;
using System.IO;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.v2.Foundation.Vectors;
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSRN chunk containing Mesh Surface Referenced Normals.
    /// </summary>
    public class MSRNChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSRN";
        public string GetSignature() => ExpectedSignature;

        public List<C3Vectori> Normals { get; private set; } = new();

        private const float InvFixedScale = 1f / 8192f; // 2^13 – empirical scale for unit normals

        /// <summary>
        /// Returns the normals as floating-point vectors (roughly unit length) by applying 1/8192 scaling.
        /// </summary>
        public List<Vector3> GetFloatNormals()
        {
            var list = new List<Vector3>(Normals.Count);
            foreach (var n in Normals)
            {
                list.Add(new Vector3(n.X * InvFixedScale, n.Y * InvFixedScale, n.Z * InvFixedScale));
            }
            return list;
        }

        /// <inheritdoc/>
        public uint GetSize()
        {
            // Size is dynamic based on the number of normals.
            return (uint)Normals.Count * C3Vectori.Size;
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
            long endPosition = br.BaseStream.Length;
            long size = endPosition - startPosition;

            if (size < 0) throw new InvalidOperationException("Stream size is negative.");
            if (size % C3Vectori.Size != 0)
            {
                Normals.Clear();
                Console.WriteLine($"Warning: MSRN chunk size {size} is not a multiple of {C3Vectori.Size} bytes. Data might be corrupt.");
                size -= (size % C3Vectori.Size); // Process only complete entries
            }

            int count = (int)(size / C3Vectori.Size);
            Normals = new List<C3Vectori>(count);

            for (int i = 0; i < count; i++)
            {
                if (br.BaseStream.Position + C3Vectori.Size > br.BaseStream.Length)
                {
                    Console.WriteLine($"Warning: MSRN chunk unexpected end of stream at normal {i}. Read {Normals.Count} normals out of expected {count}.");
                    break;
                }
                var normal = new C3Vectori();
                normal.Load(br);
                Normals.Add(normal);
            }

            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size + (size % C3Vectori.Size)) // Check against original expected size if padded
            {
                 Console.WriteLine($"Warning: MSRN chunk read {bytesRead} bytes, expected to process based on size {size}. Original size reported by header might have padding or corruption.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var normal in Normals)
            {
                normal.Write(bw); // Assuming C3Vectori has a Write method
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MSRN Chunk [{Normals.Count} Normals]";
        }
    }
}
