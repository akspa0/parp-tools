using System;
using System.IO;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents the MSHD chunk containing header information for the PM4 file.
    /// Based on documentation at chunkvault/chunks/PM4/M009_MSHD.md
    /// </summary>
    public class MSHDChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSHD";
        public string GetSignature() => ExpectedSignature;

        public uint Unknown_0x00 { get; set; } // Meaning TBD.
        public uint Unknown_0x04 { get; set; } // Meaning TBD.
        public uint Unknown_0x08 { get; set; } // Meaning TBD.
        public uint Unknown_0x0C { get; set; } // Meaning TBD.
        public uint Unknown_0x10 { get; set; } // Meaning TBD.
        public uint Unknown_0x14 { get; set; } // Meaning TBD.
        public uint Unknown_0x18 { get; set; } // Meaning TBD.
        public uint Unknown_0x1C { get; set; } // Meaning TBD.

        public const uint ExpectedSize = 32;

        /// <inheritdoc/>
        public uint GetSize()
        {
            return ExpectedSize;
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));
            if (chunkData.Length != ExpectedSize)
            {
                 Console.WriteLine($"Warning: MSHD chunk size is {chunkData.Length}, expected {ExpectedSize}. Data might be corrupt or incomplete.");
                 // Decide how to handle incorrect size: throw, pad, or truncate?
                 // For now, we'll attempt to read assuming the first 32 bytes are valid if > 32, 
                 // or throw if < 32.
                 if (chunkData.Length < ExpectedSize)
                 {
                    throw new InvalidDataException($"MSHD chunk size {chunkData.Length} is less than expected {ExpectedSize}.");
                 }
                 // If larger, log warning and proceed with first 32 bytes.
            }

            using var ms = new MemoryStream(chunkData, 0, (int)Math.Min(chunkData.Length, ExpectedSize));
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            // Note: Size validation is primarily done in LoadBinaryData
            // This Load assumes the stream passed is exactly the expected size.
            long startPos = br.BaseStream.Position;

            Unknown_0x00 = br.ReadUInt32();
            Unknown_0x04 = br.ReadUInt32();
            Unknown_0x08 = br.ReadUInt32();
            Unknown_0x0C = br.ReadUInt32();
            Unknown_0x10 = br.ReadUInt32();
            Unknown_0x14 = br.ReadUInt32();
            Unknown_0x18 = br.ReadUInt32();
            Unknown_0x1C = br.ReadUInt32();
            
            long bytesRead = br.BaseStream.Position - startPos;
            if (bytesRead != ExpectedSize)
            {
                // This case should ideally not happen if LoadBinaryData handles size correctly.
                 Console.WriteLine($"Warning: MSHD chunk read {bytesRead} bytes, expected {ExpectedSize}.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)ExpectedSize);
            using var bw = new BinaryWriter(ms);

            bw.Write(Unknown_0x00);
            bw.Write(Unknown_0x04);
            bw.Write(Unknown_0x08);
            bw.Write(Unknown_0x0C);
            bw.Write(Unknown_0x10);
            bw.Write(Unknown_0x14);
            bw.Write(Unknown_0x18);
            bw.Write(Unknown_0x1C);

            return ms.ToArray();
        }

        public override string ToString()
        {
             return $"MSHD: [00={Unknown_0x00:X8}, 04={Unknown_0x04:X8}, 08={Unknown_0x08:X8}, 0C={Unknown_0x0C:X8}, " +
                    $"10={Unknown_0x10:X8}, 14={Unknown_0x14:X8}, 18={Unknown_0x18:X8}, 1C={Unknown_0x1C:X8}]";
        }
    }
} 