using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using WoWMapConverter.Core.Formats.Shared;

namespace WoWMapConverter.Core.Formats.LichKing
{
    /// <summary>
    /// Ghidra-verified MCNK parser for WotLK 3.3.5.
    /// Subchunks are discovered by sequential FourCC scanning from offset 0x80
    /// (after the 128-byte header), NOT by header offset fields.
    /// See: terrain-loading-335-mcnk-deep-dive.md
    /// </summary>
    public class Mcnk
    {
        public const string Signature = "MCNK";
        public McnkHeader Header;
        public float[] Heightmap; // MCVT
        public Mcal AlphaMaps; // MCAL (3.3.5 flag-based decode)
        public byte[] McalRawData; // MCAL raw bytes (for Alpha-style sequential decode)
        public byte[] MclyRawData; // MCLY raw bytes (for Alpha-style sequential decode)
        public List<MclyEntry> TextureLayers; // MCLY
        public byte[] GeneratedNormals; // To store normals if needed

        // Raw subchunks discovered by FourCC scan
        public byte[] McvtData;
        public byte[] MccvData;  // MCV? vertex color
        public byte[] McnrData;
        public byte[] McshData;
        public byte[] MclqData;  // MCLQ legacy liquid

        public Mcnk(byte[] data)
        {
            Parse(data);
        }

        private void Parse(byte[] data)
        {
            if (data.Length < 128)
                throw new InvalidDataException("MCNK data too short for header");

            Header = ReadHeader(data);
            ScanSubchunks(data);
        }

        /// <summary>
        /// Sequential FourCC scan starting at offset 0x80.
        /// Matches Ghidra FUN_007c3a10 behavior exactly.
        /// </summary>
        private static int _diagCount = 0;

        private void ScanSubchunks(byte[] data)
        {
            int pos = 0x80;
            int remaining = data.Length - 0x80;
            bool diag = (_diagCount++ == 0);
            var dl = diag ? new System.Collections.Generic.List<string>() : null;

            while (remaining > 8)
            {
                uint fourcc = BitConverter.ToUInt32(data, pos);
                uint size = BitConverter.ToUInt32(data, pos + 4);
                int dataStart = pos + 8;

                if (diag)
                {
                    string f = (pos + 4 <= data.Length) ? System.Text.Encoding.ASCII.GetString(data, pos, 4) : "????";
                    dl.Add($"pos=0x{pos:X} fcc='{f}' (0x{fourcc:X8}) size={size} rem={remaining}");
                }

                if (size > (uint)remaining - 8)
                {
                    if (diag) dl.Add($"BREAK: size={size} > rem-8={(uint)remaining - 8}");
                    break;
                }

                switch (fourcc)
                {
                    case 0x4D435654: // MCVT
                        Heightmap = ReadFloats(data, dataStart, size);
                        break;

                    case 0x4D434E52: // MCNR
                    {
                        // Ghidra-verified: client always consumes 0x1C0 bytes for MCNR
                        // regardless of declared size. The declared size is often smaller
                        // (e.g. 435 vs 448), and the gap is padding. We must advance by
                        // the larger value so the scan lands on the next real FourCC.
                        int mcnrConsumed = (int)Math.Max(size, 0x1C0);
                        int readSize = (int)Math.Min(mcnrConsumed, data.Length - dataStart);
                        if (readSize > 0)
                        {
                            McnrData = new byte[Math.Min(readSize, 0x1C0)];
                            Array.Copy(data, dataStart, McnrData, 0, McnrData.Length);
                        }
                        // Override advance to use the actual consumed size
                        pos = dataStart + mcnrConsumed;
                        remaining = data.Length - pos;
                        continue; // skip normal advance at bottom of loop
                    }

                    case 0x4D434C59: // MCLY
                        TextureLayers = ReadMclyData(data, dataStart, size);
                        // Also keep raw bytes for Alpha-style decode path
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MclyRawData = new byte[size];
                            Array.Copy(data, dataStart, MclyRawData, 0, (int)size);
                        }
                        break;

                    case 0x4D43414C: // MCAL
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            var mcalBytes = new byte[size];
                            Array.Copy(data, dataStart, mcalBytes, 0, (int)size);
                            McalRawData = mcalBytes; // Always keep raw
                            AlphaMaps = new Mcal(mcalBytes); // 3.3.5 flag-based wrapper
                        }
                        break;

                    case 0x4D435348: // MCSH
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McshData = new byte[size];
                            Array.Copy(data, dataStart, McshData, 0, (int)size);
                        }
                        break;

                    case 0x4D434C51: // MCLQ
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MclqData = new byte[size];
                            Array.Copy(data, dataStart, MclqData, 0, (int)size);
                        }
                        break;

                    case 0x4D434356: // MCCV vertex color
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MccvData = new byte[size];
                            Array.Copy(data, dataStart, MccvData, 0, (int)size);
                        }
                        break;
                }

                pos = dataStart + (int)size;
                remaining = data.Length - pos;
            }

            if (diag && dl != null)
            {
                dl.Add($"RESULT: MCVT={Heightmap != null} MCNR={McnrData != null} MCLY={TextureLayers?.Count ?? -1} MCAL={McalRawData != null}({McalRawData?.Length ?? 0}) MCSH={McshData != null} MCLQ={MclqData != null}");
                try { File.WriteAllLines(Path.Combine(Path.GetTempPath(), "mcnk_scan.txt"), dl); } catch { }
            }
        }

        /// <summary>
        /// Read header fields from raw bytes.
        /// Ghidra-verified offsets from data start (FUN_007c64b0).
        /// </summary>
        private static McnkHeader ReadHeader(byte[] data)
        {
            var h = new McnkHeader();
            h.Flags = (McnkFlags)BitConverter.ToUInt32(data, 0x00);
            h.IndexX = BitConverter.ToUInt32(data, 0x04);
            h.IndexY = BitConverter.ToUInt32(data, 0x08);
            h.Layers = BitConverter.ToUInt32(data, 0x0C);
            h.DoodadRefs = BitConverter.ToUInt32(data, 0x10);
            h.AreaId = BitConverter.ToUInt32(data, 0x34);
            h.MapObjRefs = BitConverter.ToUInt32(data, 0x38);
            h.Holes = BitConverter.ToUInt16(data, 0x3C);

            if (data.Length >= 0x7C)
            {
                h.Position = new float[] {
                    BitConverter.ToSingle(data, 0x70), // Z (base height)
                    BitConverter.ToSingle(data, 0x74), // X
                    BitConverter.ToSingle(data, 0x78)  // Y
                };
            }
            else
            {
                h.Position = new float[] { 0f, 0f, 0f };
            }

            return h;
        }

        private static float[] ReadFloats(byte[] data, int offset, uint size)
        {
            int count = (int)(size / 4);
            var floats = new float[count];
            for (int i = 0; i < count; i++)
                floats[i] = BitConverter.ToSingle(data, offset + i * 4);
            return floats;
        }

        private static List<MclyEntry> ReadMclyData(byte[] data, int offset, uint size)
        {
            var list = new List<MclyEntry>();
            int end = offset + (int)size;
            int pos = offset;
            while (pos + 16 <= end && pos + 16 <= data.Length)
            {
                list.Add(new MclyEntry
                {
                    TextureId = BitConverter.ToUInt32(data, pos),
                    Flags = (MclyFlags)BitConverter.ToUInt32(data, pos + 4),
                    AlphaMapOffset = BitConverter.ToUInt32(data, pos + 8),
                    EffectId = BitConverter.ToUInt32(data, pos + 12)
                });
                pos += 16;
            }
            return list;
        }
    }

    public struct McnkHeader
    {
        public McnkFlags Flags;
        public uint IndexX;
        public uint IndexY;
        public uint Layers;
        public uint DoodadRefs;
        public uint AreaId;
        public uint MapObjRefs;
        public ushort Holes;
        public float[] Position; // [0]=Z(height), [1]=X, [2]=Y at data offset 0x70
    }

    [Flags]
    public enum McnkFlags : uint
    {
        HasShadows = 0x1,
        Impassable = 0x2,
        River = 0x4,
        Ocean = 0x8,
        HasMagma = 0x10,
        HasSlime = 0x20,
        HasMccv = 0x40,
        HasBakedShadows = 0x20000
    }
}
